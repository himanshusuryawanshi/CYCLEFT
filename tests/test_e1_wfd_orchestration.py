"""Orchestration smoke test: the full E1 pipeline on synthetic WikiFactDiff
data with a mocked model — no downloads, no real training.

Exercises the ACTUAL orchestration (prepare -> baseline -> events 1-3 with
checkpoint continuation -> per-event evaluation of test/validation/sweep
populations -> event metrics -> WFD M1/M2/M3 summary) by mocking only the
model boundary (load_base_model/attach_lora/load_adapter_for_training/
train_event/evaluate_event/load_tokenizer).
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cycleft.data.wikifactdiff import build_wikifactdiff_events
from cycleft.e1.metrics import exact_match_multi, normalize_answer, token_f1_multi

CONFIG = Path(__file__).resolve().parents[1] / "configs/e1_stream_wikifactdiff.yaml"


def _subject(sid, label=None):
    return {"id": sid, "label": label or sid,
            "description": f"desc {sid}"}


def _obj(label, decision):
    return {"id": None, "label": label, "description": "", "decision": decision}


def _row(sid, rid, *, ph_new=False, replace=False, new=(), obsolete=(),
         static=(), gen=(), neighborhood=()):
    return {
        "subject": _subject(sid),
        "relation": _subject(rid, f"relation {rid}"),
        "update_prompt": f"{sid} {rid} is ____",
        "generalization_prompts": [f"{sid} paraphrase {i}" for i in gen],
        "subject_is_ph_new": ph_new,
        "relation_is_temp_func": replace,
        "is_replace": replace,
        "subject_popularity": 1.0,
        "objects": (
            [_obj(label, "new") for label in new]
            + [_obj(label, "obsolete") for label in obsolete]
            + [_obj(label, "static") for label in static]
        ),
        "neighborhood": [
            {
                "subject": _subject(nsid),
                "dist": 0.5,
                "objects": [{"object": _obj(label, "static"),
                             "prompt": prompt}],
            }
            for nsid, label, prompt in neighborhood
        ],
    }


def _synthetic_rows():
    rows = [
        _row("W1", "Q1", ph_new=True, new=["Alpha One"], gen=[1]),
        _row("W2", "Q1", ph_new=True, new=["Beta"]),
        _row("W3", "Q2", new=["Delta"], obsolete=["OldDelta"], gen=[1]),
        _row("W4", "Q2", new=["Epsilon"]),
        _row("W5", "Q1", replace=True, new=["NewPM"], obsolete=["OldPM"],
             gen=[1]),
        _row("W6", "Q3", replace=True, new=["NewCoach"], obsolete=["OldCoach"]),
        _row("W7", "Q1", new=["Zeta"], gen=[1],
             neighborhood=[("N1", "NeighborLabel", "N1 Q1 is ____")]),
    ]
    # persistence probe (obsolete-only)
    rows.append(_row("W8", "Q3", obsolete=["Vanished"]))
    # keep every subject in the train partition for a compact run: the
    # partition is subject-disjoint by salt, so force small fractions
    return rows


class _FakeTokenizer:
    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "tokenizer.json").write_text("{}")


class _FakeModel:
    def __init__(self):
        self.layer = torch.nn.Linear(2, 2)
        self.config = type("C", (), {"use_cache": True})()

    def parameters(self):
        return self.layer.parameters()

    def to(self, device):
        return self

    def train(self):
        return self

    def eval(self):
        return self

    def print_trainable_parameters(self):
        pass

    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "adapter.safetensors").write_text("{}")

    def __call__(self, *args, **kwargs):
        return self


import torch  # noqa: E402  (after fixture helpers; used by _FakeModel)


def _mock_evaluate_event(model, tokenizer, eval_rows, config, log=None):
    """Mirrors the real evaluate_event contract incl. perplexity disable
    and metadata propagation, using the real multi-reference metrics."""
    records = []
    for index, row in enumerate(eval_rows):
        question = str(row["question"])
        reference = str(row["object"])
        prediction = reference if reference else f"guess-{index}"
        valid = row.get("valid_answers") or []
        em = exact_match_multi(prediction, valid) if valid else \
            float(normalize_answer(prediction) == normalize_answer(reference))
        f1 = token_f1_multi(prediction, valid) if valid else 0.0
        record = {
            "example_index": index,
            "question": question,
            "reference": reference,
            "raw_prediction": prediction,
            "prediction": prediction,
            "prediction_normalized": normalize_answer(prediction),
            "exact_match": em,
            "token_f1": f1,
        }
        if valid:
            record["valid_answers"] = list(valid)
        for key in ("fact_key", "eval_family", "prompt_kind", "partition",
                    "event_tag", "new_answers", "static_answers",
                    "obsolete_answers"):
            if key in row:
                record[key] = row[key]
        records.append(record)
    disabled = any(row.get("perplexity_disabled") for row in eval_rows)
    if disabled:
        loss_stats = {
            "eval_loss": None,
            "perplexity": None,
            "perplexity_note": ("disabled: WikiFactDiff multi-reference "
                                "likelihood is not implemented"),
        }
    else:
        loss_stats = {"eval_loss": 1.0, "perplexity": 3.0}
    n = len(records)
    metrics = {
        "examples": n,
        "exact_match": (sum(r["exact_match"] for r in records) / n) if n else 0.0,
        "token_f1": (sum(r["token_f1"] for r in records) / n) if n else 0.0,
        **loss_stats,
        "generation_seconds": 0.0,
    }
    return {"metrics": metrics, "records": records}


def _mock_train_event(model, tokenizer, train_rows, config, log=None,
                      on_step=None):
    return {
        "examples_scheduled": len(train_rows),
        "examples_skipped": 0,
        "optimizer_steps": len(train_rows),
        "losses_per_step": [0.5] * len(train_rows),
        "average_loss": 0.5,
        "final_loss": 0.5,
        "training_seconds": 0.0,
        "trainable_parameters": 1916928,
        "total_parameters": 1235814400,
        "trainable_percentage": 0.155,
        "answer_tokens_trained": 100,
        "total_tokens_trained": 5000,
        "memory_after": {"rss_gb": 1.0},
    }


@pytest.fixture()
def orchestration_run(monkeypatch, tmp_path):
    """Runs the real pipeline end-to-end with mocked model boundaries."""
    import cycleft.e1.pipeline as pipeline

    rows = _synthetic_rows()
    events, manifest = build_wikifactdiff_events(
        rows, seed=42, mode="pilot",
        adapter_config={"pilot_eval_per_event": 10,
                        # small fractions so the 7-subject fixture populates
                        # train, validation AND test (the default 80/10/10
                        # puts every subject of this tiny set in train)
                        "split_fractions": (0.34, 0.67),
                        "sweep_limits": {"m1": 10, "persistence": 10,
                                         "neighborhood": 10}})

    monkeypatch.setattr(pipeline, "prepare_event_datasets",
                        lambda train_limit, seed, dataset, dataset_config,
                        mode=None: (events, manifest))
    monkeypatch.setattr(pipeline, "load_tokenizer", lambda model_id: _FakeTokenizer())
    monkeypatch.setattr(pipeline, "load_base_model", lambda cfg: _FakeModel())
    monkeypatch.setattr(pipeline, "attach_lora", lambda base, cfg, dev: _FakeModel())
    monkeypatch.setattr(pipeline, "load_adapter_for_training",
                        lambda base, ckpt, cfg, dev: _FakeModel())
    monkeypatch.setattr(pipeline, "train_event", _mock_train_event)
    monkeypatch.setattr(pipeline, "evaluate_event", _mock_evaluate_event)

    run_id = "wfd_smoke"
    summary = pipeline.run_e1(CONFIG, mode="pilot", run_id=run_id, seed=42,
                              device="cpu")
    return pipeline, run_id, summary


def test_orchestration_end_to_end(orchestration_run):
    module, run_id, event_metrics = orchestration_run
    results = module.RESULTS_ROOT / run_id
    checkpoints = module.CHECKPOINTS_ROOT / run_id

    # 1-3. all artifacts exist and parse (JSON docs vs JSONL logs)
    json_docs = ("dataset_manifest.json", "baseline_metrics.json",
                 "config.yaml", "runtime_metadata.json",
                 "event_metrics.json", "validation_metrics.json",
                 "wfd_summary.json",
                 "sweep_metrics_m1_trained_fact_paraphrases.json",
                 "sweep_metrics_m3_persistence_probes.json",
                 "sweep_metrics_m3_neighborhood_probes.json")
    jsonl_logs = ("predictions.jsonl", "training_log.jsonl")
    for name in json_docs:
        payload = json.loads((results / name).read_text())
        assert payload is not None, name
    for name in jsonl_logs:
        lines = [json.loads(line) for line in
                 (results / name).read_text().splitlines() if line.strip()]
        assert lines, f"{name} is empty"
    for event_id in (1, 2, 3):
        assert json.loads((results / f"event_{event_id}_eval.json").read_text())
        assert (checkpoints / f"event_{event_id}" / "adapter.safetensors").is_file()

    # 4. validation metrics are separate from final test metrics
    val = json.loads((results / "validation_metrics.json").read_text())
    test = json.loads((results / "event_1_eval.json").read_text())
    assert set(val) >= {"base", "after_event_1"}
    # validation rows are their own population (never the test rows)
    predictions = [json.loads(l) for l in
                   (results / "predictions.jsonl").read_text().splitlines()]
    val_records = [p for p in predictions if p.get("population") == "validation"]
    test_records = [p for p in predictions if p.get("population") == "test"]
    assert val_records and test_records
    assert not ({p["question"] for p in val_records}
                & {p["question"] for p in test_records})

    # 5. prediction metadata preserved
    wfd_records = [p for p in test_records if "fact_key" in p]
    assert wfd_records
    assert all("eval_family" in p and "valid_answers" in p
               for p in wfd_records)

    # 6. no empty reference evaluated; perplexity explicitly disabled
    assert all(p["reference"] != "" for p in predictions)
    for state in val.values():
        for metrics in state.values():
            assert metrics["eval_loss"] is None
            assert "disabled" in metrics["perplexity_note"]

    # 7-10. WFD summary: M1/M2/M3 generated with provenance
    summary = json.loads((results / "wfd_summary.json").read_text())
    assert "m1_trained_fact_paraphrases" in summary["families"]
    assert "m3_persistence_probes" in summary["families"]
    assert "m3_neighborhood_probes" in summary["families"]
    m1 = summary["m1_update_efficacy_final"]
    assert m1["n"] > 0 and 0.0 <= m1["update_efficacy"] <= 1.0
    m2 = summary["m2_heldout_transfer"]
    assert m2["baseline_conditioned"] is not None
    assert "initially_unknown" in m2["baseline_conditioned"]
    assert "m3_neighborhood_retention" in summary
    assert "m3_persistence" in summary
    # sweep provenance recorded
    assert summary["sweep_selection"]["m1_trained_fact_paraphrases"]["seed"] == 42

    # event metrics matrices complete (base + 3 events)
    assert event_metrics["states"] == ["base", "after_event_1",
                                       "after_event_2", "after_event_3"]
    assert len(event_metrics["accuracy_matrix"]) == 4
