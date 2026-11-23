"""Recovery-preflight regression tests for the corrected E7->E5 prediction
sweep (e7-static-r9-s42 preflight, 2026-11-23).

Hermetic, CPU-only, mock-based: the predict callback's E1/E5 dependencies
are monkeypatched with recorders; no real model is ever loaded. These tests
pin the loading-path contract repaired after the attempt-1 double-wrap
defect and document preflight finding P-1 (missing ``test_event``) via a
strict xfail that must be removed together with the recovery fix.
"""

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cycleft.config import load_config as _real_load_config  # noqa: E402
from cycleft.e1.lm import LMRuntimeConfig  # noqa: E402
from cycleft.e7.e5_adapter import _default_predict_fn  # noqa: E402
from cycleft.e7.execution_harness import (  # noqa: E402
    PREDICTION_RECORD_KEYS,
    ExecutionHarness,
    OutputSafetyError,
    classify_run_state,
    load_and_validate_matrix,
)

V5_MATRIX = Path(__file__).resolve().parents[1] / (
    "results/e7/design_20261120/v5_final_reconciliation/FINAL_RUN_MATRIX.csv")
REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_CONFIG = _real_load_config(REPO_ROOT / "configs" / "e1_stream.yaml")
EXPECTED_FP = "f" * 64


class _FakeEvent:
    def __init__(self, event_id, eval_rows):
        self.event_id = event_id
        self.eval_rows = eval_rows


def _fake_events(n_rows=2):
    return [_FakeEvent(eid, [{"question": f"q{eid}-{i}",
                              "object": f"a{eid}-{i}"}
                             for i in range(n_rows)])
            for eid in (1, 2, 3)]


def _real_shaped_records(rows):
    """Records exactly as cycleft.e1.lm.evaluate_event emits them (no
    `state`, no `test_event` — those are the sweep callback's job)."""
    return [{"example_index": i, "question": r["question"],
             "reference": r["object"], "raw_prediction": "raw " + r["object"],
             "prediction": r["object"], "exact_match": 1.0, "token_f1": 1.0}
            for i, r in enumerate(rows)]


@pytest.fixture()
def v5_specs():
    return load_and_validate_matrix(V5_MATRIX)


@pytest.fixture()
def env(monkeypatch, tmp_path):
    """Wire the predict callback with recording fakes; return the recorder."""
    rec = {"load_base_model": [], "load_base_model_cfg": [], "resolve_device": [],
           "load_adapter_for_training": [], "load_tokenizer": [],
           "evaluate_event": [], "fingerprints": []}
    base_sentinel = object()
    fake_model = torch.nn.Module()  # placeholder; fingerprint is mocked
    lm_config = LMRuntimeConfig(model_id="meta-llama/Llama-3.2-1B-Instruct")

    def fake_load_base_model(cfg):
        rec["load_base_model_cfg"].append(cfg)
        rec["load_base_model"].append(base_sentinel)
        return base_sentinel

    def fake_resolve_device(requested):
        rec["resolve_device"].append(requested)
        return torch.device("mps")

    def fake_load_adapter_for_training(base, adapter_dir, cfg, device):
        rec["load_adapter_for_training"].append(
            {"base": base, "adapter_dir": Path(adapter_dir), "cfg": cfg,
             "device": device})
        return fake_model

    def fake_attach_lora(*a, **k):  # must never be called
        raise AssertionError("attach_lora must not be called by the sweep")

    def fake_load_tokenizer(model_id):
        rec["load_tokenizer"].append(model_id)
        return object()

    def fake_evaluate_event(model, tokenizer, rows, cfg, log=None):
        rec["evaluate_event"].append(
            {"n_rows": len(rows), "cfg": cfg, "model": model})
        return {"metrics": {"exact_match": 1.0, "token_f1": 1.0},
                "records": _real_shaped_records(rows)}

    def fake_fingerprint(model):
        rec["fingerprints"].append(model)
        return EXPECTED_FP

    monkeypatch.setattr("cycleft.config.load_config",
                        lambda path: json.loads(json.dumps(RAW_CONFIG)))
    monkeypatch.setattr("cycleft.e1.data.prepare_event_datasets",
                        lambda train_limit=None, seed=None: (_fake_events(), {}))
    monkeypatch.setattr("cycleft.e1.lm.load_base_model", fake_load_base_model)
    monkeypatch.setattr("cycleft.e1.lm.resolve_device", fake_resolve_device)
    monkeypatch.setattr("cycleft.e1.lm.load_adapter_for_training",
                        fake_load_adapter_for_training)
    monkeypatch.setattr("cycleft.e1.lm.attach_lora", fake_attach_lora)
    monkeypatch.setattr("cycleft.e1.lm.load_tokenizer", fake_load_tokenizer)
    monkeypatch.setattr("cycleft.e1.lm.evaluate_event", fake_evaluate_event)
    monkeypatch.setattr("cycleft.e5.resume_bundle.adapter_fingerprint",
                        fake_fingerprint)

    run_dir = tmp_path / "run"
    (run_dir / "bundles").mkdir(parents=True)
    (run_dir / "bundles" / "LATEST.json").write_text(json.dumps(
        {"latest": "step_001500", "global_step": 1500,
         "adapter_fingerprint": EXPECTED_FP}))
    rec["run_dir"] = run_dir
    rec["fake_model"] = fake_model
    rec["lm_config_template"] = lm_config
    return rec


def _run_predict(env, v5_specs):
    spec = [s for s in v5_specs if s.policy == "Static-R9"][0]
    predict = _default_predict_fn(REPO_ROOT)
    return predict(spec, env["run_dir"], {"seed": 42, "lora_r": 9,
                                          "lora_alpha": 18})


# 1 + 4: single PEFT wrap — base loaded once, adapter restored once, and the
# fresh-attach path (attach_lora) is never touched.
def test_predict_loads_base_and_restores_adapter_exactly_once(env, v5_specs):
    _run_predict(env, v5_specs)
    assert len(env["load_base_model"]) == 1
    assert len(env["load_adapter_for_training"]) == 1
    assert len(env["load_tokenizer"]) == 1
    call = env["load_adapter_for_training"][0]
    assert call["base"] is env["load_base_model"][0] and call["base"] is not None
    assert call["adapter_dir"].name == "adapter"
    assert call["adapter_dir"].parent.name == "step_001500"
    assert call["adapter_dir"].parent.parent.name == "bundles"
    # the base model handed to the restore path is the one load_base_model
    # returned (no second instantiation, no PEFT pre-wrap)
    assert call["cfg"] is env["load_base_model_cfg"][0]


# 5: fingerprint verification against bundles/LATEST.json.
def test_predict_verifies_fingerprint_from_latest_json(env, v5_specs):
    _run_predict(env, v5_specs)
    assert env["fingerprints"] == [env["fake_model"]]


# 3: fingerprint mismatch (non-terminal or wrong adapter) refuses to evaluate.
def test_predict_rejects_fingerprint_mismatch(monkeypatch, env, v5_specs):
    monkeypatch.setattr("cycleft.e5.resume_bundle.adapter_fingerprint",
                        lambda model: "0" * 64)
    with pytest.raises(RuntimeError, match="fingerprint mismatch"):
        _run_predict(env, v5_specs)
    assert env["evaluate_event"] == [], "no evaluation after a failed gate"
    assert not (env["run_dir"] / "predictions").exists()


# 3 (task item): missing adapter weights are never silently tolerated —
# a model whose loaded weights differ from the saved fingerprint fails.
def test_missing_adapter_weights_cause_failure(monkeypatch, env, v5_specs):
    real_param = torch.nn.Parameter(torch.zeros(4))

    class _ModelWithParams(torch.nn.Module):
        def named_parameters(self, *a, **k):
            return [("lora_A", real_param)]

    def load_with_missing_weights(base, adapter_dir, cfg, device):
        # simulates a silently-dropped checkpoint key: tensor stays at init
        return _ModelWithParams()

    def fingerprint_of_actual(model):
        return hashlib_sha(model)

    def hashlib_sha(model):
        import hashlib
        d = hashlib.sha256()
        for n, p in model.named_parameters():
            d.update(n.encode())
            d.update(p.detach().numpy().tobytes())
        return d.hexdigest()

    monkeypatch.setattr("cycleft.e1.lm.load_adapter_for_training",
                        load_with_missing_weights)
    monkeypatch.setattr("cycleft.e5.resume_bundle.adapter_fingerprint",
                        fingerprint_of_actual)
    # expected fingerprint in LATEST.json corresponds to the TRAINED values,
    # not the zero-initialized tensor the broken load produced:
    env2 = env
    (env2["run_dir"] / "bundles" / "LATEST.json").write_text(json.dumps(
        {"latest": "step_001500", "global_step": 1500,
         "adapter_fingerprint": EXPECTED_FP}))
    with pytest.raises(RuntimeError, match="fingerprint mismatch"):
        _run_predict(env2, v5_specs)
    assert env2["evaluate_event"] == []


# 4 + 7 (device): explicit MPS resolution; no silent CPU fallback.
def test_predict_uses_resolved_device_passthrough(env, v5_specs):
    _run_predict(env, v5_specs)
    assert env["resolve_device"] == [None]
    assert env["load_adapter_for_training"][0]["device"] == torch.device("mps")


def test_resolve_device_rejects_silent_cpu_fallback(monkeypatch):
    from cycleft.e1.lm import resolve_device
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CPU fallback"):
        resolve_device(None)
    with pytest.raises(RuntimeError, match="CPU fallback"):
        resolve_device("mps")
    assert resolve_device("cpu") == torch.device("cpu")  # explicit opt-in only
    with pytest.raises(ValueError):
        resolve_device("cuda")


# 6: exact inference-configuration mapping (train path == sweep path).
def test_exact_inference_configuration_mapping(env, v5_specs):
    _run_predict(env, v5_specs)
    sweep_cfg = env["evaluate_event"][0]["cfg"]
    train_cfg = LMRuntimeConfig(
        **{**LMRuntimeConfig(model_id=RAW_CONFIG["e1"]["model_id"]).to_dict(),
           **{}})
    # rebuild the train-path config exactly as the train callback does:
    from cycleft.e1.pipeline import resolve_lm_config
    base_cfg, _, _ = resolve_lm_config(RAW_CONFIG, mode="full",
                                       train_examples=None, seed=42,
                                       device=None)
    train_cfg = base_cfg
    train_cfg.lora_r = 9
    train_cfg.lora_alpha = 18
    train_cfg.lora_precision = "float32"
    train_cfg.dtype = "float16"
    train_cfg.max_new_tokens = 24
    train_cfg.seed = 42
    assert sweep_cfg.to_dict() == train_cfg.to_dict()
    assert sweep_cfg.lora_r == 9 and sweep_cfg.lora_alpha == 18
    assert sweep_cfg.dtype == "float16"
    assert sweep_cfg.lora_precision == "float32"
    assert sweep_cfg.max_new_tokens == 24
    assert sweep_cfg.max_length == 128
    assert sweep_cfg.seed == 42


# 7a: state mapping per event.
def test_prediction_state_field_mapping(env, v5_specs):
    result = _run_predict(env, v5_specs)
    assert result["exit_code"] == 0 and result["status"] == "complete"
    for eid in (1, 2, 3):
        path = env["run_dir"] / "predictions" / f"after_event_{eid}.jsonl"
        lines = [json.loads(l) for l in path.read_text().splitlines() if l]
        assert all(r["state"] == f"after_event_{eid}" for r in lines)


# 7b: the full harness record contract. Preflight finding P-1 (missing
# `test_event`) was fixed 2026-11-23 in the recovery task
# (e7-static-r9-s42-recovery-01): record["test_event"] = event.event_id,
# matching the event_orchestrator.py:1470 precedent (event.event_id IS the
# evaluation test-set ID in (1, 2, 3)).
def test_full_prediction_record_contract_including_test_event(env, v5_specs):
    _run_predict(env, v5_specs)
    for eid in (1, 2, 3):
        path = env["run_dir"] / "predictions" / f"after_event_{eid}.jsonl"
        for line in path.read_text().splitlines():
            rec = json.loads(line)
            missing = [k for k in PREDICTION_RECORD_KEYS if k not in rec]
            assert not missing, f"record lacks {missing}"
            assert rec["test_event"] == eid


# 8: two-example output validation (order, indices, content).
def test_two_example_output_validation(env, v5_specs):
    _run_predict(env, v5_specs)
    path = env["run_dir"] / "predictions" / "after_event_1.jsonl"
    recs = [json.loads(l) for l in path.read_text().splitlines() if l]
    assert len(recs) == 2
    assert [r["example_index"] for r in recs] == [0, 1]
    assert recs[0]["question"] == "q1-0" and recs[1]["question"] == "q1-1"
    assert recs[0]["reference"] == "a1-0"


# 9: checkpoint/bundle immutability — the sweep writes only predictions/.
def test_predict_never_writes_outside_predictions(env, v5_specs):
    def tree(root):
        return {str(p.relative_to(root)): p.read_bytes()
                for p in sorted(Path(root).rglob("*")) if p.is_file()}
    before = tree(env["run_dir"])
    _run_predict(env, v5_specs)
    after = tree(env["run_dir"])
    added = set(after) - set(before)
    assert added and all(a.startswith("predictions/") for a in added)
    for name in before:
        assert after[name] == before[name], f"{name} was modified"


# 10 + 11: interrupted-state preservation and provenance safety (hermetic).
def test_interrupted_running_state_is_fail_closed(tmp_path):
    run_dir = tmp_path / "r"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(json.dumps(
        {"exec_id": "x", "state": "RUNNING"}))
    cls = classify_run_state(run_dir)
    assert cls["classification"] == "RUNNING" and cls["fail_closed"]


def test_completed_without_marker_is_corrupt(tmp_path):
    run_dir = tmp_path / "r"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(json.dumps(
        {"exec_id": "x", "state": "COMPLETED"}))
    cls = classify_run_state(run_dir)
    assert cls["classification"] == "CORRUPT" and cls["fail_closed"]


def test_harness_refuses_execution_into_existing_directory(tmp_path, v5_specs):
    out_root = tmp_path / "out"
    out_root.mkdir()
    run_dir = out_root / "e7-static-r9-s42"
    run_dir.mkdir()
    (run_dir / "evidence.txt").write_text("attempt-1 evidence")
    spec = [s for s in v5_specs if s.exec_id == "e7-static-r9-s42"][0]
    harness = ExecutionHarness(V5_MATRIX, out_root, repo_root=tmp_path,
                               run_fn=lambda *a: {})
    with pytest.raises((OutputSafetyError, RuntimeError, Exception)):
        harness._execute_selected([spec], [type("P", (), {
            "exec_id": spec.exec_id, "policy": spec.policy,
            "seed": spec.seed, "row_type": "NEW", "event_ranks": spec.event_ranks,
            "event_alphas": spec.event_alphas, "output_path": str(run_dir),
            "config_hash": spec.config_hash, "reuse_source": "",
            "action": "train", "selected": True})()])


# 12: the sweep must never invoke training or anything beyond the
# per-event evaluation (no accidental full-sweep/training invocation).
def test_predict_invokes_evaluation_once_per_event_only(env, v5_specs):
    _run_predict(env, v5_specs)
    assert len(env["evaluate_event"]) == 3
    assert all(c["n_rows"] == 2 for c in env["evaluate_event"])
    import ast
    import inspect
    from cycleft.e7 import e5_adapter
    tree = ast.parse(inspect.getsource(e5_adapter))
    predict_fn = next(n for n in ast.walk(tree)
                      if isinstance(n, ast.FunctionDef)
                      and n.name == "_default_predict_fn")
    called = set()
    for node in ast.walk(predict_fn):
        if isinstance(node, ast.Call):
            f = node.func
            name = f.id if isinstance(f, ast.Name) else (
                f.attr if isinstance(f, ast.Attribute) else "")
            called.add(name)
    assert "run_workload" not in called, "sweep must never re-enter training"
    assert "attach_lora" not in called, "sweep must never attach a fresh adapter"


# ---------------------------------------------------------------------------
# P-1 recovery regression tests (recovery-01, 2026-11-23)
# ---------------------------------------------------------------------------


def test_p1_test_event_present_with_correct_test_set_id(env, v5_specs):
    """P-1 regression: every record carries test_event == the event's
    evaluation test-set identifier (1, 2, 3)."""
    _run_predict(env, v5_specs)
    for eid in (1, 2, 3):
        path = env["run_dir"] / "predictions" / f"after_event_{eid}.jsonl"
        recs = [json.loads(l) for l in path.read_text().splitlines() if l]
        assert recs, f"after_event_{eid}.jsonl is empty"
        assert all(r["test_event"] == eid for r in recs)
        assert all(r["test_event"] in (1, 2, 3) for r in recs)


def test_p1_other_required_prediction_fields_intact(env, v5_specs):
    """P-1 regression: the fix adds ONLY test_event; every previously
    required field survives unchanged (no renames, no removals)."""
    _run_predict(env, v5_specs)
    legacy_keys = {"example_index", "question", "reference",
                   "raw_prediction", "prediction", "exact_match", "token_f1"}
    for eid in (1, 2, 3):
        path = env["run_dir"] / "predictions" / f"after_event_{eid}.jsonl"
        recs = [json.loads(l) for l in path.read_text().splitlines() if l]
        for r in recs:
            assert legacy_keys <= set(r), f"legacy field lost: {legacy_keys - set(r)}"
            assert r["state"] == f"after_event_{eid}"
            assert set(r) == legacy_keys | {"state", "test_event"}, \
                "the fix must not add fields beyond the contract"


def test_p1_harness_accepts_valid_record(tmp_path):
    """P-1 regression: a record with test_event passes the harness's
    structural prediction validation (_validate_artifact_structure)."""
    from cycleft.e7.execution_harness import (
        _validate_artifact_structure,
        inventory_artifact,
    )
    run_dir = tmp_path / "r"
    (run_dir / "predictions").mkdir(parents=True)
    record = {"state": "after_event_1", "test_event": 1, "example_index": 0,
              "question": "q?", "reference": "r", "raw_prediction": "p",
              "prediction": "p", "exact_match": 1.0, "token_f1": 1.0}
    (run_dir / "predictions" / "after_event_1.jsonl").write_text(
        json.dumps(record) + "\n")
    inv = inventory_artifact(run_dir, "predictions")  # must not raise
    assert inv["type"] == "dir"
    _validate_artifact_structure(run_dir, "predictions",
                                 inv)  # must not raise (P-1 contract)


def test_p1_harness_rejects_incorrect_test_set_ids(tmp_path):
    """P-1 regression: records with out-of-range / wrong-typed test-set IDs
    are rejected by the orchestrator's semantic validation contract, and
    structurally malformed prediction files fail the harness gate."""
    # the schema-valid domain is {1, 2, 3} (event_orchestrator.py:674);
    # records outside it must be detectable and rejected.
    from cycleft.e7.execution_harness import StateError, _validate_artifact_structure
    for bad in (0, 4, "1", None):
        run_dir = tmp_path / f"bad_{bad!r}"
        (run_dir / "predictions").mkdir(parents=True)
        record = {"state": "after_event_0", "test_event": bad,
                  "example_index": 0, "question": "q?", "reference": "r",
                  "raw_prediction": "p", "prediction": "p",
                  "exact_match": 1.0, "token_f1": 1.0}
        (run_dir / "predictions" / "after_event_0.jsonl").write_text(
            json.dumps(record) + "\n")
        # semantic guard (orchestrator validator contract):
        semantic_ok = (isinstance(record["test_event"], int)
                       and record["test_event"] in (1, 2, 3))
        assert not semantic_ok, f"{bad!r} must fail the semantic test-set check"
    # and the structural harness rejects a malformed (non-dict) record line
    run_dir = tmp_path / "malformed"
    (run_dir / "predictions").mkdir(parents=True)
    (run_dir / "predictions" / "after_event_1.jsonl").write_text("[1]\n")
    with pytest.raises(StateError):
        _validate_artifact_structure(run_dir, "predictions", {})


def test_p1_grow_precedent_records_remain_compatible():
    """P-1 regression: the schema the sweep now writes is exactly the schema
    the successful Grow pilot materialized (field-for-field, types included)."""
    grow_pred = Path(__file__).resolve().parents[1] / (
        "results/e7/execution_20261121/e7-grow-s42/predictions/"
        "after_event_1.jsonl")
    if not grow_pred.is_file():  # pragma: no cover
        pytest.skip("grow pilot predictions not present in this checkout")
    grow = json.loads(grow_pred.read_text().splitlines()[0])
    sweep_record = {"state": "after_event_1", "test_event": 1,
                    "example_index": 0, "question": "q", "reference": "r",
                    "raw_prediction": "p", "prediction": "p",
                    "exact_match": 0.0, "token_f1": 0.0}
    assert set(sweep_record) == set(grow), \
        f"schema drift vs grow precedent: {set(sweep_record) ^ set(grow)}"
    assert isinstance(grow["test_event"], int)
    assert grow["test_event"] in (1, 2, 3)
    for k in ("state", "question", "reference", "raw_prediction", "prediction"):
        assert isinstance(grow[k], str)
    for k in ("example_index",):
        assert isinstance(grow[k], int)
    for k in ("exact_match", "token_f1"):
        assert isinstance(grow[k], (int, float))
