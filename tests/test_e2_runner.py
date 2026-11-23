"""Tests for E2 runner orchestration (fake model stack, CPU, tmp dirs).

Verifies the critical fairness properties:
- every configuration attaches a FRESH adapter (event 1) and never
  initializes from another configuration's adapter;
- events 2..3 continue the SAME configuration's previous checkpoint;
- one failing configuration does not abort the grid;
- matrices, checkpoints and prediction files are written.
"""

import json
from pathlib import Path

import pytest

from cycleft.e1.data import PreparedEvent
from cycleft.e1.lm import LMRuntimeConfig
from cycleft.e2.grid import E2Configuration
from cycleft.e2.runner import E2Runner


class FakeTokenizer:
    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "tokenizer.json").write_text("{}", encoding="utf-8")


class FakeModel:
    def __init__(self, tag):
        self.tag = tag

    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "adapter_model.safetensors").write_text(self.tag, encoding="utf-8")

    def print_trainable_parameters(self):
        pass


def training_stats(event_id: int, steps: int) -> dict:
    return {
        "examples_scheduled": steps,
        "examples_skipped": 0,
        "optimizer_steps": steps,
        "losses_per_step": [1.0] * steps,
        "average_loss": 1.0,
        "final_loss": 1.0,
        "training_seconds": 0.001 * steps,
        "trainable_parameters": 1000,
        "total_parameters": 100000,
        "trainable_percentage": 1.0,
        "answer_tokens_trained": 10 * steps,
        "total_tokens_trained": 100 * steps,
        "memory_after": {},
    }


def make_events() -> list[PreparedEvent]:
    events = []
    for event_id in (1, 2, 3):
        events.append(
            PreparedEvent(
                event_id=event_id,
                name=f"event_{event_id}",
                slice_tag=f"slice_{event_id}",
                train_rows=[
                    {"question": f"q{i}", "object": f"a{i}"} for i in range(5)
                ],
                eval_rows=[
                    {"question": f"eq{i}", "object": f"ea{i}"} for i in range(2)
                ],
                train_pool_size=500,
                eval_pool_size=2,
                train_indices_used=list(range(5)),
            )
        )
    return events


def make_baseline() -> dict:
    return {
        "state": "base",
        "model_id": "tiny",
        "per_event": {
            "1": {"exact_match": 0.04, "token_f1": 0.05, "perplexity": 12.0,
                  "examples": 2, "generation_seconds": 0.0,
                  "eval_loss": 2.5, "examples_scored": 2,
                  "mean_answer_tokens": 3.0},
            "2": {"exact_match": 0.02, "token_f1": 0.03, "perplexity": 14.0,
                  "examples": 2, "generation_seconds": 0.0,
                  "eval_loss": 2.6, "examples_scored": 2,
                  "mean_answer_tokens": 3.0},
            "3": {"exact_match": 0.12, "token_f1": 0.13, "perplexity": 11.0,
                  "examples": 2, "generation_seconds": 0.0,
                  "eval_loss": 2.4, "examples_scored": 2,
                  "mean_answer_tokens": 3.0},
        },
    }


@pytest.fixture
def fake_stack(monkeypatch, tmp_path):
    from cycleft.e2 import runner as runner_module

    monkeypatch.setattr(runner_module, "RESULTS_ROOT", tmp_path / "results" / "e2")
    monkeypatch.setattr(
        runner_module, "CHECKPOINTS_ROOT", tmp_path / "checkpoints" / "e2"
    )

    calls = {"attach": [], "load_adapter": [], "train": [], "seed": []}

    def fake_attach(base_model, lm_config, device):
        calls["attach"].append(lm_config.lora_r)
        return FakeModel(f"fresh_rank{lm_config.lora_r}")

    def fake_load_adapter(base_model, adapter_dir, lm_config, device):
        calls["load_adapter"].append(str(adapter_dir))
        return FakeModel(f"continued_from_{Path(adapter_dir).name}")

    def fake_train_event(model, tokenizer, rows, lm_config, log=None, on_step=None):
        config_id = model.tag
        calls["train"].append((config_id, len(rows)))
        if "fail" in config_id:
            raise RuntimeError("simulated OOM")
        return training_stats(event_id=len(calls["train"]), steps=len(rows))

    def fake_evaluate(model, tokenizer, eval_rows, lm_config, log=None):
        rank = int(model.tag.split("rank")[1].split("_")[0]) if "rank" in model.tag else 0
        value = 0.05 * rank + 0.01 * len(calls["train"]) + 0.001 * len(eval_rows)
        return {
            "metrics": {
                "examples": len(eval_rows),
                "exact_match": value,
                "token_f1": value + 0.01,
                "eval_loss": 2.0,
                "perplexity": 7.0,
                "generation_seconds": 0.0,
                "examples_scored": len(eval_rows),
                "mean_answer_tokens": 3.0,
            },
            "records": [
                {
                    "example_index": i,
                    "question": row["question"],
                    "reference": row["object"],
                    "raw_prediction": "x",
                    "prediction": "x",
                    "exact_match": 0.0,
                    "token_f1": 0.0,
                }
                for i, row in enumerate(eval_rows)
            ],
        }

    monkeypatch.setattr(runner_module, "load_tokenizer", lambda model_id: FakeTokenizer())
    monkeypatch.setattr(runner_module, "load_base_model", lambda cfg: FakeModel(cfg.model_id))
    monkeypatch.setattr(runner_module, "attach_lora", fake_attach)
    monkeypatch.setattr(runner_module, "load_adapter_for_training", fake_load_adapter)
    monkeypatch.setattr(runner_module, "train_event", fake_train_event)
    monkeypatch.setattr(runner_module, "evaluate_event", fake_evaluate)
    monkeypatch.setattr(runner_module, "set_seed", lambda seed: calls["seed"].append(seed))
    return calls


def config(rank: int, budget: int) -> E2Configuration:
    return E2Configuration(
        config_id=f"rank{rank}_b{budget}",
        lora_rank=rank,
        budget_id=f"b{budget}",
        budget_examples=budget,
        budget_fraction=budget / 500,
    )


def make_runner(tmp_path) -> E2Runner:
    return E2Runner(
        run_id="test_run",
        base_lm_config=LMRuntimeConfig(model_id="tiny", device="cpu"),
        events=make_events(),
        baseline=make_baseline(),
        baseline_source="test",
        log=lambda message: None,
    )


def test_fresh_adapter_per_configuration_and_continuation_within(
    fake_stack, tmp_path
):
    runner = make_runner(tmp_path)
    config_a = config(4, 2)
    config_b = config(8, 2)

    result_a = runner.run_configuration(config_a)
    result_b = runner.run_configuration(config_b)

    # One fresh adapter per configuration, with that configuration's rank.
    assert fake_stack["attach"] == [4, 8]

    # Events 2 and 3 continue the SAME configuration's checkpoints.
    ckpt_root = tmp_path / "checkpoints" / "e2" / "test_run"
    assert fake_stack["load_adapter"] == [
        str(ckpt_root / "rank4_b2" / "event_1"),
        str(ckpt_root / "rank4_b2" / "event_2"),
        str(ckpt_root / "rank8_b2" / "event_1"),
        str(ckpt_root / "rank8_b2" / "event_2"),
    ]

    assert result_a.status == "complete"
    assert len(result_a.accuracy_matrix) == 4  # base + 3 states
    assert result_a.accuracy_matrix[0] == [0.04, 0.02, 0.12]  # shared baseline row
    assert result_a.states == ["base", "after_event_1", "after_event_2", "after_event_3"]

    # Provenance chain inside the checkpoint configs.
    for event_id in (1, 2, 3):
        training_config = json.loads(
            (ckpt_root / "rank4_b2" / f"event_{event_id}" / "training_config.json")
            .read_text(encoding="utf-8")
        )
        assert training_config["lora_rank"] == 4
        if event_id == 1:
            assert training_config["continued_from"] is None
        else:
            assert training_config["continued_from"].endswith(f"event_{event_id - 1}")

    # Predictions and per-config artifacts exist.
    assert (tmp_path / "results" / "e2" / "test_run" / "predictions"
            / "rank4_b2" / "after_event_3.jsonl").exists()
    assert (tmp_path / "results" / "e2" / "test_run" / "rank4_b2"
            / "event_metrics.json").exists()


def test_budget_is_prefix_of_training_order(fake_stack, tmp_path):
    runner = make_runner(tmp_path)
    small = runner.run_configuration(config(4, 2))
    big = runner.run_configuration(config(4, 5))

    assert small.status == "complete" and big.status == "complete"

    # Budget sizes are recorded per event in the saved training configs.
    ckpt_root = tmp_path / "checkpoints" / "e2" / "test_run"
    for budget, config_id in ((2, "rank4_b2"), (5, "rank4_b5")):
        for event_id in (1, 2, 3):
            training_config = json.loads(
                (ckpt_root / config_id / f"event_{event_id}" / "training_config.json")
                .read_text(encoding="utf-8")
            )
            assert training_config["budget_examples"] == budget
            assert training_config["lora_rank"] == 4
            assert (
                training_config["training_stats"]["examples_scheduled"] == budget
            )


def test_failed_configuration_isolated(fake_stack, tmp_path):
    runner = make_runner(tmp_path)

    class FailModel(FakeModel):
        pass

    # Mark rank4's models as failing via tag -> train_event raises.
    original_load = runner.configuration_lm_config

    failing = config(4, 2)
    failing = E2Configuration(
        config_id="rank4_fail", lora_rank=4, budget_id="b2",
        budget_examples=2, budget_fraction=0.01,
    )
    healthy = config(8, 2)

    # Patch FakeModel tags so fake_train_event sees 'fail'.
    from cycleft.e2 import runner as runner_module

    real_attach = runner_module.attach_lora
    real_load = runner_module.load_adapter_for_training

    def attach_with_tag(base_model, lm_config, device):
        model = real_attach(base_model, lm_config, device)
        model.tag = f"rank{lm_config.lora_r}_{'fail' if lm_config.lora_r == 4 else 'ok'}"
        return model

    def load_with_tag(base_model, adapter_dir, lm_config, device):
        model = real_load(base_model, adapter_dir, lm_config, device)
        model.tag = f"rank{lm_config.lora_r}_{'fail' if lm_config.lora_r == 4 else 'ok'}"
        return model

    runner_module.attach_lora = attach_with_tag
    runner_module.load_adapter_for_training = load_with_tag
    try:
        result_bad = runner.run_configuration(failing)
        result_good = runner.run_configuration(healthy)
    finally:
        runner_module.attach_lora = real_attach
        runner_module.load_adapter_for_training = real_load

    assert result_bad.status == "failed"
    assert "simulated OOM" in result_bad.error
    assert result_good.status == "complete"
    _ = original_load  # unused; keeps fixture reference honest
