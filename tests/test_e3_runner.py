"""Tests for the E3 runner orchestration (fake model stack, CPU, tmp dirs).

Verifies the critical execution properties without loading the real model:
- the shared bootstrap convention (strategies continue the same adapter
  lineage; the bootstrap is excluded from detection accounting);
- trigger strategies actually adapt (fake training called) and save
  checkpoints with a continuation chain;
- the oracle reference adapts exactly at event starts and its detection
  accounting attributes adaptations to the boundaries they respond to;
- the periodic schedule produces the documented adaptations;
- warm-up and minimum-gap suppression works;
- Regime-A detectors never receive answer strings (leakage check);
- the delayed-feedback error stream is prequential (predict-then-score);
- one failing strategy does not abort the run.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from cycleft.e1.data import PreparedEvent
from cycleft.e1.lm import LMRuntimeConfig
from cycleft.e3.runner import (
    E3Config as _E3Config,
)
from cycleft.e3.runner import (
    ModelController,
    STRATEGY_REGIMES,
    run_strategy,
)
from cycleft.e3.stream import E3Stream, WINDOW_SIZE


def make_stream(n_per_event: int = 500) -> E3Stream:
    rows = []
    event_ids = []
    for event in (1, 2, 3):
        for i in range(n_per_event):
            position = (event - 1) * n_per_event + i
            rows.append(
                {
                    "question": f"q{position}",
                    "object": f"a{position}",
                    "_position": position,
                }
            )
            event_ids.append(event)
    return E3Stream(rows=rows, event_ids=event_ids, window_size=WINDOW_SIZE, seed=42)


def make_events() -> list[PreparedEvent]:
    events = []
    for event_id in (1, 2, 3):
        events.append(
            PreparedEvent(
                event_id=event_id,
                name=f"event_{event_id}",
                slice_tag=f"slice_{event_id}",
                train_rows=[{"question": "q", "object": "a"}],
                eval_rows=[
                    {"question": f"eq{i}", "object": f"ea{i}"} for i in range(3)
                ],
                train_pool_size=500,
                eval_pool_size=3,
                train_indices_used=list(range(1)),
            )
        )
    return events


BASELINE_ROWS = (
    [0.04, 0.02, 0.12],
    [0.057, 0.060, 0.141],
    [12.8, 16.8, 12.1],
)


def fake_training_stats(rows: list[dict]) -> dict:
    steps = len(rows)
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


@pytest.fixture
def fake_stack(monkeypatch):
    from cycleft.e3 import runner as runner_module

    calls = {
        "attach": [],
        "train": [],
        "eval_states": [],
        "error_positions": [],
        "seeds": [],
    }
    counter = {"models": 0}

    class FakeTokenizer:
        def save_pretrained(self, path):
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "tokenizer.json").write_text("{}", encoding="utf-8")

    class FakeModel:
        def __init__(self, tag):
            self.tag = tag

        def to(self, device):
            return self

        def train(self):
            return self

        def eval(self):
            return self

        def named_parameters(self):
            # required by the init-fingerprint provenance helper
            return [("adapter.fake", torch.nn.Parameter(torch.zeros(2)))]

        def save_pretrained(self, path):
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "adapter_model.safetensors").write_text(
                self.tag, encoding="utf-8"
            )

    def fake_attach(base_model, lm_config, device):
        counter["models"] += 1
        calls["attach"].append(lm_config.lora_r)
        return FakeModel(f"adapter_{counter['models']}")

    def fake_train_event(model, tokenizer, rows, lm_config, log=None, on_step=None):
        calls["train"].append((model.tag, len(rows)))
        if "fail" in model.tag:
            raise RuntimeError("simulated training failure")
        return fake_training_stats(rows)

    def fake_evaluate(model, tokenizer, eval_rows, lm_config, log=None):
        value = 0.01 * len(calls["train"]) + 0.001 * len(eval_rows)
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

    def fake_error(self, row, position):
        """Prequential error: 0.85 in event 1, 0.97 after boundary 500."""
        calls["error_positions"].append(position)
        wobble = 0.01 * np.sin(position / 9.0)
        return 0.85 + wobble if position < 500 else 0.97 + wobble

    monkeypatch.setattr(runner_module, "load_tokenizer", lambda model_id: FakeTokenizer())
    monkeypatch.setattr(runner_module, "load_base_model", lambda cfg: FakeModel("base"))
    monkeypatch.setattr(runner_module, "attach_lora", fake_attach)
    monkeypatch.setattr(runner_module, "train_event", fake_train_event)
    monkeypatch.setattr(runner_module, "evaluate_event", fake_evaluate)
    monkeypatch.setattr(
        runner_module, "resolve_device", lambda cfg=None: torch.device("cpu")
    )
    monkeypatch.setattr(
        runner_module, "set_seed", lambda seed: calls["seeds"].append(seed)
    )
    monkeypatch.setattr(ModelController, "prequential_error", fake_error)
    return calls


def make_config(**overrides) -> _E3Config:
    defaults = dict(
        run_id="test_run",
        adapt_budget=25,
        n_calibration_windows=3,
        warmup_windows=6,
        min_gap_examples=100,
    )
    defaults.update(overrides)
    return _E3Config(**defaults)


def make_base_error_stream(n: int = 1500) -> dict[int, float]:
    return {position: 0.9 for position in range(n)}


def make_bootstrap_error_stream() -> dict[int, float]:
    return {position: 0.85 for position in range(25, 300)}


def run(
    strategy: str,
    fake_stack,
    tmp_path,
    config: _E3Config | None = None,
    n_per_event: int = 500,
):
    config = config or make_config()
    rng = np.random.default_rng(0)
    embeddings = rng.normal(size=(3 * n_per_event, 8))
    calibration = [[0.86] * 50, [0.86] * 50, [0.86] * 50]
    return run_strategy(
        strategy=strategy,
        config=config,
        lm_config=LMRuntimeConfig(model_id="tiny", device="cpu", seed=42),
        stream=make_stream(n_per_event),
        events=make_events(),
        baseline_rows=BASELINE_ROWS,
        embeddings=embeddings,
        base_error_stream=make_base_error_stream(3 * n_per_event),
        bootstrap_error_stream=make_bootstrap_error_stream(),
        shared_calibration_errors=calibration,
        bootstrap_checkpoint=None,
        bootstrap_seconds=1.0,
        results_dir=tmp_path / "results",
        checkpoints_dir=tmp_path / "checkpoints",
        log=lambda message: None,
    )


def test_config_defaults_documented():
    config = _E3Config(run_id="x")
    assert config.tolerance_examples == 100
    assert config.adapt_rank == 4
    assert config.adapt_budget == 125
    assert config.window_size == 50
    assert config.warmup_examples == 300
    assert config.calibration_window_indices == (3, 4, 5)
    assert config.mmd_alpha == 0.10


def test_loss_threshold_triggers_at_drift(fake_stack, tmp_path):
    result = run("loss_threshold", fake_stack, tmp_path)
    assert result.status == "complete"
    assert result.n_adaptations >= 1
    first_after_drift = [
        position
        for position in result.adaptation_positions
        if 500 <= position <= 700
    ]
    assert first_after_drift, "error trigger must fire shortly after the drift"
    assert len(result.error_curve) == 1500
    assert result.detection["true_positives"] >= 1
    assert "500" in result.detection["detection_delays_examples"]


def test_prequential_error_predicts_before_scoring(fake_stack, tmp_path):
    result = run("loss_threshold", fake_stack, tmp_path)
    # Online errors only from the warm-up boundary onward; the shared
    # caches cover earlier positions (no model calls there).
    assert min(fake_stack["error_positions"]) >= 300
    predictions = result.prediction_log
    for record in predictions:
        # The prediction is recorded with the reference only post-feedback;
        # error is a plain function of both.
        assert record["error"] in (0.0, 1.0)


def test_checkpoint_continuation_chain(fake_stack, tmp_path):
    result = run("loss_threshold", fake_stack, tmp_path)
    records = result.adaptation_records
    assert records[0]["continued_from"] is None
    for previous, record in zip(records, records[1:]):
        assert record["continued_from"] == previous["checkpoint_dir"]
        assert Path(record["continued_from"]).exists()
    for record in records:
        checkpoint = Path(record["checkpoint_dir"])
        assert (checkpoint / "adapter_model.safetensors").exists()
        provenance = json.loads(
            (checkpoint / "adaptation_provenance.json").read_text(encoding="utf-8")
        )
        assert provenance["trigger"]["trigger"] == "loss_threshold"
        assert provenance["buffer_size"] <= 25


def test_always_adapt_oracle_positions_and_attribution(fake_stack, tmp_path):
    result = run("always_adapt", fake_stack, tmp_path)
    assert result.status == "complete"
    # Bootstrap (position 24) is shared, not a strategy adaptation; the
    # oracle adapts after the first adapt_budget examples of events 2/3.
    assert result.adaptation_positions == [524, 1024]
    assert result.n_adaptations == 2
    detection = result.detection
    assert detection["true_positives"] == 2
    assert detection["detection_delays_examples"] == {"500": 0, "1000": 0}
    assert detection["false_positives"] == 0
    assert "oracle" in detection["attribution_note"]
    assert [record["buffer_size"] for record in result.adaptation_records] == [25, 25]


def test_periodic_k5_schedule_downstream(fake_stack, tmp_path):
    result = run("periodic_k5", fake_stack, tmp_path)
    assert result.status == "complete"
    # Windows 4 (position 249) is inside the warm-up (300 examples).
    assert result.adaptation_positions == [499, 749, 999, 1249, 1499]
    assert result.n_adaptations == 5
    assert result.detection["n_triggers_total"] == 5


def test_warmup_and_min_gap_suppression(fake_stack, tmp_path):
    result = run("periodic_k3", fake_stack, tmp_path)
    # Scheduled windows 2 (149) and 5 (299) fall inside the warm-up.
    suppressed = [
        window for window in result.window_log if window.get("suppressed_warmup")
    ]
    assert [window["position"] for window in suppressed] == [149, 299]
    assert result.adaptation_positions == [
        449, 599, 749, 899, 1049, 1199, 1349, 1499,
    ]


def test_mmd_regime_a_never_sees_answers(fake_stack, tmp_path):
    result = run("mmd", fake_stack, tmp_path)
    assert result.status == "complete"
    payload = json.dumps(result.detector_log + result.window_log)
    assert "a12" not in payload and "ea12" not in payload, (
        "answer strings must never appear in regime-A detector inputs/logs"
    )
    assert all(entry["regime"] == "A_unlabeled" for entry in result.detector_log)
    assert result.error_curve == [], "regime-A strategy must not consume answers"


def test_combined_logs_both_components(fake_stack, tmp_path):
    result = run("combined", fake_stack, tmp_path)
    assert result.status == "complete"
    detectors = {entry["detector"] for entry in result.detector_log}
    assert {"adwin", "mmd", "combined"} <= detectors


def test_no_adapt_uses_shared_base_curve(fake_stack, tmp_path):
    result = run("no_adapt", fake_stack, tmp_path)
    assert result.status == "complete"
    assert result.n_adaptations == 0
    assert result.detection["n_triggers_total"] == 0
    assert len(result.error_curve) == 1500
    assert fake_stack["error_positions"] == [], (
        "no_adapt must reuse the shared base stream, not generate predictions"
    )
    # Deterministic identical state: baseline rows reused for all states.
    assert result.accuracy_matrix.count(result.accuracy_matrix[0]) == 4


def test_boundary_evaluations_and_matrices(fake_stack, tmp_path):
    result = run("always_adapt", fake_stack, tmp_path)
    assert result.states == [
        "base", "after_segment_1", "after_segment_2", "after_segment_3",
    ]
    assert len(result.accuracy_matrix) == 4
    assert all(len(row) == 3 for row in result.accuracy_matrix)
    assert set(result.eval_points) == {
        "after_segment_1", "after_segment_2", "after_segment_3",
    }
    assert (tmp_path / "results" / "predictions" / "always_adapt").exists()
    cl = result.cl_metrics
    assert "backward_transfer_conventional" in cl
    assert "average_forgetting" in cl


def test_failing_strategy_isolated(fake_stack, tmp_path):
    from cycleft.e3 import runner as runner_module

    original = runner_module.train_event

    def failing_train(model, tokenizer, rows, lm_config, log=None, on_step=None):
        raise RuntimeError("simulated training failure")

    runner_module.train_event = failing_train
    try:
        result = run("loss_threshold", fake_stack, tmp_path)
    finally:
        runner_module.train_event = original
    assert result.status == "failed"
    assert "simulated training failure" in result.error
    assert (tmp_path / "results" / "loss_threshold_status.json").exists()


def test_strategy_regimes_declared():
    assert STRATEGY_REGIMES["mmd"] == "A_unlabeled (question-embedding MMD^2)"
    assert (
        STRATEGY_REGIMES["adwin"]
        == "B_delayed_feedback (per-example prequential error)"
    )
    assert "A+B" in STRATEGY_REGIMES["combined"]
