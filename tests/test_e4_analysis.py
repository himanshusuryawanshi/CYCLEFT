"""Tests for E4 analysis: policy comparison, metric rows, CSV writing."""

import pytest

from cycleft.e4.analysis import (
    benchmark_metric_rows,
    checkpoint_overhead_rows,
    continual_learning_rows,
    policy_comparison_rows,
    resource_comparison_rows,
    write_csv,
)


def make_benchmark(workload="A_economical", resource="local_mps", status="complete"):
    return {
        "workload": workload,
        "resource_id": resource,
        "device": "mps",
        "status": status,
        "states": ["base", "after_stage_1", "after_stage_2", "after_stage_3"],
        "accuracy_matrix": [[0.04, 0.02, 0.12], [0.10, 0.04, 0.16], [0.16, 0.04, 0.14], [0.18, 0.06, 0.20]],
        "token_f1_matrix": [[0.05, 0.06, 0.14], [0.20, 0.12, 0.21], [0.22, 0.21, 0.18], [0.23, 0.21, 0.21]],
        "totals": {
            "optimizer_steps": 375,
            "training_examples": 375,
            "training_tokens": 29309,
            "training_seconds": 300.0,
            "model_load_seconds": 30.0,
            "checkpoint_save_seconds": 3.0,
            "checkpoint_load_seconds": 60.0,
            "eval_seconds": 280.0,
            "total_completion_seconds": 673.0,
            "steps_per_second": 1.25,
            "tokens_per_second": 97.7,
            "peak_memory_gb": 2.5,
            "n_transitions": 2,
        },
        "cl_metrics": {
            "average_forgetting": 0.0067,
            "backward_transfer_conventional": 0.04,
        },
        "transitions": [
            {
                "transition": "transition_1",
                "from_stage": 1,
                "to_stage": 2,
                "checkpoint_save_seconds": 1.5,
                "checkpoint_load_seconds": 30.0,
                "restore_verified": True,
                "note": "test",
            },
            {
                "transition": "transition_2",
                "from_stage": 2,
                "to_stage": 3,
                "checkpoint_save_seconds": 1.5,
                "checkpoint_load_seconds": 30.0,
                "restore_verified": True,
                "note": "test",
            },
        ],
        "stages": [
            {
                "stage": 1, "kind": "event1_bootstrap", "state": "after_stage_1",
                "model_load_seconds": 10.0, "training_seconds": 100.0,
                "checkpoint_save_seconds": 1.5, "eval_seconds": 90.0,
                "optimizer_steps": 125, "examples": 125, "total_tokens": 9770,
                "steps_per_second": 1.25, "tokens_per_second": 97.7,
                "average_loss": 2.0, "peak_memory_gb": 2.5,
                "checkpoint_dir": "/tmp/a1", "continued_from": None,
            },
            {
                "stage": 2, "kind": "transition_1", "state": "after_stage_2",
                "model_load_seconds": 10.0, "training_seconds": 100.0,
                "checkpoint_save_seconds": 1.5, "eval_seconds": 90.0,
                "optimizer_steps": 125, "examples": 125, "total_tokens": 9770,
                "steps_per_second": 1.25, "tokens_per_second": 97.7,
                "average_loss": 1.9, "peak_memory_gb": 2.5,
                "checkpoint_dir": "/tmp/a2", "continued_from": "/tmp/a1",
            },
            {
                "stage": 3, "kind": "transition_2", "state": "after_stage_3",
                "model_load_seconds": 10.0, "training_seconds": 100.0,
                "checkpoint_save_seconds": 1.5, "eval_seconds": 90.0,
                "optimizer_steps": 125, "examples": 125, "total_tokens": 9770,
                "steps_per_second": 1.25, "tokens_per_second": 97.7,
                "average_loss": 1.8, "peak_memory_gb": 2.5,
                "checkpoint_dir": "/tmp/a3", "continued_from": "/tmp/a2",
            },
        ],
    }


def make_decisions(policy="dynamic", resource="local_mps", workload="A_economical"):
    return [
        {
            "policy": policy,
            "event_index": event_index,
            "chosen_resource": resource,
            "feasible_resources": [resource],
            "infeasible": {},
            "estimated_seconds": 100.0,
            "estimated_cost": {"cost_type": "owned_no_metered_cost", "cost_usd": None},
            "reason": "only feasible resource",
            "context": {"workload": workload, "event_index": event_index},
        }
        for event_index in range(3)
    ]


def test_benchmark_metric_rows_flatten_stages():
    rows = benchmark_metric_rows([make_benchmark()])
    assert len(rows) == 3
    assert rows[1]["continued_from"] == "/tmp/a1"
    assert rows[0]["steps_per_second"] == 1.25


def test_checkpoint_overhead_rows():
    rows = checkpoint_overhead_rows([make_benchmark()])
    assert len(rows) == 2
    assert all(row["restore_verified"] for row in rows)
    # Save = from-stage's checkpoint save; load = to-stage model-load step.
    assert rows[0]["checkpoint_save_seconds"] == 1.5
    assert rows[0]["checkpoint_load_seconds"] == 10.0
    assert rows[1]["checkpoint_load_seconds"] == 10.0


def test_policy_comparison_degenerate_single_resource():
    decisions = []
    for policy in ("fixed_cheapest", "fixed_fastest", "fixed_preferred", "dynamic", "oracle_offline"):
        decisions.extend(make_decisions(policy=policy))
    rows = policy_comparison_rows(decisions, [make_benchmark()])
    assert len(rows) == 5
    for row in rows:
        assert row["degenerate_single_resource"] is True
        assert row["chosen_resources"] == "local_mps"
        assert row["final_avg_exact_match"] == pytest.approx(
            (0.18 + 0.06 + 0.20) / 3
        )
        assert row["training_seconds"] == 300.0


def test_policy_comparison_without_measured_benchmark():
    decisions = make_decisions(policy="dynamic", resource="cloud_t4")
    rows = policy_comparison_rows(decisions, [make_benchmark()])
    row = rows[0]
    assert row["final_avg_exact_match"] is None
    assert row["chosen_resources"] == "cloud_t4"


def test_resource_comparison_flags_pending_tiers():
    profiles = {
        "local_mps": {
            "kind": "local", "device": "mps", "gpu_model": "Apple M1",
            "gpu_memory_gb": None, "ram_gb": 8.0, "measured": True,
            "available": True, "price_usd_per_hour": None,
            "price_type": "owned_hardware_no_metered_cost",
            "price_source": "owned", "price_timestamp": "2026-11-17",
        },
        "cloud_t4": {
            "kind": "cloud", "device": "cuda:0", "gpu_model": "NVIDIA T4",
            "gpu_memory_gb": 16.0, "ram_gb": None, "measured": False,
            "available": False, "price_usd_per_hour": None,
            "price_type": None, "price_source": "pending",
            "price_timestamp": None,
        },
    }
    rows = resource_comparison_rows(profiles, [make_benchmark()])
    by_id = {row["resource_id"]: row for row in rows}
    assert by_id["local_mps"]["measured"] is True
    assert by_id["local_mps"]["tokens_per_second"] == pytest.approx(97.7)
    assert by_id["cloud_t4"]["measured"] is False
    assert by_id["cloud_t4"]["tokens_per_second"] is None, (
        "pending tiers must carry null measurements, never invented values"
    )


def test_continual_learning_rows():
    rows = continual_learning_rows([make_benchmark()])
    states = [row["state"] for row in rows]
    assert "base" in states and "after_stage_3" in states
    assert "backward_transfer_conventional" in states


def test_write_csv(tmp_path):
    path = write_csv(tmp_path / "out.csv", [{"a": 1, "b": 1.5, "c": None}])
    lines = path.read_text(encoding="utf-8").strip().split("\n")
    assert lines[0] == "a,b,c"
    assert lines[1] == "1,1.500000,"


def test_write_csv_rejects_empty(tmp_path):
    with pytest.raises(ValueError, match="No rows"):
        write_csv(tmp_path / "empty.csv", [])
