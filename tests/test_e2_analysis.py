"""Tests for E2 result aggregation and CSV writing."""

import pytest

from cycleft.e2.analysis import (
    _final_avg,
    configuration_table,
    metrics_rows,
    pareto_table,
    resource_rows,
    write_csv,
)


def payload(config_id, rank, final_em, seconds, tokens, forgetting=0.0):
    states = ["base", "after_event_1", "after_event_2", "after_event_3"]
    base_em = 0.06
    matrix = [[0.04, 0.02, 0.12]]
    for state_index in range(1, 4):
        progress = final_em + (base_em - final_em) * (3 - state_index) / 3
        matrix.append([progress, progress - 0.01, progress + 0.02])
    token_f1 = [[v + 0.01 for v in row] for row in matrix]
    ppl = [[10.0 for _ in row] for row in matrix]
    return {
        "event_metrics": {
            "config_id": config_id,
            "status": "complete",
            "states": states,
            "test_events": [1, 2, 3],
            "accuracy_matrix": matrix,
            "token_f1_matrix": token_f1,
            "perplexity_matrix": ppl,
            "continual_learning": {
                "average_forgetting": forgetting,
                "backward_transfer_mean_all_events": 0.01,
                "backward_transfer_conventional": 0.015,
            },
            "training": {
                "1": {"trainable_parameters": 1703936 * rank // 8},
            },
        },
        "resource": {
            "totals": {
                "optimizer_steps": 3 * seconds,
                "examples_processed": 3 * seconds,
                "total_tokens": tokens,
                "training_seconds": seconds,
                "wall_seconds_including_eval": seconds * 2,
                "peak_mps_allocated_gb": 2.5,
                "peak_rss_gb": 3.0,
                "adapter_checkpoint_bytes": 6800000,
            },
            "events": [
                {
                    "event_id": 1,
                    "optimizer_steps": seconds,
                    "examples_processed": seconds,
                    "total_tokens": tokens // 3,
                    "training_seconds": seconds / 3,
                    "tokens_per_second": 100.0,
                    "checkpoint_size_bytes": 6800000,
                    "continued_from": None,
                    "memory": {"peak_mps_allocated_gb": 2.5, "peak_rss_gb": 3.0,
                               "samples": 10},
                }
            ],
        },
    }


def results_fixture():
    return {
        "rank8_b125": payload("rank8_b125", 8, 0.12, 100.0, 30_000),
        "rank8_b500": payload("rank8_b500", 8, 0.17, 400.0, 120_000),
        "rank16_b125": payload("rank16_b125", 16, 0.13, 120.0, 30_000),
        "rank16_b500": payload("rank16_b500", 16, 0.16, 450.0, 120_000),
    }


def test_configuration_table_computes_efficiency():
    rows = configuration_table(None, results_fixture())
    by_id = {row["config_id"]: row for row in rows}

    r8b125 = by_id["rank8_b125"]
    assert r8b125["trainable_parameters"] == 1_703_936
    assert r8b125["quality_gain_per_training_second"] == pytest.approx(
        (r8b125["final_avg_exact_match"] - r8b125["base_avg_exact_match"]) / 100.0
    )
    assert r8b125["quality_gain_per_million_tokens"] == pytest.approx(
        (r8b125["final_avg_exact_match"] - r8b125["base_avg_exact_match"]) / 0.03
    )
    assert r8b125["peak_mps_allocated_gb"] == 2.5


def test_final_avg_helper():
    assert _final_avg([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]) == pytest.approx(0.5)


def test_pareto_table_flags_frontier():
    rows = configuration_table(None, results_fixture())
    table = pareto_table(rows, "training_seconds")
    flags = {row["config_id"]: row["on_pareto_frontier"] for row in table}

    # rank8_b125: fast and cheap; rank16_b500: highest quality... but
    # rank16_b125 (0.13 @120s) dominates nothing here; rank8_b500 (0.17)
    # beats rank16_b500 (0.16) with LESS time -> rank16_b500 dominated.
    assert flags["rank8_b125"] is True
    assert flags["rank8_b500"] is True
    assert flags["rank16_b500"] is False


def test_metrics_rows_long_format():
    rows = metrics_rows(results_fixture())
    rank8 = [row for row in rows if row["config_id"] == "rank8_b125"]
    assert len(rank8) == 12  # 4 states x 3 test events
    base_rows = [row for row in rank8 if row["state"] == "base"]
    assert all(row["exact_match"] in (0.04, 0.02, 0.12) for row in base_rows)


def test_resource_rows_include_totals():
    rows = resource_rows(results_fixture())
    totals = [row for row in rows if row["event_id"] == "TOTAL"]
    assert len(totals) == 4  # one per configuration
    r8b500 = next(row for row in totals if row["config_id"] == "rank8_b500")
    assert r8b500["training_seconds"] == 400.0


def test_write_csv_roundtrip(tmp_path):
    rows = [
        {"a": 1, "b": 0.123456789, "c": "text,with,commas", "d": None},
        {"a": 2, "b": 3.0, "c": "plain", "d": "x"},
    ]
    path = write_csv(tmp_path / "out.csv", rows)
    lines = path.read_text(encoding="utf-8").strip().split("\n")
    assert lines[0] == "a,b,c,d"
    assert lines[1] == "1,0.123457,text;with;commas,"
    assert lines[2] == "2,3.000000,plain,x"


def test_write_csv_rejects_empty(tmp_path):
    with pytest.raises(ValueError, match="No rows"):
        write_csv(tmp_path / "empty.csv", [])
