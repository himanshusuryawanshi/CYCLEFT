"""Tests for E2 resource measurement and aggregation."""

import json

import pytest

from cycleft.e2.resources import (
    EventResourceRecord,
    ResourceTracker,
    event_resource_record,
    total_optimizer_steps,
    total_training_seconds,
    total_training_tokens,
)


def test_resource_tracker_records_peak():
    readings = iter([0.5, 1.0, 0.75, 2.0])
    rss = iter([0.9, 1.5, 1.2, 1.8])
    tracker = ResourceTracker(
        allocated_sampler=lambda: next(readings),
        rss_sampler=lambda: next(rss),
    )
    for _ in range(4):
        tracker.sample()
    assert tracker.peak_allocated_gb == 2.0
    assert tracker.peak_rss_gb == 1.8
    assert tracker.samples == 4


def test_resource_tracker_default_sampler_never_negative():
    tracker = ResourceTracker(
        allocated_sampler=lambda: 0.0, rss_sampler=lambda: 0.0
    )
    tracker.sample()
    assert tracker.snapshot() == {
        "peak_mps_allocated_gb": 0.0,
        "peak_rss_gb": 0.0,
        "samples": 1,
    }


def training_stats_fixture(**overrides):
    stats = {
        "event_id": 1,
        "optimizer_steps": 125,
        "examples_scheduled": 125,
        "examples_skipped": 0,
        "total_tokens_trained": 10000,
        "answer_tokens_trained": 600,
        "training_seconds": 100.0,
        "average_loss": 2.0,
        "final_loss": 1.5,
    }
    stats.update(overrides)
    return stats


def test_event_resource_record_computes_throughput(tmp_path):
    tracker = ResourceTracker(allocated_sampler=lambda: 1.25, rss_sampler=lambda: 1.0)
    tracker.sample()
    record = event_resource_record(
        config_id="rank8_b125",
        lora_rank=8,
        budget_examples=125,
        training_stats=training_stats_fixture(),
        checkpoint_dir=tmp_path,
        continued_from=None,
        tracker=tracker,
    )
    assert record.tokens_per_second == pytest.approx(100.0)  # 10000 tokens / 100 s
    assert record.optimizer_steps == 125
    assert record.examples_processed == 125
    assert record.checkpoint_size_bytes == 0  # empty dir
    assert record.memory["peak_mps_allocated_gb"] == 1.25


def test_event_resource_record_counts_skipped_examples(tmp_path):
    record = event_resource_record(
        config_id="rank4_b250",
        lora_rank=4,
        budget_examples=250,
        training_stats=training_stats_fixture(
            examples_scheduled=250, examples_skipped=3, optimizer_steps=247
        ),
        checkpoint_dir=tmp_path,
        continued_from="event_1",
        tracker=ResourceTracker(lambda: 0.0, lambda: 0.0),
    )
    assert record.examples_processed == 247
    assert record.examples_skipped == 3


def test_totals_aggregate_across_events():
    records = [
        EventResourceRecord(
            config_id="c", event_id=e, lora_rank=8, budget_examples=125,
            optimizer_steps=125, examples_processed=125, examples_skipped=0,
            total_tokens=1000 * e, answer_tokens=100,
            training_seconds=10.0 * e, tokens_per_second=0.0,
            checkpoint_size_bytes=0, checkpoint_dir="", continued_from=None,
        )
        for e in (1, 2, 3)
    ]
    assert total_optimizer_steps(records) == 375
    assert total_training_seconds(records) == pytest.approx(60.0)
    assert total_training_tokens(records) == 6000


def test_directory_size_bytes(tmp_path):
    (tmp_path / "a.bin").write_bytes(b"x" * 100)
    (tmp_path / "b.json").write_text("{}")
    from cycleft.e2.resources import directory_size_bytes

    assert directory_size_bytes(tmp_path) == 100 + 2
