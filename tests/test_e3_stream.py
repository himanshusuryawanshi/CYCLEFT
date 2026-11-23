"""Tests for the E3 temporal stream construction and audit."""

import numpy as np
import pytest

from cycleft.e3.stream import (
    EXAMPLES_PER_EVENT,
    WINDOW_SIZE,
    E3Stream,
    build_stream,
    dataset_audit,
    load_slice_order_check,
    summarise_audit,
)


def make_synthetic_stream(n_per_event: int = 500) -> E3Stream:
    rows = [
        {"question": f"q{position}", "object": f"a{position}"}
        for position in range(3 * n_per_event)
    ]
    event_ids = [1] * n_per_event + [2] * n_per_event + [3] * n_per_event
    return E3Stream(rows=rows, event_ids=event_ids, window_size=WINDOW_SIZE, seed=42)


def test_stream_shape_and_boundaries():
    stream = make_synthetic_stream()
    assert stream.n_examples == 1500
    assert stream.n_windows == 30
    assert stream.boundaries == [500, 1000]
    assert stream.boundary_windows == [10, 20]


def test_windows_are_contiguous_and_sized():
    stream = make_synthetic_stream()
    covered = []
    for window in range(stream.n_windows):
        indices = stream.window(window)
        assert len(indices) == WINDOW_SIZE
        covered.extend(indices)
    assert covered == list(range(1500))


def test_stream_event_ids_match_boundaries():
    stream = make_synthetic_stream()
    assert stream.event_of_position(0) == 1
    assert stream.event_of_position(499) == 1
    assert stream.event_of_position(500) == 2
    assert stream.event_of_position(999) == 2
    assert stream.event_of_position(1000) == 3
    assert stream.event_of_position(1499) == 3


def test_real_stream_matches_prepared_event_order():
    """The stream must reuse the E1/E2 seed-42 order, event by event."""
    stream, events, _manifest = build_stream(seed=42)
    assert stream.n_examples == 1500
    offset = 0
    for event in events:
        for i, row in enumerate(event.train_rows):
            assert stream.rows[offset + i]["question"] == row["question"]
        offset += len(event.train_rows)
    assert offset == 1500


def test_real_stream_boundaries_are_two():
    stream, _events, _manifest = build_stream(seed=42)
    assert stream.boundaries == [500, 1000]
    order_check = load_slice_order_check(stream)
    assert order_check["matches_chronological_order"] is True
    assert order_check["contiguous"] is True


def test_audit_reports_two_boundaries_and_limitations():
    stream, events, manifest = build_stream(seed=42)
    audit = dataset_audit(stream, events, manifest)
    assert audit["known_boundaries"] == [500, 1000]
    assert audit["temporal_columns"]["per_example_timestamps"] is None
    assert audit["answer_column"] == "object"
    assert any("stationarity" in limitation for limitation in audit["limitations"])
    assert len(audit["cross_event_overlap"]) == 3
    for entry in audit["cross_event_overlap"]:
        # All shared facts change answers between events (temporal drift).
        assert entry["shared_facts_with_changed_answer"] == (
            entry["shared_subject_relation_facts"]
        )
    rendered = summarise_audit(audit, load_slice_order_check(stream))
    assert "TWO" in rendered
    assert "stationarity" in rendered
    assert "No per-example timestamps" in rendered


def test_audit_dataset_fingerprint_present():
    stream, events, manifest = build_stream(seed=42)
    audit = dataset_audit(stream, events, manifest)
    assert audit["dataset_id"] == manifest["dataset_id"]
