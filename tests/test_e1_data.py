"""Tests for E1 dataset preparation, split integrity and determinism."""

import pytest
from datasets import Dataset

from cycleft.e1.data import (
    SLICE_TAGS,
    check_train_eval_overlap,
    deterministic_subset,
    prepare_event_datasets,
)


def make_rows(slice_tag: str, split: str, n: int, prefix: str) -> list[dict]:
    # Test questions are phrased differently (paraphrases), mirroring the
    # real TemporalWiki-CL split design.
    question_template = (
        "{prefix} question {i} as of 2026-01-01?"
        if split == "train"
        else "{prefix} asks about item {i}: which one?"
    )
    return [
        {
            "question": question_template.format(prefix=prefix, i=i),
            "object": f"answer {i}",
            "slice_tag": slice_tag,
            "subject": f"{prefix} subject {i}",
            "relation": "member of",
            "qa_style": "train_paraphrase" if split == "train" else "paraphrase",
        }
        for i in range(n)
    ]


def fake_loader(monkeypatch, train_by_slice, test_by_slice):
    """Patch datasets.load_dataset used inside cycleft.e1.data."""

    def fake_load_dataset(_dataset_id, split):
        rows = []
        for slice_tag, slice_rows in (train_by_slice if split == "train" else test_by_slice).items():
            rows.extend(slice_rows)
        return Dataset.from_list(rows)

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)


def test_deterministic_subset_is_reproducible():
    rows = make_rows("pair_a", "train", 50, "x")
    first, indices_first = deterministic_subset(rows, 10, seed=42)
    second, indices_second = deterministic_subset(rows, 10, seed=42)
    assert indices_first == indices_second
    assert first == second
    assert len(first) == 10
    assert len({id(row) for row in first}) == 10

    other, _ = deterministic_subset(rows, 10, seed=7)
    assert first != other


def test_deterministic_subset_full_shuffles_order():
    rows = make_rows("pair_a", "train", 20, "x")
    shuffled, indices = deterministic_subset(rows, None, seed=3)
    assert len(shuffled) == 20
    assert sorted(indices) == list(range(20))


def test_deterministic_subset_rejects_bad_limit():
    with pytest.raises(ValueError, match="limit"):
        deterministic_subset([{"a": 1}], 0, seed=1)


def test_check_train_eval_overlap_counts():
    train = make_rows("pair_a", "train", 5, "x")
    eval_rows = make_rows("pair_a", "test", 3, "x")
    # Same subjects/relations but distinct (paraphrased) question strings.
    overlap = check_train_eval_overlap(train, eval_rows)
    assert overlap["exact_question_overlap"] == 0
    assert overlap["fact_overlap"] == 3
    assert overlap["eval_rows"] == 3

    eval_rows.append({"question": train[0]["question"], "object": "answer 0",
                      "slice_tag": "pair_a", "subject": "x subject 0",
                      "relation": "member of", "qa_style": "paraphrase"})
    overlap = check_train_eval_overlap(train, eval_rows)
    assert overlap["exact_question_overlap"] == 1


def test_prepare_event_datasets_builds_manifest(monkeypatch):
    slice_tags = ("pair_a", "pair_b")
    train_by_slice = {
        "pair_a": make_rows("pair_a", "train", 20, "a"),
        "pair_b": make_rows("pair_b", "train", 20, "b"),
    }
    test_by_slice = {
        "pair_a": make_rows("pair_a", "test", 6, "a"),
        "pair_b": make_rows("pair_b", "test", 6, "b"),
    }
    fake_loader(monkeypatch, train_by_slice, test_by_slice)

    events, manifest = prepare_event_datasets(
        slice_tags=slice_tags, train_limit=8, seed=42
    )

    assert [e.event_id for e in events] == [1, 2]
    assert manifest["dataset_id"] == "saxenan3/temporalwiki-drift-cl-easy"
    assert manifest["seed"] == 42
    assert manifest["train_limit"] == 8
    assert manifest["question_column"] == "question"
    assert manifest["answer_column"] == "object"

    for event, event_manifest in zip(events, manifest["events"]):
        assert len(event.train_rows) == 8
        assert len(event.eval_rows) == 6
        assert event.eval_pool_size == 6
        assert len(event.train_indices_used) == 8
        assert event_manifest["n_train_used"] == 8
        assert event_manifest["n_eval"] == 6
        # Deterministic: same seed reproduces identical row selection.
        events_again, _ = prepare_event_datasets(
            slice_tags=slice_tags, train_limit=8, seed=42
        )
        assert [r["question"] for r in events_again[event.event_id - 1].train_rows] == [
            r["question"] for r in event.train_rows
        ]


def test_prepare_event_datasets_preserves_provided_split(monkeypatch):
    slice_tags = ("pair_a",)
    train_by_slice = {"pair_a": make_rows("pair_a", "train", 10, "a")}
    test_by_slice = {"pair_a": make_rows("pair_a", "test", 4, "a")}
    fake_loader(monkeypatch, train_by_slice, test_by_slice)

    events, _ = prepare_event_datasets(slice_tags=slice_tags, train_limit=None, seed=1)
    eval_questions = {row["question"] for row in events[0].eval_rows}
    train_questions = {row["question"] for row in events[0].train_rows}
    assert eval_questions.isdisjoint(train_questions)


def test_prepare_event_datasets_rejects_eval_limit():
    with pytest.raises(ValueError, match="eval_limit"):
        prepare_event_datasets(eval_limit=5)


def test_slice_tags_constant_matches_stream_config():
    assert SLICE_TAGS == (
        "pair_20251120_20251201",
        "pair_20251201_20260101",
        "pair_20260101_20260201",
    )
