
import pytest

from cycleft.data.temporalwiki import (
    SLICE_TAGS,
    to_training_example,
    load_temporal_slice,
)


def test_temporal_slices_are_ordered():
    assert len(SLICE_TAGS) == 3
    assert len(set(SLICE_TAGS)) == 3


def test_training_row_mapping():
    row = {
        "question": "Who is the coach?",
        "object": "Alex Smith",
        "slice_tag": SLICE_TAGS[0],
        "snapshot_new": "2025-12-01",
    }

    result = to_training_example(row)

    assert result["prompt"] == "Who is the coach?"
    assert result["completion"] == "Alex Smith"
    assert result["slice_tag"] == SLICE_TAGS[0]


def test_unknown_slice_rejected():
    with pytest.raises(ValueError, match="Unknown temporal slice"):
        load_temporal_slice("not_a_real_slice")
