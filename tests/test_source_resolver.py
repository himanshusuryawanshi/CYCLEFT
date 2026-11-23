
import pytest

from cycleft.data.stream import build_temporal_stream, load_event_data
from cycleft.data.temporalwiki import (
    SLICE_TAGS,
    resolve_temporal_source,
)


def test_resolver_rejects_invalid_source():
    with pytest.raises(ValueError, match="Invalid source"):
        resolve_temporal_source("invalid-source")


def test_resolver_rejects_unsupported_split():
    with pytest.raises(ValueError, match="Unsupported dataset split"):
        resolve_temporal_source(f"validation:{SLICE_TAGS[0]}")


def test_event_data_loads_train_and_test():
    event = build_temporal_stream()[0]
    train_data, eval_data = load_event_data(event)

    assert len(train_data) > 0
    assert len(eval_data) > 0
    assert train_data[0]["slice_tag"] == SLICE_TAGS[0]
    assert eval_data[0]["slice_tag"] == SLICE_TAGS[0]
