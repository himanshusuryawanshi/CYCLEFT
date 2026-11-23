
from cycleft.data.stream import build_temporal_stream
from cycleft.data.temporalwiki import SLICE_TAGS


def test_temporal_stream_has_three_ordered_updates():
    stream = build_temporal_stream()

    assert len(stream) == 3
    assert [event.event_id for event in stream] == [1, 2, 3]


def test_temporal_stream_preserves_slice_order():
    stream = build_temporal_stream()

    for event, slice_tag in zip(stream, SLICE_TAGS):
        assert event.train_source == f"train:{slice_tag}"
        assert event.eval_source == f"test:{slice_tag}"
