import pytest

from cycleft.data.stream import ContinualStream, StreamEvent


def test_continual_stream():
    events = [
        StreamEvent(
            event_id=1,
            name="event_a",
            train_source="data/e1/event_a/train",
            eval_source="data/e1/event_a/eval",
        ),
        StreamEvent(
            event_id=2,
            name="event_b",
            train_source="data/e1/event_b/train",
            eval_source="data/e1/event_b/eval",
        ),
    ]

    stream = ContinualStream(events)

    assert len(stream) == 2
    assert stream[0].name == "event_a"
    assert stream[1].name == "event_b"


def test_empty_stream_rejected():
    with pytest.raises(ValueError):
        ContinualStream([])

def test_event_order():
    events = [
        StreamEvent(
            event_id=0,
            name="base",
            train_source="none",
            eval_source="none",
        ),
        StreamEvent(
            event_id=1,
            name="event_1",
            train_source="train_1",
            eval_source="eval_1",
        ),
        StreamEvent(
            event_id=2,
            name="event_2",
            train_source="train_2",
            eval_source="eval_2",
        ),
    ]

    stream = ContinualStream(events)

    assert [event.event_id for event in stream] == [0, 1, 2]