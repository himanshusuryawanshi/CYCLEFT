from dataclasses import dataclass
from typing import Iterator, Sequence
from cycleft.data.temporalwiki import SLICE_TAGS

@dataclass(frozen=True)
class StreamEvent:
    """One continual-learning adaptation event."""

    event_id: int
    name: str
    train_source: str
    eval_source: str


class ContinualStream:
    """Ordered sequence of continual-learning events."""

    def __init__(self, events: Sequence[StreamEvent]):
        if not events:
            raise ValueError("Continual stream must contain at least one event.")

        self._events = tuple(events)

    def __len__(self) -> int:
        return len(self._events)

    def __iter__(self) -> Iterator[StreamEvent]:
        return iter(self._events)

    def __getitem__(self, index: int) -> StreamEvent:
        return self._events[index]
    
def build_temporal_stream() -> ContinualStream:
    """Build an ordered stream from verified TemporalWiki slices."""
    events = [
        StreamEvent(
            event_id=i + 1,
            name=f"temporal_update_{i + 1}",
            train_source=f"train:{slice_tag}",
            eval_source=f"test:{slice_tag}",
        )
        for i, slice_tag in enumerate(SLICE_TAGS)
    ]

    return ContinualStream(events)


def load_event_data(event: StreamEvent):
    """Load the training and evaluation datasets for one event."""
    from cycleft.data.temporalwiki import resolve_temporal_source

    train_data = resolve_temporal_source(event.train_source)
    eval_data = resolve_temporal_source(event.eval_source)

    if len(train_data) == 0:
        raise ValueError(f"Event {event.event_id} has no training rows.")

    if len(eval_data) == 0:
        raise ValueError(f"Event {event.event_id} has no evaluation rows.")

    return train_data, eval_data
