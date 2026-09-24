from dataclasses import dataclass
from typing import Iterator, Sequence


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