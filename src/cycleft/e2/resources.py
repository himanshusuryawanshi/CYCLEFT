"""Resource measurement for E2: memory peaks, timing, token accounting.

All values are *measured* on the executing machine. Monetary figures are
never produced here; if a downstream report converts time to cost it must
state an explicit hardware + hourly-price assumption and label it as
modeled, not billed.
"""

from __future__ import annotations

import resource
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


def _mps_allocated_gb() -> float:
    import torch

    if torch.backends.mps.is_available():
        return torch.mps.current_allocated_memory() / 1e9
    return 0.0


def _rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9


class ResourceTracker:
    """Track peak MPS allocation and peak RSS across a configuration run.

    ``sample()`` is cheap and is called after every optimizer step (via
    the ``train_event`` ``on_step`` hook) and around evaluation stages.
    """

    def __init__(
        self,
        allocated_sampler: Callable[[], float] | None = None,
        rss_sampler: Callable[[], float] | None = None,
    ):
        self._allocated_sampler = allocated_sampler or _mps_allocated_gb
        self._rss_sampler = rss_sampler or _rss_gb
        self.peak_allocated_gb = 0.0
        self.peak_rss_gb = 0.0
        self.samples = 0

    def sample(self) -> None:
        self.peak_allocated_gb = max(self.peak_allocated_gb, self._allocated_sampler())
        self.peak_rss_gb = max(self.peak_rss_gb, self._rss_sampler())
        self.samples += 1

    def snapshot(self) -> dict:
        return {
            "peak_mps_allocated_gb": round(self.peak_allocated_gb, 4),
            "peak_rss_gb": round(self.peak_rss_gb, 4),
            "samples": self.samples,
        }


@dataclass
class EventResourceRecord:
    """Measured resource consumption for one (configuration, event) pair."""

    config_id: str
    event_id: int
    lora_rank: int
    budget_examples: int
    optimizer_steps: int
    examples_processed: int
    examples_skipped: int
    total_tokens: int
    answer_tokens: int
    training_seconds: float
    tokens_per_second: float
    checkpoint_size_bytes: int
    checkpoint_dir: str
    continued_from: str | None
    memory: dict = field(default_factory=dict)


def event_resource_record(
    config_id: str,
    lora_rank: int,
    budget_examples: int,
    training_stats: dict,
    checkpoint_dir: Path,
    continued_from: str | None,
    tracker: ResourceTracker,
) -> EventResourceRecord:
    """Build a resource record from measured training statistics."""
    total_tokens = training_stats["total_tokens_trained"]
    seconds = training_stats["training_seconds"]
    return EventResourceRecord(
        config_id=config_id,
        event_id=training_stats.get("event_id", 0),
        lora_rank=lora_rank,
        budget_examples=budget_examples,
        optimizer_steps=training_stats["optimizer_steps"],
        examples_processed=training_stats["examples_scheduled"]
        - training_stats["examples_skipped"],
        examples_skipped=training_stats["examples_skipped"],
        total_tokens=total_tokens,
        answer_tokens=training_stats["answer_tokens_trained"],
        training_seconds=seconds,
        tokens_per_second=total_tokens / seconds if seconds > 0 else 0.0,
        checkpoint_size_bytes=directory_size_bytes(checkpoint_dir),
        checkpoint_dir=str(checkpoint_dir),
        continued_from=continued_from,
        memory=tracker.snapshot(),
    )


def directory_size_bytes(path: Path) -> int:
    """Total size of all files in a checkpoint directory."""
    return sum(f.stat().st_size for f in path.iterdir() if f.is_file())


def total_training_seconds(records: list[EventResourceRecord]) -> float:
    return sum(record.training_seconds for record in records)


def total_training_tokens(records: list[EventResourceRecord]) -> int:
    return sum(record.total_tokens for record in records)


def total_optimizer_steps(records: list[EventResourceRecord]) -> int:
    return sum(record.optimizer_steps for record in records)
