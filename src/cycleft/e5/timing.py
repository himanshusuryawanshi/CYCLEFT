"""E5 timing instrumentation (Phase 2).

perf_counter_ns-based category registry. Active training time is kept
separate from process wall-clock; evaluation, checkpointing, restoration and
loading are separate categories. For 'train' intervals the MPS device is
synchronized before and after the timed interval (per the Phase 2
instrumentation brief).
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any

CATEGORIES = (
    "train",              # active optimizer-step execution
    "model_load",         # base-model construction/loading
    "lora_attach",        # LoRA attach / adapter-weight load into the model
    "optimizer_restore",  # optimizer state_dict load
    "rng_restore",        # RNG state restoration
    "ckpt_serialize",     # bundle serialization (adapter + optimizer + rng write)
    "ckpt_hash_verify",   # hashing / integrity verification
    "ckpt_fsync_commit",  # fsync + atomic rename + LATEST pointer
    "durable_copy",       # local durable-transfer EMULATION (labeled; not cloud latency)
    "ckpt_restore",       # bundle load + verification on resume
    "eval_generation",    # evaluation generation + scoring
    "process_wall",       # total process wall-clock (informational; not training time)
)


class TimingRegistry:
    """Accumulates per-category elapsed nanoseconds via perf_counter_ns."""

    def __init__(self, exec_id: str = "", mps_sync: bool = True):
        self.exec_id = exec_id
        self.mps_sync = mps_sync
        self.totals_ns: dict[str, int] = {c: 0 for c in CATEGORIES}
        self.counts: dict[str, int] = {c: 0 for c in CATEGORIES}

    def start(self, category: str) -> int:
        if category not in self.totals_ns:
            raise KeyError(f"unknown timing category {category!r}")
        if category == "train" and self.mps_sync and torch_mps_available():
            import torch

            torch.mps.synchronize()
        return time.perf_counter_ns()

    def stop(self, category: str, token: int) -> int:
        if category == "train" and self.mps_sync and torch_mps_available():
            import torch

            torch.mps.synchronize()
        elapsed = time.perf_counter_ns() - token
        self.totals_ns[category] += elapsed
        self.counts[category] += 1
        return elapsed

    @contextmanager
    def timed(self, category: str):
        token = self.start(category)
        try:
            yield token
        finally:
            self.stop(category, token)

    def add(self, category: str, elapsed_ns: int, count: int = 1) -> None:
        self.totals_ns[category] += elapsed_ns
        self.counts[category] += count

    def merge(self, other: "TimingRegistry") -> None:
        """Aggregate another process's registry (per-process records retained
        by the caller; this merges totals only)."""
        for category in CATEGORIES:
            self.totals_ns[category] += other.totals_ns.get(category, 0)
            self.counts[category] += other.counts.get(category, 0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "exec_id": self.exec_id,
            "clock": "time.perf_counter_ns",
            "mps_sync_around_train": self.mps_sync,
            "categories": {
                c: {
                    "total_ns": self.totals_ns[c],
                    "total_s": round(self.totals_ns[c] / 1e9, 6),
                    "count": self.counts[c],
                }
                for c in CATEGORIES
                if self.counts[c] or self.totals_ns[c]
            },
            "note": (
                "train = active optimizer-step execution (device-synchronized "
                "intervals); process downtime, waiting and evaluation are "
                "excluded; process_wall is informational and never added to train"
            ),
        }


def torch_mps_available() -> bool:
    import torch

    return torch.backends.mps.is_available()
