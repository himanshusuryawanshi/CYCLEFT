"""Dataset adapter protocol for CYCLEFT event streams.

E1 (and downstream E2–E7) consume ``PreparedEvent`` objects produced by
``cycleft.e1.data.prepare_event_datasets``.  Historically that function was
hard-wired to the TemporalWiki drift stream.  This module introduces a
lightweight adapter contract so multiple datasets can feed the *same*
event interface without touching the training/evaluation pipeline.

Design rules (frozen after the WikiFactDiff migration audits):

* An adapter turns one dataset into an ordered list of continual-learning
  events plus a serializable manifest.  It owns all dataset-specific
  logic: schema mapping, event definition, partitioning, leakage masks
  and answer normalization.
* The adapter MUST keep ``PreparedEvent`` backward-compatible: training
  rows carry ``question``/``object`` (consumed by
  ``cycleft.e1.lm.train_event``/``evaluate_event``); adapter-specific
  metadata travels as extra keys inside the row dictionaries so the
  frozen ``PreparedEvent`` dataclass and every downstream consumer keep
  working unchanged.
* Determinism: every stochastic decision must be derived from
  ``(salt, seed)`` through :mod:`hashlib` (never :func:`hash`, whose
  salting differs between Python processes) and must be re-derivable
  from the manifest alone.
* Leakage protection is the adapter's responsibility: train/eval prompt
  masks, subject disjointness, neighborhood exclusions, and excluded-
  record bookkeeping all belong in the manifest with explicit reasons.

Only the TemporalWiki adapter exists in the baseline; the WikiFactDiff
adapter registers itself in :data:`ADAPTERS`.  Unknown names fail closed.
"""

from __future__ import annotations

import hashlib
from typing import Any, Callable

DATASET_TEMPORALWIKI = "temporalwiki"
DATASET_WIKIFACTDIFF = "wikifactdiff"

# Registry populated by the adapter modules (kept lazy so importing this
# module never pulls dataset-specific dependencies such as `datasets`).
ADAPTERS: dict[str, Callable[[], Callable[..., tuple[list, dict]]]] = {}


def register_adapter(name: str, factory: Callable) -> None:
    """Register an adapter factory under ``name`` (fail on duplicates)."""
    if name in ADAPTERS:
        raise ValueError(f"Dataset adapter already registered: {name!r}")
    ADAPTERS[name] = factory


def stable_hash(payload: str) -> str:
    """Deterministic hex digest for partition assignment.

    :func:`hash` is salted per Python process and therefore not
    reproducible; every dataset-level partition decision must use this
    instead.
    """
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def stable_partition(subject_id: str, salt: str, fractions: tuple[float, ...]) -> str:
    """Assign a subject to one of ``fractions`` partitions deterministically.

    ``fractions`` are cumulative cut points, e.g. ``(0.8, 0.9)`` for an
    80/10/10 train/val/test split.  The subject is hashed with the salt
    into the unit interval; NOTE that this base helper is a plain hash
    assignment — relation-aware stratification is implemented by the
    WikiFactDiff adapter on top of it (a bare modulo/hash split does NOT
    stratify by relation and must not be claimed to).
    """
    u = int(stable_hash(f"{salt}|{subject_id}"), 16) / 2**256
    for name, cut in zip(("train", "val", "test"), fractions):
        if u < cut:
            return name
    return "test"


def require_mapping(value: Any, what: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{what} must be a mapping, got {type(value).__name__}.")
    return value
