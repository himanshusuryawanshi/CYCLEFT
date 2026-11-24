"""Dataset preparation and integrity checks for E1.

Loads the TemporalWiki drift stream, verifies split integrity, selects a
deterministic training subset, and produces a dataset manifest.

The dataset's own train/test split is preserved as-is. The QA target is
the ``object`` column (the actual answer entity); metadata columns such
as ``continuous_reward`` are never used as targets.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from cycleft.data.base import DATASET_TEMPORALWIKI, DATASET_WIKIFACTDIFF
from cycleft.data.temporalwiki import DATASET_ID, SLICE_TAGS

QUESTION_COLUMN = "question"
ANSWER_COLUMN = "object"
SLICE_COLUMN = "slice_tag"


@dataclass(frozen=True)
class PreparedEvent:
    """One event with its (deterministic) training subset and eval sets.

    ``validation_rows`` is an optional, strictly separate model-selection
    population (added for the WikiFactDiff migration).  It defaults to an
    empty list, so every existing constructor — TemporalWiki and the
    E2–E7 consumers — keeps working unchanged.  ``eval_rows`` remain the
    held-out TEST population; validation is never mixed into them.
    """

    event_id: int
    name: str
    slice_tag: str
    train_rows: list[dict] = field(repr=False)
    eval_rows: list[dict] = field(repr=False)
    train_pool_size: int
    eval_pool_size: int
    train_indices_used: list[int]
    validation_rows: list[dict] = field(default_factory=list, repr=False)
    validation_pool_size: int = 0


def _dataset_fingerprint(dataset) -> str:
    """Best-effort stable identifier for the loaded dataset state."""
    fingerprint = getattr(dataset, "_fingerprint", None)
    version = getattr(getattr(dataset, "info", None), "version", None)
    parts = [str(fingerprint)]
    if version is not None:
        parts.append(str(version))
    return "|".join(parts)


def check_train_eval_overlap(
    train_rows: list[dict],
    eval_rows: list[dict],
) -> dict[str, int]:
    """Quantify accidental train/eval overlap without altering the split.

    Returns counts of (a) exact duplicate question strings and
    (b) duplicated (subject, relation, slice) facts. The dataset
    intentionally re-uses facts across splits with paraphrased prompts
    (train: ``direct``/``train_paraphrase``; test: ``paraphrase``), so a
    nonzero fact overlap is expected and reported, not corrected.
    """
    train_questions = {str(row[QUESTION_COLUMN]) for row in train_rows}
    eval_questions = {str(row[QUESTION_COLUMN]) for row in eval_rows}

    train_facts = {
        (row["subject"], row["relation"], row[SLICE_COLUMN]) for row in train_rows
    }
    eval_facts = {
        (row["subject"], row["relation"], row[SLICE_COLUMN]) for row in eval_rows
    }

    return {
        "exact_question_overlap": len(train_questions & eval_questions),
        "fact_overlap": len(train_facts & eval_facts),
        "eval_rows": len(eval_rows),
    }


def deterministic_subset(
    rows: list[dict],
    limit: int | None,
    seed: int,
) -> tuple[list[dict], list[int]]:
    """Select a deterministic, seed-shuffled subset of ``rows``.

    A seeded shuffle (rather than taking the head of the dataset) avoids
    subject-clustering bias when the dataset is sorted. Returns the
    selected rows in training order plus their original indices.
    """
    if limit is None or limit >= len(rows):
        indices = list(range(len(rows)))
        random.Random(seed).shuffle(indices)
        return [rows[i] for i in indices], indices

    if limit <= 0:
        raise ValueError(f"limit must be a positive integer or None, got {limit!r}.")

    indices = sorted(random.Random(seed).sample(range(len(rows)), limit))
    selected = [rows[i] for i in indices]
    random.Random(seed).shuffle(selected)
    return selected, indices


def load_slice_rows(
    split: str,
    slice_tags: tuple[str, ...] = SLICE_TAGS,
) -> tuple[dict[str, list[dict]], str]:
    """Load one split of the TemporalWiki dataset grouped by slice tag.

    Returns the grouped rows and a dataset fingerprint string.
    """
    from datasets import load_dataset

    dataset = load_dataset(DATASET_ID, split=split)
    grouped: dict[str, list[dict]] = {tag: [] for tag in slice_tags}
    for row in dataset:
        tag = row[SLICE_COLUMN]
        if tag in grouped:
            grouped[tag].append(dict(row))

    missing = [tag for tag, rows in grouped.items() if not rows]
    if missing and split != "stable":
        raise ValueError(f"No rows found for slices {missing} in split {split}.")
    return grouped, _dataset_fingerprint(dataset)


def prepare_event_datasets(
    slice_tags: tuple[str, ...] = SLICE_TAGS,
    train_limit: int | None = None,
    eval_limit: int | None = None,
    seed: int = 42,
    dataset: str = DATASET_TEMPORALWIKI,
    dataset_config: dict[str, Any] | None = None,
    mode: str | None = None,
) -> tuple[list[PreparedEvent], dict[str, Any]]:
    """Prepare all continual-learning events plus a serializable manifest.

    Dataset dispatch: the default ``dataset="temporalwiki"`` preserves the
    original E1 behaviour byte-for-byte (E2–E7 call this function without
    the new keyword arguments).  ``dataset="wikifactdiff"`` delegates to
    the WikiFactDiff adapter (``cycleft.data.wikifactdiff``), which
    implements its own event definitions, subject-disjoint partitioning
    and leakage masks; ``slice_tags`` is ignored for it.

    Args:
        slice_tags: Temporal slices in stream order (TemporalWiki only).
        train_limit: Training examples per event (``None`` = all).
        eval_limit: Reserved for debugging; must stay ``None`` in real
            runs. The evaluation set is never reduced to inflate scores.
        seed: Seed controlling the deterministic training subset.
        mode: 'pilot' or 'full' — derives the evaluation population when
            the adapter config does not pin one explicitly.
        dataset: Registered dataset adapter name.
        dataset_config: Adapter-specific options (pilot eval populations,
            split fractions, subset name, ...).
    """
    if eval_limit is not None:
        raise ValueError(
            "eval_limit is only for throwaway debugging; E1 always "
            "evaluates the full held-out test set."
        )

    if dataset == DATASET_WIKIFACTDIFF:
        from cycleft.data.wikifactdiff import prepare_wikifactdiff_events

        return prepare_wikifactdiff_events(
            train_limit=train_limit,
            seed=seed,
            mode=mode,
            dataset_config=dataset_config or {},
        )
    if dataset != DATASET_TEMPORALWIKI:
        raise ValueError(
            f"Unknown dataset adapter {dataset!r}; expected "
            f"{DATASET_TEMPORALWIKI!r} or {DATASET_WIKIFACTDIFF!r}."
        )

    train_splits, train_fingerprint = load_slice_rows("train", slice_tags)
    test_splits, test_fingerprint = load_slice_rows("test", slice_tags)

    events: list[PreparedEvent] = []
    manifest_events: list[dict[str, Any]] = []

    for event_id, slice_tag in enumerate(slice_tags, start=1):
        train_pool = train_splits[slice_tag]
        eval_rows = test_splits[slice_tag]

        train_rows, train_indices = deterministic_subset(train_pool, train_limit, seed)
        overlap = check_train_eval_overlap(train_pool, eval_rows)

        events.append(
            PreparedEvent(
                event_id=event_id,
                name=f"temporal_update_{event_id}",
                slice_tag=slice_tag,
                train_rows=train_rows,
                eval_rows=eval_rows,
                train_pool_size=len(train_pool),
                eval_pool_size=len(eval_rows),
                train_indices_used=train_indices,
            )
        )

        manifest_events.append(
            {
                "event_id": event_id,
                "name": f"temporal_update_{event_id}",
                "slice_tag": slice_tag,
                "train_split": f"train:slice_tag={slice_tag}",
                "eval_split": f"test:slice_tag={slice_tag}",
                "n_train_pool": len(train_pool),
                "n_train_used": len(train_rows),
                "n_eval": len(eval_rows),
                "train_indices_used": train_indices,
                "question_column": QUESTION_COLUMN,
                "answer_column": ANSWER_COLUMN,
                "qa_style_train": dict(Counter(r["qa_style"] for r in train_pool)),
                "qa_style_eval": dict(Counter(r["qa_style"] for r in eval_rows)),
                "train_eval_overlap": overlap,
            }
        )

    manifest = {
        "dataset_id": DATASET_ID,
        "slices": list(slice_tags),
        "seed": seed,
        "train_limit": train_limit,
        "eval_limit": eval_limit,
        "question_column": QUESTION_COLUMN,
        "answer_column": ANSWER_COLUMN,
        "events": manifest_events,
        "dataset_fingerprint": {
            "train": train_fingerprint,
            "test": test_fingerprint,
        },
        "notes": [
            "The provided train/test split is preserved; evaluation rows are never trained on.",
            "Answers come from the 'object' column; 'continuous_reward' is dataset metadata, not a target.",
            "Test prompts are paraphrases; train prompts include 'direct' and 'train_paraphrase' styles.",
            "Test rows cover the same (subject, relation) facts as train rows by dataset design; exact "
            "question-string overlap is reported per event.",
        ],
    }
    return events, manifest
