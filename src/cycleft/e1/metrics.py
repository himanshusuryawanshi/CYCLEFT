"""Evaluation and continual-learning metrics for E1.

All metrics operate on plain Python floats so they can be unit-tested
without loading any model.

Sign conventions (documented in the E1 README as well):

* ``forgetting(j) = max over k >= j of A(k, j)  -  A(T, j)``

  where ``A(k, j)`` is the accuracy on test event ``j`` after the model
  has been trained on events ``1..k`` (``k = 0`` is the base model) and
  ``T`` is the final event. Positive values mean the model's accuracy on
  event ``j`` dropped between its best post-training point and the end
  of the stream (i.e. forgetting happened). Zero or negative values
  mean no forgetting (or late-run improvement) on that event.

* ``backward_transfer = mean over j of ( A(T, j) - A(j, j) )``

  i.e. the average change in accuracy on each event's test set from the
  moment the model *just finished learning that event* to the end of the
  stream. Negative values indicate net forgetting; positive values
  indicate that later training helped earlier events.
"""

from __future__ import annotations

import re
import string
from typing import Sequence

_PUNCTUATION_TABLE = str.maketrans("", "", string.punctuation)


def normalize_answer(text: str) -> str:
    """Normalize an answer for case-, punctuation- and whitespace-insensitive matching."""
    text = str(text).strip().lower()
    text = text.translate(_PUNCTUATION_TABLE)
    return " ".join(text.split())


def exact_match(prediction: str, reference: str) -> float:
    """Return 1.0 when normalized prediction equals normalized reference."""
    return float(normalize_answer(prediction) == normalize_answer(reference))


def token_f1(prediction: str, reference: str) -> float:
    """Token-level F1 between prediction and reference (SQuAD-style).

    Both strings are normalized with :func:`normalize_answer` before
    tokenization. Returns 1.0 only when the token bags match exactly.
    """
    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = normalize_answer(reference).split()

    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    common = {}
    for token in pred_tokens:
        common[token] = common.get(token, 0) + 1

    overlap = 0
    for token in ref_tokens:
        if common.get(token, 0) > 0:
            overlap += 1
            common[token] -= 1

    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2.0 * precision * recall / (precision + recall)


_EXTRACTION_PREFIXES = (
    "the answer is",
    "answer:",
    "answer is",
    "a:",
)


def extract_answer(raw_generation: str) -> str:
    """Transparent, state-independent answer extraction.

    Rules (applied identically to every prediction of every model state,
    never hand-edited per example):

    1. strip surrounding whitespace;
    2. cut at the first newline;
    3. strip a small fixed list of leading prefixes such as
       ``"The answer is"`` or ``"Answer:"``;
    4. strip surrounding quotes and one trailing period.
    """
    text = str(raw_generation).strip()
    text = text.split("\n", 1)[0].strip()

    lowered = text.lower()
    for prefix in _EXTRACTION_PREFIXES:
        if lowered.startswith(prefix):
            text = text[len(prefix):].strip()
            break

    text = text.strip("\"'").strip()
    if text.endswith("."):
        text = text[:-1].strip()
    return text


def validate_accuracy_matrix(
    matrix: Sequence[Sequence[float]],
) -> tuple[int, int]:
    """Validate a (T + 1) x T accuracy matrix; return (T, num_rows - 1)."""
    if not matrix:
        raise ValueError("Accuracy matrix must be non-empty.")
    if any(len(row) != len(matrix[0]) for row in matrix):
        raise ValueError("Accuracy matrix must be rectangular.")
    if len(matrix[0]) != len(matrix) - 1:
        raise ValueError(
            "Expected (T + 1) rows (base + T training states) and T columns "
            f"(test events); got {len(matrix)} rows x {len(matrix[0])} columns."
        )
    return len(matrix[0]), len(matrix)


def _state_indices_after_event(t: int, t_final: int) -> list[int]:
    """Row indices k (states) with k >= t, where row 0 is the base model.

    Row ``t`` is the state right after training on event ``t`` (rows are
    0 = base, 1 = after event 1, ..., T = after event T). For event ``t``
    the relevant states are rows t..T inclusive.
    """
    return list(range(t, t_final + 1))


def forgetting(
    matrix: Sequence[Sequence[float]],
    event: int,
) -> float:
    """Forgetting for 1-based event ``event``.

    ``forgetting(event) = max over k >= event of A(k, event) - A(T, event)``.

    Row ``k`` of ``matrix`` is the accuracy on all test events after the
    model was trained on events ``1..k`` (row 0 = base model). The result
    is always >= 0 by construction (the max includes the final state):
    positive means forgetting, zero means no forgetting relative to the
    best post-training point for that event.
    """
    t_final, _ = validate_accuracy_matrix(matrix)
    if not 1 <= event <= t_final:
        raise ValueError(f"event must be in [1, {t_final}], got {event}.")

    column = event - 1
    best = max(matrix[k][column] for k in _state_indices_after_event(event, t_final))
    return best - matrix[t_final][column]


def average_forgetting(matrix: Sequence[Sequence[float]]) -> float:
    """Mean forgetting across events 1..T (positive = net forgetting)."""
    t_final, _ = validate_accuracy_matrix(matrix)
    return sum(forgetting(matrix, event) for event in range(1, t_final + 1)) / t_final


def backward_transfer(matrix: Sequence[Sequence[float]]) -> float:
    """Backward transfer (documented formula above).

    ``BWT = (1 / T) * sum_j ( A(T, j) - A(j, j) )``

    Negative = net forgetting; positive = later events improved earlier
    test performance. The all-events (divisor T) variant is used rather
    than the Lopez-Paz & Ranzato divisor (T - 1) so the number is the
    plain mean per-event change; both are common in the literature and
    the choice is documented here.
    """
    t_final, _ = validate_accuracy_matrix(matrix)
    deltas = [
        matrix[t_final][j] - matrix[j + 1][j] for j in range(t_final)
    ]
    return sum(deltas) / t_final


def backward_transfer_conventional(matrix: Sequence[Sequence[float]]) -> float:
    """Conventional Lopez-Paz & Ranzato (2017) backward transfer.

    ``BWT = (1 / (T - 1)) * sum_{j=1}^{T-1} ( A(T, j) - A(j, j) )``

    Averages only over events that received subsequent training (the last
    event's own delta is trivially zero and excluded, divisor T - 1).
    Negative = net forgetting. E1's ``backward_transfer`` averages over
    all T events (divisor T); both are reported side by side in E2.
    """
    t_final, _ = validate_accuracy_matrix(matrix)
    if t_final < 2:
        raise ValueError(
            "backward_transfer_conventional requires at least 2 events."
        )
    deltas = [
        matrix[t_final][j] - matrix[j + 1][j] for j in range(t_final - 1)
    ]
    return sum(deltas) / (t_final - 1)


def average_accuracy(
    matrix: Sequence[Sequence[float]],
    state: int,
) -> float:
    """Mean accuracy of model state ``state`` (0 = base) across all test events."""
    t_final, _ = validate_accuracy_matrix(matrix)
    if not 0 <= state <= t_final:
        raise ValueError(f"state must be in [0, {t_final}], got {state}.")
    return sum(matrix[state]) / t_final


def mean(values: Sequence[float]) -> float:
    """Arithmetic mean of a non-empty sequence."""
    if not values:
        raise ValueError("Cannot average an empty sequence.")
    return sum(values) / len(values)
