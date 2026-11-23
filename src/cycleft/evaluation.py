
"""Evaluation utilities for continual adaptation."""

from collections.abc import Callable, Iterable
from typing import Any


def normalize_answer(text: str) -> str:
    """Normalize an answer for case- and whitespace-insensitive matching."""
    return " ".join(str(text).strip().lower().split())


def evaluate_predictions(
    references: Iterable[str],
    predictions: Iterable[str],
) -> dict[str, Any]:
    """Compute exact-match accuracy for predictions and references."""
    references = list(references)
    predictions = list(predictions)

    if len(references) != len(predictions):
        raise ValueError(
            "References and predictions must have the same length."
        )

    if not references:
        raise ValueError("Cannot evaluate an empty dataset.")

    correct = sum(
        normalize_answer(reference) == normalize_answer(prediction)
        for reference, prediction in zip(references, predictions)
    )

    return {
        "examples": len(references),
        "correct": correct,
        "accuracy": correct / len(references),
    }


def evaluate_qa(
    dataset,
    predict_fn: Callable[[str], str],
) -> dict[str, Any]:
    """Evaluate a predictor on rows containing question and object fields."""
    references = []
    predictions = []

    for row in dataset:
        if "question" not in row or "object" not in row:
            raise ValueError(
                "Each evaluation row must contain 'question' and 'object'."
            )

        predictions.append(str(predict_fn(str(row["question"]))))
        references.append(str(row["object"]))

    return evaluate_predictions(references, predictions)
