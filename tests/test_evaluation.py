
import pytest

from cycleft.evaluation import (
    evaluate_predictions,
    evaluate_qa,
    normalize_answer,
)


def test_normalize_answer():
    assert normalize_answer("  Project   Cadmus ") == "project cadmus"


def test_exact_match_accuracy():
    result = evaluate_predictions(
        ["Project Cadmus", "Alex Smith", "Metropolis"],
        ["project cadmus", "Wrong answer", "Metropolis"],
    )
    assert result["examples"] == 3
    assert result["correct"] == 2
    assert result["accuracy"] == pytest.approx(2 / 3)


def test_empty_evaluation_rejected():
    with pytest.raises(ValueError, match="empty dataset"):
        evaluate_predictions([], [])


def test_mismatched_lengths_rejected():
    with pytest.raises(ValueError, match="same length"):
        evaluate_predictions(["answer"], [])


def test_evaluate_qa():
    rows = [
        {"question": "Who?", "object": "Alex Smith"},
        {"question": "Which project?", "object": "Project Cadmus"},
    ]

    result = evaluate_qa(rows, lambda question: {
        "Who?": "Alex Smith",
        "Which project?": "Wrong",
    }[question])

    assert result["correct"] == 1
    assert result["accuracy"] == pytest.approx(0.5)
