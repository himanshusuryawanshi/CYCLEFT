"""Tests for E1 evaluation and continual-learning metrics."""

import pytest

from cycleft.e1.metrics import (
    average_accuracy,
    average_forgetting,
    backward_transfer,
    backward_transfer_conventional,
    exact_match,
    extract_answer,
    forgetting,
    normalize_answer,
    token_f1,
    validate_accuracy_matrix,
)


def test_normalize_answer_handles_case_punctuation_whitespace():
    assert normalize_answer("  Project   Cadmus ") == "project cadmus"
    assert normalize_answer("St. Louis, MO!") == "st louis mo"
    assert normalize_answer("The-Answer") == "theanswer"


def test_exact_match_is_normalized():
    assert exact_match("Project Cadmus.", "project cadmus") == 1.0
    assert exact_match("CNN", "C.N.N.") == 1.0
    assert exact_match("wrong", "right") == 0.0


def test_token_f1_full_partial_and_zero():
    assert token_f1("Project Cadmus", "Project Cadmus") == 1.0
    # half the reference tokens predicted
    assert token_f1("Project", "Project Cadmus") == pytest.approx(2 / 3)
    assert token_f1("nothing", "shared words here") == 0.0
    assert token_f1("", "") == 1.0
    assert token_f1("", "answer") == 0.0


def test_extract_answer_rules_are_state_independent():
    assert extract_answer("Project Cadmus") == "Project Cadmus"
    assert extract_answer("Project Cadmus\nand more text") == "Project Cadmus"
    assert extract_answer("The answer is Project Cadmus.") == "Project Cadmus"
    assert extract_answer("Answer: Project Cadmus") == "Project Cadmus"
    assert extract_answer('  "Project Cadmus" ') == "Project Cadmus"
    # No hand-tuning per example: same rule for everything.
    assert extract_answer("It is definitely Project Cadmus") == (
        "It is definitely Project Cadmus"
    )


def test_extract_answer_preserves_punctuated_entities():
    assert extract_answer("St. Louis") == "St. Louis"


# Matrix layout: row 0 = base, row k = after event k. Columns = test events.
MATRIX = [
    [0.10, 0.08, 0.05],   # base
    [0.60, 0.12, 0.06],   # after event 1
    [0.50, 0.55, 0.07],   # after event 2
    [0.40, 0.45, 0.50],   # after event 3 (final)
]


def test_validate_accuracy_matrix_accepts_t_plus_one_by_t():
    assert validate_accuracy_matrix(MATRIX) == (3, 4)


def test_validate_accuracy_matrix_rejects_bad_shapes():
    with pytest.raises(ValueError, match="rectangular"):
        validate_accuracy_matrix([[0.1, 0.2], [0.3]])
    with pytest.raises(ValueError, match=r"T \+ 1"):
        validate_accuracy_matrix(
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]
        )
    with pytest.raises(ValueError, match="non-empty"):
        validate_accuracy_matrix([])


def test_forgetting_uses_max_over_k_ge_j_minus_final():
    # Event 1: best after event 1 is 0.60, final is 0.40 -> forgetting 0.20.
    assert forgetting(MATRIX, 1) == pytest.approx(0.20)
    # Event 2: best 0.55, final 0.45 -> 0.10.
    assert forgetting(MATRIX, 2) == pytest.approx(0.10)
    # Event 3: only k = 3 qualifies; final equals itself -> 0.
    assert forgetting(MATRIX, 3) == pytest.approx(0.0)


def test_forgetting_is_zero_when_final_model_is_best():
    matrix = [
        [0.0, 0.0],
        [0.5, 0.2],
        [0.6, 0.3],
    ]
    # The max over k >= j includes the final state, so forgetting is
    # always >= 0; it is exactly 0 when the final model is the best.
    assert forgetting(matrix, 1) == pytest.approx(0.0)
    assert forgetting(matrix, 2) == pytest.approx(0.0)


def test_average_forgetting():
    assert average_forgetting(MATRIX) == pytest.approx((0.20 + 0.10 + 0.0) / 3)


def test_backward_transfer_formula_documented():
    # BWT = mean over j of A(T, j) - A(j, j)
    # j=1: 0.40 - 0.60 = -0.20; j=2: 0.45 - 0.55 = -0.10; j=3: 0.50 - 0.50 = 0
    assert backward_transfer(MATRIX) == pytest.approx(-0.30 / 3)


def test_backward_transfer_conventional_uses_divisor_t_minus_one():
    # Conventional: only events 1..T-1, divisor T-1.
    # j=1: 0.40-0.60 = -0.20; j=2: 0.45-0.55 = -0.10 -> (-0.30)/2 = -0.15
    assert backward_transfer_conventional(MATRIX) == pytest.approx(-0.15)
    with pytest.raises(ValueError, match="at least 2"):
        backward_transfer_conventional([[0.1], [0.2]])


def test_average_accuracy_per_state():
    assert average_accuracy(MATRIX, 0) == pytest.approx((0.10 + 0.08 + 0.05) / 3)
    assert average_accuracy(MATRIX, 3) == pytest.approx((0.40 + 0.45 + 0.50) / 3)
    with pytest.raises(ValueError, match=r"\[0, 3\]"):
        average_accuracy(MATRIX, 4)


def test_forgetting_rejects_out_of_range_event():
    with pytest.raises(ValueError, match=r"\[1, 3\]"):
        forgetting(MATRIX, 0)
    with pytest.raises(ValueError, match=r"\[1, 3\]"):
        forgetting(MATRIX, 4)
