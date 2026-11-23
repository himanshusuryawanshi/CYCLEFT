"""Tests for the E2 grid: budgets, configurations, semantics."""

import pytest

from cycleft.e1.lm import LMRuntimeConfig
from cycleft.e2.grid import (
    BUDGET_FRACTIONS,
    RANKS,
    assert_budget_semantics,
    budget_examples_for,
    build_grid,
    configuration_row,
    make_configuration,
)


def test_budget_examples_resolve_to_expected_counts():
    assert budget_examples_for("b125") == 125
    assert budget_examples_for("b250") == 250
    assert budget_examples_for("b500") == 500


def test_unknown_budget_or_rank_rejected():
    with pytest.raises(ValueError, match="Unknown budget"):
        budget_examples_for("b10")
    with pytest.raises(ValueError, match="Unknown LoRA rank"):
        make_configuration(64, "b125")


def test_configuration_ids_and_expected_steps():
    config = make_configuration(16, "b250")
    assert config.config_id == "rank16_b250"
    assert config.budget_fraction == 0.5
    # batch_size=1, grad_accum=1 -> optimizer steps == examples.
    assert config.expected_optimizer_steps_per_event == 250


def test_build_grid_is_rank_major_budget_minor():
    grid = build_grid(ranks=(4, 8), budget_ids=("b125", "b250"))
    assert [c.config_id for c in grid] == [
        "rank4_b125",
        "rank4_b250",
        "rank8_b125",
        "rank8_b250",
    ]
    full = build_grid()
    assert len(full) == len(RANKS) * len(BUDGET_FRACTIONS) == 12
    assert len({c.config_id for c in full}) == 12


def test_assert_budget_semantics():
    good = LMRuntimeConfig(model_id="m", batch_size=1,
                           gradient_accumulation_steps=1, epochs=1)
    assert_budget_semantics(good)

    bad = LMRuntimeConfig(model_id="m", gradient_accumulation_steps=4)
    with pytest.raises(ValueError, match="gradient_accumulation_steps=4"):
        assert_budget_semantics(bad)


def test_configuration_row_records_scaling():
    lm_config = LMRuntimeConfig(model_id="m", lora_alpha=16, seed=7)
    row = configuration_row(make_configuration(8, "b125"), lm_config)
    assert row["lora_scaling"] == 2.0
    assert row["seed"] == 7
    assert row["budget_examples_per_event"] == 125
    assert "q_proj" in row["lora_target_modules"]
