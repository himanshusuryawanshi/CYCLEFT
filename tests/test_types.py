import pytest
from cycleft.types import CycleResult, Domain, EvaluationResult


def test_domain():
    domain = Domain(
        name="domain_a",
        train_tokens=1000,
        eval_tokens=200,
    )

    assert domain.name == "domain_a"
    assert domain.train_tokens == 1000
    assert domain.eval_tokens == 200


def test_cycle_result():
    evaluation = EvaluationResult(
        new_domain_quality=0.85,
        old_domain_quality_pre=0.90,
        old_domain_quality_post=0.84,
    )

    result = CycleResult(
        cycle_id=1,
        domain="domain_a",
        training_tokens=1000,
        training_seconds=60.0,
        evaluation=evaluation,
    )

    assert result.cycle_id == 1
    assert result.domain == "domain_a"
    assert result.training_tokens == 1000
    assert result.evaluation.new_domain_quality == 0.85


def test_forgetting():
    evaluation = EvaluationResult(
        new_domain_quality=0.85,
        old_domain_quality_pre=0.90,
        old_domain_quality_post=0.84,
    )

    assert evaluation.forgetting == pytest.approx(0.06)


def test_no_forgetting_when_quality_improves():
    evaluation = EvaluationResult(
        new_domain_quality=0.85,
        old_domain_quality_pre=0.90,
        old_domain_quality_post=0.93,
    )

    assert evaluation.forgetting == 0.0