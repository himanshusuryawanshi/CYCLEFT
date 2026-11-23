"""Tests for Pareto dominance and frontier extraction."""

import pytest

from cycleft.e2.pareto import (
    ParetoPoint,
    dominates,
    dominated_set,
    nondominated_set,
    quality_gain_per_resource,
)


def P(label, resource, quality):
    return ParetoPoint(label=label, resource=resource, quality=quality)


def test_dominates_basic():
    faster_better = P("a", 10, 0.5)
    assert dominates(faster_better, P("b", 20, 0.4))
    assert dominates(faster_better, P("b", 10, 0.5 - 1e-9))
    assert not dominates(faster_better, P("b", 10, 0.5))  # equal: no strict gain
    assert not dominates(faster_better, P("b", 5, 0.6))   # b is better
    assert not dominates(P("a", 10, 0.5), P("b", 5, 0.6)) # mixed


def test_nondominated_set_finds_frontier():
    points = [
        P("slow_good", 100, 0.30),
        P("mid", 50, 0.20),
        P("fast_ok", 25, 0.15),
        P("fast_bad", 25, 0.10),   # dominated by fast_ok
        P("slow_bad", 100, 0.10),  # dominated by slow_good
    ]
    front = nondominated_set(points)
    assert [p.label for p in front] == ["fast_ok", "mid", "slow_good"]


def test_dominated_set_complement():
    points = [
        P("a", 10, 0.5),
        P("b", 20, 0.4),  # dominated by a
        P("c", 30, 0.3),  # dominated by a
        P("d", 25, 0.6),  # frontier (better quality)
    ]
    assert [p.label for p in dominated_set(points)] == ["b", "c"]


def test_identical_points_are_both_kept():
    points = [P("x", 10, 0.5), P("y", 10, 0.5)]
    assert {p.label for p in nondominated_set(points)} == {"x", "y"}


def test_empty_and_single():
    assert nondominated_set([]) == []
    assert [p.label for p in nondominated_set([P("only", 1, 0.1)])] == ["only"]


def test_quality_gain_per_resource():
    assert quality_gain_per_resource(0.20, 0.10, 100.0) == pytest.approx(0.001)
    assert quality_gain_per_resource(0.10, 0.10, 100.0) == 0.0
    assert quality_gain_per_resource(0.20, 0.10, 0.0) is None
    assert quality_gain_per_resource(0.20, 0.10, -5.0) is None
