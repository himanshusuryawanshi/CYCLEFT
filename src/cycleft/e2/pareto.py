"""Pareto frontier computation for E2.

A configuration ``a`` Pareto-dominates ``b`` iff ``a`` is at least as
good as ``b`` on every objective and strictly better on at least one.
Objectives: resource consumption (minimize: training seconds or tokens)
and final quality (maximize: average exact match).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class ParetoPoint:
    """One candidate on the quality-resource plane."""

    label: str
    resource: float      # minimize (training seconds or training tokens)
    quality: float       # maximize (final average exact match)
    payload: dict | None = None


def dominates(a: ParetoPoint, b: ParetoPoint) -> bool:
    """True iff ``a`` is at least as good everywhere and strictly better somewhere."""
    at_least_as_good = a.resource <= b.resource and a.quality >= b.quality
    strictly_better = a.resource < b.resource or a.quality > b.quality
    return at_least_as_good and strictly_better


def nondominated_set(points: Sequence[ParetoPoint]) -> list[ParetoPoint]:
    """Return the Pareto-nondominated points, sorted by resource cost.

    Duplicated (identical resource, quality) points are all kept: neither
    dominates the other, and the caller decides how to present ties.
    """
    if not points:
        return []
    front = [
        point
        for point in points
        if not any(dominates(other, point) for other in points if other is not point)
    ]
    return sorted(front, key=lambda p: (p.resource, -p.quality))


def dominated_set(points: Sequence[ParetoPoint]) -> list[ParetoPoint]:
    """Points that are Pareto-dominated by some other point."""
    front_labels = {p.label for p in nondominated_set(points)}
    return [p for p in points if p.label not in front_labels]


def quality_gain_per_resource(
    quality: float,
    baseline_quality: float,
    resource: float,
) -> float | None:
    """Quality improvement (absolute) per unit of measured resource.

    Returns ``None`` when the resource quantity is zero (division by
    zero is reported as missing rather than estimated).
    """
    if resource <= 0:
        return None
    return (quality - baseline_quality) / resource
