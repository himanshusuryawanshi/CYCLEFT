"""E3 detection and cost-quality metrics.

Detection accounting rules (fixed a priori, documented in the run README):

* A trigger ``matches`` a known boundary at stream position ``b`` iff its
  position ``t`` satisfies ``b <= t <= b + tolerance``. The FIRST such
  trigger is the detection; its delay is ``t - b`` examples.
* Any other trigger is a false positive (this includes early warnings,
  i.e. triggers within ``tolerance`` BEFORE a boundary, which are also
  reported separately and never counted as detections of the upcoming
  boundary).
* ``missed = n_boundaries - detected``; ``recall = detected / n_boundaries``.

With only two known boundaries, recall/delay comparisons carry minimal
statistical power; confusion counts are always reported alongside rates.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DetectionOutcome:
    """Per-strategy detection accounting against the known boundaries."""

    n_boundaries: int
    detected_boundaries: list[int]  # boundary positions detected (within tolerance)
    detection_delays: dict[int, int]  # boundary position -> delay in examples
    false_positive_positions: list[int]
    early_warning_positions: list[int]
    missed_boundaries: list[int]
    n_triggers: int
    tolerance_examples: int

    @property
    def true_positives(self) -> int:
        return len(self.detected_boundaries)

    @property
    def false_positives(self) -> int:
        return len(self.false_positive_positions)

    @property
    def missed(self) -> int:
        return len(self.missed_boundaries)

    @property
    def recall(self) -> float:
        return self.true_positives / self.n_boundaries if self.n_boundaries else 0.0

    def as_dict(self) -> dict:
        return {
            "n_boundaries": self.n_boundaries,
            "true_positives": self.true_positives,
            "detected_boundaries": self.detected_boundaries,
            "detection_delays_examples": {
                str(k): v for k, v in self.detection_delays.items()
            },
            "false_positives": self.false_positives,
            "false_positive_positions": self.false_positive_positions,
            "early_warning_positions": self.early_warning_positions,
            "missed": self.missed,
            "missed_boundaries": self.missed_boundaries,
            "recall": self.recall,
            "n_triggers_total": self.n_triggers,
            "tolerance_examples": self.tolerance_examples,
        }


def detection_accounting(
    trigger_positions: list[int],
    boundaries: list[int],
    tolerance: int,
    window_size: int = 50,
) -> DetectionOutcome:
    """Apply the accounting rules to a sorted list of trigger positions."""
    positions = sorted(set(trigger_positions))
    matched: set[int] = set()
    detected: list[int] = []
    delays: dict[int, int] = {}

    for boundary in boundaries:
        in_window = [
            t for t in positions if boundary <= t <= boundary + tolerance
        ]
        if in_window:
            first = in_window[0]
            matched.add(first)
            detected.append(boundary)
            delays[boundary] = first - boundary

    false_positives = [t for t in positions if t not in matched]
    early = [
        t
        for t in false_positives
        if any(boundary - tolerance <= t < boundary for boundary in boundaries)
    ]
    missed = [b for b in boundaries if b not in detected]
    return DetectionOutcome(
        n_boundaries=len(boundaries),
        detected_boundaries=detected,
        detection_delays=delays,
        false_positive_positions=false_positives,
        early_warning_positions=early,
        missed_boundaries=missed,
        n_triggers=len(positions),
        tolerance_examples=tolerance,
    )


def false_positive_rate_per_window(
    false_positives: int,
    n_windows: int,
    n_calibration_windows: int,
) -> float | None:
    """FPs per monitored window (calibration windows excluded from monitoring)."""
    monitored = n_windows - n_calibration_windows
    if monitored <= 0:
        return None
    return false_positives / monitored


def adaptation_efficiency(
    final_avg_em: float,
    no_adapt_avg_em: float,
    adaptation_seconds: float,
    n_adaptations: int,
    total_tokens: int,
) -> dict:
    """Quality-vs-cost efficiency figures (all measured inputs)."""
    gain = final_avg_em - no_adapt_avg_em
    return {
        "quality_gain_over_no_adapt": gain,
        "quality_gain_per_adaptation": gain / n_adaptations if n_adaptations else None,
        "quality_gain_per_training_second": (
            gain / adaptation_seconds if adaptation_seconds > 0 else None
        ),
        "quality_gain_per_million_tokens": (
            gain / (total_tokens / 1e6) if total_tokens > 0 else None
        ),
    }


def avoided_adaptations(
    n_adaptations: int,
    always_adapt_n: int,
) -> int | None:
    """Adaptations avoided relative to the always-adapt reference policy."""
    if always_adapt_n <= 0:
        return None
    return always_adapt_n - n_adaptations
