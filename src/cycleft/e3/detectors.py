"""E3 drift detectors with explicit information regimes.

Regime A (unlabeled): the detector sees only genuinely observable inputs —
question embeddings. The signal type physically carries no answer field.

Regime B (delayed feedback): the system predicts first, the answer arrives
afterwards, and the observed prequential loss is fed to the detector. The
loss of example t is computed with the model state BEFORE any update from
example t, so the answer never influences the prediction being scored.

Periodic is schedule-based (neither regime). Every detector logs its regime
in ``regime`` and every decision in its ``log``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

REGIME_UNLABELED = "A_unlabeled"
REGIME_FEEDBACK = "B_delayed_feedback"
REGIME_SCHEDULE = "scheduled"
REGIME_COMBINED = "A+B_combined"


@dataclass(frozen=True)
class UnlabeledSignal:
    """Per-window Regime-A signal: question embeddings only, by construction."""

    window_index: int
    embeddings: np.ndarray  # (window_size, dim), L2-normalised question embeddings


@dataclass(frozen=True)
class FeedbackSignal:
    """Per-window Regime-B signal: prequential prediction errors (0-1)
    observed with delayed feedback (answer arrives after the prediction)."""

    window_index: int
    example_errors: tuple[float, ...]


def rbf_kernel_median_bandwidth(reference: np.ndarray) -> float:
    """Median-heuristic RBF bandwidth from the reference sample (float64)."""
    distances = np.linalg.norm(
        reference[:, None, :] - reference[None, :, :], axis=-1
    )
    iu = np.triu_indices(len(reference), k=1)
    median = float(np.median(distances[iu]))
    if median <= 0:
        return 1.0
    return median


def _rbf_gram(a: np.ndarray, b: np.ndarray, bandwidth: float) -> np.ndarray:
    sq = (
        np.sum(a * a, axis=1)[:, None]
        + np.sum(b * b, axis=1)[None, :]
        - 2.0 * a @ b.T
    )
    return np.exp(-np.maximum(sq, 0.0) / (2.0 * bandwidth**2))


def mmd2_unbiased(
    x: np.ndarray, y: np.ndarray, bandwidth: float | None = None
) -> float:
    """Unbiased MMD^2 between samples x and y with an RBF kernel.

    ``bandwidth=None`` uses the median heuristic on ``x`` (the reference).
    """
    if bandwidth is None:
        bandwidth = rbf_kernel_median_bandwidth(x)
    n, m = len(x), len(y)
    if n < 2 or m < 2:
        raise ValueError("MMD^2 needs at least 2 samples per group.")
    kxx = _rbf_gram(x, x, bandwidth)
    kyy = _rbf_gram(y, y, bandwidth)
    kxy = _rbf_gram(x, y, bandwidth)
    term_x = (kxx.sum() - np.trace(kxx)) / (n * (n - 1))
    term_y = (kyy.sum() - np.trace(kyy)) / (m * (m - 1))
    term_xy = kxy.mean()
    return float(term_x + term_y - 2.0 * term_xy)


@dataclass
class TriggerDecision:
    """One logged detector decision (fired or not)."""

    detector: str
    regime: str
    window_index: int
    position: int  # last stream example index covered by the decision
    fired: bool
    statistic: float | None = None
    threshold: float | None = None
    reason: str = ""


class BaseTrigger:
    name: str = "base"
    regime: str = REGIME_SCHEDULE

    def __init__(self) -> None:
        self.log: list[TriggerDecision] = []

    def _record(self, decision: TriggerDecision) -> bool:
        self.log.append(decision)
        return decision.fired


class PeriodicTrigger(BaseTrigger):
    """Strategy A: fire every ``every_windows`` windows (schedule, no signal)."""

    name = "periodic"
    regime = REGIME_SCHEDULE

    def __init__(self, every_windows: int, n_windows: int):
        super().__init__()
        if every_windows <= 0:
            raise ValueError("every_windows must be positive.")
        self.every_windows = every_windows
        self.n_windows = n_windows

    def scheduled_windows(self) -> list[int]:
        """Window indices at which the schedule fires (0-based)."""
        return [
            w for w in range(self.n_windows) if (w + 1) % self.every_windows == 0
        ]

    def observe_window(self, window_index: int, position: int) -> bool:
        fired = (window_index + 1) % self.every_windows == 0
        return self._record(
            TriggerDecision(
                detector=f"{self.name}_k{self.every_windows}",
                regime=self.regime,
                window_index=window_index,
                position=position,
                fired=fired,
                statistic=float(window_index + 1),
                threshold=float(self.every_windows),
                reason="scheduled" if fired else "schedule",
            )
        )


class LossThresholdTrigger(BaseTrigger):
    """Strategy B (Regime B): prequential window-mean prediction error above a
    calibrated threshold (reference mean + z * reference std), with a
    refractory period of ``refractory_windows`` windows after every trigger."""

    name = "loss_threshold"
    regime = REGIME_FEEDBACK

    def __init__(
        self,
        reference_window_losses: Sequence[float],
        z: float = 2.0,
        refractory_windows: int = 3,
    ):
        super().__init__()
        if len(reference_window_losses) < 2:
            raise ValueError("Need >= 2 calibration windows.")
        self.reference_mean = float(np.mean(reference_window_losses))
        self.reference_std = float(np.std(reference_window_losses, ddof=0))
        self.threshold = self.reference_mean + z * self.reference_std
        self.z = z
        self.refractory_windows = refractory_windows
        self._windows_since_trigger = refractory_windows  # warm-up free

    def observe_window(self, signal: FeedbackSignal, position: int) -> bool:
        mean_error = (
            float(np.mean(signal.example_errors))
            if signal.example_errors
            else float("nan")
        )
        self._windows_since_trigger += 1
        if not np.isfinite(mean_error):
            return self._record(
                TriggerDecision(
                    self.name, self.regime, signal.window_index, position, False,
                    reason="no finite errors",
                )
            )
        fired = mean_error > self.threshold
        reason = "error above threshold"
        if fired and self._windows_since_trigger < self.refractory_windows:
            fired = False
            reason = "above threshold but refractory"
        if fired:
            self._windows_since_trigger = 0
        return self._record(
            TriggerDecision(
                self.name, self.regime, signal.window_index, position, fired,
                statistic=mean_error, threshold=self.threshold, reason=reason,
            )
        )


class AdwinTrigger(BaseTrigger):
    """Strategy C (Regime B): river's ADWIN on the per-example prequential
    loss stream, with a post-alarm cooldown measured in windows."""

    name = "adwin"
    regime = REGIME_FEEDBACK

    def __init__(
        self,
        delta: float = 0.002,
        grace_period: int = 100,
        clock: int = 32,
        min_window_length: int = 32,
        cooldown_windows: int = 2,
    ):
        super().__init__()
        from river.drift import ADWIN  # imported here; tested dependency

        self._adwin = ADWIN(
            delta=delta,
            clock=clock,
            min_window_length=min_window_length,
            grace_period=grace_period,
        )
        self.delta = delta
        self.grace_period = grace_period
        self.cooldown_windows = cooldown_windows
        self._windows_since_trigger = cooldown_windows
        self._last_window_end = -1
        self.alarms: list[int] = []  # example positions

    def observe_example(self, loss: float, position: int) -> bool:
        """Feed one prequential loss; returns True the moment ADWIN alarms."""
        self._adwin.update(float(loss))
        if self._adwin.drift_detected:
            self.alarms.append(position)
            self._windows_since_trigger = 0
            return True
        return False

    def observe_window(self, window_index: int, position: int) -> bool:
        """Window-level view used by the combined rule (with cooldown).

        Fires when ADWIN alarmed at least once since the previous window
        close and the cooldown has elapsed.
        """
        self._windows_since_trigger += 1
        window_alarms = [
            alarm for alarm in self.alarms
            if alarm > self._last_window_end and alarm <= position
        ]
        self._last_window_end = position
        fired = bool(window_alarms) and (
            self._windows_since_trigger >= self.cooldown_windows
        )
        if fired:
            self._windows_since_trigger = 0
        statistic = float(min(window_alarms)) if window_alarms else None
        return self._record(
            TriggerDecision(
                self.name, self.regime, window_index, position, fired,
                statistic=statistic,
                threshold=self.delta,
                reason="ADWIN alarm" if window_alarms else "no alarm",
            )
        )


class MMDTrigger(BaseTrigger):
    """Strategy D (Regime A): two-sample MMD^2 between the reference window
    and the recent window of question embeddings. RBF kernel with
    median-heuristic bandwidth from the reference; trigger when MMD^2
    exceeds a permutation-calibrated threshold. On a trigger the reference
    is updated to the current (recent) window and a refractory period of
    ``refractory_windows`` windows suppresses immediate re-triggers."""

    name = "mmd"
    regime = REGIME_UNLABELED

    def __init__(
        self,
        reference_embeddings: np.ndarray,
        threshold: float,
        bandwidth: float | None = None,
        update_reference_on_trigger: bool = True,
        refractory_windows: int = 5,
    ):
        super().__init__()
        self.reference = np.asarray(reference_embeddings, dtype=np.float64)
        self.bandwidth = (
            bandwidth
            if bandwidth is not None
            else rbf_kernel_median_bandwidth(self.reference)
        )
        self.threshold = float(threshold)
        self.update_reference_on_trigger = update_reference_on_trigger
        self.refractory_windows = refractory_windows
        self._windows_since_trigger = refractory_windows

    def observe_window(self, signal: UnlabeledSignal, position: int) -> bool:
        recent = np.asarray(signal.embeddings, dtype=np.float64)
        statistic = mmd2_unbiased(self.reference, recent, self.bandwidth)
        self._windows_since_trigger += 1
        fired = statistic > self.threshold
        reason = "MMD^2 above threshold"
        if fired and self._windows_since_trigger < self.refractory_windows:
            fired = False
            reason = "above threshold but refractory"
        if fired:
            self._windows_since_trigger = 0
            if self.update_reference_on_trigger:
                self.reference = recent.copy()
        return self._record(
            TriggerDecision(
                self.name, self.regime, signal.window_index, position, fired,
                statistic=statistic, threshold=self.threshold, reason=reason,
            )
        )


class CombinedTrigger(BaseTrigger):
    """Strategy E: OR rule over ADWIN (Regime B) and MMD (Regime A), with a
    shared minimum gap of ``min_gap_windows`` windows between adaptations so
    near-simultaneous alarms cause one adaptation, not two."""

    name = "combined"
    regime = REGIME_COMBINED

    def __init__(self, adwin: AdwinTrigger, mmd: MMDTrigger, min_gap_windows: int = 2):
        super().__init__()
        self.adwin = adwin
        self.mmd = mmd
        self.min_gap_windows = min_gap_windows
        self._windows_since_trigger = min_gap_windows

    def observe_window(
        self,
        window_index: int,
        position: int,
        feedback: FeedbackSignal | None,
        unlabeled: UnlabeledSignal | None,
    ) -> bool:
        self._windows_since_trigger += 1
        adwin_fired = self.adwin.observe_window(window_index, position)
        mmd_fired = self.mmd.observe_window(unlabeled, position)
        fired = (adwin_fired or mmd_fired) and (
            self._windows_since_trigger >= self.min_gap_windows
        )
        if fired:
            self._windows_since_trigger = 0
        components = []
        if adwin_fired:
            components.append("adwin")
        if mmd_fired:
            components.append("mmd")
        return self._record(
            TriggerDecision(
                self.name, self.regime, window_index, position, fired,
                statistic=None,
                threshold=None,
                reason=("components=" + "+".join(components)) if components
                else "no component fired",
            )
        )
