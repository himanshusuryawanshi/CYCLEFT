"""E3 detector calibration.

All calibration uses ONLY the warm-up/calibration period (the first
``n_calibration_windows`` windows of the stream, processed by the base
model). Final evaluation data is never used for calibration.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cycleft.e3.detectors import mmd2_unbiased


@dataclass(frozen=True)
class LossThresholdCalibration:
    """Calibrated threshold rule: reference mean + z * reference std.

    Calibration uses the prequential prediction-error stream of the
    calibration windows under the shared bootstrap model — never final
    evaluation data.
    """

    reference_mean: float
    reference_std: float
    z: float
    threshold: float
    n_calibration_windows: int

    def as_dict(self) -> dict:
        return {
            "rule": "threshold = reference_mean + z * reference_std",
            "reference_mean": self.reference_mean,
            "reference_std": self.reference_std,
            "z": self.z,
            "threshold": self.threshold,
            "n_calibration_windows": self.n_calibration_windows,
        }


def calibrate_loss_threshold(
    calibration_window_errors: list[list[float]],
    z: float = 2.0,
) -> LossThresholdCalibration:
    """Calibrate the predictive-error trigger on the stable reference period.

    ``calibration_window_errors[w]`` holds the per-example prequential 0-1
    prediction errors of calibration window ``w`` (shared bootstrap model,
    before any strategy-specific adaptation).
    """
    if len(calibration_window_errors) < 2:
        raise ValueError("Error calibration needs >= 2 windows.")
    window_means = [float(np.mean(errors)) for errors in calibration_window_errors]
    mean = float(np.mean(window_means))
    std = float(np.std(window_means, ddof=0))
    return LossThresholdCalibration(
        reference_mean=mean,
        reference_std=std,
        z=z,
        threshold=mean + z * std,
        n_calibration_windows=len(calibration_window_errors),
    )


@dataclass(frozen=True)
class MMDCalibration:
    """Permutation-calibrated MMD^2 threshold from the reference period."""

    threshold: float
    alpha: float
    n_permutations: int
    bandwidth: float
    null_quantiles: dict[str, float]
    n_reference_examples: int

    def as_dict(self) -> dict:
        return {
            "rule": (
                "threshold = (1 - alpha) quantile of the permutation null of "
                "MMD^2 computed by randomly splitting the calibration "
                "embeddings into two disjoint windows"
            ),
            "threshold": self.threshold,
            "alpha": self.alpha,
            "n_permutations": self.n_permutations,
            "kernel": "RBF, median-heuristic bandwidth on the reference",
            "bandwidth": self.bandwidth,
            "null_quantiles": self.null_quantiles,
            "n_reference_examples": self.n_reference_examples,
        }


def calibrate_mmd_threshold(
    calibration_embeddings: np.ndarray,
    window_size: int,
    alpha: float = 0.05,
    n_permutations: int = 200,
    seed: int = 42,
) -> MMDCalibration:
    """Build the MMD^2 null distribution from calibration data only.

    The calibration embeddings are repeatedly split at random into two
    disjoint ``window_size`` samples; the resulting MMD^2 values form the
    null distribution of the statistic under the (assumed) within-event
    stationarity of the calibration period. The threshold is the
    ``(1 - alpha)`` quantile of that null.
    """
    calibration_embeddings = np.asarray(calibration_embeddings, dtype=np.float64)
    n_cal = len(calibration_embeddings)
    if n_cal < 2 * window_size:
        raise ValueError(
            f"Need >= {2 * window_size} calibration embeddings, got {n_cal}."
        )
    bandwidth = float(np.median(
        np.linalg.norm(
            calibration_embeddings[:, None, :]
            - calibration_embeddings[None, :, :],
            axis=-1,
        )[np.triu_indices(n_cal, k=1)]
    ))
    if bandwidth <= 0:
        bandwidth = 1.0

    rng = np.random.default_rng(seed)
    null_stats = []
    for _ in range(n_permutations):
        permutation = rng.permutation(n_cal)
        x = calibration_embeddings[permutation[:window_size]]
        y = calibration_embeddings[permutation[window_size : 2 * window_size]]
        null_stats.append(mmd2_unbiased(x, y, bandwidth))

    null_stats = np.sort(np.asarray(null_stats))
    threshold = float(np.quantile(null_stats, 1.0 - alpha))
    return MMDCalibration(
        threshold=threshold,
        alpha=alpha,
        n_permutations=n_permutations,
        bandwidth=bandwidth,
        null_quantiles={
            "q50": float(np.quantile(null_stats, 0.50)),
            "q95": float(np.quantile(null_stats, 0.95)),
            "q99": float(np.quantile(null_stats, 0.99)),
            "max": float(null_stats[-1]),
        },
        n_reference_examples=n_cal,
    )
