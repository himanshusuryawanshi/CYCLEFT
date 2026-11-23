"""Tests for E3 detectors, calibration and detection accounting."""

import dataclasses

import numpy as np
import pytest

from cycleft.e3.calibration import calibrate_loss_threshold, calibrate_mmd_threshold
from cycleft.e3.detectors import (
    AdwinTrigger,
    CombinedTrigger,
    FeedbackSignal,
    LossThresholdTrigger,
    MMDTrigger,
    PeriodicTrigger,
    UnlabeledSignal,
    mmd2_unbiased,
)
from cycleft.e3.metrics import (
    adaptation_efficiency,
    avoided_adaptations,
    detection_accounting,
    false_positive_rate_per_window,
)


# -- Periodic trigger --------------------------------------------------------


def test_periodic_schedule_fires_every_k_windows():
    trigger = PeriodicTrigger(every_windows=5, n_windows=30)
    fired = [
        window
        for window in range(30)
        if trigger.observe_window(window, (window + 1) * 50 - 1)
    ]
    assert fired == [4, 9, 14, 19, 24, 29]
    assert trigger.log[-1].reason == "scheduled"


def test_periodic_k3_and_k10_schedules():
    assert PeriodicTrigger(3, 30).scheduled_windows() == [2, 5, 8, 11, 14, 17, 20, 23, 26, 29]
    assert PeriodicTrigger(10, 30).scheduled_windows() == [9, 19, 29]


def test_periodic_rejects_non_positive_k():
    with pytest.raises(ValueError):
        PeriodicTrigger(0, 30)


# -- Regime A: information-leakage prevention by construction ------------------


def test_unlabeled_signal_has_no_answer_field():
    field_names = {field.name for field in dataclasses.fields(UnlabeledSignal)}
    assert field_names == {"window_index", "embeddings"}


def test_mmd_trigger_rejects_feedback_signal():
    rng = np.random.default_rng(0)
    reference = rng.normal(size=(50, 4))
    trigger = MMDTrigger(reference, threshold=0.1)
    with pytest.raises(Exception):
        trigger.observe_window(
            FeedbackSignal(window_index=0, example_losses=(1.0, 2.0)),  # type: ignore[arg-type]
            position=49,
        )


# -- Loss threshold trigger ----------------------------------------------------


def test_loss_threshold_rule_and_trigger():
    calibration = [[2.0] * 50, [2.2] * 50, [2.1] * 50]
    trigger = LossThresholdTrigger(
        [float(np.mean(w)) for w in calibration], z=2.0, refractory_windows=3
    )
    assert trigger.reference_mean == pytest.approx(2.1)
    reference_std = float(np.std([2.0, 2.2, 2.1]))
    assert trigger.threshold == pytest.approx(2.1 + 2.0 * reference_std)

    below = trigger.observe_window(FeedbackSignal(3, (2.0,) * 50), 199)
    assert below is False
    above = trigger.observe_window(FeedbackSignal(4, (2.5,) * 50), 249)
    assert above is True


def test_loss_threshold_refractory_suppresses_repeat_trigger():
    trigger = LossThresholdTrigger([[2.0] * 50, [2.0] * 50], z=1.0,
                                   refractory_windows=3)
    assert trigger.observe_window(FeedbackSignal(2, (3.0,) * 50), 149) is True
    assert trigger.observe_window(FeedbackSignal(3, (3.0,) * 50), 199) is False
    assert trigger.observe_window(FeedbackSignal(4, (3.0,) * 50), 249) is False
    assert trigger.observe_window(FeedbackSignal(5, (3.0,) * 50), 299) is True


def test_loss_threshold_calibrated_on_reference_only():
    calibration = calibrate_loss_threshold([[2.0] * 50, [2.2] * 50], z=2.0)
    rule = calibration.as_dict()
    assert rule["rule"].startswith("threshold = reference_mean")
    assert rule["threshold"] == pytest.approx(2.3)
    assert rule["n_calibration_windows"] == 2


# -- ADWIN ---------------------------------------------------------------------


def test_adwin_detects_synthetic_step_change():
    rng = np.random.default_rng(7)
    losses = np.concatenate([
        rng.normal(2.0, 0.2, 400),
        rng.normal(3.0, 0.2, 400),
    ])
    trigger = AdwinTrigger(delta=0.002, grace_period=100, cooldown_windows=2)
    alarms = []
    for position, loss in enumerate(losses):
        if trigger.observe_example(float(loss), position):
            alarms.append(position)
    assert alarms, "ADWIN must alarm on a large step change"
    assert any(400 <= alarm <= 520 for alarm in alarms), (
        f"first alarm should be near the change point, got {alarms[:3]}"
    )


def test_adwin_stays_quiet_on_stationary_stream():
    rng = np.random.default_rng(11)
    losses = rng.normal(2.0, 0.2, 900)
    trigger = AdwinTrigger(delta=0.002, grace_period=100, cooldown_windows=2)
    alarms = [
        position
        for position, loss in enumerate(losses)
        if trigger.observe_example(float(loss), position)
    ]
    assert not alarms


def test_adwin_window_view_respects_cooldown():
    trigger = AdwinTrigger(delta=0.002, cooldown_windows=2)
    trigger.alarms = [120]  # injected alarm inside window 2 (positions 100-149)
    assert trigger.observe_window(2, 149) is True
    trigger.alarms.append(160)
    assert trigger.observe_window(3, 199) is False  # cooldown active
    trigger.alarms.append(210)  # new alarm inside window 4
    assert trigger.observe_window(4, 249) is True  # cooldown elapsed


# -- MMD ------------------------------------------------------------------------


def test_mmd_small_for_same_distribution_large_for_shifted():
    rng = np.random.default_rng(3)
    reference = rng.normal(size=(150, 8)) / np.sqrt(8)
    reference = reference / np.linalg.norm(reference, axis=1, keepdims=True)
    same = rng.normal(size=(50, 8))
    same = same / np.linalg.norm(same, axis=1, keepdims=True)
    shifted = rng.normal(loc=1.5, size=(50, 8))
    shifted = shifted / np.linalg.norm(shifted, axis=1, keepdims=True)
    bandwidth = 1.0
    stat_same = mmd2_unbiased(reference, same, bandwidth)
    stat_shifted = mmd2_unbiased(reference, shifted, bandwidth)
    assert stat_same < 0.05
    assert stat_shifted > 0.1
    assert stat_shifted > stat_same


def test_mmd_trigger_fires_and_updates_reference():
    rng = np.random.default_rng(5)
    reference = rng.normal(size=(150, 8)) / 3
    calibration = calibrate_mmd_threshold(
        reference, window_size=50, alpha=0.05, n_permutations=100, seed=1
    )
    trigger = MMDTrigger(reference, threshold=calibration.threshold,
                         bandwidth=calibration.bandwidth)
    same = rng.normal(size=(50, 8)) / 3
    before_reference = trigger.reference.copy()
    assert trigger.observe_window(
        UnlabeledSignal(3, same), position=199
    ) is False
    assert np.array_equal(trigger.reference, before_reference)

    shifted = rng.normal(loc=2.0, size=(50, 8)) / 3
    fired = trigger.observe_window(UnlabeledSignal(10, shifted), position=549)
    assert fired is True
    assert not np.array_equal(trigger.reference, before_reference), (
        "reference must update on trigger (documented policy)"
    )
    assert trigger.log[-1].statistic > calibration.threshold


def test_mmd_calibration_threshold_controls_false_alarm_rate():
    rng = np.random.default_rng(13)
    reference = rng.normal(size=(150, 8)) / 3
    calibration = calibrate_mmd_threshold(
        reference, window_size=50, alpha=0.05, n_permutations=200, seed=2
    )
    exceed = 0
    trials = 100
    for _ in range(trials):
        fresh = rng.normal(size=(50, 8)) / 3
        if mmd2_unbiased(reference, fresh, calibration.bandwidth) > calibration.threshold:
            exceed += 1
    assert exceed / trials <= 0.15, (
        f"per-window false alarm rate too high: {exceed / trials:.2f}"
    )


def test_mmd_calibration_requires_two_windows():
    with pytest.raises(ValueError):
        calibrate_mmd_threshold(np.zeros((60, 4)), window_size=50)


# -- Combined trigger ------------------------------------------------------------


def test_combined_or_rule_and_shared_gap():
    adwin = AdwinTrigger(delta=0.002, cooldown_windows=2)
    rng = np.random.default_rng(0)
    reference = rng.normal(size=(150, 8)) / 3
    mmd = MMDTrigger(reference, threshold=10.0)  # never fires
    combined = CombinedTrigger(adwin, mmd, min_gap_windows=2)

    adwin.alarms = [120]
    fired = combined.observe_window(
        2, 149, FeedbackSignal(2, (2.0,) * 50),
        UnlabeledSignal(2, rng.normal(size=(50, 8)) / 3),
    )
    assert fired is True
    assert "adwin" in combined.log[-1].reason

    adwin.alarms.append(300)
    fired = combined.observe_window(
        6, 349, FeedbackSignal(6, (2.0,) * 50),
        UnlabeledSignal(6, rng.normal(size=(50, 8)) / 3),
    )
    assert fired is False, "min gap must suppress a second immediate trigger"
    fired = combined.observe_window(
        7, 399, FeedbackSignal(7, (2.0,) * 50),
        UnlabeledSignal(7, rng.normal(size=(50, 8)) / 3),
    )
    assert fired is False, "alarm at 300 belongs to window 6, already consumed"


# -- Detection accounting ----------------------------------------------------------


def test_detection_accounting_tp_delay_fp_early_warning():
    outcome = detection_accounting(
        trigger_positions=[450, 530, 700, 1030],
        boundaries=[500, 1000],
        tolerance=100,
    )
    assert outcome.true_positives == 2
    assert outcome.detection_delays == {500: 30, 1000: 30}
    assert outcome.missed == 0
    assert outcome.missed_boundaries == []
    assert outcome.recall == 1.0
    # 450 is an early warning before boundary 500 (also counted as FP);
    # 700 is a plain false positive.
    assert outcome.false_positive_positions == [450, 700]
    assert outcome.early_warning_positions == [450]


def test_detection_accounting_missed_and_out_of_tolerance():
    outcome = detection_accounting(
        trigger_positions=[650, 1400],
        boundaries=[500, 1000],
        tolerance=100,
    )
    assert outcome.true_positives == 0
    assert outcome.missed_boundaries == [500, 1000]
    assert outcome.false_positives == 2


def test_detection_accounting_first_trigger_in_window_counts():
    outcome = detection_accounting(
        trigger_positions=[500, 540], boundaries=[500], tolerance=100
    )
    assert outcome.detection_delays == {500: 0}
    assert outcome.false_positive_positions == [540]


def test_false_positive_rate_excludes_calibration_windows():
    assert false_positive_rate_per_window(2, 30, 3) == pytest.approx(2 / 27)
    assert false_positive_rate_per_window(2, 3, 3) is None


def test_adaptation_efficiency_and_avoided():
    efficiency = adaptation_efficiency(
        final_avg_em=0.14,
        no_adapt_avg_em=0.06,
        adaptation_seconds=100.0,
        n_adaptations=2,
        total_tokens=20_000,
    )
    assert efficiency["quality_gain_over_no_adapt"] == pytest.approx(0.08)
    assert efficiency["quality_gain_per_training_second"] == pytest.approx(0.0008)
    assert efficiency["quality_gain_per_million_tokens"] == pytest.approx(4.0)
    assert avoided_adaptations(3, 3) == 0
    assert avoided_adaptations(1, 3) == 2
    assert avoided_adaptations(0, 0) is None
