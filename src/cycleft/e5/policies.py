"""E5 Phase-2 checkpoint policies (deterministic; specified in E5_DESIGN.md §4
and DESIGN_AMENDMENTS.md A2/A3)."""

from __future__ import annotations

import math
from typing import Any

POLICIES = ("P0", "P1", "P2", "P3", "P5")

SCHEDULES = {
    "INT-2x-A": [100, 250],
    "INT-3x-A": [62, 137, 311],
    "INT-2x-B": [400, 1000],
}

NOTICE_LEAD_STEPS = 10          # compressed local stress window (~10 s locally)
ADAPTIVE_LAMBDA = 1e-3          # synthetic interruptions per optimizer step
ADAPTIVE_K_MIN = 25
ADAPTIVE_K_MAX = 250


def adaptive_interval(
    ckpt_cost_s: float,
    step_cost_s: float,
    lambda_per_step: float = ADAPTIVE_LAMBDA,
    k_min: int = ADAPTIVE_K_MIN,
    k_max: int = ADAPTIVE_K_MAX,
) -> int:
    """Deterministic P3 rule: k* = clip(round(sqrt(2*C_ckpt/(lambda*c_step))),
    k_min, k_max). Units: C_ckpt [s], c_step [s/step], lambda [1/step],
    k* [steps]. Heuristic, not RL; lambda is a scenario input (no future
    knowledge of the realized schedule)."""
    if step_cost_s <= 0 or lambda_per_step <= 0:
        return k_max
    k = round(math.sqrt(2.0 * ckpt_cost_s / (lambda_per_step * step_cost_s)))
    return int(max(k_min, min(k_max, k)))


def next_kill_position(
    schedule: list[int],
    start_position: int,
    fired_positions: list[int],
) -> int | None:
    """Next scheduled interruption strictly after the start position that has
    not already fired. The schedule pointer never resets on restart."""
    for position in schedule:
        if position > start_position and position not in fired_positions:
            return position
    return None


def policy_checkpoint_due(
    policy: str,
    global_step: int,
    last_checkpoint_step: int,
    *,
    ckpt_every: int,
    stage_end: bool,
    kill_at: int | None,
    ckpt_cost_s: float,
    step_cost_s: float,
) -> tuple[bool, str]:
    """Whether a checkpoint commits at this step boundary, and why.

    Ordering rule (fairness): the interruption at position p fires AFTER
    optimizer step p and BEFORE any checkpoint scheduled at p — so a
    checkpoint at p never survives an interruption at p.
    """
    if policy == "P0":
        return False, "P0 never checkpoints"
    if policy == "P5":
        return False, "P5 checkpoints only at oracle interruption positions"
    if kill_at is not None and global_step == kill_at:
        if policy == "P5":
            return True, "oracle checkpoint at the interruption position"
        return False, "interruption fires before any checkpoint at this step"
    if stage_end:
        return True, "stage-end checkpoint (continual-learning convention)"
    if policy == "P1":
        due = ckpt_every > 0 and (global_step - last_checkpoint_step) >= ckpt_every
        return due, f"fixed interval k={ckpt_every}"
    if policy == "P2":
        due = ckpt_every > 0 and (global_step - last_checkpoint_step) >= ckpt_every
        return due, f"periodic baseline k={ckpt_every} (notice checkpoint handled separately)"
    if policy == "P3":
        k = adaptive_interval(ckpt_cost_s, step_cost_s)
        due = (global_step - last_checkpoint_step) >= k
        return due, f"adaptive k*={k}"
    raise ValueError(f"unknown policy {policy!r}")


def notice_position(kill_at: int, lead: int = NOTICE_LEAD_STEPS) -> int | None:
    """P2 notice fires `lead` steps before the scheduled interruption."""
    if kill_at is None:
        return None
    return max(1, kill_at - lead)
