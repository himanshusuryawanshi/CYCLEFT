"""E4 resource-selection policies: fixed baselines, dynamic scheduler, oracle.

Dynamic policy (documented rule, no learning, no tuning on final evaluation):

    At each adaptation point:
    1. Estimate the workload (optimizer steps, memory requirement).
    2. Filter candidates to FEASIBLE resources
       (``feasibility_report``: available, measured, memory fits, and —
       when a deadline is configured — measured throughput + switching
       overhead meets the deadline).
    3. Choose the least expensive feasible resource; tie-break on the
       shorter estimated time.
    4. If nothing is feasible, fail loudly (no silent fallback, no
       fabricated measurements).

With no configured deadline the problem is explicitly unconstrained cost
minimization over feasible resources.

The oracle is an OFFLINE reference: it knows the measured performance and
cost of every resource in advance (information advantage an online
scheduler does not have) and minimizes cost subject to a completion-time
bound computed from the true measured numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cycleft.e4.resources import (
    ResourceProfile,
    event_cost_usd,
    estimated_memory_requirement_gb,
    feasibility_report,
)


@dataclass
class SchedulingContext:
    """Everything the scheduler knows at one decision point."""

    event_index: int  # 0-based adaptation index within the run
    total_steps: int  # optimizer steps of the upcoming adaptation
    base_model_gb: float
    lora_overhead_gb: float
    deadline_seconds: float | None = None  # None = unconstrained cost minimization

    @property
    def required_memory_gb(self) -> float:
        return estimated_memory_requirement_gb(
            self.base_model_gb, self.lora_overhead_gb
        )


@dataclass
class SchedulerDecision:
    """One logged decision with its full reasoning."""

    policy: str
    event_index: int
    chosen_resource: str | None
    feasible_resources: list[str]
    infeasible: dict[str, list[str]]
    estimated_seconds: float | None
    estimated_cost: dict | None
    reason: str
    context: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "policy": self.policy,
            "event_index": self.event_index,
            "chosen_resource": self.chosen_resource,
            "feasible_resources": self.feasible_resources,
            "infeasible": self.infeasible,
            "estimated_seconds": self.estimated_seconds,
            "estimated_cost": self.estimated_cost,
            "reason": self.reason,
            "context": self.context,
        }


def _feasible_with_reasons(
    profiles: dict[str, ResourceProfile], context: SchedulingContext
) -> tuple[list[str], dict[str, list[str]], dict[str, float]]:
    feasible: list[str] = []
    infeasible: dict[str, list[str]] = {}
    estimates: dict[str, float] = {}
    for resource_id, profile in profiles.items():
        ok, reasons = feasibility_report(
            profile,
            required_memory_gb=context.required_memory_gb,
            deadline_seconds=context.deadline_seconds,
            total_steps=context.total_steps,
            switching_overhead_seconds=(
                (profile.checkpoint_load_seconds or 0.0)
                + (profile.checkpoint_save_seconds or 0.0)
            ),
        )
        if ok:
            feasible.append(resource_id)
            estimate = profile.estimated_seconds(context.total_steps)
            if estimate is not None:
                estimates[resource_id] = estimate
        else:
            infeasible[resource_id] = reasons
    return feasible, infeasible, estimates


def select_resource_dynamic(
    profiles: dict[str, ResourceProfile],
    context: SchedulingContext,
) -> SchedulerDecision:
    """Policy D: least expensive feasible resource; tie-break on time."""
    feasible, infeasible, estimates = _feasible_with_reasons(profiles, context)
    if not feasible:
        return SchedulerDecision(
            policy="dynamic",
            event_index=context.event_index,
            chosen_resource=None,
            feasible_resources=[],
            infeasible=infeasible,
            estimated_seconds=None,
            estimated_cost=None,
            reason=(
                "no feasible resource (no silent fallback; missing "
                "measurements are never replaced with fabricated values)"
            ),
            context=context.__dict__,
        )

    def cost_key(resource_id: str) -> tuple[float, float]:
        profile = profiles[resource_id]
        if profile.price_usd_per_hour is not None:
            price = profile.price_usd_per_hour
        else:
            price = float("inf")
        return (price, estimates.get(resource_id, float("inf")))

    chosen = min(feasible, key=cost_key)
    chosen_profile = profiles[chosen]
    usage = estimates.get(
        chosen, (context.total_steps / chosen_profile.steps_per_second)
        if chosen_profile.steps_per_second else 0.0
    )
    return SchedulerDecision(
        policy="dynamic",
        event_index=context.event_index,
        chosen_resource=chosen,
        feasible_resources=feasible,
        infeasible=infeasible,
        estimated_seconds=estimates.get(chosen),
        estimated_cost=event_cost_usd(chosen_profile, usage),
        reason=(
            "least expensive feasible resource"
            + (
                " (tie-break: shorter estimated time)"
                if len(feasible) > 1
                else " (only feasible resource)"
            )
            + (
                ""
                if context.deadline_seconds is None
                else f"; deadline {context.deadline_seconds:.0f}s satisfied"
            )
        ),
        context=context.__dict__,
    )


def select_resource_fixed(
    policy_name: str,
    fixed_resource_id: str,
    profiles: dict[str, ResourceProfile],
    context: SchedulingContext,
) -> SchedulerDecision:
    """Policies A/B/C: always use the pre-declared resource.

    The fixed choice still must be feasible; an infeasible fixed resource
    fails the decision loudly instead of degrading silently.
    """
    profile = profiles.get(fixed_resource_id)
    if profile is None:
        return SchedulerDecision(
            policy=policy_name,
            event_index=context.event_index,
            chosen_resource=None,
            feasible_resources=[],
            infeasible={fixed_resource_id: ["resource not in profile registry"]},
            estimated_seconds=None,
            estimated_cost=None,
            reason="fixed resource unknown",
            context=context.__dict__,
        )
    feasible, infeasible, estimates = _feasible_with_reasons(profiles, context)
    if fixed_resource_id not in feasible:
        return SchedulerDecision(
            policy=policy_name,
            event_index=context.event_index,
            chosen_resource=None,
            feasible_resources=feasible,
            infeasible=infeasible,
            estimated_seconds=None,
            estimated_cost=None,
            reason=(
                f"fixed resource '{fixed_resource_id}' is infeasible for this "
                "workload; refusing to degrade silently"
            ),
            context=context.__dict__,
        )
    usage = estimates.get(fixed_resource_id)
    return SchedulerDecision(
        policy=policy_name,
        event_index=context.event_index,
        chosen_resource=fixed_resource_id,
        feasible_resources=feasible,
        infeasible=infeasible,
        estimated_seconds=estimates.get(fixed_resource_id),
        estimated_cost=event_cost_usd(profile, usage or 0.0),
        reason="fixed policy resource",
        context=context.__dict__,
    )


def select_resource_oracle(
    profiles: dict[str, ResourceProfile],
    context: SchedulingContext,
    completion_time_bound_seconds: float | None = None,
) -> SchedulerDecision:
    """Policy E (OFFLINE REFERENCE, not an implementable scheduler).

    Knows the true measured time and cost of every resource and minimizes
    cost subject to an optional completion-time bound on the measured
    times. Its information advantage: perfect knowledge of measured
    throughput and prices before the run starts.
    """
    measured = {
        resource_id: profile
        for resource_id, profile in profiles.items()
        if profile.measured and profile.available
    }
    if not measured:
        return SchedulerDecision(
            policy="oracle_offline",
            event_index=context.event_index,
            chosen_resource=None,
            feasible_resources=[],
            infeasible={rid: ["no measured resources exist"] for rid in profiles},
            estimated_seconds=None,
            estimated_cost=None,
            reason="oracle requires measured profiles",
            context=context.__dict__,
        )

    def meets_bound(profile: ResourceProfile) -> bool:
        if completion_time_bound_seconds is None:
            return True
        estimate = profile.estimated_seconds(
            context.total_steps,
            (profile.checkpoint_load_seconds or 0.0)
            + (profile.checkpoint_save_seconds or 0.0),
        )
        return estimate is not None and estimate <= completion_time_bound_seconds

    candidates = {
        resource_id: profile
        for resource_id, profile in measured.items()
        if meets_bound(profile)
    }
    if not candidates:
        return SchedulerDecision(
            policy="oracle_offline",
            event_index=context.event_index,
            chosen_resource=None,
            feasible_resources=sorted(measured),
            infeasible={
                resource_id: ["exceeds the oracle completion-time bound"]
                for resource_id in measured
            },
            estimated_seconds=None,
            estimated_cost=None,
            reason="no measured resource meets the bound",
            context=context.__dict__,
        )
    chosen = min(
        candidates,
        key=lambda resource_id: (
            candidates[resource_id].price_usd_per_hour
            if candidates[resource_id].price_usd_per_hour is not None
            else float("inf"),
            candidates[resource_id].estimated_seconds(
                context.total_steps,
                (candidates[resource_id].checkpoint_load_seconds or 0.0)
                + (candidates[resource_id].checkpoint_save_seconds or 0.0),
            ) or float("inf"),
        ),
    )
    profile = candidates[chosen]
    usage = profile.estimated_seconds(
        context.total_steps,
        (profile.checkpoint_load_seconds or 0.0)
        + (profile.checkpoint_save_seconds or 0.0),
    )
    return SchedulerDecision(
        policy="oracle_offline",
        event_index=context.event_index,
        chosen_resource=chosen,
        feasible_resources=sorted(candidates),
        infeasible={
            resource_id: ["more expensive than the chosen resource"]
            for resource_id in measured
            if resource_id != chosen
        },
        estimated_seconds=usage,
        estimated_cost=event_cost_usd(profile, usage or 0.0),
        reason=(
            "offline oracle: minimum measured cost within the completion-time "
            "bound (information advantage: perfect knowledge of measured "
            "performance and prices)"
        ),
        context=context.__dict__,
    )


POLICY_REGISTRY = {
    "fixed_cheapest": "Policy A: fixed lowest-cost feasible resource",
    "fixed_fastest": "Policy B: fixed fastest feasible resource",
    "fixed_preferred": "Policy C: one resource chosen before the experiment",
    "dynamic": "Policy D: rule-based dynamic selection (documented rule)",
    "oracle_offline": "Policy E: offline oracle reference (not implementable online)",
}


def fixed_resource_for_policy(
    policy_name: str,
    profiles: dict[str, ResourceProfile],
    context: SchedulingContext,
) -> str | None:
    """Pre-declared resource for policies A/B/C given the measured profiles.

    Cheapest = minimum hourly price among feasible; fastest = minimum
    estimated time among feasible. Declared BEFORE the experiment, from
    measured profiles only.
    """
    feasible, _infeasible, estimates = _feasible_with_reasons(profiles, context)
    if not feasible:
        return None
    if policy_name == "fixed_cheapest":
        return min(
            feasible,
            key=lambda resource_id: (
                profiles[resource_id].price_usd_per_hour
                if profiles[resource_id].price_usd_per_hour is not None
                else float("inf"),
                estimates.get(resource_id, float("inf")),
            ),
        )
    if policy_name == "fixed_fastest":
        return min(
            feasible,
            key=lambda resource_id: estimates.get(resource_id, float("inf")),
        )
    return None
