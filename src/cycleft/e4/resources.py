"""E4 resource profiles, feasibility checks and cost semantics.

A :class:`ResourceProfile` describes one compute tier. Two facts matter for
integrity:

* ``measured`` — the profile carries throughput/memory numbers measured by
  the E4 benchmark harness on that resource. Only measured profiles are
  eligible for real scheduling decisions.
* ``available`` — the resource is reachable from this host right now.

Advertised specifications (vendor datasheets) are recorded with their
source and timestamp but never treated as measurements; such profiles are
``measured=False`` and are excluded from real policy decisions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


PRICE_TYPE_MEASURED = "measured_billed"
PRICE_TYPE_ADVERTISED = "advertised_unverified"
PRICE_TYPE_OWNED = "owned_hardware_no_metered_cost"


@dataclass
class ResourceProfile:
    """One compute resource tier (measured or candidate)."""

    resource_id: str
    kind: str  # "local" | "cloud"
    device: str  # torch device string, e.g. "mps", "cuda:0", "cpu"
    description: str
    gpu_model: str | None = None
    gpu_memory_gb: float | None = None
    cpu_cores: int | None = None
    ram_gb: float | None = None
    precision_support: tuple[str, ...] = ("float16",)
    framework_versions: dict[str, str] = field(default_factory=dict)
    # --- benchmark-derived (None until actually measured) ---
    measured: bool = False
    available: bool = False
    tokens_per_second: float | None = None
    steps_per_second: float | None = None
    peak_memory_gb: float | None = None
    measured_workload: str | None = None
    measured_timestamp: str | None = None
    # --- pricing ---
    price_usd_per_hour: float | None = None
    price_type: str | None = None
    price_source: str | None = None
    price_timestamp: str | None = None
    price_instance_type: str | None = None
    price_region: str | None = None
    # --- switching overhead ---
    checkpoint_save_seconds: float | None = None
    checkpoint_load_seconds: float | None = None

    def validate(self) -> list[str]:
        """Return a list of integrity problems (empty = valid)."""
        problems: list[str] = []
        if self.measured:
            if self.tokens_per_second is None or self.tokens_per_second <= 0:
                problems.append("measured profile requires positive tokens_per_second")
            if self.peak_memory_gb is None or self.peak_memory_gb <= 0:
                problems.append("measured profile requires positive peak_memory_gb")
            if not self.available:
                problems.append("measured profile should also be available")
        if self.price_usd_per_hour is not None:
            if self.price_type not in (
                PRICE_TYPE_MEASURED, PRICE_TYPE_ADVERTISED, PRICE_TYPE_OWNED,
            ):
                problems.append(f"unknown price_type {self.price_type!r}")
            if self.price_type == PRICE_TYPE_MEASURED and self.price_source is None:
                problems.append("measured_billed price requires a source")
            if self.price_type == PRICE_TYPE_ADVERTISED and self.price_source is None:
                problems.append("advertised price requires a source and timestamp")
            if self.price_type != PRICE_TYPE_OWNED and self.price_timestamp is None:
                problems.append("cloud price requires a pricing timestamp")
            if self.price_usd_per_hour < 0:
                problems.append("price must be non-negative")
        elif self.kind == "cloud":
            problems.append("cloud resource requires a price entry (or explicit null + note)")
        return problems

    def memory_fits(self, required_gb: float) -> bool:
        if self.peak_memory_gb is None or self.gpu_memory_gb is None:
            return False
        return self.peak_memory_gb <= self.gpu_memory_gb

    def estimated_seconds(
        self, total_steps: int, switching_overhead_seconds: float = 0.0
    ) -> float | None:
        """Estimated training wall-clock from the MEASURED throughput."""
        if self.steps_per_second is None or self.steps_per_second <= 0:
            return None
        return total_steps / self.steps_per_second + switching_overhead_seconds

    def to_dict(self) -> dict[str, Any]:
        payload = dict(self.__dict__)
        payload["precision_support"] = list(self.precision_support)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ResourceProfile":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        data = {k: v for k, v in payload.items() if k in known}
        if isinstance(data.get("precision_support"), list):
            data["precision_support"] = tuple(data["precision_support"])
        return cls(**data)


def estimated_memory_requirement_gb(base_model_gb: float, lora_overhead_gb: float) -> float:
    """Documented a-priori memory estimate (base weights + optimizer state)."""
    return base_model_gb + lora_overhead_gb


def feasibility_report(
    profile: ResourceProfile,
    required_memory_gb: float,
    deadline_seconds: float | None = None,
    total_steps: int = 0,
    switching_overhead_seconds: float = 0.0,
) -> tuple[bool, list[str]]:
    """Why a resource is (in)feasible for a workload; used by the scheduler.

    A resource is feasible iff it is available, has MEASURED performance
    (no advertised-spec scheduling), its measured peak memory fits the
    requirement, and — when a deadline is set — its measured throughput
    plus switching overhead meets the deadline.
    """
    reasons: list[str] = []
    if not profile.available:
        reasons.append("resource not available from this host")
    if not profile.measured:
        reasons.append("no measured benchmark on this resource (advertised specs are not schedulable)")
    else:
        if profile.peak_memory_gb is not None and profile.peak_memory_gb > 0:
            # For local unified memory the OS limit is the RAM; for GPUs the VRAM.
            capacity = profile.gpu_memory_gb or profile.ram_gb
            if capacity is not None and profile.peak_memory_gb > capacity:
                reasons.append("recorded peak memory exceeds device capacity")
        if required_memory_gb > 0:
            capacity = profile.gpu_memory_gb or profile.ram_gb
            if capacity is not None and required_memory_gb > capacity:
                reasons.append(
                    f"estimated memory requirement {required_memory_gb:.2f} GB "
                    f"exceeds capacity {capacity:.2f} GB"
                )
        if deadline_seconds is not None:
            estimate = profile.estimated_seconds(total_steps, switching_overhead_seconds)
            if estimate is None:
                reasons.append("no measured throughput to verify the deadline")
            elif estimate > deadline_seconds:
                reasons.append(
                    f"estimated {estimate:.1f}s exceeds deadline {deadline_seconds:.1f}s"
                )
    return (not reasons, reasons)


def event_cost_usd(
    profile: ResourceProfile,
    usage_seconds: float,
) -> dict[str, Any]:
    """Cost of one usage period, keeping billed and modeled strictly apart.

    * measured_billed: actual billing data exists (none in this project yet).
    * advertised_unverified: modeled cost from a documented advertised rate —
      clearly labeled as modeled, never presented as billed.
    * owned_hardware_no_metered_cost: local hardware has no metered price;
      usage_seconds is the measurable quantity and no dollar figure is
      invented for it.
    """
    hours = usage_seconds / 3600.0
    if profile.price_type == PRICE_TYPE_MEASURED:
        return {
            "cost_type": "measured_billed",
            "cost_usd": round(hours * (profile.price_usd_per_hour or 0.0), 6),
            "usage_seconds": usage_seconds,
            "note": "actual billing data",
        }
    if profile.price_type == PRICE_TYPE_ADVERTISED:
        return {
            "cost_type": "modeled_advertised",
            "cost_usd": round(hours * (profile.price_usd_per_hour or 0.0), 6),
            "usage_seconds": usage_seconds,
            "price_source": profile.price_source,
            "price_timestamp": profile.price_timestamp,
            "price_instance_type": profile.price_instance_type,
            "price_region": profile.price_region,
            "note": (
                "MODELED cost from an advertised on-demand rate; not billed, "
                "not measured"
            ),
        }
    if profile.price_type == PRICE_TYPE_OWNED:
        return {
            "cost_type": "owned_no_metered_cost",
            "cost_usd": None,
            "usage_seconds": usage_seconds,
            "note": (
                "owned hardware; no metered dollar cost is attributable and "
                "none is invented; usage reported in seconds"
            ),
        }
    return {"cost_type": "unknown", "cost_usd": None, "usage_seconds": usage_seconds,
            "note": "no pricing information recorded"}
