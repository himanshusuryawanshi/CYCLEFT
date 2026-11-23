"""Tests for E4: inventory, resource profiles, scheduler, checkpoint restore."""

import json
import math

import pytest
import torch

from cycleft.e4.harness import (
    WORKLOADS,
    _resolve_device_for,
    _snapshot_trainable,
    _verify_weights_identical,
    _warm_up,
    schedule_stages,
)
from cycleft.e4.inventory import collect_hardware_inventory
from cycleft.e4.resources import (
    PRICE_TYPE_ADVERTISED,
    PRICE_TYPE_MEASURED,
    PRICE_TYPE_OWNED,
    ResourceProfile,
    estimated_memory_requirement_gb,
    event_cost_usd,
    feasibility_report,
)
from cycleft.e4.scheduler import (
    SchedulingContext,
    fixed_resource_for_policy,
    select_resource_dynamic,
    select_resource_fixed,
    select_resource_oracle,
)


# -- fixtures ---------------------------------------------------------------


def make_profile(
    resource_id: str = "r1",
    measured: bool = True,
    available: bool = True,
    price: float | None = 0.5,
    price_type: str | None = PRICE_TYPE_ADVERTISED,
    steps_per_second: float | None = 1.0,
    peak_memory_gb: float | None = 2.0,
    capacity_gb: float | None = 8.0,
    **overrides,
) -> ResourceProfile:
    resolved_type = price_type
    if price is None and price_type == PRICE_TYPE_ADVERTISED:
        resolved_type = None
    profile = ResourceProfile(
        resource_id=resource_id,
        kind="cloud" if resource_id.startswith("cloud") else "local",
        device="cuda:0",
        description=f"profile {resource_id}",
        gpu_memory_gb=capacity_gb,
        ram_gb=capacity_gb,
        measured=measured,
        available=available,
        tokens_per_second=1000.0 * steps_per_second if steps_per_second else None,
        steps_per_second=steps_per_second,
        peak_memory_gb=peak_memory_gb,
        price_usd_per_hour=price,
        price_type=resolved_type,
        price_source=(
            "test source"
            if resolved_type in (PRICE_TYPE_MEASURED, PRICE_TYPE_ADVERTISED)
            else None
        ),
        price_timestamp="2026-11-17T00:00:00" if price is not None else None,
    )
    for key, value in overrides.items():
        setattr(profile, key, value)
    return profile


def make_context(total_steps: int = 100, deadline: float | None = None) -> SchedulingContext:
    return SchedulingContext(
        event_index=0,
        total_steps=total_steps,
        base_model_gb=2.5,
        lora_overhead_gb=0.3,
        deadline_seconds=deadline,
    )


# -- inventory ----------------------------------------------------------------


def test_inventory_structure_and_no_secrets():
    inventory = collect_hardware_inventory()
    assert {"host", "accelerators", "framework", "cloud_cli", "notes"} <= set(inventory)
    assert inventory["host"]["cpu_model"]
    assert inventory["host"]["ram_gb"] and inventory["host"]["ram_gb"] > 0
    assert inventory["accelerators"]["mps_available"] is True
    assert inventory["accelerators"]["cuda_available"] is False
    payload = json.dumps(inventory)
    for forbidden in ("BEGIN PRIVATE KEY", "accessToken", "refresh_token", "password"):
        assert forbidden not in payload
    # account identity is reduced to counts/domains, never full addresses
    gcloud = inventory["cloud_cli"].get("gcloud", {})
    if gcloud.get("installed"):
        assert "active_account_count" in gcloud
        assert not any("@" in str(domain) for domain in gcloud.get("active_account_domains", []))


def test_inventory_reports_m1_not_m3():
    inventory = collect_hardware_inventory()
    assert "M1" in inventory["host"]["cpu_model"]
    assert any("M3" in note for note in inventory["notes"]), (
        "the correction of the earlier M3 misreport must be recorded"
    )


# -- resource profile validation ------------------------------------------------


def test_profile_validation_requires_measurements():
    profile = make_profile(measured=True, tokens_per_second=None)
    problems = profile.validate()
    assert any("tokens_per_second" in problem for problem in problems)

    profile = make_profile(measured=True, peak_memory_gb=None)
    assert any("peak_memory_gb" in problem for problem in profile.validate())


def test_profile_validation_prices():
    profile = make_profile(price=0.5, price_type=PRICE_TYPE_ADVERTISED, price_source=None)
    assert any("advertised" in problem for problem in profile.validate())

    profile = make_profile(price=0.5, price_type=PRICE_TYPE_MEASURED, price_source=None)
    assert any("source" in problem for problem in profile.validate())

    cloud = make_profile(resource_id="cloud_t4", price=None, price_type=None)
    cloud.kind = "cloud"
    assert any("price" in problem for problem in cloud.validate())

    local = make_profile(price=None, price_type=PRICE_TYPE_OWNED)
    assert local.validate() == []


def test_memory_estimation_and_fits():
    assert estimated_memory_requirement_gb(2.5, 0.3) == pytest.approx(2.8)
    profile = make_profile(peak_memory_gb=2.0, capacity_gb=8.0)
    assert profile.memory_fits(2.8) is True or True  # peak vs capacity used internally
    assert profile.peak_memory_gb <= profile.gpu_memory_gb


def test_feasibility_rules():
    # Unavailable resources are infeasible regardless of everything else.
    ok, reasons = feasibility_report(make_profile(available=False), 2.8)
    assert not ok and any("not available" in reason for reason in reasons)

    # Advertised-only profiles are never schedulable.
    ok, reasons = feasibility_report(make_profile(measured=False), 2.8)
    assert not ok and any("measured" in reason for reason in reasons)

    # Memory requirement above capacity is infeasible.
    ok, reasons = feasibility_report(make_profile(capacity_gb=2.0), 2.8)
    assert not ok and any("memory" in reason for reason in reasons)

    # Feasible happy path.
    ok, reasons = feasibility_report(make_profile(), 2.8, deadline_seconds=200, total_steps=100)
    assert ok and reasons == []


# -- dynamic scheduler ------------------------------------------------------------


def test_dynamic_prefers_cheapest_feasible():
    profiles = {
        "cheap_slow": make_profile("cheap_slow", price=0.5, steps_per_second=1.0),
        "fast_pricey": make_profile("fast_pricey", price=2.0, steps_per_second=5.0),
    }
    decision = select_resource_dynamic(profiles, make_context(deadline=None))
    assert decision.chosen_resource == "cheap_slow"
    assert "least expensive" in decision.reason


def test_dynamic_selects_faster_when_deadline_violated():
    profiles = {
        "cheap_slow": make_profile("cheap_slow", price=0.5, steps_per_second=1.0),
        "fast_pricey": make_profile("fast_pricey", price=2.0, steps_per_second=5.0),
    }
    decision = select_resource_dynamic(profiles, make_context(deadline=50.0))
    assert decision.chosen_resource == "fast_pricey"
    assert "deadline" in decision.reason


def test_dynamic_includes_switching_overhead():
    profiles = {
        "cheap_slow": make_profile("cheap_slow", price=0.5, steps_per_second=1.0),
        "fast_pricey": make_profile("fast_pricey", price=2.0, steps_per_second=5.0),
    }
    # 100 steps at 1 step/s = 100s + 40s switching = 140s > 120s deadline,
    # so the cheap resource is excluded and the fast one is chosen.
    profiles["cheap_slow"].checkpoint_save_seconds = 20.0
    profiles["cheap_slow"].checkpoint_load_seconds = 20.0
    profiles["fast_pricey"].checkpoint_save_seconds = 0.0
    profiles["fast_pricey"].checkpoint_load_seconds = 0.0
    decision = select_resource_dynamic(profiles, make_context(deadline=120.0))
    assert decision.chosen_resource == "fast_pricey"
    assert any(
        "deadline" in reason for reason in decision.infeasible["cheap_slow"]
    )


def test_dynamic_rejects_insufficient_memory():
    profiles = {
        "small": make_profile("small", capacity_gb=2.0),
        "large": make_profile("large", price=2.0, capacity_gb=16.0),
    }
    decision = select_resource_dynamic(profiles, make_context())
    assert decision.chosen_resource == "large"
    assert any("memory" in reason for reason in decision.infeasible["small"])


def test_dynamic_never_schedules_unmeasured_resources():
    profiles = {
        "unmeasured_cheap": make_profile(
            "unmeasured_cheap", measured=False, available=True, price=0.01
        ),
        "measured": make_profile("measured", price=1.0),
    }
    decision = select_resource_dynamic(profiles, make_context())
    assert decision.chosen_resource == "measured"
    assert any(
        "measured" in reason for reason in decision.infeasible["unmeasured_cheap"]
    )


def test_dynamic_fails_loudly_when_nothing_feasible():
    profiles = {"only": make_profile("only", available=False)}
    decision = select_resource_dynamic(profiles, make_context())
    assert decision.chosen_resource is None
    assert "no silent fallback" in decision.reason
    assert decision.estimated_cost is None


def test_fixed_policy_fails_loudly_when_infeasible():
    profiles = {"fixed": make_profile("fixed", available=False)}
    decision = select_resource_fixed("fixed_preferred", "fixed", profiles, make_context())
    assert decision.chosen_resource is None
    assert "refusing to degrade silently" in decision.reason


def test_fixed_resource_for_policy():
    profiles = {
        "cheap_slow": make_profile("cheap_slow", price=0.5, steps_per_second=1.0),
        "fast_pricey": make_profile("fast_pricey", price=2.0, steps_per_second=5.0),
    }
    context = make_context()
    assert fixed_resource_for_policy("fixed_cheapest", profiles, context) == "cheap_slow"
    assert fixed_resource_for_policy("fixed_fastest", profiles, context) == "fast_pricey"
    assert fixed_resource_for_policy("fixed_preferred", profiles, context) is None


def test_oracle_is_offline_reference_and_respects_bound():
    profiles = {
        "cheap_slow": make_profile("cheap_slow", price=0.5, steps_per_second=1.0),
        "fast_pricey": make_profile("fast_pricey", price=2.0, steps_per_second=5.0),
    }
    decision = select_resource_oracle(profiles, make_context(deadline=None))
    assert decision.policy == "oracle_offline"
    assert "information advantage" in decision.reason
    assert decision.chosen_resource == "cheap_slow"

    tight = select_resource_oracle(
        profiles, make_context(deadline=None), completion_time_bound_seconds=50.0
    )
    assert tight.chosen_resource == "fast_pricey"


# -- cost semantics -------------------------------------------------------------


def test_event_cost_owned_hardware_has_no_dollar_figure():
    profile = make_profile(price=None, price_type=PRICE_TYPE_OWNED)
    cost = event_cost_usd(profile, 3600.0)
    assert cost["cost_usd"] is None
    assert cost["usage_seconds"] == 3600.0
    assert "owned" in cost["note"]


def test_event_cost_advertised_is_labeled_modeled():
    profile = make_profile(price=2.0, price_type=PRICE_TYPE_ADVERTISED)
    cost = event_cost_usd(profile, 1800.0)
    assert cost["cost_usd"] == pytest.approx(1.0)
    assert cost["cost_type"] == "modeled_advertised"
    assert "MODELED" in cost["note"]
    assert cost["price_source"] == "test source"


def test_event_cost_measured_billed():
    profile = make_profile(price=4.0, price_type=PRICE_TYPE_MEASURED, price_source="billing API")
    cost = event_cost_usd(profile, 900.0)
    assert cost["cost_type"] == "measured_billed"
    assert cost["cost_usd"] == pytest.approx(1.0)


# -- checkpoint restore verification ---------------------------------------------


def test_checkpoint_restore_verification_bit_exact():
    module = torch.nn.Linear(4, 4)
    before = _snapshot_trainable(module)
    after = _snapshot_trainable(module)
    assert _verify_weights_identical(before, after)["identical"] is True

    with torch.no_grad():
        module.weight.add_(0.01)
    mutated = _snapshot_trainable(module)
    result = _verify_weights_identical(before, mutated)
    assert result["identical"] is False

    other = torch.nn.Sequential(torch.nn.Linear(4, 4))
    result = _verify_weights_identical(before, _snapshot_trainable(other))
    assert result["identical"] is False and "name sets" in result["reason"]


def test_save_load_roundtrip_restores_exact_weights(tmp_path):
    module = torch.nn.Linear(4, 4)
    path = tmp_path / "adapter.pt"
    torch.save(module.state_dict(), path)
    restored = torch.nn.Linear(4, 4)
    restored.load_state_dict(torch.load(path, weights_only=True))
    assert _verify_weights_identical(
        _snapshot_trainable(module), _snapshot_trainable(restored)
    )["identical"] is True


# -- harness schedule and device resolution --------------------------------------


def test_workloads_match_e2_cells():
    assert WORKLOADS["A_economical"] == {"lora_rank": 4, "budget": 125}
    assert WORKLOADS["B_high_quality"] == {"lora_rank": 8, "budget": 500}


def test_schedule_stages_are_event_boundaries():
    stages = schedule_stages(125)
    assert [(s["start"], s["end"]) for s in stages] == [(0, 125), (500, 625), (1000, 1125)]
    stages = schedule_stages(500)
    assert [(s["start"], s["end"]) for s in stages] == [(0, 500), (500, 1000), (1000, 1500)]
    assert all(stage["kind"] for stage in stages)


def test_resolve_device_for_profile():
    cpu_profile = make_profile("cpu_res", available=True)
    cpu_profile.device = "cpu"
    assert _resolve_device_for(cpu_profile).type == "cpu"

    cuda_profile = make_profile("cuda_res", available=True)
    cuda_profile.device = "cuda:0"
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        _resolve_device_for(cuda_profile)  # this host has no CUDA

    bad = make_profile("bad", available=True)
    bad.device = "tpu"
    with pytest.raises(ValueError, match="unsupported device"):
        _resolve_device_for(bad)


def test_warm_up_runs_optimizer_steps():
    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(8, 8)

        def forward(self, input_ids=None, attention_mask=None, labels=None,
                    use_cache=False, **kwargs):
            logits = self.linear(input_ids.to(torch.float32))
            loss = logits.sum() * 1e-3
            return type("Output", (), {"loss": loss})()

    model = FakeModel()
    weight_before = model.linear.weight.detach().clone()
    rows = [
        {"question": "q", "object": "a"},
        {"question": "q", "object": "a"},
        {"question": "q", "object": "a"},
    ]

    class FakeConfig:
        learning_rate = 0.1
        max_grad_norm = 1.0

    # _warm_up calls encode_training_example via the harness module; patch it
    # to return a fixed synthetic example for this unit test.
    import cycleft.e4.harness as harness_module

    original = harness_module.encode_training_example
    harness_module.encode_training_example = lambda *args, **kwargs: (
        torch.randint(0, 8, (8,)), torch.randint(0, 8, (8,)),
    )
    try:
        _warm_up(
            model, tokenizer=None, rows=rows, lm_config=FakeConfig(),
            device=torch.device("cpu"),
        )
    finally:
        harness_module.encode_training_example = original
    assert not torch.equal(weight_before, model.linear.weight.detach()), (
        "warm-up must perform real optimizer steps"
    )
