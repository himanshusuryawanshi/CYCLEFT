"""E4 local benchmark orchestration.

Sequence (documented in the run README):
1. Hardware inventory (measured on this host).
2. Resource profiles: local MPS (to be measured) + cloud candidates that are
   recorded with advertised specs only — measured=False, available=False.
3. Local benchmarks: workload A (rank4_b125, with a repeat run) and workload
   B (rank8_b500), using the fixed event-boundary schedule on local MPS.
4. The measured profile is written back, and ALL policies are evaluated
   against it (with a single measured resource every policy selects it —
   the degenerate case, recorded explicitly).

No paid cloud resource is provisioned or launched by this script.

Usage:
    uv run python scripts/e4_benchmark.py --run-id local_20261117
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cycleft.e1.pipeline import write_json
from cycleft.e4.harness import (
    CHECKPOINTS_ROOT,
    RESULTS_ROOT,
    WORKLOADS,
    load_baseline_rows,
    prepare_workload_config,
    run_benchmark,
)
from cycleft.e4.inventory import collect_hardware_inventory
from cycleft.e4.resources import (
    PRICE_TYPE_OWNED,
    ResourceProfile,
)
from cycleft.e3.stream import build_stream


def cloud_candidate_profiles() -> dict[str, ResourceProfile]:
    """Candidate cloud tiers: advertised specs only, explicitly NOT measured.

    Prices are deliberately null until captured at provision time from the
    cloud pricing APIs; inventing list prices now would be fabrication.
    """
    candidates = {
        "cloud_t4": ResourceProfile(
            resource_id="cloud_t4",
            kind="cloud",
            device="cuda:0",
            description="NVIDIA T4 candidate (pending provisioning/benchmark)",
            gpu_model="NVIDIA T4 (advertised datasheet, not verified on-host)",
            gpu_memory_gb=16.0,
            precision_support=("float16", "float32"),
        ),
        "cloud_v100": ResourceProfile(
            resource_id="cloud_v100",
            kind="cloud",
            device="cuda:0",
            description="NVIDIA V100 candidate (pending provisioning/benchmark)",
            gpu_model="NVIDIA V100 (advertised datasheet, not verified on-host)",
            gpu_memory_gb=16.0,
            precision_support=("float16", "float32"),
        ),
        "cloud_a10g": ResourceProfile(
            resource_id="cloud_a10g",
            kind="cloud",
            device="cuda:0",
            description="NVIDIA A10G candidate (pending provisioning/benchmark)",
            gpu_model="NVIDIA A10G (advertised datasheet, not verified on-host)",
            gpu_memory_gb=24.0,
            precision_support=("float16", "float32"),
        ),
    }
    for profile in candidates.values():
        profile.measured = False
        profile.available = False
        profile.price_usd_per_hour = None
        profile.price_type = None
        profile.price_source = (
            "pending: capture from cloud pricing API/CLI at provision time"
        )
        profile.price_timestamp = None
    return candidates


def local_profile_from_inventory(inventory: dict) -> ResourceProfile:
    host = inventory["host"]
    accel = inventory["accelerators"]
    return ResourceProfile(
        resource_id="local_mps",
        kind="local",
        device="mps",
        description=(
            f"Local Apple Silicon ({host['cpu_model']}), unified memory, "
            "MPS backend"
        ),
        gpu_model=accel.get("mps_device_name"),
        gpu_memory_gb=None,
        cpu_cores=host.get("cpu_cores"),
        ram_gb=host.get("ram_gb"),
        precision_support=("float16", "float32"),
        available=accel.get("mps_available", False),
        measured=False,  # updated after the benchmark
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--workloads",
        default="A_economical,A_economical_repeat,B_high_quality",
        help="Comma-separated workload runs (A repeat measures run-to-run noise)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = RESULTS_ROOT / args.run_id
    checkpoints_dir = CHECKPOINTS_ROOT / args.run_id
    results_dir.mkdir(parents=True, exist_ok=True)

    def log(message: str) -> None:
        print(f"[e4:{args.run_id}] {message}", flush=True)

    # 1. Hardware inventory.
    inventory = collect_hardware_inventory()
    write_json(results_dir / "hardware_inventory.json", inventory)
    log(
        f"host: {inventory['host']['cpu_model']}, "
        f"{inventory['host']['ram_gb']} GB RAM, "
        f"MPS={inventory['accelerators']['mps_available']}, "
        f"CUDA={inventory['accelerators']['cuda_available']}"
    )

    # 2. Resource profiles.
    local_profile = local_profile_from_inventory(inventory)
    profiles = {"local_mps": local_profile, **cloud_candidate_profiles()}

    # 3. Benchmark configuration artifact (YAML, as specified).
    import yaml

    benchmark_config = {
        "run_id": args.run_id,
        "seed": args.seed,
        "workloads": WORKLOADS,
        "schedule": (
            "fixed event-boundary oracle convention (E3): adaptation on "
            "the first `budget` examples of each event; the adapter "
            "continues across events within a run; fresh pretrained base "
            "per run; the drift detector is NOT part of E4"
        ),
        "evaluation": "E1/E2 greedy protocol on the 3 held-out test sets",
        "precision_policy": (
            "fp16 base + fp32 LoRA on MPS; identical protocol on CUDA "
            "hosts (fp16); documented per resource"
        ),
        "workload_runs": args.workloads.split(","),
        "cloud_benchmarks": "PENDING (no authorized GPU access)",
        "checkpoint_layout": (
            "checkpoints/e4/<run_id>/measured/<workload_run>/adapt_<n>/ "
            "(single degenerate policy: every policy maps to the same "
            "measured local run)"
        ),
    }
    (results_dir / "benchmark_config.yaml").write_text(
        yaml.safe_dump(benchmark_config, sort_keys=False), encoding="utf-8"
    )

    # 4. Local benchmarks.
    stream, events, manifest = build_stream(seed=args.seed)
    write_json(
        results_dir / "dataset_manifest.json",
        {**manifest, "note": "same data/splits/order as E1-E3"},
    )
    baseline_rows = load_baseline_rows()
    workload_runs = [w.strip() for w in args.workloads.split(",") if w.strip()]

    # Map "A_economical_repeat" -> workload A with a run label.
    benchmark_payloads = []
    for run_name in workload_runs:
        if run_name.endswith("_repeat"):
            workload = run_name[: -len("_repeat")]
            label = " (repeat run)"
        else:
            workload = run_name
            label = ""
        if workload not in WORKLOADS:
            raise SystemExit(f"unknown workload run: {run_name}")
        benchmark_payloads.append((workload, label, run_name))

    lm_config = prepare_workload_config("A_economical", seed=args.seed)
    all_results = []
    for workload, label, run_name in benchmark_payloads:
        lm_config = prepare_workload_config(workload, seed=args.seed)
        log(f"=== benchmark {run_name} (rank={lm_config.lora_r}, "
            f"budget={WORKLOADS[workload]['budget']}){label} ===")
        result = run_benchmark(
            workload=workload,
            resource_profile=local_profile,
            lm_config=lm_config,
            events=events,
            baseline_rows=baseline_rows,
            stream_rows=stream.rows,
            checkpoints_dir=checkpoints_dir / "measured" / run_name,
            log=log,
            run_label=label,
        )
        all_results.append(result)
        if result.status != "complete":
            log(f"benchmark {run_name} FAILED: {result.error}")
        else:
            totals = result.totals
            log(
                f"benchmark {run_name}: {totals['optimizer_steps']} steps, "
                f"{totals['training_seconds']:.1f}s train, "
                f"{totals['tokens_per_second']:.0f} tok/s, "
                f"peak {totals['peak_memory_gb']:.2f} GB"
            )

    # 5. Update the local profile with measured values (workload A run 1).
    first_a = next(
        (r for r in all_results if r.workload == "A_economical" and r.status == "complete"),
        None,
    )
    first_b = next(
        (r for r in all_results if r.workload == "B_high_quality" and r.status == "complete"),
        None,
    )
    reference = first_b or first_a
    if reference is not None:
        local_profile.measured = True
        local_profile.tokens_per_second = reference.totals["tokens_per_second"]
        local_profile.steps_per_second = reference.totals["steps_per_second"]
        local_profile.peak_memory_gb = reference.totals["peak_memory_gb"]
        local_profile.measured_workload = reference.workload
        local_profile.measured_timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        local_profile.price_type = PRICE_TYPE_OWNED
        local_profile.price_usd_per_hour = None
        local_profile.price_source = (
            "owned hardware; no metered price (usage reported in seconds)"
        )

    profiles_payload = {
        resource_id: profile.to_dict()
        for resource_id, profile in profiles.items()
    }
    write_json(results_dir / "resource_profiles.json", profiles_payload)

    # 6. Scheduler decisions for every policy against the MEASURED profiles.
    from cycleft.e4.scheduler import (
        POLICY_REGISTRY,
        SchedulingContext,
        fixed_resource_for_policy,
        select_resource_dynamic,
        select_resource_fixed,
        select_resource_oracle,
    )

    decisions = []
    for workload in ("A_economical", "B_high_quality"):
        budget = WORKLOADS[workload]["budget"]
        rank = WORKLOADS[workload]["lora_rank"]
        for stage in range(3):
            context = SchedulingContext(
                event_index=stage,
                total_steps=budget,
                base_model_gb=2.5,
                lora_overhead_gb=0.3,
                deadline_seconds=None,  # unconstrained cost minimization
            )
            context.workload = workload  # type: ignore[attr-defined]
            for policy in POLICY_REGISTRY:
                if policy == "fixed_cheapest":
                    resource_id = fixed_resource_for_policy(
                        "fixed_cheapest", profiles, context
                    ) or "local_mps"
                    decision = select_resource_fixed(
                        "fixed_cheapest", resource_id, profiles, context
                    )
                elif policy == "fixed_fastest":
                    resource_id = fixed_resource_for_policy(
                        "fixed_fastest", profiles, context
                    ) or "local_mps"
                    decision = select_resource_fixed(
                        "fixed_fastest", resource_id, profiles, context
                    )
                elif policy == "fixed_preferred":
                    decision = select_resource_fixed(
                        "fixed_preferred", "local_mps", profiles, context
                    )
                elif policy == "dynamic":
                    decision = select_resource_dynamic(profiles, context)
                else:
                    decision = select_resource_oracle(profiles, context)
                decision.context["workload"] = workload
                decisions.append(decision.to_dict())

    with (results_dir / "scheduler_decisions.jsonl").open(
        "w", encoding="utf-8"
    ) as file:
        for decision in decisions:
            file.write(json.dumps(decision) + "\n")

    # 7. Raw benchmark results for the analyzer.
    benchmark_dir = results_dir / "benchmark_results"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    for result, (workload, label, run_name) in zip(all_results, benchmark_payloads):
        payload = dict(result.__dict__)
        payload["run_name"] = run_name
        write_json(benchmark_dir / f"{run_name}.json", payload)

    log("benchmark orchestration complete; run scripts/e4_analyze.py next")


if __name__ == "__main__":
    main()
