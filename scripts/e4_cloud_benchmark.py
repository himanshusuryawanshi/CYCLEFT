"""E4 portable cloud benchmark harness.

Runs the IDENTICAL E4 protocol on any torch device so that cloud GPU tiers
(T4/V100/A10G/...) can be profiled and merged into the E4 comparison.

THIS SCRIPT NEVER PROVISIONS INFRASTRUCTURE. Run it ON an already-created
GPU host (manual or user-authorized provisioning). Requirements on the host:
- Python 3.11+ with the CYCLEFT repo (uv sync), the HF model cache or
  network access to download meta-llama/Llama-3.2-1B-Instruct (gated: run
  `huggingface-cli login` once), and the temporalwiki dataset cache.
- A CUDA-capable torch installation.

Usage on the GPU host:
    uv run python scripts/e4_cloud_benchmark.py --run-id cloud_t4_YYYYMMDD \
        --device cuda:0 --workloads A_economical,B_high_quality

Then copy the produced results/e4/<run-id>/ directory back into this repo
before running scripts/e4_analyze.py, and mark the corresponding resource
profile as measured (the script writes everything needed).

Example provisioning (NOT executed by this project; requires explicit
user authorization and billing):
    # GCP T4 (Compute Engine API must be enabled first):
    gcloud services enable compute.googleapis.com
    gcloud compute instances create cycleft-t4 \
        --zone=us-central1-a --machine-type=n1-standard-4 \
        --accelerator=type=nvidia-tesla-t4,count=1 --maintenance-policy=TERMINATE \
        --image-family=pytorch-latest-cu124 --image-project=deeplearning-platform-images
    # Azure NCasT4_v3 (requires az login and quota):
    az vm create -g cycleft -n cycleft-t4 \
        --image Canonical:ubuntu-24_04-lts:server:latest \
        --size Standard_NC4as_T4_v3
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cycleft.e1.pipeline import write_json
from cycleft.e4.harness import (
    RESULTS_ROOT,
    WORKLOADS,
    load_baseline_rows,
    prepare_workload_config,
    run_benchmark,
)
from cycleft.e4.inventory import collect_hardware_inventory
from cycleft.e4.resources import ResourceProfile
from cycleft.e3.stream import build_stream


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workloads", default="A_economical,B_high_quality")
    parser.add_argument(
        "--baseline-from",
        default="results/e1/full_20261113/baseline_metrics.json",
        help="Path to the shared E1 baseline on this host",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is not available on this host; this script benchmarks "
            "cloud GPU tiers and must run on a CUDA host."
        )
    results_dir = RESULTS_ROOT / args.run_id
    checkpoints_dir = results_dir.parent.parent / "checkpoints" / "e4" / args.run_id
    results_dir.mkdir(parents=True, exist_ok=True)

    def log(message: str) -> None:
        print(f"[e4-cloud:{args.run_id}] {message}", flush=True)

    inventory = collect_hardware_inventory()
    inventory["accelerators"]["gpu_name"] = torch.cuda.get_device_name(0)
    inventory["accelerators"]["gpu_memory_gb"] = (
        torch.cuda.get_device_properties(0).total_memory / 1e9
    )
    write_json(results_dir / "hardware_inventory.json", inventory)
    log(f"GPU: {inventory['accelerators']['gpu_name']}")

    # On the cloud host the resource is measured and available by definition.
    profile = ResourceProfile(
        resource_id=args.run_id,
        kind="cloud",
        device=args.device,
        description=f"cloud benchmark on {inventory['accelerators']['gpu_name']}",
        gpu_model=inventory["accelerators"]["gpu_name"],
        gpu_memory_gb=inventory["accelerators"]["gpu_memory_gb"],
        ram_gb=inventory["host"]["ram_gb"],
        cpu_cores=inventory["host"]["cpu_cores"],
        precision_support=("float16", "float32"),
        available=True,
        measured=False,
        price_usd_per_hour=None,
        price_type=None,
        price_source="pending: capture from the cloud billing/pricing API",
    )
    from cycleft.e2.resources import ResourceTracker  # noqa: F401

    stream, events, manifest = build_stream(seed=args.seed)
    write_json(
        results_dir / "dataset_manifest.json",
        {**manifest, "note": "same data/splits/order as E1-E3"},
    )
    import json

    baseline = json.loads(Path(args.baseline_from).read_text(encoding="utf-8"))
    per_event = baseline["per_event"]
    baseline_rows = (
        [per_event[str(e)]["exact_match"] for e in sorted(per_event, key=int)],
        [per_event[str(e)]["token_f1"] for e in sorted(per_event, key=int)],
        [per_event[str(e)]["perplexity"] for e in sorted(per_event, key=int)],
    )

    results = []
    for workload in [w.strip() for w in args.workloads.split(",") if w.strip()]:
        if workload not in WORKLOADS:
            raise SystemExit(f"unknown workload: {workload}")
        lm_config = prepare_workload_config(workload, seed=args.seed, device_request="cuda")
        log(f"=== {workload} on {args.device} ===")
        result = run_benchmark(
            workload=workload,
            resource_profile=profile,
            lm_config=lm_config,
            events=events,
            baseline_rows=baseline_rows,
            stream_rows=stream.rows,
            checkpoints_dir=checkpoints_dir / workload,
            log=log,
        )
        results.append(result)
        totals = result.totals
        log(
            f"{workload}: {totals.get('optimizer_steps')} steps, "
            f"{totals.get('training_seconds'):.1f}s, "
            f"{totals.get('tokens_per_second'):.0f} tok/s"
        )

    # Mark measured with the reference workload and write everything back.
    reference = next(
        (r for r in results if r.status == "complete" and r.workload == "B_high_quality"),
        next((r for r in results if r.status == "complete"), None),
    )
    if reference is not None:
        profile.measured = True
        profile.tokens_per_second = reference.totals["tokens_per_second"]
        profile.steps_per_second = reference.totals["steps_per_second"]
        profile.peak_memory_gb = reference.totals["peak_memory_gb"]
        profile.measured_workload = reference.workload
        profile.measured_timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")

    write_json(results_dir / "resource_profiles.json", {args.run_id: profile.to_dict()})
    benchmark_dir = results_dir / "benchmark_results"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    for result in results:
        write_json(benchmark_dir / f"{result.workload}.json", dict(result.__dict__))
    log("cloud benchmark complete; copy results/e4/%s back to the main repo" % args.run_id)


if __name__ == "__main__":
    main()
