"""E7 first real experiment: e7-grow-s42 (Grow 4→8→16, seed 42).

Authorized single-run pilot executed through the guarded harness and the
Option B event orchestrator, using the existing E5 workflow with its real
default callbacks (run_workload on MPS; bundle-reload evaluation sweeps;
model-based rank transitions). Exactly one NEW run; no fallback to the
matrix. Token-gated. Emits a telemetry JSON for the pilot evidence package.
"""
import json
import os
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

MATRIX = REPO / "results/e7/design_20261120/v5_final_reconciliation/FINAL_RUN_MATRIX.csv"
OUT_ROOT = REPO / "results/e7/execution_20261121"
TOKEN = "i-authorize-e7-phase2-execution"
RUN_ID = "e7-grow-s42"
TELEMETRY_PATH = REPO / "results/e7/pilot_grow_s42_20261121/pilot_telemetry.json"


def gate(label, ok, detail=""):
    print(f"[gate] {'PASS' if ok else 'FAIL'} {label}" + (f" — {detail}" if detail else ""),
          flush=True)
    if not ok:
        raise SystemExit(f"PRE-EXECUTION GATE FAILED: {label}")


def preflight():
    import torch
    from cycleft.e7.execution_harness import load_and_validate_matrix, select_run_id

    spec = select_run_id(load_and_validate_matrix(MATRIX), RUN_ID)
    gate("run identity (NEW/Grow/seed 42/ranks 4-8-16/alpha 2r)",
         spec.row_type == "NEW" and spec.policy == "Grow" and spec.seed == 42
         and [int(r) for r in spec.event_ranks] == [4, 8, 16]
         and [int(a) for a in spec.event_alphas] == [8, 16, 32])
    gate("config hash", len(spec.config_hash) == 64, spec.config_hash)

    gate("MPS available", torch.backends.mps.is_available())
    gate("CUDA absent", not torch.cuda.is_available())
    gate("no MPS CPU-fallback permission",
         os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "1")

    hub = Path.home() / ".cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct"
    snaps = list((hub / "snapshots").glob("*")) if (hub / "snapshots").exists() else []
    weights = [s / "model.safetensors" for s in snaps if (s / "model.safetensors").exists()]
    gate("local base model present (no download)", bool(weights), str(weights[0].parent.name))
    ds = Path.home() / ".cache/huggingface/datasets/saxenan3___temporalwiki-drift-cl-easy"
    gate("local dataset present (no download)", ds.is_dir())

    df = subprocess.run(["df", "-k", "/System/Volumes/Data"], capture_output=True, text=True)
    free_gb = int(df.stdout.splitlines()[-1].split()[3]) / 1e6
    gate("disk space", free_gb > 5.0, f"{free_gb:.1f} GB free")

    run_dir = OUT_ROOT / RUN_ID
    gate("output dir absent", not run_dir.exists())
    resolved = run_dir.resolve()
    gate("path inside authorized root",
         (REPO / "results/e7").resolve() in resolved.parents)
    return spec


def main():
    spec = preflight()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    import torch
    from cycleft.e7.event_orchestrator import make_dynamic_run_fn
    from cycleft.e7.execution_harness import ExecutionHarness

    harness = ExecutionHarness(MATRIX, OUT_ROOT, repo_root=REPO,
                               run_fn=make_dynamic_run_fn(repo_root=REPO))
    wall0 = time.perf_counter()
    telemetry = harness.execute(TOKEN, run_id=RUN_ID)  # token-gated, single-run
    wall_s = time.perf_counter() - wall0

    # harness.execute returns the executed PlannedRun; the authoritative
    # telemetry lives in the run artifacts (manifest.json / marker)
    run_dir = OUT_ROOT / RUN_ID
    harness_manifest = json.loads((run_dir / "manifest.json").read_text())
    telemetry = {
        "training": harness_manifest["telemetry"]["training"],
        "evaluation": harness_manifest["telemetry"]["evaluation"],
        "final_rank_schedule": harness_manifest["telemetry"]["final_rank_schedule"],
        "master_manifest_sha256": harness_manifest["telemetry"].get(
            "master_manifest_sha256")}
    mps_alloc_mb = 0.0
    try:
        mps_alloc_mb = torch.mps.current_allocated_memory() / 1048576
    except Exception:
        pass
    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # macOS: KB
    du = subprocess.run(["du", "-sk", str(run_dir)], capture_output=True, text=True)
    disk_kb = int(du.stdout.split()[0])

    telemetry_out = {
        "run_id": RUN_ID,
        "executed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "wall_clock_s": round(wall_s, 1),
        "peak_rss_mb": round(peak_rss_mb, 1),
        "mps_allocated_at_exit_mb": round(mps_alloc_mb, 1),
        "run_dir_disk_kb": disk_kb,
        "device": "mps",
        "harness_state": "COMPLETED (per harness manifest/completion marker)",
        "orchestrator_telemetry": telemetry,
    }
    TELEMETRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    TELEMETRY_PATH.write_text(json.dumps(telemetry_out, indent=2))
    print(json.dumps({k: v for k, v in telemetry_out.items()
                      if k != "orchestrator_telemetry"}, indent=2), flush=True)
    print("PILOT EXECUTION COMPLETE", flush=True)


if __name__ == "__main__":
    main()
