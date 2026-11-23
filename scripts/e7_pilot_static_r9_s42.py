"""E7 static baseline: e7-static-r9-s42 (rank 9, alpha 18, seed 42).

Authorized single-run pilot via the guarded harness and the E5 adapter's
single-invocation static path (E5 workflow: 3-stage b500 training with
fresh AdamW per stage, in-workload final evaluation, then the prediction
sweep). Final-only evaluation per the frozen v5 static schedule.
"""
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

MATRIX = REPO / "results/e7/design_20261120/v5_final_reconciliation/FINAL_RUN_MATRIX.csv"
OUT_ROOT = REPO / "results/e7/execution_20261123"
TOKEN = "i-authorize-e7-phase2-execution"
RUN_ID = "e7-static-r9-s42"
TELEMETRY_PATH = REPO / "results/e7/pilot_static_r9_s42_20261123/pilot_telemetry.json"


def gate(label, ok, detail=""):
    print(f"[gate] {'PASS' if ok else 'FAIL'} {label}" + (f" — {detail}" if detail else ""),
          flush=True)
    if not ok:
        raise SystemExit(f"PRE-EXECUTION GATE FAILED: {label}")


def preflight():
    import torch
    from cycleft.e7.execution_harness import load_and_validate_matrix, select_run_id

    spec = select_run_id(load_and_validate_matrix(MATRIX), RUN_ID)
    gate("run identity (NEW/Static-R9/seed 42/rank 9x3/alpha 18=2r)",
         spec.row_type == "NEW" and spec.policy == "Static-R9" and spec.seed == 42
         and [int(r) for r in spec.event_ranks] == [9, 9, 9]
         and [int(a) for a in spec.event_alphas] == [18, 18, 18])
    gate("config hash", len(spec.config_hash) == 64, spec.config_hash)

    gate("MPS available", torch.backends.mps.is_available())
    gate("CUDA absent", not torch.cuda.is_available())
    gate("no MPS CPU-fallback permission",
         os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "1")

    hub = Path.home() / ".cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct"
    snaps = list((hub / "snapshots").glob("*")) if (hub / "snapshots").exists() else []
    weights = [s / "model.safetensors" for s in snaps if (s / "model.safetensors").exists()]
    gate("local base model present (no download)", bool(weights))
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
    from cycleft.e7.e5_adapter import make_run_fn
    from cycleft.e7.execution_harness import ExecutionHarness

    harness = ExecutionHarness(MATRIX, OUT_ROOT, repo_root=REPO,
                               run_fn=make_run_fn(repo_root=REPO))  # real E5 defaults
    wall0 = time.perf_counter()
    planned = harness.execute(TOKEN, run_id=RUN_ID)  # token-gated, single-run
    wall_s = time.perf_counter() - wall0

    run_dir = OUT_ROOT / RUN_ID
    harness_manifest = json.loads((run_dir / "manifest.json").read_text())
    telemetry = harness_manifest["telemetry"]
    mps_alloc_mb = 0.0
    try:
        mps_alloc_mb = torch.mps.current_allocated_memory() / 1048576
    except Exception:
        pass
    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
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
        "training": telemetry.get("training"),
        "evaluation": telemetry.get("evaluation"),
        "final_em": harness_manifest.get("final_avg_em"),
    }
    TELEMETRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    TELEMETRY_PATH.write_text(json.dumps(telemetry_out, indent=2))
    print(json.dumps({k: v for k, v in telemetry_out.items()
                      if k not in ("training", "evaluation")}, indent=2), flush=True)
    print("PILOT EXECUTION COMPLETE", flush=True)


if __name__ == "__main__":
    main()
