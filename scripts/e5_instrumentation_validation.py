"""E5 Phase-2 instrumentation validation (authorized: deterministic fixtures).

V1: in-process instrumented run (reduced 6-step schedule) — verifies nonzero
    training timing, audit-by-default, split checkpoint accounting, event log.
V2: fresh-process SIGKILL fixture (kill at step 4, deterministic fallback to
    the step-3 bundle) — verifies event-log persistence across SIGKILL,
    supervisor signal recording, replay accounting, position coverage.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

WORKER = ROOT / "scripts" / "e5_phase1_worker.py"
OUT = ROOT / "results" / "e5" / "phase2_instrumentation_20261119"
REDUCED = 6


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary: dict = {"validated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                     "reduced_schedule_steps": REDUCED}

    # fresh per-launch output (event logs are append-only; a stale log from
    # a previous launch would corrupt the position accounting)
    import shutil as _shutil

    for sub in ("v1_timing", "v2_sigkill"):
        if (OUT / sub).exists():
            _shutil.rmtree(OUT / sub)

    # ---- V1: in-process instrumented run ----
    from cycleft.e1.data import prepare_event_datasets
    from cycleft.e1.pipeline import resolve_lm_config
    from cycleft.e5.event_log import read_events
    from cycleft.e5.position_audit import validate_positions
    from cycleft.e5.resumable_trainer import run_workload

    raw_config = resolve_lm_config(
        load_config("configs/e1_stream.yaml"),
        mode="full", train_examples=None, seed=42, device=None,
    )[0]
    events, _ = prepare_event_datasets(train_limit=None, seed=42)
    rows = [dict(r) for r in events[0].train_rows[:REDUCED]]
    schedule = [{"stage": 1, "event_id": 1, "start": 0, "end": REDUCED}]
    v1_out = OUT / "v1_timing"
    summary_v1 = run_workload(
        lm_config=raw_config, stream_rows=rows, out_dir=v1_out,
        exec_id="e5p2-v1-timing", attempt=1, ckpt_every=3,
        stage_schedule=schedule, log=lambda m: None,
    )
    timing = summary_v1["timing"]["categories"]
    checks_v1 = {
        "train_timing_nonzero": timing["train"]["total_s"] > 0,
        "ckpt_serialize_nonzero": timing["ckpt_serialize"]["total_s"] > 0,
        "hash_verify_nonzero": timing["ckpt_hash_verify"]["total_s"] > 0,
        "fsync_commit_nonzero": timing["ckpt_fsync_commit"]["total_s"] > 0,
        "durable_copy_nonzero": timing["durable_copy"]["total_s"] > 0,
        "process_wall_recorded": summary_v1["process_wall_s"] > 0,
        "audit_default_present": (v1_out / "audit_log.jsonl").exists(),
        "event_log_persisted": (v1_out / "event_log.jsonl").exists(),
    }
    audit_entries = [json.loads(l) for l in (v1_out / "audit_log.jsonl").read_text().splitlines()]
    checks_v1["audit_entries"] = len(audit_entries)
    checks_v1["audit_enriched_fields"] = all(
        {"exec_id", "pid", "attempt", "logical_step",
         "dataset_example_position", "ts_epoch"} <= set(e)
        for e in audit_entries
    )
    summary["v1"] = {"checks": checks_v1, "timing": summary_v1["timing"]}
    print("V1:", {k: v for k, v in checks_v1.items() if k != "audit_entries"})

    # ---- V2: fresh-process SIGKILL fixture with deterministic fallback ----
    v2_out = OUT / "v2_sigkill"
    log_file = open(OUT / "v2_worker.log", "a")
    cmd_kill = ["uv", "run", "python", str(WORKER), "--mode", "killrun",
                "--out", str(v2_out), "--seed", "42",
                "--ckpt-every", "3", "--kill-step", "4",
                "--reduced-schedule", str(REDUCED), "--attempt", "1"]
    child = subprocess.Popen(cmd_kill, stdout=log_file, stderr=subprocess.STDOUT, cwd=ROOT)
    sentinel = v2_out / "KILL_READY"
    deadline = time.time() + 600
    while time.time() < deadline and not sentinel.exists():
        if child.poll() is not None:
            raise SystemExit("killrun worker exited before sentinel")
        time.sleep(0.5)
    time.sleep(2.0)  # grace (matches Phase 1 driver)
    child.kill()
    child.wait()
    from cycleft.e5.event_log import supervisor_termination_record

    sup = supervisor_termination_record(
        OUT / "supervisor_log.jsonl", exec_id="e5p2-v2-killrun",
        child_pid=child.pid, mode="killrun", returncode=child.returncode,
        extra={"killed_after_sentinel": True},
    )
    # deterministic fallback fixture: remove the newest bundle (step 4) so
    # recovery falls back to the step-3 bundle -> position 4 is replayed
    newest = v2_out / "bundles" / "step_000004"
    if newest.exists():
        import shutil

        shutil.rmtree(newest)
    checks_v2 = {"supervisor_recorded_signal": sup["signal"] == 9,
                 "supervisor_exit_status": sup["exit_status"]}

    cmd_resume = ["uv", "run", "python", str(WORKER), "--mode", "resume",
                  "--out", str(v2_out), "--seed", "42",
                  "--ckpt-every", "3", "--reduced-schedule", str(REDUCED),
                  "--attempt", "2"]
    recovery = subprocess.Popen(cmd_resume, stdout=log_file, stderr=subprocess.STDOUT, cwd=ROOT)
    recovery.wait()
    sup_resume = supervisor_termination_record(
        OUT / "supervisor_log.jsonl", exec_id="e5p2-v2-resume",
        child_pid=recovery.pid, mode="resume", returncode=recovery.returncode,
    )
    checks_v2["resume_clean_exit"] = sup_resume["returncode"] == 0

    # position accounting over BOTH processes' step events
    events_all = [
        e for e in read_events(v2_out / "event_log.jsonl") if e["event"] == "step"
    ]
    position = validate_positions(events_all, target_positions=REDUCED,
                                  interruption_boundaries=[2])
    checks_v2["position_exactly_once"] = position["exactly_once_composite"]
    checks_v2["reexecuted_entries"] = position["reexecuted_entries"]
    checks_v2["valid_replays"] = len(position["valid_replays"])
    checks_v2["missing_logical_positions"] = position["missing_logical_positions"]
    checks_v2["physical_step_attempts"] = position["physical_step_attempts"]
    checks_v2["unique_logical_positions_completed"] = position[
        "unique_logical_positions_completed"
    ]

    # timing aggregation across processes (train only; per-process retained)
    per_process = []
    for run_dir in (v2_out,):
        pass
    events_full = read_events(v2_out / "event_log.jsonl")
    summary["v2"] = {
        "checks": checks_v2,
        "supervisor_records": [sup, sup_resume],
        "position_audit": position,
        "final_eval_present": (v2_out / "final_eval.json").exists(),
    }
    print("V2:", checks_v2)

    # ---- aggregate ----
    summary["all_checks_pass"] = (
        all(checks_v1.values())
        and checks_v2["supervisor_recorded_signal"]
        and checks_v2["resume_clean_exit"]
        and checks_v2["position_exactly_once"]
        and checks_v2["valid_replays"] >= 1
        and checks_v2["missing_logical_positions"] == []
    )
    (OUT / "VALIDATION_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    print("ALL CHECKS PASS:", summary["all_checks_pass"])


from cycleft.config import load_config  # noqa: E402

if __name__ == "__main__":
    main()
