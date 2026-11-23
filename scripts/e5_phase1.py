"""E5 Phase-1 driver: correctness executions (anchor, kill+resume, clean
handoff, audit log) with subprocess-based SIGKILL injection and T0
verification.

Usage:
    uv run python scripts/e5_phase1.py --out results/e5/phase1_20261119
"""

from __future__ import annotations

import argparse
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


def run_worker(mode: str, out: Path, *, seed: int = 42, ckpt_every: int = 25,
               kill_step: int | None = None, stop_step: int | None = None,
               audit: bool = False, log_path: Path | None = None) -> subprocess.Popen:
    cmd = [
        "uv", "run", "python", str(WORKER),
        "--mode", mode, "--out", str(out), "--seed", str(seed),
        "--ckpt-every", str(ckpt_every),
    ]
    if kill_step is not None:
        cmd += ["--kill-step", str(kill_step)]
    if stop_step is not None:
        cmd += ["--stop-step", str(stop_step)]
    if audit:
        cmd += ["--audit"]
    log_file = open(log_path or (out.parent / f"{out.name}.log"), "a")
    log_file.write(f"\n===== worker {mode} -> {out} ({time.strftime('%H:%M:%S')}) =====\n")
    log_file.flush()
    process = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, cwd=ROOT)
    return process


SUPERVISOR_LOG: Path | None = None


def supervisor_record(exec_id: str, process: subprocess.Popen, mode: str, extra=None):
    record = supervisor_termination_record(
        SUPERVISOR_LOG, exec_id, process.pid, mode, process.returncode, extra
    )
    print(f"[supervisor] {exec_id}: {record['exit_status']}", flush=True)
    return record


def wait_for_sentinel(sentinel: Path, process: subprocess.Popen, timeout_s: float = 600.0):
    """Poll for the pre-kill sentinel, then SIGKILL the worker."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if sentinel.exists():
            time.sleep(2.0)  # small grace so the sentinel write is fully durable
            os.kill(process.pid, signal.SIGKILL)
            process.wait()
            return True
        if process.poll() is not None:
            return False  # worker exited on its own (failure)
        time.sleep(0.5)
    raise TimeoutError("kill sentinel never appeared")


def fingerprints(out: Path) -> dict:
    return json.loads((out / "stage_end_fingerprints.json").read_text())


def t0_check(anchor: dict, resumed: dict) -> dict:
    equal = all(
        anchor[f"stage_{s}"] == resumed[f"stage_{s}"] for s in (1, 2, 3)
    )
    return {"bit_identical": equal, "anchor": anchor, "resumed": resumed}


def main() -> None:
    from cycleft.e5.event_log import supervisor_termination_record

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/e5/phase1_20261119")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    global SUPERVISOR_LOG
    base = Path(args.out)
    base.mkdir(parents=True, exist_ok=True)
    SUPERVISOR_LOG = base / "supervisor_log.jsonl"
    results: dict[str, dict] = {}

    # ---- e5p1-anchor: uninterrupted ----
    out = base / "e5p1-anchor"
    if not (out / "execution_summary.json").exists():
        print("=== e5p1-anchor (uninterrupted) ===", flush=True)
        p = run_worker("anchor", out, seed=args.seed, ckpt_every=25)
        p.wait()
    results["e5p1-anchor"] = json.loads((out / "execution_summary.json").read_text())

    # ---- e5p1-killresume: SIGKILL at step 62 (bundle at 50; 12 steps lost) ----
    out = base / "e5p1-killresume"
    if not (out / "execution_summary.json").exists():
        print("=== e5p1-killresume (kill at 62; last bundle at 50) ===", flush=True)
        p = run_worker("killrun", out, seed=args.seed, ckpt_every=25, kill_step=62)
        ok = wait_for_sentinel(out / "KILL_READY", p)
        supervisor_record("e5p1-killresume.process1", p, "killrun",
                          {"killed_after_sentinel": ok})
        if not ok:
            raise SystemExit("killrun worker exited before the sentinel appeared")
        print("worker SIGKILLed after pre-kill bundle; launching recovery", flush=True)
        p2 = run_worker("resume", out, seed=args.seed, ckpt_every=25)
        p2.wait()
    results["e5p1-killresume"] = json.loads((out / "execution_summary.json").read_text())

    # ---- e5p1-cleanhandoff: clean stop at 62, restore in a fresh process ----
    out = base / "e5p1-cleanhandoff"
    if not (out / "execution_summary.json").exists():
        print("=== e5p1-cleanhandoff (clean stop at 62) ===", flush=True)
        p = run_worker("cleanstop", out, seed=args.seed, ckpt_every=25, stop_step=62)
        p.wait()
        p2 = run_worker("resume", out, seed=args.seed, ckpt_every=25)
        p2.wait()
    results["e5p1-cleanhandoff"] = json.loads((out / "execution_summary.json").read_text())

    # ---- e5p1-auditlog: kill at 62 with example-index audit ----
    out = base / "e5p1-auditlog"
    if not (out / "execution_summary.json").exists():
        print("=== e5p1-auditlog (kill at 62; audit log) ===", flush=True)
        p = run_worker("killrun", out, seed=args.seed, ckpt_every=25, kill_step=62,
                       audit=True)
        ok = wait_for_sentinel(out / "KILL_READY", p)
        supervisor_record("e5p1-auditlog.process1", p, "killrun",
                          {"killed_after_sentinel": ok})
        if not ok:
            raise SystemExit("auditlog worker exited before the sentinel appeared")
        p2 = run_worker("resume", out, seed=args.seed, ckpt_every=25, audit=True)
        p2.wait()
    results["e5p1-auditlog"] = json.loads((out / "execution_summary.json").read_text())

    # ---- T0 verification ----
    t0 = t0_check(fingerprints(base / "e5p1-anchor"),
                  fingerprints(base / "e5p1-killresume"))
    t0_clean = t0_check(fingerprints(base / "e5p1-anchor"),
                        fingerprints(base / "e5p1-cleanhandoff"))

    anchor_em = results["e5p1-anchor"].get("final_avg_em")
    kill_em = results["e5p1-killresume"].get("final_avg_em")
    clean_em = results["e5p1-cleanhandoff"].get("final_avg_em")

    # audit-log verification: final pass covers each example exactly once
    audit_path = base / "e5p1-auditlog" / "audit_log.jsonl"
    audit_check = {"audit_file_present": audit_path.exists()}
    if audit_path.exists():
        entries = [json.loads(l) for l in audit_path.read_text().splitlines() if l]
        last_kill = max(i for i, e in enumerate(entries)
                        if e["global_step"] == 62) if entries else -1
        # the post-resume final pass = entries after the last step-62 event
        final_pass = entries[last_kill + 1:] if last_kill >= 0 else entries
        steps_final = [e["global_step"] for e in final_pass]
        audit_check.update({
            "total_entries": len(entries),
            "final_pass_entries": len(final_pass),
            "final_pass_covers_1_to_375_exactly_once": steps_final == list(range(1, 376)),
            "reexecuted_entries_before_kill": len(entries) - len(final_pass),
        })

    summary = {
        "phase": "E5 Phase 1 — checkpoint correctness",
        "executions": results,
        "t0_killresume": t0,
        "t0_cleanhandoff": t0_clean,
        "final_em": {"anchor": anchor_em, "killresume": kill_em,
                     "cleanhandoff": clean_em},
        "audit_check": audit_check,
        "verdict": {
            "t0_bit_identical": t0["bit_identical"] and t0_clean["bit_identical"],
            "quality_parity": (
                anchor_em is not None and kill_em is not None
                and abs(anchor_em - kill_em) < 1e-9
                and clean_em is not None and abs(anchor_em - clean_em) < 1e-9
            ),
        },
    }
    (base / "phase1_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps({
        "T0 killresume bit-identical": t0["bit_identical"],
        "T0 cleanhandoff bit-identical": t0_clean["bit_identical"],
        "final EM (anchor/killresume/cleanhandoff)":
            f"{anchor_em}/{kill_em}/{clean_em}",
        "audit check": audit_check,
    }, indent=1, default=str))
    print("PHASE 1 COMPLETE")


if __name__ == "__main__":
    main()
