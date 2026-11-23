"""E5 Phase-2 driver: policy × interruption-schedule matrix (sequential).

For each policy: loop attempts (fresh / bundle-resume / scratch-restart per
policy semantics), SIGKILL at the scheduled logical positions, aggregate
position accounting + timing + quality, and emit per-policy records.
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

from cycleft.e5.event_log import (
    EventLog,
    read_events,
    supervisor_termination_record,
)
from cycleft.e5.policies import (
    ADAPTIVE_LAMBDA,
    NOTICE_LEAD_STEPS,
    SCHEDULES,
    next_kill_position,
)
from cycleft.e5.position_audit import validate_positions

WORKER = ROOT / "scripts" / "e5_phase2_worker.py"


class PolicyRun:
    """One policy's sequential multi-attempt execution."""

    def __init__(self, policy: str, schedule_id: str, workload: str,
                 base: Path, ckpt_every: int, seed: int = 42, log=None):
        self.policy = policy
        self.schedule_id = schedule_id
        self.schedule = SCHEDULES[schedule_id]
        self.workload = workload
        self.ckpt_every = ckpt_every
        self.seed = seed
        self.base = base
        self.log = log or print
        # include the checkpoint interval in the id: two policies can share
        # (policy family, schedule, workload) with different k values
        self.run_id = (
            f"e5p2-{policy.lower()}-k{ckpt_every}-{schedule_id.lower()}-{workload.lower()}"
        )
        self.out = base / self.run_id
        self.out.mkdir(parents=True, exist_ok=True)
        self.bundles = self.out / "bundles"
        self.event_log_path = self.out / "event_log.jsonl"
        self.supervisor_log = base / "supervisor_log.jsonl"
        self.fired: list[int] = []
        self.attempt = 0
        self.records: list[dict] = []

    def _newest_bundle_step(self) -> int | None:
        latest = self.bundles / "LATEST.json"
        if latest.exists():
            return json.loads(latest.read_text())["global_step"]
        return None

    def _worker(self, kill_at: int | None, resume: bool, attempt: int,
                final_eval: bool) -> subprocess.Popen:
        cmd = [
            "uv", "run", "python", str(WORKER),
            "--mode", "resume" if resume else "fresh",
            "--policy", self.policy,
            "--out", str(self.out),
            "--seed", str(self.seed),
            "--ckpt-every", str(self.ckpt_every),
            "--workload", self.workload,
            "--attempt", str(attempt),
            "--exec-id", self.run_id,
            "--notice-lead", str(NOTICE_LEAD_STEPS),
            "--adaptive-lambda", str(ADAPTIVE_LAMBDA),
        ]
        if kill_at is not None:
            cmd += ["--kill-at", str(kill_at)]
        log_path = self.base / f"{self.run_id}.log"
        handle = open(log_path, "a")
        handle.write(f"\n===== attempt {attempt} kill={kill_at} "
                     f"({time.strftime('%H:%M:%S')}) =====\n")
        handle.flush()
        # start_new_session: the worker becomes a process-group leader so the
        # supervisor can SIGKILL the whole group (uv wrapper + python child)
        return subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT,
                                cwd=ROOT, start_new_session=True)

    def execute(self, timeout_s: float = 7200.0) -> dict:
        sentinel = self.out / "KILL_READY"
        existing = self.out / "policy_result.json"
        if existing.exists():
            self.log(f"[{self.run_id}] policy_result.json exists; skipping")
            return json.loads(existing.read_text())
        """Run attempts until the workload completes or attempts are exhausted."""
        max_attempts = 12
        target = 375 if self.workload == "A" else 1500
        resume = False
        while self.attempt < max_attempts:
            self.attempt += 1
            start_position = self._newest_bundle_step() or 0
            kill_at = next_kill_position(self.schedule, start_position, self.fired)
            # P0 never resumes from durable state
            resume = self.policy != "P0" and start_position > 0
            self.log(
                f"[{self.run_id}] attempt {self.attempt}: start={start_position} "
                f"kill={kill_at} policy={self.policy} resume={resume}"
            )
            # remove any stale sentinel from a previous attempt (the poll
            # below waits for THIS attempt's sentinel)
            sentinel.unlink(missing_ok=True)
            process = self._worker(kill_at, resume, self.attempt,
                                   final_eval=(kill_at is None))
            if kill_at is None:
                process.wait()
            else:
                deadline = time.time() + timeout_s
                fired_seen = False
                while time.time() < deadline:
                    if sentinel.exists():
                        time.sleep(2.0)
                        try:
                            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        process.wait()
                        fired_seen = True
                        break
                    if process.poll() is not None:
                        break
                    time.sleep(0.5)
                if not fired_seen:
                    raise RuntimeError(
                        f"{self.run_id}: worker exited before the kill sentinel"
                    )
                self.fired.append(kill_at)
                self.log(f"[{self.run_id}] SIGKILL delivered at {kill_at}")
            supervisor_termination_record(
                self.supervisor_log, exec_id=self.run_id,
                child_pid=process.pid, mode=f"attempt{self.attempt}",
                returncode=process.returncode,
                extra={"policy": self.policy, "kill_at": kill_at},
            )
            try:
                summary = json.loads(
                    (self.out / "execution_summary.json").read_text()
                )
                # keep only the newest
            except FileNotFoundError:
                summary = None
            if kill_at is None and summary and summary.get("status") == "complete":
                self.records.append({
                    "attempt": self.attempt, "start_position": start_position,
                    "kill_at": None, "completed": True,
                    "summary": summary,
                })
                break
            self.records.append({
                "attempt": self.attempt, "start_position": start_position,
                "kill_at": kill_at, "completed": False,
            })
        else:
            raise RuntimeError(f"{self.run_id}: exhausted {max_attempts} attempts")

        # ---- position accounting over ALL attempts ----
        events = [e for e in read_events(self.event_log_path) if e["event"] == "step"]
        boundaries = list(range(2, self.attempt + 1))
        position = validate_positions(events, target_positions=target,
                                      interruption_boundaries=boundaries)
        # ---- timing aggregation across processes ----
        timing_total = {"train_s": 0.0, "ckpt_serialize_s": 0.0,
                        "ckpt_hash_verify_s": 0.0, "ckpt_fsync_commit_s": 0.0,
                        "durable_copy_s": 0.0, "ckpt_restore_s": 0.0,
                        "eval_generation_s": 0.0, "model_load_s": 0.0,
                        "lora_attach_s": 0.0, "optimizer_restore_s": 0.0,
                        "rng_restore_s": 0.0, "process_wall_s": 0.0}
        all_events = read_events(self.event_log_path)
        final_eval = None
        attempts_summaries = []
        # execution summaries are overwritten per attempt; reconstruct from
        # event-log timing categories instead
        cat_totals: dict[str, float] = {}
        wall_total = 0.0
        for e in all_events:
            if e["event"] == "process_exit":
                wall_total += e.get("process_wall_s", 0.0)
        # timing categories: aggregate via the last execution summary only if
        # present, else zero (per-process records are in worker summaries)
        for record in self.records:
            if record.get("completed") and record["summary"]:
                t = record["summary"].get("timing", {}).get("categories", {})
                for category, entry in t.items():
                    cat_totals[category] = cat_totals.get(category, 0.0) + entry["total_s"]
                final_em = record["summary"].get("final_avg_em")
        # NOTE: non-completing attempts' registry totals are lost with the
        # SIGKILL (Phase 2 fix pending: per-attempt summary writes by the
        # supervisor from worker stdout). Recorded as a known limitation.

        result = {
            "run_id": self.run_id,
            "policy": self.policy,
            "schedule_id": self.schedule_id,
            "workload": self.workload,
            "attempts": self.attempt,
            "fired_positions": self.fired,
            "final_avg_em": final_em,
            "position_audit": position,
            "timing_category_totals_s": {
                k: round(v, 3) for k, v in cat_totals.items()
            },
            "wall_total_s": round(wall_total, 1),
            "completed": True,
        }
        (self.out / "policy_result.json").write_text(
            json.dumps(result, indent=2, default=str), encoding="utf-8"
        )
        self.log(f"[{self.run_id}] COMPLETE: attempts={self.attempt} "
                 f"position_exactly_once={position['exactly_once_composite']} "
                 f"reexecuted={position['reexecuted_entries']}")
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="results/e5/phase2_20261119")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--runs",
        default="e5p2-policy0,e5p2-policy1-k125,e5p2-policy2-notice,"
                "e5p2-policy3-adaptive,e5p2-policy1k62-abrupt,e5p2-policy5-oracle,"
                "e5p2-stress-nonaligned,e5p2-scale-B-nockpt,e5p2-scale-B-k125",
    )
    args = parser.parse_args()

    base = Path(args.base)
    base.mkdir(parents=True, exist_ok=True)
    specs = {
        "e5p2-policy0": ("P0", "INT-2x-A", "A", 0),
        "e5p2-policy1-k125": ("P1", "INT-2x-A", "A", 125),
        "e5p2-policy2-notice": ("P2", "INT-2x-A", "A", 125),
        "e5p2-policy3-adaptive": ("P3", "INT-2x-A", "A", 0),
        "e5p2-policy1k62-abrupt": ("P1", "INT-2x-A", "A", 62),
        "e5p2-policy5-oracle": ("P5", "INT-2x-A", "A", 0),
        "e5p2-stress-nonaligned": ("P1", "INT-3x-A", "A", 125),
        "e5p2-scale-B-nockpt": ("P0", "INT-2x-B", "B", 0),
        "e5p2-scale-B-k125": ("P1", "INT-2x-B", "B", 125),
    }
    selected = [s.strip() for s in args.runs.split(",") if s.strip()]
    outputs = {}
    for run_id in selected:
        policy, schedule_id, workload, ckpt_every = specs[run_id]
        run = PolicyRun(policy, schedule_id, workload, base, ckpt_every,
                        seed=args.seed)
        outputs[run_id] = run.execute()
    (base / "phase2_outputs.json").write_text(
        json.dumps(outputs, indent=2, default=str), encoding="utf-8"
    )
    print("PHASE 2 COMPLETE")


if __name__ == "__main__":
    main()
