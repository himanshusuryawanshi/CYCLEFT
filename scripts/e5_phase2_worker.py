"""E5 Phase-2 worker: one process attempt under a checkpointing policy."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cycleft.e1.data import prepare_event_datasets
from cycleft.e1.pipeline import resolve_lm_config, write_json
from cycleft.e5.policies import SCHEDULES
from cycleft.e5.resumable_trainer import run_workload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True,
                        choices=["fresh", "resume"])
    parser.add_argument("--policy", required=True, choices=["P0", "P1", "P2", "P3", "P5"])
    parser.add_argument("--out", required=True)
    parser.add_argument("--config", default="configs/e1_stream.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ckpt-every", type=int, default=125)
    parser.add_argument("--kill-at", type=int, default=None,
                        help="logical position of THIS attempt's interruption")
    parser.add_argument("--notice-lead", type=int, default=10)
    parser.add_argument("--adaptive-lambda", type=float, default=1e-3)
    parser.add_argument("--workload", default="A", choices=["A", "B"])
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--exec-id", default=None)
    parser.add_argument("--reduced-schedule", type=int, default=None)
    args = parser.parse_args()

    from cycleft.config import load_config

    raw_config = load_config(args.config)
    lm_config, _, _ = resolve_lm_config(
        raw_config, mode="full", train_examples=None, seed=args.seed, device=None
    )

    events, _manifest = prepare_event_datasets(train_limit=None, seed=args.seed)
    rows = []
    for event in events:
        rows.extend(dict(r) for r in event.train_rows)
    stage_schedule = None
    if args.workload == "B":
        from cycleft.e5.resumable_trainer import WORKLOAD_B_STAGES

        stage_schedule = WORKLOAD_B_STAGES
    if args.reduced_schedule:
        stage_schedule = [{"stage": 1, "event_id": 1, "start": 0,
                           "end": args.reduced_schedule}]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    exec_id = args.exec_id or out.name
    # killrun/cleanstop-style processes stop mid-work; they do not run the
    # final evaluation (the completing process does)
    final_eval_events = None if args.kill_at is not None else events

    summary = run_workload(
        lm_config=lm_config,
        stream_rows=rows,
        out_dir=out,
        exec_id=exec_id,
        attempt=args.attempt,
        policy=args.policy,
        ckpt_every=args.ckpt_every,
        kill_at_step=args.kill_at,
        notice_lead=args.notice_lead,
        adaptive_lambda=args.adaptive_lambda,
        resume=args.mode == "resume",
        durable_copy=True,
        log=lambda m: print(f"[e5worker:{args.policy}/{args.mode}] {m}", flush=True),
        final_eval_events=final_eval_events,
        stage_schedule=stage_schedule,
    )
    write_json(out / "worker_summary.json", summary)
    print(f"[e5worker] policy={args.policy} mode={args.mode} "
          f"status={summary['status']} global_step={summary['global_step']} "
          f"final_em={summary['final_avg_em']}", flush=True)


if __name__ == "__main__":
    main()
