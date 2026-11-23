"""E5 Phase-1 worker: one training execution (anchor / killrun / cleanstop / resume)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cycleft.e1.data import prepare_event_datasets
from cycleft.e1.pipeline import resolve_lm_config, write_json
from cycleft.e5.resumable_trainer import run_workload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True,
                        choices=["anchor", "killrun", "cleanstop", "resume"])
    parser.add_argument("--out", required=True)
    parser.add_argument("--config", default="configs/e1_stream.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ckpt-every", type=int, default=25)
    parser.add_argument("--kill-step", type=int, default=None)
    parser.add_argument("--stop-step", type=int, default=None)
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--exec-id", default=None)
    parser.add_argument("--reduced-schedule", type=int, default=None,
                        help="Instrumentation validation: 1 stage of N steps "
                             "instead of the full workload-A schedule")
    args = parser.parse_args()

    raw = json.loads((Path("configs/e1_stream.yaml")).read_text()) if False else None
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
    if args.reduced_schedule:
        stage_schedule = [{"stage": 1, "event_id": 1, "start": 0,
                           "end": args.reduced_schedule}]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    exec_id = args.exec_id or out.name
    audit_path = out / "audit_log.jsonl"  # ENABLED BY DEFAULT (Phase 2)
    # killrun/cleanstop processes stop mid-stage; they must not run the final eval
    final_eval_events = None if args.mode in ("killrun", "cleanstop") else events

    summary = run_workload(
        lm_config=lm_config,
        stream_rows=rows,
        out_dir=out,
        exec_id=exec_id,
        attempt=args.attempt,
        ckpt_every=args.ckpt_every,
        kill_at_step=args.kill_step if args.mode == "killrun" else None,
        clean_stop_at_step=args.stop_step if args.mode == "cleanstop" else None,
        resume=args.mode == "resume",
        audit_path=audit_path,
        log=lambda m: print(f"[e5worker:{args.mode}] {m}", flush=True),
        final_eval_events=final_eval_events,
        stage_schedule=stage_schedule,
    )
    write_json(out / "worker_summary.json", summary)
    print(f"[e5worker] mode={args.mode} status={summary['status']} "
          f"global_step={summary['global_step']} final_em={summary['final_avg_em']}",
          flush=True)


if __name__ == "__main__":
    main()
