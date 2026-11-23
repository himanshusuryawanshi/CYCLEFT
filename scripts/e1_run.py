"""E1 sequential experiment runner.

Runs the baseline evaluation, then sequential LoRA fine-tuning on
TemporalWiki events 1 -> 2 -> 3 with evaluation on all test sets after
every stage.

Usage:
    uv run python scripts/e1_run.py --mode pilot
    uv run python scripts/e1_run.py --mode full --run-id full_20261113
    uv run python scripts/e1_run.py --mode pilot --start-event 3 --run-id <id>   # resume
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cycleft.e1.pipeline import run_e1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/e1_stream.yaml")
    parser.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    parser.add_argument("--run-id", default=None,
                        help="Stable run id (default: <mode>_<timestamp>)")
    parser.add_argument("--train-examples", type=int, default=None,
                        help="Override the per-event training limit")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", choices=["mps", "cpu"], default=None,
                        help="cpu must be requested explicitly (no silent fallback)")
    parser.add_argument("--start-event", type=int, choices=[1, 2, 3], default=1,
                        help="Resume the stream at this event (requires prior "
                             "checkpoints and baseline in the run directory)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    event_metrics = run_e1(
        config_path=args.config,
        mode=args.mode,
        run_id=args.run_id,
        train_examples=args.train_examples,
        seed=args.seed,
        device=args.device,
        start_event=args.start_event,
    )

    print("\n=== E1 accuracy matrix (exact match) ===")
    for state, row in zip(event_metrics["states"], event_metrics["accuracy_matrix"]):
        cells = "  ".join(f"test{j}={value:.4f}" for j, value in
                          zip(event_metrics["test_events"], row))
        print(f"{state:>16}: {cells}")
    print(f"\nArtifacts: results/e1/{args.run_id or '(timestamped run)'}")


if __name__ == "__main__":
    main()
