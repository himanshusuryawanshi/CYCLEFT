"""E3 runner CLI.

Usage:
    # Pilot (5 detector strategies, reduced adaptation budget):
    uv run python scripts/e3_run.py --run-id pilot_20261116 --pilot-budget 25

    # Full comparison (all 9 strategies incl. references, rank4_b125 action):
    uv run python scripts/e3_run.py --run-id full_20261116
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cycleft.e3.runner import RESULTS_ROOT, run_e3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/e1_stream.yaml")
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--strategies",
        default=None,
        help="Comma-separated subset of strategies (default: full set)",
    )
    parser.add_argument(
        "--pilot-budget",
        type=int,
        default=None,
        help="Reduced adaptation budget (optimizer steps) for the pilot",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", choices=["mps", "cpu"], default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    strategies = (
        tuple(s.strip() for s in args.strategies.split(",") if s.strip())
        if args.strategies
        else None
    )
    results = run_e3(
        config_path=args.config,
        run_id=args.run_id,
        strategies=strategies,  # type: ignore[arg-type]
        pilot_budget=args.pilot_budget,
        seed=args.seed,
        device=args.device,
    )
    print("\n=== E3 strategy status ===")
    for strategy, result in results.items():
        line = f"{strategy:>16}: {result.status}"
        if result.status == "complete":
            line += (
                f"  adaptations={result.n_adaptations}"
                f"  detection={result.detection.get('true_positives')}/2 TP,"
                f" {result.detection.get('false_positives')} FP"
            )
        else:
            line += f"  ({result.error})"
        print(line)
    print(f"\nResults: {RESULTS_ROOT / args.run_id}")


if __name__ == "__main__":
    main()
