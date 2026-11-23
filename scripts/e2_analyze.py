"""E2 post-run analysis: aggregation CSVs, Pareto frontier, plots, README.

Usage:
    uv run python scripts/e2_analyze.py --results-dir results/e2/<run_id>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cycleft.e2.analysis import analyze_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True,
                        help="e.g. results/e2/full_20261113")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outcome = analyze_run(args.results_dir)

    print(f"Analysis complete for {args.results_dir}")
    for name, path in outcome["files"].items():
        if name == "plots":
            print(f"  plots: {len(path)} files")
        else:
            print(f"  {name}: {path}")

    summary = outcome["summary"]
    print(f"\n  configurations completed: {summary['n_configurations_completed']}"
          f"/{summary['n_configurations_total']}")
    print(f"  Pareto frontier (training time): {summary['pareto_frontier_by_training_time']}")
    print(f"  Pareto frontier (tokens):        {summary['pareto_frontier_by_tokens']}")


if __name__ == "__main__":
    main()
