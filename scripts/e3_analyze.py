"""E3 post-run analysis.

Usage:
    uv run python scripts/e3_analyze.py --results-dir results/e3/<run_id>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cycleft.e3.analysis import analyze_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--tolerance", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outcome = analyze_run(args.results_dir, tolerance=args.tolerance)
    print(f"Analysis complete for {args.results_dir}")
    for name, path in outcome["files"].items():
        if name == "plots":
            print(f"  plots: {len(path)} files")
        else:
            print(f"  {name}: {path}")
    summary = outcome["summary"]
    print(
        f"\n  strategies completed: {summary['n_strategies_completed']}"
        f"/{summary['n_strategies_total']}"
    )
    print(f"  best final avg EM: {summary['best_final_avg_exact_match']}")
    print(f"  caveat: {summary['statistical_caveat']}")


if __name__ == "__main__":
    main()
