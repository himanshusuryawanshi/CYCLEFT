"""E1 post-run analysis: CL metrics, matrix CSV, plots, summary, README.

Usage:
    uv run python scripts/e1_analyze.py --results-dir results/e1/<run_id>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cycleft.e1.analysis import analyze_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True,
                        help="e.g. results/e1/pilot_20261113_120000")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outcome = analyze_run(args.results_dir)

    summary = outcome["summary"]
    cl = summary["continual_learning"]
    print(f"Analysis complete for {args.results_dir}")
    print(f"  matrix CSV: {outcome['matrix_csv']}")
    print(f"  plots: {len(outcome['plots'])} files")
    print(f"  summary: {Path(args.results_dir) / 'summary.json'}")
    print(f"  readme: {outcome['readme']}")
    print(f"\n  average forgetting: {cl['average_forgetting']:.4f}")
    print(f"  backward transfer:  {cl['backward_transfer']:.4f}")


if __name__ == "__main__":
    main()
