"""E4 post-run analysis.

Usage:
    uv run python scripts/e4_analyze.py --results-dir results/e4/<run_id>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cycleft.e4.analysis import analyze_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True)
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
    print(f"\n  benchmarks completed: {summary['n_benchmarks_completed']}"
          f"/{summary['n_benchmarks_total']}")
    print(f"  measured resources: {summary['measured_resources']}")
    print(f"  multi-resource comparison: {summary['multi_resource_comparison']}")


if __name__ == "__main__":
    main()
