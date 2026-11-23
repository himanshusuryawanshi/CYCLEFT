"""E2 grid runner CLI.

Usage:
    # Pilot (rank 8 and 16 at 25% and 100% budgets):
    uv run python scripts/e2_run.py --run-id pilot_20261113 \
        --ranks 8,16 --budgets b125,b500

    # Full grid:
    uv run python scripts/e2_run.py --run-id full_20261113
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cycleft.e2.runner import RESULTS_ROOT, run_e2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/e1_stream.yaml")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--ranks", default="4,8,16,32",
                        help="Comma-separated LoRA ranks (default: 4,8,16,32)")
    parser.add_argument("--budgets", default="b125,b250,b500",
                        help="Comma-separated budget ids (default: b125,b250,b500)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", choices=["mps", "cpu"], default=None)
    parser.add_argument("--baseline-from", default=None,
                        help="Path to a baseline_metrics.json to reuse "
                             "(default: the E1 full-run baseline)")
    parser.add_argument("--alpha", type=int, default=None,
                        help="Override lora_alpha for this run "
                             "(remediation Block A; default: configured 16)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ranks = tuple(int(rank) for rank in args.ranks.split(",") if rank.strip())
    budgets = tuple(budget.strip() for budget in args.budgets.split(",") if budget.strip())
    baseline_from = Path(args.baseline_from) if args.baseline_from else None

    results = run_e2(
        config_path=args.config,
        run_id=args.run_id,
        ranks=ranks,
        budget_ids=budgets,
        seed=args.seed,
        device=args.device,
        baseline_from=baseline_from,
        alpha=args.alpha,
    )

    print("\n=== E2 grid status ===")
    for config_id, result in results.items():
        line = f"{config_id:>16}: {result.status}"
        if result.status == "complete":
            line += (
                f"  final_avg_EM={result.final_avg_exact_match:.4f}  "
                f"train_time={sum(r.training_seconds for r in result.resource_records):.0f}s"
            )
        else:
            line += f"  ({result.error})"
        print(line)
    print(f"\nResults: {RESULTS_ROOT / args.run_id}")


if __name__ == "__main__":
    main()
