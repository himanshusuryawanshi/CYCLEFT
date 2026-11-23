"""Merge the completed E2 pilot configurations into the full-grid run directory.

The four pilot cells (rank8/rank16 x b125/b500) were executed on 2026-11-13
under run_id ``pilot_20261113`` with byte-identical configuration semantics
(seed 42, shared E1 baseline, same code path). The remaining eight cells were
executed on 2026-11-15 under ``full_20261115``. This script copies the pilot
per-configuration result artifacts and predictions into the full run
directory so that ``scripts/e2_analyze.py`` sees all 12 configurations, and
writes a PROVENANCE.md documenting exactly what came from where. Checkpoints
are NOT duplicated; their original locations are recorded in the provenance.

Usage:
    uv run python scripts/e2_merge_pilot.py --full-run full_20261115 \
        --pilot-run pilot_20261113
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cycleft.e2.runner import RESULTS_ROOT

CONFIG_ARTIFACTS = (
    "event_metrics.json",
    "resource_metrics.json",
    "continual_learning_matrix.csv.json",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-run", required=True)
    parser.add_argument("--pilot-run", default="pilot_20261113")
    args = parser.parse_args()

    full_dir = RESULTS_ROOT / args.full_run
    pilot_dir = RESULTS_ROOT / args.pilot_run
    if not full_dir.exists():
        raise SystemExit(f"missing run directory: {full_dir}")

    merged: list[dict] = []
    for config_dir in sorted(pilot_dir.iterdir()):
        if not config_dir.is_dir() or config_dir.name == "predictions":
            continue
        event_metrics_path = config_dir / "event_metrics.json"
        if not event_metrics_path.exists():
            continue
        event_metrics = json.loads(event_metrics_path.read_text(encoding="utf-8"))
        if event_metrics.get("status") != "complete":
            print(f"skip {config_dir.name}: not complete")
            continue

        target = full_dir / config_dir.name
        if target.exists():
            print(f"skip {config_dir.name}: already present in {full_dir.name}")
            continue
        target.mkdir(parents=True)
        for name in CONFIG_ARTIFACTS:
            source = config_dir / name
            if source.exists():
                shutil.copy2(source, target / name)
        pilot_predictions = pilot_dir / "predictions" / config_dir.name
        if pilot_predictions.exists():
            shutil.copytree(
                pilot_predictions,
                full_dir / "predictions" / config_dir.name,
                dirs_exist_ok=True,
            )
        merged.append(
            {
                "config_id": config_dir.name,
                "source_run_id": args.pilot_run,
                "source_results_dir": str(config_dir),
                "source_checkpoints_dir": str(
                    Path("checkpoints") / "e2" / args.pilot_run / config_dir.name
                ),
                "executed": "2026-11-13",
                "status": event_metrics.get("status"),
            }
        )
        print(f"merged {config_dir.name} <- {config_dir}")

    provenance_path = full_dir / "PROVENANCE.md"
    lines = [
        f"# Provenance for run `{args.full_run}`",
        "",
        "This run directory contains results from two execution windows with",
        "identical configuration semantics (same model, seed 42, dataset",
        "manifest, prompt template, optimizer settings, decoding settings,",
        "shared pretrained baseline from `results/e1/full_20261113/`):",
        "",
        f"- Pilot cells executed 2026-11-13 under run id `{args.pilot_run}`:",
    ]
    lines += [f"  - `{entry['config_id']}`" for entry in merged]
    lines += [
        f"- Remaining cells executed 2026-11-15 under run id `{args.full_run}`:",
        "  - all `(rank, budget)` cells not listed above (rank4 x3, rank8_b250,",
        "    rank16_b250, rank32 x3).",
        "",
        "Checkpoints for the pilot cells remain at",
        f"`checkpoints/e2/{args.pilot_run}/<config_id>/event_<t>/`; checkpoints",
        f"for the remaining cells are at `checkpoints/e2/{args.full_run}/`.",
        "No result numbers were recomputed, edited, or fabricated during the",
        "merge; only the artifact files listed above were copied.",
    ]
    provenance_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {provenance_path} ({len(merged)} configs merged)")


if __name__ == "__main__":
    main()
