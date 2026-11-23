"""E3 pre-flight diagnostic: does the bootstrap model's prequential error
jump at the known boundary (stream position 500)?

Produces the exact shared artifacts the full run reuses:
- checkpoints/e3/<run_id>/bootstrap/           (shared bootstrap adapter)
- results/e3/<run_id>/base_error_stream.json   (base-model error, all 1500)
- results/e3/<run_id>/bootstrap_calibration_errors.json (positions 125-299)
plus a diagnostic extension (positions 300-649) saved separately.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from cycleft.config import load_config
from cycleft.e1.lm import LMRuntimeConfig
from cycleft.e1.pipeline import resolve_lm_config, write_json
from cycleft.e2.resources import ResourceTracker
from cycleft.e3.runner import (
    CHECKPOINTS_ROOT,
    RESULTS_ROOT,
    BootstrapRunner,
    compute_base_error_stream,
    compute_bootstrap_calibration_errors,
)
from cycleft.e3.stream import build_stream


def window_means(errors: dict[int, float], positions: list[int], size: int = 50):
    return [
        round(float(np.mean([errors[p] for p in positions[i : i + size]])), 4)
        for i in range(0, len(positions), size)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="full_20261116")
    parser.add_argument("--start", type=int, default=125)
    parser.add_argument("--end", type=int, default=650)
    args = parser.parse_args()

    lm_config, _, _ = resolve_lm_config(
        load_config("configs/e1_stream.yaml"),
        mode="full",
        train_examples=None,
        seed=42,
        device=None,
    )
    lm_config.lora_r = 4

    def log(message: str) -> None:
        print(f"[e3-diag] {message}", flush=True)

    stream, _events, _manifest = build_stream(seed=42)
    results_dir = RESULTS_ROOT / args.run_id
    checkpoints_dir = CHECKPOINTS_ROOT / args.run_id
    results_dir.mkdir(parents=True, exist_ok=True)

    tracker = ResourceTracker()

    # 1. Shared bootstrap adaptation (reused by the full run).
    bootstrap_path = checkpoints_dir / "bootstrap"
    if (bootstrap_path / "adaptation_provenance.json").exists():
        provenance = json.loads(
            (bootstrap_path / "adaptation_provenance.json").read_text()
        )
        bootstrap_seconds = provenance["training_stats"]["training_seconds"]
        log(f"bootstrap already exists ({bootstrap_seconds:.1f}s)")
    else:
        bootstrap_runner = BootstrapRunner(
            lm_config, stream, 125, checkpoints_dir, log
        )
        bootstrap_checkpoint, record = bootstrap_runner.run(tracker)
        bootstrap_seconds = record["training_seconds"]

    # 2. Base error stream (all positions; reused by the full run).
    base_path = results_dir / "base_error_stream.json"
    if base_path.exists():
        base_errors = {
            int(k): v
            for k, v in json.loads(base_path.read_text())["errors"].items()
        }
        log(f"base error stream cached ({len(base_errors)} positions)")
    else:
        base_errors = compute_base_error_stream(lm_config, stream.rows, log)
        write_json(
            base_path,
            {
                "model_state": "pretrained base (no adapter)",
                "signal": "prequential 0-1 prediction error",
                "feedback_delay_windows": 0,
                "errors": {str(k): v for k, v in base_errors.items()},
            },
        )

    # 3. Bootstrap errors for positions 125..end (calibration + diagnostic).
    bootstrap_errors = compute_bootstrap_calibration_errors(
        lm_config, stream.rows, bootstrap_path, start=args.start, end=args.end, log=log
    )
    write_json(
        results_dir / "bootstrap_calibration_errors.json",
        {
            "model_state": "shared bootstrap adapter",
            "signal": "prequential 0-1 prediction error",
            "positions": [125, 300],
            "calibration_window_indices": [3, 4, 5],
            "errors": {
                str(k): v
                for k, v in bootstrap_errors.items()
                if 125 <= k < 300
            },
        },
    )
    write_json(
        results_dir / "diagnostic_bootstrap_errors.json",
        {
            "model_state": "shared bootstrap adapter",
            "positions": [args.start, args.end],
            "errors": {str(k): v for k, v in bootstrap_errors.items()},
        },
    )

    # 4. Report the signal around the boundary.
    all_bootstrap = dict(bootstrap_errors)
    print("\n=== window mean prequential error (bootstrap model) ===")
    for start in range(150, min(args.end, 650), 50):
        window = list(range(start, start + 50))
        values = [all_bootstrap[p] for p in window]
        marker = ""
        if start == 500:
            marker = "  <== boundary 500 (event 2 begins)"
        if start == 550:
            marker = "  (first full event-2 window)"
        print(f"  positions {start:>4}-{start + 49:>4}: {np.mean(values):.4f}{marker}")
    pre = [all_bootstrap[p] for p in range(300, 500)]
    post = [all_bootstrap[p] for p in range(500, min(args.end, 650))]
    print(f"\npre-boundary  (300-499) mean error: {np.mean(pre):.4f}")
    print(f"post-boundary (500-{min(args.end, 650) - 1}) mean error: {np.mean(post):.4f}")
    base_pre = [base_errors[p] for p in range(300, 500)]
    base_post = [base_errors[p] for p in range(500, min(args.end, 650))]
    print(f"base model   (300-499): {np.mean(base_pre):.4f}  "
          f"(500-{min(args.end, 650) - 1}): {np.mean(base_post):.4f}")


if __name__ == "__main__":
    main()
