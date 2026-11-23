"""E6 matrix driver: executes the NEW rows of FINAL_RUN_MATRIX.csv sequentially."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# The 15 NEW executions (FINAL_RUN_MATRIX.csv): E6-A 9 (3 depths x 3 seeds,
# D100 reused) + E6-B 9 (Early/Middle/Uniform x 3 seeds; Late25 = D25 shared)
NEW_RUNS = []
for seed in (42, 43, 44):
    for cfg in ("D75", "D50", "D25"):
        NEW_RUNS.append((f"e6a-{cfg.lower()}-s{seed}", cfg, seed))
    for cfg in ("Early25", "Middle25", "Uniform25"):
        NEW_RUNS.append((f"e6b-{cfg.lower()}-s{seed}", cfg, seed))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="results/e6/execution_20261119")
    parser.add_argument("--only", default=None,
                        help="Comma-separated exec_ids to (re)run; default: all NEW rows")
    args = parser.parse_args()

    base = Path(args.base)
    base.mkdir(parents=True, exist_ok=True)
    selected = (
        [s.strip() for s in args.only.split(",") if s.strip()]
        if args.only else [exec_id for exec_id, _, _ in NEW_RUNS]
    )
    spec = {exec_id: (cfg, seed) for exec_id, cfg, seed in NEW_RUNS}

    outputs = {}
    for exec_id in selected:
        cfg, seed = spec[exec_id]
        out = base / exec_id
        if (out / "execution_summary.json").exists():
            print(f"[e6matrix] {exec_id}: already complete, skipping", flush=True)
            outputs[exec_id] = json.loads((out / "execution_summary.json").read_text())
            continue
        print(f"[e6matrix] === {exec_id} ({cfg}, seed {seed}) ===", flush=True)
        cmd = [
            "uv", "run", "python", str(ROOT / "scripts" / "e6_worker.py"),
            "--config-name", cfg, "--out", str(out),
            "--seed", str(seed), "--exec-id", exec_id,
        ]
        log = open(base / f"{exec_id}.log", "a")
        log.write(f"\n===== {exec_id} ({time.strftime('%H:%M:%S')}) =====\n")
        log.flush()
        process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
        process.wait()
        summary = json.loads((out / "execution_summary.json").read_text())
        outputs[exec_id] = summary
        print(f"[e6matrix] {exec_id}: {summary['status']} "
              f"final_em={summary['final_avg_em']} "
              f"train={summary['timing']['categories'].get('train', {}).get('total_s', 0):.0f}s",
              flush=True)

    (base / "matrix_outputs.json").write_text(
        json.dumps(outputs, indent=2, default=str), encoding="utf-8"
    )
    print("E6 MATRIX COMPLETE", flush=True)


if __name__ == "__main__":
    main()
