"""E1 dataset preparation: build the manifest and verify split integrity.

Usage:
    uv run python scripts/e1_prepare.py --mode pilot
    uv run python scripts/e1_prepare.py --mode full --train-examples 128
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cycleft.config import load_config, validate_stream_config
from cycleft.e1.data import prepare_event_datasets
from cycleft.e1.pipeline import REPO_ROOT, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/e1_stream.yaml")
    parser.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    parser.add_argument("--train-examples", type=int, default=None,
                        help="Override the per-event training limit")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", default=None,
                        help="Where to write dataset_manifest.json "
                             "(default: data/e1/dataset_manifest.json)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_config = load_config(args.config)
    validate_stream_config(raw_config)

    e1_section = raw_config.get("e1", {})
    mode_settings = e1_section.get("modes", {}).get(args.mode, {}) or {}

    train_limit = args.train_examples
    if train_limit is None:
        configured = mode_settings.get("train_examples_per_event")
        train_limit = None if configured in (None, "null") else int(configured)

    seed = args.seed if args.seed is not None else int(e1_section.get("seed", 42))

    stream = raw_config.get("stream", {})
    dataset_name = stream.get("dataset", {}).get("name", "temporalwiki")
    dataset_config = stream.get("dataset", {}).get("adapter", {})

    events, manifest = prepare_event_datasets(
        train_limit=train_limit,
        seed=seed,
        dataset=dataset_name,
        dataset_config=dataset_config,
    )

    out_path = Path(args.out) if args.out else REPO_ROOT / "data" / "e1" / "dataset_manifest.json"
    write_json(out_path, manifest)

    print(f"Manifest written to {out_path}")
    for event in events:
        event_meta = manifest["events"][event.event_id - 1]
        n_train = len(event.train_rows)
        n_eval = event.eval_pool_size
        line = (f"  event {event.event_id} ({event.slice_tag}): "
                f"train {n_train}, eval {n_eval}")
        overlap = event_meta.get("train_eval_overlap")
        if overlap:
            line += f", exact question overlap train/test = {overlap['exact_question_overlap']}"
        print(line)


if __name__ == "__main__":
    main()
