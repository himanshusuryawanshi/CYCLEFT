"""E6 worker: one depth/placement configuration x one seed (b500 protocol).

Uses the E2-rank8_b500 protocol (3 stages x 500 steps, continued adapter,
fresh optimizer per stage) with the E5 instrumentation, and selects LoRA
target layers via the deterministic E6 regex.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cycleft.e1.data import prepare_event_datasets
from cycleft.e1.pipeline import resolve_lm_config, write_json
from cycleft.e5.resumable_trainer import run_workload
from cycleft.e6.layer_selection import (
    ALL_CONFIGS,
    expected_trainable_params,
    lora_target_config,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", required=True, choices=sorted(ALL_CONFIGS))
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--exec-id", default=None)
    args = parser.parse_args()

    from cycleft.config import load_config

    raw_config = load_config("configs/e1_stream.yaml")
    lm_config, _, _ = resolve_lm_config(
        raw_config, mode="full", train_examples=None, seed=args.seed, device=None
    )

    indices = ALL_CONFIGS[args.config_name]
    target_kwargs = lora_target_config(indices)
    lm_config.lora_target_modules = tuple(target_kwargs["target_modules"])
    lm_config.lora_layers_to_transform = target_kwargs["layers_to_transform"]
    expected_params = expected_trainable_params(indices, rank=8)

    events, _manifest = prepare_event_datasets(train_limit=None, seed=args.seed)
    rows = []
    for event in events:
        rows.extend(dict(r) for r in event.train_rows)
    # b500 protocol: 3 stages x 500 steps over the seed-42 stream
    stage_schedule = [
        {"stage": 1, "event_id": 1, "start": 0, "end": 500},
        {"stage": 2, "event_id": 2, "start": 500, "end": 1000},
        {"stage": 3, "event_id": 3, "start": 1000, "end": 1500},
    ]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    exec_id = args.exec_id or out.name
    write_json(
        out / "e6_config.json",
        {
            "config_name": args.config_name,
            "layer_indices_0indexed": indices,
            "lora_target_config": target_kwargs,
            "expected_trainable_params": expected_params,
            "rank": 8,
            "alpha": 16,
            "budget_per_event": 500,
            "seed": args.seed,
            "attach_device_note": (
                "E3/E5 order: load_base_model -> .to(device) -> attach_lora "
                "(LoRA init on device); init fingerprint recorded post-attach"
            ),
        },
    )

    summary = run_workload(
        lm_config=lm_config,
        stream_rows=rows,
        out_dir=out,
        exec_id=exec_id,
        attempt=args.attempt,
        policy="P1",
        ckpt_every=500,  # stage-end commits only (E2 convention)
        resume=False,
        log=lambda m: print(f"[e6worker:{args.config_name}/s{args.seed}] {m}", flush=True),
        final_eval_events=events,
        stage_schedule=stage_schedule,
    )
    # post-run verification: the saved adapter must resolve to exactly the
    # configured layers (regex correctness evidence)
    from safetensors.torch import load_file

    latest = json.loads((out / "bundles/LATEST.json").read_text())
    adapter = load_file(str(out / "bundles" / latest["latest"] / "adapter/adapter_model.safetensors"))
    resolved = sorted({
        int(k.split("layers.")[1].split(".")[0]) for k in adapter if "layers." in k
    })
    write_json(out / "resolved_layers_verified.json", {
        "resolved_layers": resolved,
        "expected_layers": indices,
        "match": resolved == sorted(indices),
    })
    if resolved != sorted(indices):
        raise SystemExit(f"resolved layers {resolved} != spec {indices}")
    print(f"[e6worker] config={args.config_name} seed={args.seed} "
          f"final_em={summary['final_avg_em']} resolved_layers={resolved}",
          flush=True)


if __name__ == "__main__":
    main()
