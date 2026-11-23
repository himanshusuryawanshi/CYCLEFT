"""E7-to-E5 execution adapter (lazy-importing; delegates to the existing E5
workflow — no new trainer).

The adapter translates an authoritative E7 matrix row into the exact
configuration surface supported by ``cycleft.e5.resumable_trainer.run_workload``
(the same call pattern as ``scripts/e6_worker.py``: E1 config resolution,
``prepare_event_datasets`` seed-42 stream, b500 stage schedule, ``P1``
stage-end checkpoints at ``ckpt_every=500``, ``final_eval_events``).

LAZY IMPORTS: this module's top level imports only the standard library.
torch / transformers / peft / cycleft.e1 / cycleft.e5 are imported inside
the execution callbacks, so dry-run and matrix-inspection paths never load
training dependencies. Only the separately authorized execution path
calls the injected callbacks.

INTERFACE LIMITATION (documented per task §8 — not worked around):
``run_workload`` constructs the model, LoRA adapter, and optimizer ONCE
per invocation with a single ``lora_r``/``lora_alpha`` (fresh AdamW per
STAGE, rank fixed across stages). E7 STATIC policies (one fixed rank)
map directly onto this surface. E7 DYNAMIC policies — Grow (4→8→16),
Reduce (16→8→4), and Resource-Aware (runtime-decided ranks) — require a
rank change at event boundaries, which the existing E5 interface cannot
express without modifying ``resumable_trainer.py`` (not authorized) or
building a new per-event orchestration (a new trainer — prohibited).
Integration for the 7 dynamic NEW runs is therefore STOPPED FOR REVIEW:
``row_to_e5_plan`` raises ``E5AdapterLimitation`` for them.

Per-event predictions note (same limitation family): ``run_workload``
persists per-event METRICS (``final_eval.json``) but discards per-example
records, so the adapter performs a post-training final-state prediction
sweep by reloading the final checkpoint bundle through the existing E5
APIs (``bundles/LATEST.json`` -> ``load_adapter_for_training`` ->
``evaluate_event``) and writing E6-schema ``predictions/after_event_*.jsonl``
files. First real invocation (e7-static-r9-s42, 2026-11-23) exposed a
double-wrap defect — ``attach_lora`` followed by ``load_adapter_for_
training`` produced nested PEFT models whose checkpoint keys matched
nothing (PEFT "missing adapter keys"; empirically reproduced with a tiny
mock: all lora_B tensors stayed at init), so the sweep would have scored
the untrained base model. Fixed 2026-11-23: the trained adapter is loaded
directly into the base model (single wrap, the run_workload resume path)
and verified against the bundle fingerprint before any prediction is
written. The attempt-1 interruption is documented in
results/e7/pilot_static_r9_s42_20261123/.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

__all__ = [
    "E5AdapterLimitation",
    "SUPPORTED_POLICIES",
    "UNSUPPORTED_DYNAMIC_POLICIES",
    "RUN_WORKLOAD_RANK_LIMITATION",
    "make_run_fn",
    "row_to_e5_plan",
]

# Fixed-rank policies that map directly onto run_workload's single-rank
# invocation surface.
SUPPORTED_POLICIES = {"Static-R4", "Static-R9", "Static-R16", "Static-R32",
                      "Static-R8"}
UNSUPPORTED_DYNAMIC_POLICIES = {"Grow", "Reduce", "Resource-Aware"}

RUN_WORKLOAD_RANK_LIMITATION = (
    "cycleft.e5.resumable_trainer.run_workload constructs the model, LoRA "
    "adapter, and optimizer once per invocation with a single fixed "
    "lora_r/lora_alpha (fresh AdamW per stage; rank constant across "
    "stages). E7 dynamic policies (Grow 4->8->16, Reduce 16->8->4, "
    "Resource-Aware runtime-decided ranks) require an event-boundary rank "
    "transition, which this interface cannot express without modifying E5 "
    "(not authorized) or creating a new per-event trainer (prohibited). "
    "Integration of the 7 dynamic NEW runs is stopped for review per task "
    "section 8.")


class E5AdapterLimitation(Exception):
    """Raised when an E7 requirement cannot be met by the existing E5
    interface without unauthorized changes."""


def row_to_e5_plan(spec) -> dict:
    """Translate a validated E7 matrix row into an E5 execution plan.

    Pure standard-library validation and mapping (no torch, no model).
    Raises E5AdapterLimitation for dynamic policies (see module docstring).
    """
    if spec.policy in UNSUPPORTED_DYNAMIC_POLICIES:
        raise E5AdapterLimitation(
            f"{spec.exec_id} ({spec.policy}): {RUN_WORKLOAD_RANK_LIMITATION}")
    if spec.policy not in SUPPORTED_POLICIES:
        raise E5AdapterLimitation(
            f"{spec.exec_id}: policy {spec.policy!r} is not in the supported "
            f"set {sorted(SUPPORTED_POLICIES)}")
    if spec.row_type != "NEW":
        raise E5AdapterLimitation(
            f"{spec.exec_id}: REUSE anchors are read-only references and are "
            f"never executed")
    if spec.depth != "D100":
        raise E5AdapterLimitation(
            f"{spec.exec_id}: depth {spec.depth!r} unsupported by this adapter "
            f"(E7 v5 protocol fixes D100)")
    if spec.budget_per_event != 500:
        raise E5AdapterLimitation(
            f"{spec.exec_id}: budget_per_event {spec.budget_per_event} != 500 "
            f"(b500 protocol)")

    rank = int(spec.event_ranks[0])
    alpha = int(spec.event_alphas[0])
    if any(int(r) != rank for r in spec.event_ranks):
        raise E5AdapterLimitation(
            f"{spec.exec_id}: static policy has non-constant rank schedule "
            f"{spec.event_ranks}")
    if alpha != 2 * rank:
        raise E5AdapterLimitation(
            f"{spec.exec_id}: alpha {alpha} != 2 * rank {rank} (E7 v5 protocol)")

    # b500 protocol: 3 stages x 500 steps over the seed-order stream
    # (identical to scripts/e6_worker.py and E2 rank8_b500).
    stage_schedule = [
        {"stage": 1, "event_id": 1, "start": 0, "end": 500},
        {"stage": 2, "event_id": 2, "start": 500, "end": 1000},
        {"stage": 3, "event_id": 3, "start": 1000, "end": 1500},
    ]
    return {
        "exec_id": spec.exec_id,
        "policy": spec.policy,
        "lora_r": rank,
        "lora_alpha": alpha,
        "seed": spec.seed,
        "depth": "D100",
        "lora_precision": "float32",
        "dtype": "float16",
        "max_new_tokens": 24,
        "budget_per_event": 500,
        "stage_schedule": stage_schedule,
        "checkpoint_policy": "P1 (stage-end commits, ckpt_every=500)",
        "fresh_adamw_per_event": True,
        "note": ("fresh AdamW per stage is run_workload's native E1/E2/E5 "
                 "semantics; no optimizer state crosses event boundaries "
                 "(OPTIMIZER_LIFECYCLE_FINAL.md)"),
    }


def _default_train_fn(repo_root: Path) -> Callable:
    """Real E5-backed training callback (lazy imports; execution path only)."""

    def train(spec, run_dir: Path, plan: dict) -> dict:
        import torch  # noqa: F401  (lazy: authorized execution path only)

        from cycleft.config import load_config
        from cycleft.e1.data import prepare_event_datasets
        from cycleft.e1.pipeline import resolve_lm_config
        from cycleft.e5.resumable_trainer import run_workload

        raw_config = load_config(repo_root / "configs" / "e1_stream.yaml")
        lm_config, _, _ = resolve_lm_config(
            raw_config, mode="full", train_examples=None,
            seed=plan["seed"], device=None)
        # E7 v5 overrides (validated in row_to_e5_plan):
        lm_config.lora_r = plan["lora_r"]
        lm_config.lora_alpha = plan["lora_alpha"]
        lm_config.lora_precision = plan["lora_precision"]
        lm_config.dtype = plan["dtype"]
        lm_config.max_new_tokens = plan["max_new_tokens"]
        lm_config.seed = plan["seed"]
        # D100 = all 16 decoder layers: default q/k/v/o target modules with
        # layers_to_transform=None (the E2/E5/E6 full-depth convention).

        events, _manifest = prepare_event_datasets(train_limit=None,
                                                   seed=plan["seed"])
        rows = [dict(r) for event in events for r in event.train_rows]

        summary = run_workload(
            lm_config=lm_config,
            stream_rows=rows,
            out_dir=Path(run_dir),
            exec_id=plan["exec_id"],
            attempt=1,
            policy="P1",
            ckpt_every=500,   # stage-end commits only (E2 convention)
            resume=False,
            log=lambda m: print(f"[e7:{plan['exec_id']}] {m}", flush=True),
            final_eval_events=events,
            stage_schedule=plan["stage_schedule"],
        )
        status = summary.get("status")
        # Fail closed: only an explicitly complete workload yields exit 0.
        if status != "complete":
            return {"exit_code": 1, "status": str(status),
                    "summary_ref": "execution_summary.json",
                    "detail": f"workload did not complete (status={status!r})"}
        return {
            "exit_code": 0,
            "status": "complete",
            "global_step": summary.get("global_step"),
            "final_avg_em": summary.get("final_avg_em"),
            "summary_ref": "execution_summary.json",
            "final_eval_ref": "final_eval.json",
        }

    return train


def _default_predict_fn(repo_root: Path) -> Callable:
    """Final-state prediction sweep via the existing E5 checkpoint bundle.

    Reloads the final adapter (bundles/LATEST.json -> load_adapter_for_
    training) and runs evaluate_event per event, writing E6-schema
    predictions/after_event_{event_id}.jsonl. Lazy imports; execution only.
    """

    def predict(spec, run_dir: Path, plan: dict) -> dict:
        from cycleft.config import load_config
        from cycleft.e1.data import prepare_event_datasets
        from cycleft.e1.lm import (
            evaluate_event,
            load_adapter_for_training,
            load_base_model,
            load_tokenizer,
            resolve_device,
        )
        from cycleft.e1.pipeline import resolve_lm_config
        from cycleft.e5.resume_bundle import adapter_fingerprint

        raw_config = load_config(repo_root / "configs" / "e1_stream.yaml")
        lm_config, _, _ = resolve_lm_config(
            raw_config, mode="full", train_examples=None,
            seed=plan["seed"], device=None)
        lm_config.lora_r = plan["lora_r"]
        lm_config.lora_alpha = plan["lora_alpha"]

        latest = json.loads((Path(run_dir) / "bundles" / "LATEST.json").read_text())
        adapter_dir = Path(run_dir) / "bundles" / latest["latest"] / "adapter"

        events, _ = prepare_event_datasets(train_limit=None, seed=plan["seed"])
        base = load_base_model(lm_config)
        # load_base_model leaves the model on CPU; run_workload explicitly
        # moves it to the resolved device before training. resolve_device
        # prefers MPS and raises when it is unavailable (silent CPU
        # fallback is disallowed by E1 convention and the E7 protocol).
        device = resolve_device(None)
        # Load the FINAL TRAINED adapter into the base model (single PEFT
        # wrap — the exact run_workload resume path). Calling attach_lora
        # here would double-wrap the model: the checkpoint keys then match
        # nothing, PEFT warns "Found missing adapter keys", and the sweep
        # would silently evaluate the untrained base model (empirically
        # reproduced 2026-11-23; e7-static-r9-s42 attempt 1 interrupted).
        model = load_adapter_for_training(base, adapter_dir, lm_config, device)
        fingerprint = adapter_fingerprint(model)
        expected = latest.get("adapter_fingerprint")
        if expected and fingerprint != expected:
            raise RuntimeError(
                "final adapter fingerprint mismatch ("
                f"{fingerprint[:12]} != {expected[:12]}); refusing to "
                "evaluate a non-terminal adapter")
        tokenizer = load_tokenizer(lm_config.model_id)

        predictions_dir = Path(run_dir) / "predictions"
        predictions_dir.mkdir(parents=True, exist_ok=True)
        written = []
        for event in events:
            evaluation = evaluate_event(model, tokenizer, event.eval_rows,
                                        lm_config)
            out = predictions_dir / f"after_event_{event.event_id}.jsonl"
            with open(out, "w", encoding="utf-8") as f:
                for record in evaluation["records"]:
                    record = dict(record)
                    record["state"] = f"after_event_{event.event_id}"
                    # E5's evaluate_event records carry no test-set
                    # identifier; the E7 predictions schema requires it
                    # (PREDICTION_RECORD_KEYS; event_orchestrator.py:1470
                    # precedent — event.event_id IS the evaluation
                    # test-set ID in (1, 2, 3)). Preflight finding P-1.
                    record["test_event"] = event.event_id
                    f.write(json.dumps(record) + "\n")
            written.append(out.name)
        return {
            "exit_code": 0,
            "status": "complete",
            "predictions_files": written,
            "metrics_ref": "final_eval.json",
        }

    return predict


def make_run_fn(repo_root: Path,
                train_fn: Callable | None = None,
                predict_fn: Callable | None = None) -> Callable:
    """Compose the harness ``run_fn`` from a training and an evaluation
    callback. Defaults delegate to the existing E5 workflow (lazy imports).
    Tests inject synthetic callbacks; nothing here is invoked during
    design-only tasks.

    Callback contract (strict — feeds Finding-1 completion gates):
        train_fn(spec, run_dir, plan)   -> {"exit_code": int, "status": str, ...}
        predict_fn(spec, run_dir, plan) -> {"exit_code": int, "status": str, ...}
    """
    resolved_train = train_fn if train_fn is not None \
        else _default_train_fn(Path(repo_root))
    resolved_predict = predict_fn if predict_fn is not None \
        else _default_predict_fn(Path(repo_root))

    def run_fn(spec, run_dir: Path) -> dict:
        plan = row_to_e5_plan(spec)  # validation happens before any model work
        training = resolved_train(spec, Path(run_dir), plan)
        evaluation = resolved_predict(spec, Path(run_dir), plan)
        return {
            "training": training,
            "evaluation": evaluation,
            "e5_plan": plan,
            "artifact_refs": {
                "execution_summary": "execution_summary.json",
                "final_eval": "final_eval.json",
                "event_log": "event_log.jsonl",
                "audit_log": "audit_log.jsonl",
                "bundles": "bundles",
                "predictions": "predictions",
            },
        }

    return run_fn
