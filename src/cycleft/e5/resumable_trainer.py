"""E5 ResumableTrainer — continual LoRA workload with faithful mid-stage resume,
full Phase-2 instrumentation, and Phase-2 policy semantics (P0/P1/P2/P3/P5).

Mirrors the E1/E2 training semantics (fresh AdamW per stage, set_seed at each
stage start, batch 1, GA 1, clip 1.0, same encoding/loss) and adds:
- TimingRegistry (perf_counter_ns; MPS sync around train intervals) with
  separated train / model_load / lora_attach / optimizer_restore / rng_restore
  / ckpt_* / durable_copy / eval_generation / process_wall categories.
- EventLog: append-only JSONL flushed at boundaries (survives SIGKILL); the
  parent supervisor records termination status/signals.
- Audit logging ENABLED BY DEFAULT with enriched step events.
- Checkpoint accounting split: serialize / hash_verify / fsync_commit /
  durable_copy (LOCAL transfer emulation) / restore.
- Phase-2 policies (e5/policies.py): P0 no-checkpoint; P1 fixed interval
  (+ stage ends); P2 = P1 + notice checkpoint `lead` steps before the kill;
  P3 adaptive interval k*; P5 oracle checkpoint exactly at the kill position.
  Interruption ordering rule: the kill at position p fires AFTER optimizer
  step p and BEFORE any checkpoint scheduled at p.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import torch

from cycleft.e1.lm import (
    LMRuntimeConfig,
    attach_lora,
    cast_lora_to_fp32,
    encode_training_example,
    evaluate_event,
    load_adapter_for_training,
    load_base_model,
    load_tokenizer,
    set_seed,
)
from cycleft.e5.event_log import EventLog
from cycleft.e5.policies import (
    NOTICE_LEAD_STEPS,
    ADAPTIVE_LAMBDA,
    ADAPTIVE_K_MIN,
    ADAPTIVE_K_MAX,
    adaptive_interval,
    policy_checkpoint_due,
)
from cycleft.e5.resume_bundle import (
    adapter_fingerprint,
    config_hash,
    load_latest_bundle,
    restore_rng_states,
    save_bundle,
)
from cycleft.e5.timing import TimingRegistry

WORKLOAD_A_STAGES = [
    {"stage": 1, "event_id": 1, "start": 0, "end": 125},
    {"stage": 2, "event_id": 2, "start": 500, "end": 625},
    {"stage": 3, "event_id": 3, "start": 1000, "end": 1125},
]

WORKLOAD_B_STAGES = [
    {"stage": 1, "event_id": 1, "start": 0, "end": 500},
    {"stage": 2, "event_id": 2, "start": 500, "end": 1000},
    {"stage": 3, "event_id": 3, "start": 1000, "end": 1500},
]


def run_workload(
    lm_config: LMRuntimeConfig,
    stream_rows: list[dict],
    out_dir: Path,
    *,
    exec_id: str = "",
    attempt: int = 1,
    policy: str = "P1",
    ckpt_every: int = 125,
    kill_at_step: int | None = None,
    clean_stop_at_step: int | None = None,
    notice_lead: int = NOTICE_LEAD_STEPS,
    adaptive_lambda: float = ADAPTIVE_LAMBDA,
    resume: bool = False,
    audit_path: Path | None = None,
    event_log_path: Path | None = None,
    durable_copy: bool = True,
    device: Any = None,
    log=None,
    final_eval_events=None,
    stage_schedule: list[dict] | None = None,
) -> dict:
    """Run the continual workload with resumable state + instrumentation."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bundles_dir = out_dir / "bundles"
    stages = stage_schedule or WORKLOAD_A_STAGES
    c_hash = config_hash(lm_config.to_dict())
    device = device or torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    emit = log or (lambda m: None)

    wall_start_ns = time.perf_counter_ns()
    timing = TimingRegistry(exec_id=exec_id or out_dir.name)
    event_log = EventLog(
        event_log_path or out_dir / "event_log.jsonl",
        exec_id=exec_id or out_dir.name,
    )
    event_log.emit("process_start", {
        "attempt": attempt, "mode": ("resume" if resume else "fresh"),
        "policy": policy, "seed": lm_config.seed, "device": str(device),
    }, fsync=True)

    # audit logging ENABLED BY DEFAULT
    if audit_path is None:
        audit_path = out_dir / "audit_log.jsonl"
    audit_file = open(audit_path, "a", encoding="utf-8")

    def audit(global_step, stage, intra_idx, row, dataset_pos):
        row_hash = hashlib.sha256(
            (str(row["question"]) + "|" + str(row["object"])).encode()
        ).hexdigest()[:16]
        audit_file.write(
            json.dumps(
                {
                    "exec_id": exec_id or out_dir.name,
                    "pid": os.getpid(),
                    "attempt": attempt,
                    "stage": stage,
                    "logical_step": global_step,
                    "dataset_example_position": dataset_pos,
                    "row_hash": row_hash,
                    "ts_epoch": time.time(),
                }
            )
            + "\n"
        )
        audit_file.flush()

    # ---- resume state BEFORE model construction ----
    resume_state = None
    if resume:
        t0 = timing.start("ckpt_restore")
        resume_state = load_latest_bundle(bundles_dir, c_hash)
        timing.stop("ckpt_restore", t0)
        event_log.emit("checkpoint_restore", {
            "bundle": str(resume_state["dir"].name),
            "global_step": resume_state["progress"]["global_step"],
            "fell_back": resume_state["fell_back"],
            "restore_timing_s": resume_state.get("restore_timing_s"),
        }, fsync=True)

    # ---- model loading (timed separately from attach) ----
    t0 = timing.start("model_load")
    tokenizer = load_tokenizer(lm_config.model_id)
    model = load_base_model(lm_config)
    model.to(device)
    timing.stop("model_load", t0)

    t0 = timing.start("lora_attach")
    if resume_state is not None:
        model = load_adapter_for_training(
            model, resume_state["adapter_dir"], lm_config, device
        )
    else:
        model = attach_lora(model, lm_config, device)
    cast_lora_to_fp32(model)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    timing.stop("lora_attach", t0)

    if resume_state is not None:
        restored_fp = adapter_fingerprint(model)
        if restored_fp != resume_state["fingerprint"]:
            raise RuntimeError(
                "restore verification failed: adapter fingerprint mismatch "
                f"({restored_fp[:12]} != {resume_state['fingerprint'][:12]})"
            )
        event_log.emit("resume_position", {
            "global_step": resume_state["progress"]["global_step"],
            "stage_index": resume_state["progress"]["stage_index"],
            "intra_stage_index": resume_state["progress"]["intra_stage_index"],
            "bundle": str(resume_state["dir"].name),
        }, fsync=True)
        emit(
            f"resumed from {resume_state['dir'].name} at global step "
            f"{resume_state['progress']['global_step']} (fingerprint verified)"
        )

    start_global = resume_state["progress"]["global_step"] if resume_state else 0
    start_stage_idx = resume_state["progress"]["stage_index"] if resume_state else 0
    start_intra = resume_state["progress"]["intra_stage_index"] if resume_state else 0

    trainable = [p for p in model.parameters() if p.requires_grad]
    totals = {"ckpt_s": 0.0, "steps": 0, "examples": 0}
    ckpt_accounting: dict[str, float] = {}
    stage_end_fingerprints: dict[str, str] = {}
    prior_fp = out_dir / "stage_end_fingerprints.json"
    if resume_state and prior_fp.exists():
        stage_end_fingerprints = json.loads(prior_fp.read_text())
    killed = False
    clean_stopped = False
    global_step = start_global
    kill_sentinel = out_dir / "KILL_READY"

    # P3 adaptive-interval running estimates
    step_cost_sum = 0.0
    step_cost_n = 0
    ckpt_cost_sum = 0.0
    ckpt_cost_n = 0
    last_checkpoint_step = start_global

    def commit_bundle(stage_index, intra_done, dataset_pos, reason):
        nonlocal ckpt_cost_sum, ckpt_cost_n, last_checkpoint_step
        event_log.emit("checkpoint_start", {
            "global_step": global_step, "stage_index": stage_index,
            "intra_done": intra_done, "reason": reason,
        })
        accounting: dict[str, float] = {}
        t0 = timing.start("ckpt_serialize")
        progress = {
            "global_step": global_step,
            "stage_index": stage_index,
            "intra_stage_index": intra_done,
            "config_hash": c_hash,
        }
        save_bundle(
            bundles_dir, model, optimizer, progress,
            {"device": str(device), "lm_config": lm_config.to_dict()},
            accounting=accounting,
            durable_copy=durable_copy,
        )
        timing.stop("ckpt_serialize", t0)
        mapping = {
            "hash_verify": "ckpt_hash_verify",
            "fsync_commit": "ckpt_fsync_commit",
            "durable_copy": "durable_copy",
        }
        for bundle_key, registry_key in mapping.items():
            timing.add(registry_key, int(accounting.get(bundle_key, 0.0) * 1e9))
        commit_total = sum(accounting.values())
        totals["ckpt_s"] += commit_total
        ckpt_cost_sum += commit_total
        ckpt_cost_n += 1
        last_checkpoint_step = global_step
        event_log.emit("checkpoint_end", {
            "global_step": global_step,
            "reason": reason,
            "accounting_s": {k: round(v, 6) for k, v in accounting.items()},
        }, fsync=True)

    for stage_index, stage in enumerate(stages):
        if stage_index < start_stage_idx:
            continue
        set_seed(lm_config.seed)
        optimizer = torch.optim.AdamW(trainable, lr=lm_config.learning_rate)
        mid_stage_resume = (
            resume_state is not None
            and stage_index == start_stage_idx
            and start_intra > 0
        )
        if mid_stage_resume:
            t0 = timing.start("optimizer_restore")
            optimizer.load_state_dict(resume_state["optimizer_state"])
            for group in optimizer.param_groups:
                group["lr"] = lm_config.learning_rate
            timing.stop("optimizer_restore", t0)
            t0 = timing.start("rng_restore")
            restore_rng_states(resume_state["rng"])
            timing.stop("rng_restore", t0)
            emit(
                f"stage {stage['stage']}: mid-stage resume at intra {start_intra} "
                "(optimizer + RNG states restored)"
            )
        rows = stream_rows[stage["start"] : stage["end"]]
        begin_intra = start_intra if mid_stage_resume else 0
        event_log.emit("train_segment_start", {
            "stage_index": stage_index, "begin_intra": begin_intra,
            "end_intra": len(rows),
        })
        model.train()
        segment_token = timing.start("train")
        for intra in range(begin_intra, len(rows)):
            step_t0 = time.perf_counter_ns()
            row = rows[intra]
            dataset_pos = stage["start"] + intra
            encoded = encode_training_example(
                tokenizer, str(row["question"]), str(row["object"]), lm_config
            )
            if encoded is None:
                continue
            input_ids, labels = encoded
            outputs = model(
                input_ids=input_ids.unsqueeze(0).to(device),
                attention_mask=torch.ones_like(input_ids).unsqueeze(0).to(device),
                labels=labels.unsqueeze(0).to(device),
                use_cache=False,
            )
            loss = outputs.loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at global step {global_step + 1}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, lm_config.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            totals["steps"] += 1
            totals["examples"] += 1
            step_cost_sum += (time.perf_counter_ns() - step_t0) / 1e9
            step_cost_n += 1
            audit(global_step, stage["stage"], intra, row, dataset_pos)
            event_log.emit("step", {
                "position": global_step, "attempt": attempt,
                "stage": stage["stage"], "dataset_example_position": dataset_pos,
            })
            del outputs, loss, input_ids, labels

            # interruption at this position? (fires BEFORE any checkpoint here)
            if kill_at_step is not None and global_step == kill_at_step:
                timing.stop("train", segment_token)
                event_log.emit("train_segment_end", {
                    "global_step": global_step, "reason": "interruption_point",
                })
                if policy == "P5":
                    commit_bundle(stage_index, intra + 1, dataset_pos,
                                  "oracle_checkpoint_at_interruption")
                event_log.emit("interruption_request", {
                    "global_step": global_step, "kind": "SIGKILL_scheduled",
                    "policy": policy,
                }, fsync=True)
                killed = True
                kill_sentinel.write_text(
                    json.dumps({
                        "global_step": global_step, "policy": policy,
                        "notice_checkpoint_at": (
                            kill_at_step - notice_lead if policy == "P2" else None
                        ),
                    }),
                    encoding="utf-8",
                )
                emit(
                    f"interruption at global step {global_step} (policy {policy}); "
                    "waiting for SIGKILL (timeout 300 s)"
                )
                deadline = time.time() + 300
                while time.time() < deadline:
                    time.sleep(0.5)
                raise RuntimeError("kill was never delivered (timeout)")

            # P2 notice checkpoint: `lead` steps before the scheduled kill
            if (
                policy == "P2"
                and kill_at_step is not None
                and global_step == kill_at_step - notice_lead
                and global_step > last_checkpoint_step
            ):
                timing.stop("train", segment_token)
                event_log.emit("interruption_request", {
                    "global_step": global_step, "kind": "notice_received",
                    "notice_lead_steps": notice_lead,
                })
                commit_bundle(stage_index, intra + 1, dataset_pos,
                              "notice_checkpoint")
                segment_token = timing.start("train")

            # policy checkpoint cadence
            step_cost_mean = (
                step_cost_sum / step_cost_n if step_cost_n else 1.0
            )
            ckpt_cost_mean = (
                ckpt_cost_sum / ckpt_cost_n if ckpt_cost_n else 1.2
            )
            due, reason = policy_checkpoint_due(
                policy, global_step, last_checkpoint_step,
                ckpt_every=ckpt_every,
                stage_end=(intra == len(rows) - 1),
                kill_at=kill_at_step,
                ckpt_cost_s=ckpt_cost_mean,
                step_cost_s=step_cost_mean,
            )
            if due:
                timing.stop("train", segment_token)
                event_log.emit("train_segment_end", {
                    "global_step": global_step, "reason": reason,
                })
                commit_bundle(stage_index, intra + 1, dataset_pos, reason)
                segment_token = timing.start("train")

        else:
            timing.stop("train", segment_token)
            event_log.emit("train_segment_end", {
                "stage_index": stage_index, "reason": "stage_complete",
                "global_step": global_step,
            })

        stage_end_fingerprints[f"stage_{stage['stage']}"] = adapter_fingerprint(model)
        (out_dir / "stage_end_fingerprints.json").write_text(
            json.dumps(stage_end_fingerprints, indent=2), encoding="utf-8"
        )
        if clean_stopped:
            break

    final_eval_out = None
    if final_eval_events is not None:
        event_log.emit("eval_start", {"n_events": len(final_eval_events)})
        model.eval()
        per_event = {}
        for event in final_eval_events:
            evaluation = evaluate_event(
                model, tokenizer, event.eval_rows, lm_config, log=emit
            )
            per_event[str(event.event_id)] = evaluation["metrics"]
        final_eval_out = per_event
        event_log.emit("eval_end", {"events": list(per_event)}, fsync=True)
        (out_dir / "final_eval.json").write_text(
            json.dumps(per_event, indent=2, default=str), encoding="utf-8"
        )

    process_wall_s = (time.perf_counter_ns() - wall_start_ns) / 1e9
    timing.add("process_wall", int(process_wall_s * 1e9), 1)
    event_log.emit("process_exit", {
        "status": ("killed" if killed else
                   "clean_stopped" if clean_stopped else "complete"),
        "global_step": global_step,
        "process_wall_s": round(process_wall_s, 3),
        "policy": policy,
        "attempt": attempt,
    }, fsync=True)
    event_log.close()
    if audit_file:
        audit_file.close()

    summary = _summary(totals, timing, killed, clean_stopped,
                       stage_end_fingerprints, global_step, final_eval_out,
                       process_wall_s, attempt, policy,
                       ckpt_cost=(ckpt_cost_sum / ckpt_cost_n if ckpt_cost_n else None),
                       step_cost=(step_cost_sum / step_cost_n if step_cost_n else None))
    (out_dir / "execution_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    return summary


def _summary(totals, timing: TimingRegistry, killed, clean_stopped,
             stage_end_fingerprints, global_step, final_eval, process_wall_s,
             attempt, policy, ckpt_cost, step_cost):
    final_em = None
    if final_eval:
        final_em = sum(m["exact_match"] for m in final_eval.values()) / len(final_eval)
    status = (
        "killed" if killed else ("clean_stopped" if clean_stopped else "complete")
    )
    return {
        "status": status,
        "policy": policy,
        "attempt": attempt,
        "global_step": global_step,
        "steps": totals["steps"],
        "examples": totals["examples"],
        "checkpoint_overhead_s": round(totals["ckpt_s"], 3),
        "mean_ckpt_cost_s": round(ckpt_cost, 4) if ckpt_cost else None,
        "mean_step_cost_s": round(step_cost, 4) if step_cost else None,
        "timing": timing.to_dict(),
        "process_wall_s": round(process_wall_s, 3),
        "final_avg_em": final_em,
        "stage_end_fingerprints": stage_end_fingerprints,
    }
