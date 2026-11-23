"""E4 portable benchmark harness.

Runs the FIXED adaptation schedule (E3's event-boundary oracle convention:
a shared bootstrap-style first-segment adaptation plus the same adaptation
operation at each temporal transition — resource allocation is the only
variable; the drift detector and trigger schedule are held constant) on any
torch device, with identical data/protocol to E1/E2/E3:

* stream = E1/E2 per-event seed-42 shuffled training pools;
* adaptation stages for budget B: rows[0:B] (event 1), rows[500:500+B]
  (transition 1), rows[1000:1000+B] (transition 2);
* within a policy run the adapter CONTINUES across events (checkpoint
  save -> load -> bit-exact verification at every transition; never a
  fresh adapter between events);
* different policy runs always start from a fresh pretrained base;
* evaluation = the exact E1/E2 greedy protocol on the 3 held-out test sets
  at each segment end (matrix row 0 reuses the verified E1 baseline).

The harness is device-parameterized: the same code path runs on MPS, CUDA
or CPU, so cloud benchmarks use the identical protocol.

Every timing is recorded separately (model load, training, checkpoint
save, checkpoint load, verification, evaluation); checkpoint overhead is
never hidden inside training time.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from cycleft.e1.data import PreparedEvent
from cycleft.e1.lm import (
    LMRuntimeConfig,
    attach_lora,
    encode_training_example,
    evaluate_event,
    load_adapter_for_training,
    load_base_model,
    load_tokenizer,
    resolve_device,
    set_seed,
    train_event,
)
from cycleft.e1.metrics import (
    average_accuracy,
    average_forgetting,
    backward_transfer_conventional,
    forgetting,
)
from cycleft.e1.pipeline import REPO_ROOT, software_versions, write_json
from cycleft.e2.resources import ResourceTracker
from cycleft.e3.stream import EXAMPLES_PER_EVENT, build_stream

RESULTS_ROOT = REPO_ROOT / "results" / "e4"
CHECKPOINTS_ROOT = REPO_ROOT / "checkpoints" / "e4"
DEFAULT_BASELINE = (
    REPO_ROOT / "results" / "e1" / "full_20261113" / "baseline_metrics.json"
)

WORKLOADS = {
    "A_economical": {"lora_rank": 4, "budget": 125},
    "B_high_quality": {"lora_rank": 8, "budget": 500},
}


def schedule_stages(budget: int) -> list[dict]:
    """The fixed adaptation schedule (documented; identical across resources)."""
    return [
        {"stage": 1, "kind": "event1_bootstrap", "start": 0, "end": budget},
        {
            "stage": 2,
            "kind": "transition_1",
            "start": EXAMPLES_PER_EVENT,
            "end": EXAMPLES_PER_EVENT + budget,
        },
        {
            "stage": 3,
            "kind": "transition_2",
            "start": 2 * EXAMPLES_PER_EVENT,
            "end": 2 * EXAMPLES_PER_EVENT + budget,
        },
    ]


def _resolve_device_for(profile) -> torch.device:
    """Resolve the profile's device string, failing loudly when unavailable."""
    requested = profile.device
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("profile requires MPS but MPS is unavailable")
        return torch.device("mps")
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"profile requires {requested} but CUDA is unavailable on this host"
            )
        return torch.device(requested)
    raise ValueError(f"unsupported device string: {requested!r}")


@torch.no_grad()
def _snapshot_trainable(model) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().to("cpu").clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def _warm_up(model, tokenizer, rows: list[dict], lm_config, device) -> None:
    """3 real optimizer steps on the first rows (untimed kernel warm-up)."""
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("no trainable parameters for warm-up")
    optimizer = torch.optim.AdamW(trainable, lr=lm_config.learning_rate)
    model.train()
    for row in rows[:3]:
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
        outputs.loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, lm_config.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        del outputs


def _verify_weights_identical(
    before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]
) -> dict[str, Any]:
    if set(before) != set(after):
        return {"identical": False, "reason": "parameter name sets differ"}
    for name, tensor in before.items():
        if not torch.equal(tensor, after[name]):
            return {"identical": False, "reason": f"tensor {name} differs"}
    return {"identical": True, "reason": "all trainable tensors bit-identical"}


def _peak_memory_gb(device: str) -> float | None:
    if device.startswith("cuda") and torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e9
    if device == "mps" and torch.backends.mps.is_available():
        return torch.mps.current_allocated_memory() / 1e9
    return None


@dataclass
class BenchmarkResult:
    """All measurements of one (workload, resource) benchmark."""

    workload: str
    resource_id: str
    device: str
    status: str = "running"
    stages: list[dict] = field(default_factory=list)
    accuracy_matrix: list[list[float]] = field(default_factory=list)
    f1_matrix: list[list[float]] = field(default_factory=list)
    ppl_matrix: list[list[float]] = field(default_factory=list)
    states: list[str] = field(default_factory=list)
    transitions: list[dict] = field(default_factory=list)
    totals: dict = field(default_factory=dict)
    cl_metrics: dict = field(default_factory=dict)
    error: str | None = None


def run_benchmark(
    workload: str,
    resource_profile,  # ResourceProfile
    lm_config: LMRuntimeConfig,
    events: list[PreparedEvent],
    baseline_rows: tuple[list[float], list[float], list[float]],
    stream_rows: list[dict],
    checkpoints_dir: Path,
    log: Any,
    run_label: str = "",
) -> BenchmarkResult:
    """Execute one sequential three-event benchmark on one resource."""
    spec = WORKLOADS[workload]
    budget = spec["budget"]
    device = _resolve_device_for(resource_profile)
    result = BenchmarkResult(
        workload=workload,
        resource_id=resource_profile.resource_id,
        device=str(device),
    )
    tracker = ResourceTracker()
    tracker.sample()

    def log_message(message: str) -> None:
        log(f"[{workload}@{resource_profile.resource_id}]{run_label} {message}")

    try:
        # Warm-up: 3 real optimizer steps, untimed, then discarded.
        log_message("warm-up (3 optimizer steps, untimed)")
        set_seed(lm_config.seed)
        tokenizer = load_tokenizer(lm_config.model_id)
        warm_base = load_base_model(lm_config)
        warm_model = attach_lora(warm_base, lm_config, device)
        _warm_up(
            warm_model, tokenizer, stream_rows[:6], lm_config, device
        )
        del warm_model, warm_base
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        total_train_seconds = 0.0
        total_ckpt_save = 0.0
        total_ckpt_load = 0.0
        total_model_load = 0.0
        total_eval_seconds = 0.0
        total_steps = 0
        total_examples = 0
        total_tokens = 0
        peak_memory = 0.0

        accuracy_rows = [list(baseline_rows[0])]
        f1_rows = [list(baseline_rows[1])]
        ppl_rows = [list(baseline_rows[2])]
        states = ["base"]
        previous_checkpoint: Path | None = None
        carried_weights: dict[str, torch.Tensor] | None = None

        for stage in schedule_stages(budget):
            stage_label = f"stage_{stage['stage']}_{stage['kind']}"
            log_message(f"=== {stage_label} ===")
            set_seed(lm_config.seed)

            # 1. Fresh pretrained base (per stage, as in E2/E3) + adapter.
            load_started = time.perf_counter()
            tokenizer = load_tokenizer(lm_config.model_id)
            base_model = load_base_model(lm_config)
            model = load_adapter_for_training(
                base_model, previous_checkpoint, lm_config, device
            ) if previous_checkpoint is not None else attach_lora(
                base_model, lm_config, device
            )
            model_load_seconds = time.perf_counter() - load_started
            total_model_load += model_load_seconds

            # 2. Restore verification: the loaded adapter must carry the
            #    exact weights the previous stage saved.
            if carried_weights is not None:
                current = _snapshot_trainable(model)
                verification = _verify_weights_identical(carried_weights, current)
                if not verification["identical"]:
                    raise RuntimeError(
                        f"checkpoint restore verification failed: "
                        f"{verification['reason']}"
                    )
                log_message("adapter restore verified bit-identical")

            # 3. Timed training on the scheduled slice.
            rows = stream_rows[stage["start"] : stage["end"]]
            stats = train_event(
                model, tokenizer, rows, lm_config, log=log_message,
                on_step=lambda _s, _l: tracker.sample(),
            )
            total_train_seconds += stats["training_seconds"]
            total_steps += stats["optimizer_steps"]
            total_examples += stats["examples_scheduled"] - stats["examples_skipped"]
            total_tokens += stats["total_tokens_trained"]
            peak = _peak_memory_gb(str(device))
            if peak is not None:
                peak_memory = max(peak_memory, peak)

            # 4. Checkpoint save (timed separately from training).
            checkpoint_dir = checkpoints_dir / f"adapt_{stage['stage']}"
            save_started = time.perf_counter()
            model.save_pretrained(str(checkpoint_dir))
            tokenizer.save_pretrained(str(checkpoint_dir))
            save_seconds = time.perf_counter() - save_started
            total_ckpt_save += save_seconds
            write_json(
                checkpoint_dir / "stage_provenance.json",
                {
                    "workload": workload,
                    "resource_id": resource_profile.resource_id,
                    "device": str(device),
                    "stage": stage,
                    "model_load_seconds": model_load_seconds,
                    "training_seconds": stats["training_seconds"],
                    "checkpoint_save_seconds": save_seconds,
                    "optimizer_steps": stats["optimizer_steps"],
                    "total_tokens": stats["total_tokens_trained"],
                    "lm_config": lm_config.to_dict(),
                    "software": software_versions(),
                },
            )

            # 5. Boundary evaluation (timed separately).
            segment = stage["stage"]
            state_name = f"after_stage_{segment}"
            eval_started = time.perf_counter()
            per_event: dict[str, dict] = {}
            for event in events:
                evaluation = evaluate_event(
                    model, tokenizer, event.eval_rows, lm_config, log=log_message
                )
                per_event[str(event.event_id)] = evaluation["metrics"]
            eval_seconds = time.perf_counter() - eval_started
            total_eval_seconds += eval_seconds

            accuracy_rows.append(
                [per_event[str(e.event_id)]["exact_match"] for e in events]
            )
            f1_rows.append(
                [per_event[str(e.event_id)]["token_f1"] for e in events]
            )
            ppl_rows.append(
                [per_event[str(e.event_id)]["perplexity"] for e in events]
            )
            states.append(state_name)

            # 6. Record the carried weights for the next transition's
            #    bit-exact restore verification.
            carried_weights = _snapshot_trainable(model)
            previous_checkpoint = checkpoint_dir

            result.stages.append(
                {
                    **stage,
                    "state": state_name,
                    "model_load_seconds": model_load_seconds,
                    "training_seconds": stats["training_seconds"],
                    "checkpoint_save_seconds": save_seconds,
                    "eval_seconds": eval_seconds,
                    "optimizer_steps": stats["optimizer_steps"],
                    "examples": stats["examples_scheduled"] - stats["examples_skipped"],
                    "total_tokens": stats["total_tokens_trained"],
                    "steps_per_second": stats["optimizer_steps"]
                    / stats["training_seconds"]
                    if stats["training_seconds"] > 0
                    else None,
                    "tokens_per_second": stats["total_tokens_trained"]
                    / stats["training_seconds"]
                    if stats["training_seconds"] > 0
                    else None,
                    "average_loss": stats["average_loss"],
                    "peak_memory_gb": peak,
                    "checkpoint_dir": str(checkpoint_dir),
                    "continued_from": (
                        str(previous_checkpoint)
                        if stage["stage"] > 1
                        else None
                    ),
                }
            )
            log_message(
                f"{stage_label}: {stats['optimizer_steps']} steps in "
                f"{stats['training_seconds']:.1f}s "
                f"({stats['optimizer_steps'] / max(stats['training_seconds'], 1e-9):.3f} steps/s)"
            )

            # 7. Transition record: the save side belongs to the PREVIOUS
            #    stage's checkpoint (attached post-loop); the load side is
            #    this stage's model-load step.
            if stage["stage"] > 1:
                result.transitions.append(
                    {
                        "transition": stage["kind"],
                        "from_stage": stage["stage"] - 1,
                        "to_stage": stage["stage"],
                        "checkpoint_save_seconds": None,  # attached post-loop
                        "checkpoint_load_seconds": None,  # attached post-loop
                        "restore_verified": True,
                    }
                )

            del model, base_model, tokenizer
            if device.type == "mps":
                torch.mps.empty_cache()
            elif device.type == "cuda":
                torch.cuda.empty_cache()

        # Attach the save/load overhead to each transition: save = the
        # previous stage's checkpoint save; load = the following stage's
        # model-load step (fresh base load + adapter restore, not separable
        # on this stack, reported in full).
        for index, stage in enumerate(result.stages):
            if index == 0:
                continue
            transition = result.transitions[index - 1]
            transition["checkpoint_save_seconds"] = result.stages[index - 1][
                "checkpoint_save_seconds"
            ]
            transition["checkpoint_load_seconds"] = stage["model_load_seconds"]
            transition["note"] = (
                "load time = fresh base load + adapter restore (not "
                "separable on this stack); reported in full"
            )
        total_ckpt_load = sum(
            transition["checkpoint_load_seconds"] or 0.0
            for transition in result.transitions
        )

        t_final = len(accuracy_rows) - 1
        result.accuracy_matrix = accuracy_rows
        result.f1_matrix = f1_rows
        result.ppl_matrix = ppl_rows
        result.states = states
        result.cl_metrics = {
            "average_accuracy_per_state": {
                f"state_{index}": average_accuracy(accuracy_rows, index)
                for index in range(len(accuracy_rows))
            },
            "forgetting_per_event": {
                f"event_{j}": forgetting(accuracy_rows, j)
                for j in range(1, t_final + 1)
            },
            "average_forgetting": average_forgetting(accuracy_rows),
            "backward_transfer_conventional": backward_transfer_conventional(
                accuracy_rows
            ),
        }
        result.totals = {
            "optimizer_steps": total_steps,
            "training_examples": total_examples,
            "training_tokens": total_tokens,
            "training_seconds": total_train_seconds,
            "model_load_seconds": total_model_load,
            "checkpoint_save_seconds": total_ckpt_save,
            "checkpoint_load_seconds": total_ckpt_load,
            "eval_seconds": total_eval_seconds,
            "total_completion_seconds": (
                total_train_seconds
                + total_model_load
                + total_ckpt_save
                + total_eval_seconds
            ),
            "steps_per_second": total_steps / total_train_seconds
            if total_train_seconds > 0
            else None,
            "tokens_per_second": total_tokens / total_train_seconds
            if total_train_seconds > 0
            else None,
            "peak_memory_gb": peak_memory,
            "n_transitions": len(result.transitions),
        }
        result.status = "complete"
    except Exception as error:  # noqa: BLE001
        import traceback

        result.status = "failed"
        result.error = f"{type(error).__name__}: {error}"
        log_message(f"BENCHMARK FAILED: {result.error}")
        log_message(traceback.format_exc())
    return result


def load_baseline_rows() -> tuple[list[float], list[float], list[float]]:
    baseline = json.loads(DEFAULT_BASELINE.read_text(encoding="utf-8"))
    per_event = baseline["per_event"]
    test_events = sorted(int(key) for key in per_event)
    return (
        [per_event[str(e)]["exact_match"] for e in test_events],
        [per_event[str(e)]["token_f1"] for e in test_events],
        [per_event[str(e)]["perplexity"] for e in test_events],
    )


def prepare_workload_config(
    workload: str, seed: int = 42, device_request: str | None = None
) -> LMRuntimeConfig:
    """LM runtime config for a workload (E2 semantics; rank from workload)."""
    from cycleft.config import load_config, validate_stream_config
    from cycleft.e1.pipeline import resolve_lm_config

    raw_config = load_config(REPO_ROOT / "configs" / "e1_stream.yaml")
    validate_stream_config(raw_config)
    lm_config, _limit, _resolved = resolve_lm_config(
        raw_config, mode="full", train_examples=None, seed=seed, device=device_request
    )
    spec = WORKLOADS[workload]
    lm_config.lora_r = spec["lora_rank"]
    return lm_config
