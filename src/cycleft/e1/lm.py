"""Model runtime for E1: device handling, prompting, LoRA training, evaluation.

Design notes:
* The base model is loaded once per pipeline stage and released before the
  next stage, so only one model copy is ever resident in memory.
* Event k > 1 training loads its adapter from the event k-1 checkpoint
  directory via ``PeftModel.from_pretrained(..., is_trainable=True)``,
  which makes sequential continuation explicit and auditable on disk.
* Prompts use the tokenizer's chat template with a frozen
  ``date_string`` so prompts do not silently change between runs.
* The frozen base model runs in float16 while LoRA adapter weights are
  cast to float32: fp16 AdamW second-moment underflow destabilized
  generation in earlier runs (documented in
  ``results/e1/pilot_20261113_aborted_lr1e4``). PEFT casts the LoRA
  input path per-adapter dtype, so this mixed precision is exact for
  the adapter and adds no base-model memory.
"""

from __future__ import annotations

import math
import random
import resource  # noqa: F401  (used for RSS reporting on macOS/Linux)
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from cycleft.e1.metrics import extract_answer

DEFAULT_SYSTEM_PROMPT = (
    "You are a precise question answering assistant. "
    "Answer with the exact answer only, no explanation."
)

# The chat template stamps a "Today Date" line; freezing it keeps prompts
# (and therefore results) reproducible across calendar days.
FROZEN_CHAT_DATE = "01 Jan 2026"


@dataclass
class LMRuntimeConfig:
    """Runtime configuration shared by training and evaluation."""

    model_id: str
    max_length: int = 128
    max_new_tokens: int = 24
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    chat_date_string: str = FROZEN_CHAT_DATE
    dtype: str = "float16"
    lora_precision: str = "float32"
    device: str | None = None  # None -> auto (MPS preferred)
    seed: int = 42
    memory_guard_fraction: float = 0.90
    learning_rate: float = 2e-5
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    epochs: int = 1
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    lora_layers_to_transform: list[int] | None = None  # E6: selective layers

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, payload: dict) -> "LMRuntimeConfig":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in payload.items() if k in known})


def resolve_device(requested: str | None = None) -> torch.device:
    """Resolve the compute device.

    Resolution rules:

    * ``requested="cpu"``  -> CPU (explicit opt-in, always allowed).
    * ``requested="cuda"`` -> CUDA, or an explicit RuntimeError when no
      CUDA device is available (never a silent downgrade).
    * ``requested="mps"``  -> Apple MPS, or an explicit RuntimeError when
      unavailable (never a silent downgrade).
    * ``requested=None/"auto"`` -> automatic selection preferring CUDA,
      then MPS, then CPU (HPC-first ordering; CPU is the last resort of
      the *automatic* path only — an explicitly requested accelerator is
      never silently downgraded).

    Results must never be reported as an accelerator run if they ran on
    CPU: the resolved device is recorded in every manifest.
    """
    requested_normalized = (requested or "auto").lower()
    if requested_normalized == "cpu":
        return torch.device("cpu")
    if requested_normalized == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError(
            "CUDA was requested but is not available "
            f"(torch.cuda.is_available()=False, torch={torch.__version__}). "
            "No silent fallback: pass device='cpu' explicitly or fix CUDA."
        )
    if requested_normalized == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        raise RuntimeError(
            "Apple MPS is not available and automatic CPU fallback is "
            "disabled. Pass device='cpu' explicitly to run on CPU."
        )
    if requested_normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    raise ValueError(f"Unsupported device requested: {requested!r}")


def gpu_metadata() -> dict:
    """GPU/CUDA metadata for run manifests (all None when CUDA is absent).

    Distinguishes *measured* fields (name, capabilities) from fields that
    require an active CUDA context.  Never raises.
    """
    info: dict[str, Any] = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime_version": torch.version.cuda,
        "torch_version": torch.__version__,
        "device_name": None,
        "device_capability": None,
        "total_memory_bytes": None,
        "free_memory_bytes": None,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    if torch.cuda.is_available():
        try:
            index = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(index)
            info["device_name"] = props.name
            info["device_capability"] = f"{props.major}.{props.minor}"
            info["total_memory_bytes"] = props.total_memory
            free, _total = torch.cuda.mem_get_info(index)
            info["free_memory_bytes"] = free
        except Exception as error:  # pragma: no cover - metadata only
            info["metadata_error"] = str(error)
    return info


def release_device_memory(device: torch.device | str | None = None) -> None:
    """Free accelerator caches between stages (MPS + CUDA aware)."""
    if isinstance(device, str):
        device = torch.device(device)
    if torch.cuda.is_available() and (device is None or device.type == "cuda"):
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    if torch.backends.mps.is_available() and (device is None or device.type == "mps"):
        torch.mps.empty_cache()


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and Torch RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def memory_status() -> dict:
    """Snapshot of accelerator allocator usage and process RSS."""
    status: dict = {"rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9}
    if torch.backends.mps.is_available():
        status["mps_allocated_gb"] = torch.mps.current_allocated_memory() / 1e9
        if hasattr(torch.mps, "recommended_max_memory"):
            status["mps_recommended_gb"] = torch.mps.recommended_max_memory() / 1e9
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        status["cuda_allocated_gb"] = allocated / 1e9
        status["cuda_reserved_gb"] = reserved / 1e9
        try:
            free, total = torch.cuda.mem_get_info()
            status["cuda_free_gb"] = free / 1e9
            status["cuda_total_gb"] = total / 1e9
        except Exception:  # pragma: no cover - informational only
            pass
    return status


def check_memory_guard(config: LMRuntimeConfig, stage: str) -> None:
    """Stop safely (with a clear error) when MPS memory pressure is excessive."""
    if not torch.backends.mps.is_available():
        return
    allocated = torch.mps.current_allocated_memory()
    if not hasattr(torch.mps, "recommended_max_memory"):
        return
    recommended = torch.mps.recommended_max_memory()
    if allocated > config.memory_guard_fraction * recommended:
        raise RuntimeError(
            f"MPS memory pressure at stage '{stage}': "
            f"{allocated / 1e9:.2f} GB allocated exceeds "
            f"{config.memory_guard_fraction:.0%} of the recommended "
            f"{recommended / 1e9:.2f} GB. Reduce train examples, max_length, "
            "or close other GPU-using processes, then retry."
        )


def load_tokenizer(model_id: str):
    """Load tokenizer, preferring the local HF cache (no credential use)."""
    try:
        return AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    except OSError:
        return AutoTokenizer.from_pretrained(model_id)


def load_base_model(config: LMRuntimeConfig):
    """Load the frozen base causal LM in float16 with eager attention."""
    dtype = getattr(torch, config.dtype)
    load_kwargs = dict(
        dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    try:
        model = AutoModelForCausalLM.from_pretrained(
            config.model_id, local_files_only=True, **load_kwargs
        )
    except OSError:
        model = AutoModelForCausalLM.from_pretrained(config.model_id, **load_kwargs)
    model.config.use_cache = False
    return model


def build_messages(question: str, config: LMRuntimeConfig) -> list[dict]:
    return [
        {"role": "system", "content": config.system_prompt},
        {"role": "user", "content": str(question)},
    ]


def render_prompt(tokenizer, question: str, config: LMRuntimeConfig) -> str:
    """Render the generation prompt (chat template, frozen date)."""
    return tokenizer.apply_chat_template(
        build_messages(question, config),
        tokenize=False,
        add_generation_prompt=True,
        date_string=config.chat_date_string,
    )


def encode_training_example(
    tokenizer,
    question: str,
    answer: str,
    config: LMRuntimeConfig,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Build prompt-masked causal LM inputs; ``None`` if truncation ate the answer.

    The loss is computed only on answer tokens (labels = -100 elsewhere),
    matching the evaluation-time loss computation.
    """
    prompt_text = render_prompt(tokenizer, question, config)
    full_text = tokenizer.apply_chat_template(
        build_messages(question, config)
        + [{"role": "assistant", "content": str(answer)}],
        tokenize=False,
        add_generation_prompt=False,
        date_string=config.chat_date_string,
    )

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    encoded = tokenizer(
        full_text,
        max_length=config.max_length,
        truncation=True,
        add_special_tokens=False,
    )

    input_ids = torch.tensor(encoded["input_ids"], dtype=torch.long)
    labels = input_ids.clone()
    labels[: min(len(prompt_ids), len(labels))] = -100

    if not (labels != -100).any():
        return None
    return input_ids, labels


def make_lora_config(config: LMRuntimeConfig) -> LoraConfig:
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=list(config.lora_target_modules),
        bias="none",
        layers_to_transform=getattr(config, "lora_layers_to_transform", None),
    )


def cast_lora_to_fp32(model) -> None:
    """Cast trainable LoRA weights to float32 (base stays float16).

    float16 AdamW moment estimates underflow on this workload and
    degenerate generation quality; float32 adapter weights fix the
    optimizer numerics at negligible memory cost. PEFT's LoRA forward
    casts the input to the adapter dtype and casts the result back, so
    mixed precision is handled correctly.
    """
    for name, parameter in model.named_parameters():
        if "lora_" in name and parameter.dtype != torch.float32:
            parameter.data = parameter.data.to(torch.float32)


def _prepare_peft_model(model, config: LMRuntimeConfig, device: torch.device):
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    cast_lora_to_fp32(model)
    model.to(device)
    return model


def attach_lora(base_model, config: LMRuntimeConfig, device: torch.device):
    """Attach a fresh LoRA adapter (event 1 only)."""
    model = get_peft_model(base_model, make_lora_config(config))
    return _prepare_peft_model(model, config, device)


def load_adapter_for_training(
    base_model,
    adapter_dir: str | Path,
    config: LMRuntimeConfig,
    device: torch.device,
):
    """Continue from a previously saved adapter (events 2+).

    Loads the adapter weights from ``adapter_dir`` so training continues
    the same LoRA rather than initializing a fresh one.
    """
    model = PeftModel.from_pretrained(
        base_model, str(adapter_dir), is_trainable=True
    )
    return _prepare_peft_model(model, config, device)


def train_event(
    model,
    tokenizer,
    train_rows: Sequence[dict],
    config: LMRuntimeConfig,
    log: Callable[[str], None] | None = None,
    on_step: Callable[[int, float], None] | None = None,
) -> dict:
    """Train one continual-learning event; return training statistics.

    The model is trained in place. Every ``gradient_accumulation_steps``
    micro-batches (batch size 1 each) one optimizer step is taken.
    ``on_step(optimizer_steps, step_loss)`` fires after each optimizer
    step (used by E2 for resource sampling).
    """
    if config.batch_size != 1:
        raise ValueError("E1 uses batch_size=1 (memory-efficient setting).")

    emit = log or (lambda message: None)
    device = next(model.parameters()).device
    set_seed(config.seed)

    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise RuntimeError("No trainable parameters; LoRA attachment failed.")

    optimizer = torch.optim.AdamW(trainable_parameters, lr=config.learning_rate)

    # Snapshot a few adapter tensors so we can prove training updates them.
    update_check_indices = list(range(min(4, len(trainable_parameters))))
    update_check_before = [
        trainable_parameters[i].detach().clone() for i in update_check_indices
    ]

    ga = max(1, config.gradient_accumulation_steps)
    scheduled = [
        row
        for _ in range(max(1, config.epochs))
        for row in train_rows
    ]
    expected_steps = math.ceil(len(scheduled) / ga)

    per_step_losses: list[float] = []
    total_loss = 0.0
    successful_steps = 0
    skipped = 0
    answer_tokens = 0
    total_tokens = 0
    started = time.perf_counter()
    optimizer_steps = 0

    model.train()
    micro_step_loss = 0.0
    micro_step_count = 0

    for index, row in enumerate(scheduled):
        example = encode_training_example(
            tokenizer, row["question"], row["object"], config
        )
        if example is None:
            skipped += 1
            emit(f"step {index + 1}/{len(scheduled)}: skipped (answer truncated away)")
            continue

        input_ids, labels = example
        answer_tokens += int((labels != -100).sum())
        total_tokens += int(labels.numel())

        input_ids = input_ids.unsqueeze(0).to(device)
        labels = labels.unsqueeze(0).to(device)
        attention_mask = torch.ones_like(input_ids)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
        )
        loss = outputs.loss

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite training loss at example {index}: {loss.item()}"
            )

        (loss / ga).backward()
        micro_step_loss += loss.item()
        micro_step_count += 1

        if micro_step_count == ga or index == len(scheduled) - 1:
            torch.nn.utils.clip_grad_norm_(trainable_parameters, config.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1

            step_loss = micro_step_loss / micro_step_count
            per_step_losses.append(step_loss)
            total_loss += step_loss
            successful_steps += 1
            emit(
                f"step {successful_steps}/{expected_steps} "
                f"loss={step_loss:.4f}"
            )
            micro_step_loss = 0.0
            micro_step_count = 0
            if on_step is not None:
                on_step(optimizer_steps, step_loss)

        check_memory_guard(config, "training")
        del outputs, loss, input_ids, labels, attention_mask

    if successful_steps == 0:
        raise RuntimeError("No training examples were successfully encoded/trained.")

    for index_before, index_param in enumerate(update_check_indices):
        if torch.equal(
            update_check_before[index_before],
            trainable_parameters[index_param].detach(),
        ):
            raise RuntimeError(
                "Trainable parameters did not change during training; "
                "the optimizer is not updating the LoRA adapter."
            )

    elapsed = time.perf_counter() - started
    trainable_count = sum(p.numel() for p in trainable_parameters)
    total_count = sum(p.numel() for p in model.parameters())

    return {
        "examples_scheduled": len(scheduled),
        "examples_skipped": skipped,
        "optimizer_steps": optimizer_steps,
        "losses_per_step": per_step_losses,
        "average_loss": total_loss / successful_steps,
        "final_loss": per_step_losses[-1],
        "training_seconds": elapsed,
        "trainable_parameters": trainable_count,
        "total_parameters": total_count,
        "trainable_percentage": 100.0 * trainable_count / total_count,
        "answer_tokens_trained": answer_tokens,
        "total_tokens_trained": total_tokens,
        "memory_after": memory_status(),
    }


@torch.no_grad()
def generate_answer(
    model,
    tokenizer,
    question: str,
    config: LMRuntimeConfig,
) -> str:
    """Deterministic greedy generation for one question."""
    device = next(model.parameters()).device
    encoded = tokenizer(
        render_prompt(tokenizer, question, config),
        return_tensors="pt",
        add_special_tokens=False,
    ).to(device)
    prompt_ids = {key: value for key, value in encoded.items()}

    output = model.generate(
        **prompt_ids,
        max_new_tokens=config.max_new_tokens,
        do_sample=False,
        num_beams=1,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_tokens = output[0][prompt_ids["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


@torch.no_grad()
def eval_set_loss(
    model,
    tokenizer,
    eval_rows: Sequence[dict],
    config: LMRuntimeConfig,
    log: Callable[[str], None] | None = None,
) -> dict:
    """Teacher-forced answer-token loss (macro-averaged) on held-out rows.

    Uses the same prompt masking as training, so the loss covers exactly
    the answer tokens; perplexity is exp(mean loss) over that token set.
    """
    emit = log or (lambda message: None)
    device = next(model.parameters()).device

    losses: list[float] = []
    token_counts: list[int] = []

    was_training = model.training
    model.eval()

    for index, row in enumerate(eval_rows):
        example = encode_training_example(
            tokenizer, row["question"], row["object"], config
        )
        if example is None:
            emit(f"eval loss: skipped row {index} (answer truncated away)")
            continue

        input_ids, labels = example
        input_ids = input_ids.unsqueeze(0).to(device)
        labels = labels.unsqueeze(0).to(device)
        attention_mask = torch.ones_like(input_ids)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
        )
        loss_value = float(outputs.loss)
        if math.isfinite(loss_value):
            losses.append(loss_value)
            token_counts.append(int((labels != -100).sum()))

        del outputs, input_ids, labels, attention_mask

    if was_training:
        model.train()

    if not losses:
        raise RuntimeError("Eval loss could not be computed on any example.")

    mean_loss = sum(losses) / len(losses)
    return {
        "eval_loss": mean_loss,
        "perplexity": math.exp(min(mean_loss, 20.0)),
        "examples_scored": len(losses),
        "mean_answer_tokens": sum(token_counts) / len(token_counts),
    }


@torch.no_grad()
def evaluate_event(
    model,
    tokenizer,
    eval_rows: Sequence[dict],
    config: LMRuntimeConfig,
    log: Callable[[str], None] | None = None,
) -> dict:
    """Greedy-decode all eval rows and compute EM / token-F1 / loss / ppl.

    Returns per-event metrics plus per-example prediction records for
    later inspection.
    """
    from cycleft.e1.metrics import exact_match, exact_match_multi, token_f1, token_f1_multi

    emit = log or (lambda message: None)
    started = time.perf_counter()

    # Generation and scoring must happen in eval mode: leaving LoRA
    # dropout active during inference corrupts predictions (this exact
    # bug produced the degenerate outputs in the two aborted runs).
    was_training = model.training
    model.eval()

    records: list[dict] = []
    for index, row in enumerate(eval_rows):
        question = str(row["question"])
        reference = str(row["object"])
        valid_answers = row.get("valid_answers")
        raw = generate_answer(model, tokenizer, question, config)
        prediction = extract_answer(raw)

        if valid_answers:
            # Multi-reference scoring (WikiFactDiff rows with several
            # simultaneously valid answers).  ``object`` remains the
            # primary answer for provenance; any valid answer counts.
            em = exact_match_multi(prediction, valid_answers)
            f1 = token_f1_multi(prediction, valid_answers)
        else:
            em = exact_match(prediction, reference)
            f1 = token_f1(prediction, reference)
        records.append(
            {
                "example_index": index,
                "question": question,
                "reference": reference,
                "raw_prediction": raw,
                "prediction": prediction,
                "exact_match": em,
                "token_f1": f1,
                **(
                    {"valid_answers": list(valid_answers)}
                    if valid_answers
                    else {}
                ),
            }
        )
        if (index + 1) % 10 == 0:
            emit(f"  evaluated {index + 1}/{len(eval_rows)}")

    loss_stats = eval_set_loss(model, tokenizer, eval_rows, config, log=emit)

    if was_training:
        model.train()

    n = len(records)
    metrics = {
        "examples": n,
        "exact_match": sum(r["exact_match"] for r in records) / n,
        "token_f1": sum(r["token_f1"] for r in records) / n,
        **loss_stats,
        "generation_seconds": time.perf_counter() - started,
    }
    return {"metrics": metrics, "records": records}
