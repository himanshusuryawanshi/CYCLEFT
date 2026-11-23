"""E3 runner: trigger strategies controlling real LoRA adaptation on the stream.

Operational semantics (documented in the run README; revised after the
pilot `pilot_20261116`, whose findings are recorded there):

* **Bootstrap.** Every adapting strategy starts from a SHARED bootstrap
  adaptation: a fresh rank-4 LoRA trained on the first ``adapt_budget``
  stream examples (positions 0..124 for budget 125), performed once per
  run. This mirrors deployment (a system is adapted on initial data
  before serving) and gives the Regime-B error signal a meaningful
  operating point: the raw pretrained model is uniformly ignorant of every
  event's answers, so its prequential error/NLL is flat and cannot reveal
  drift (measured in the pilot). All strategies continue the SAME adapter
  lineage (bootstrap checkpoint -> strategy adaptations), and no strategy
  starts from a different model state than another.
* **Regime B (delayed feedback).** Signal = prequential 0-1 prediction
  error: the model generates a prediction FIRST, the answer arrives
  afterwards (feedback delay 0 windows in this simulation, recorded), and
  the exact-match error is fed to the detectors. Errors at positions
  0..124 come from the shared base-model prediction cache; positions
  125..299 from the shared bootstrap-model cache; from position
  ``warmup_examples`` (300) each strategy computes its own online errors,
  because its model may have diverged after its first own adaptation. The
  answer of example t never influences the prediction made for example t.
* **Regime A (unlabeled).** Signal = L2-normalised mean-pooled base-model
  embeddings of the question text, computed once per run; answers are
  structurally inaccessible to these detectors.
* **Warm-up / calibration.** No trigger may fire before position
  ``warmup_examples`` (default 300 = 6 windows). The loss-threshold rule
  is calibrated on the shared bootstrap-model errors of calibration
  windows 3-5 (positions 150-299). MMD's reference is the question
  embeddings of positions 0-149 with a permutation-calibrated threshold.
  No threshold uses final-evaluation data.
* **Minimum adaptation gap.** Trigger strategies require
  ``min_gap_examples`` since the previous adaptation (bootstrap included).
* **Evaluation.** Reuses the exact E1/E2 protocol (``evaluate_event``) on
  all three held-out test sets after each stream segment; matrix row 0
  reuses the verified E1 baseline (identical deterministic protocol).
* **Reference policies.** ``always_adapt`` = shared bootstrap + oracle
  adaptation with the first ``adapt_budget`` examples of each later event
  (positions 624/1024 for budget 125; the E2 ``rank4_b125`` continuation).
  ``no_adapt`` never adapts and reuses the deterministic baseline
  evaluations for all states.
"""

from __future__ import annotations

import copy
import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cycleft.e1.data import PreparedEvent
from cycleft.e1.lm import (
    LMRuntimeConfig,
    attach_lora,
    evaluate_event,
    generate_answer,
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
    backward_transfer,
    backward_transfer_conventional,
    exact_match,
    extract_answer,
    forgetting,
)
from cycleft.e1.pipeline import REPO_ROOT, software_versions, write_json
from cycleft.e2.resources import ResourceTracker
from cycleft.e3.calibration import (
    calibrate_loss_threshold,
    calibrate_mmd_threshold,
)
from cycleft.e3.detectors import (
    AdwinTrigger,
    CombinedTrigger,
    FeedbackSignal,
    LossThresholdTrigger,
    MMDTrigger,
    PeriodicTrigger,
    UnlabeledSignal,
    mmd2_unbiased,
)
from cycleft.e3.metrics import (
    adaptation_efficiency,
    detection_accounting,
    false_positive_rate_per_window,
)
from cycleft.e3.stream import (
    EXAMPLES_PER_EVENT,
    build_stream,
    dataset_audit,
    load_slice_order_check,
    summarise_audit,
)

RESULTS_ROOT = REPO_ROOT / "results" / "e3"
CHECKPOINTS_ROOT = REPO_ROOT / "checkpoints" / "e3"
N_SEGMENTS = 3

STRATEGY_REGIMES = {
    "no_adapt": "reference (never adapts)",
    "always_adapt": "reference (oracle event boundaries)",
    "periodic_k3": "scheduled (every 3 windows = 150 examples)",
    "periodic_k5": "scheduled (every 5 windows = 250 examples)",
    "periodic_k10": "scheduled (every 10 windows = 500 examples)",
    "loss_threshold": "B_delayed_feedback (prequential window-mean error)",
    "adwin": "B_delayed_feedback (per-example prequential error)",
    "mmd": "A_unlabeled (question-embedding MMD^2)",
    "combined": "A+B_combined (OR of ADWIN and MMD)",
}
ERROR_STREAM_STRATEGIES = {"loss_threshold", "adwin", "combined"}
MMD_STRATEGIES = {"mmd", "combined"}
PILOT_STRATEGIES = ("periodic_k5", "loss_threshold", "adwin", "mmd", "combined")
FULL_STRATEGIES = (
    "no_adapt",
    "always_adapt",
    "periodic_k3",
    "periodic_k5",
    "periodic_k10",
    "loss_threshold",
    "adwin",
    "mmd",
    "combined",
)


def _adapter_fingerprint(model) -> str:
    """sha256 over trainable-parameter bytes (pre-training state snapshot).

    Records the exact initialization/restore state so that RNG-order
    questions (see FINAL_VERIFICATION.md Check 1) are decidable from
    provenance alone.
    """
    import hashlib

    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        if parameter.requires_grad:
            digest.update(name.encode())
            digest.update(parameter.detach().to("cpu").contiguous().numpy().tobytes())
    return digest.hexdigest()


INITIALIZATION_ORDER_E3 = (
    "load_base_model -> model.to(device) -> attach_lora (LoRA init on "
    "device) -> set_seed -> train_event (re-seeds internally)"
)


@dataclass
class E3Config:
    """E3 run configuration (mirrored into detector_config.yaml)."""

    run_id: str
    seed: int = 42
    window_size: int = 50
    n_calibration_windows: int = 3  # calibration windows 3-5 (bootstrap model)
    warmup_windows: int = 6  # no trigger before position 300
    calibration_start_window: int = 3
    tolerance_examples: int = 100  # detection tolerance AFTER a boundary
    # Adaptation action: E2 Pareto cell rank4_b125.
    adapt_rank: int = 4
    adapt_budget: int = 125  # max optimizer steps per adaptation
    # Detector hyperparameters. Fixed a priori for the full run; MMD alpha
    # was raised from 0.05 to 0.10 after the PILOT diagnosis only (stream
    # signals + calibration null), never using final-evaluation data.
    loss_z: float = 2.0
    loss_refractory_windows: int = 3
    adwin_delta: float = 0.002
    adwin_grace_period: int = 100
    adwin_clock: int = 32
    adwin_min_window_length: int = 32
    adwin_cooldown_windows: int = 2
    mmd_alpha: float = 0.10
    mmd_n_permutations: int = 500
    mmd_refractory_windows: int = 5
    combined_min_gap_windows: int = 2
    min_gap_examples: int = 2 * 50
    periodic_ks: tuple[int, ...] = (3, 5, 10)
    pilot: bool = False

    @property
    def warmup_examples(self) -> int:
        return self.warmup_windows * self.window_size

    @property
    def calibration_window_indices(self) -> tuple[int, ...]:
        return tuple(
            range(
                self.calibration_start_window,
                self.calibration_start_window + self.n_calibration_windows,
            )
        )

    def with_pilot_budget(self, budget: int) -> "E3Config":
        config = copy.deepcopy(self)
        config.pilot = True
        config.adapt_budget = budget
        return config

    def as_dict(self) -> dict:
        payload = dict(self.__dict__)
        payload["periodic_ks"] = list(self.periodic_ks)
        payload["calibration_window_indices"] = list(
            self.calibration_window_indices
        )
        return payload


class ModelController:
    """Owns the single resident model; triggers control its adaptation.

    ``initial_adapter`` (the shared bootstrap checkpoint) is loaded at
    startup when provided, so the strategy's first own adaptation
    continues the bootstrap adapter — the same lineage for every strategy.
    """

    def __init__(
        self,
        lm_config: LMRuntimeConfig,
        device: torch.device,
        checkpoints_dir: Path,
        tracker: ResourceTracker,
        log: Any,
        initial_adapter: Path | None = None,
    ):
        self.lm_config = lm_config
        self.device = device
        self.checkpoints_dir = checkpoints_dir
        self.tracker = tracker
        self.log = log
        self.initial_adapter = initial_adapter
        self.n_adaptations = 0
        self._model = None
        self._tokenizer = None
        self.adapter_attached = initial_adapter is not None
        self.last_checkpoint = initial_adapter
        self.adaptation_records: list[dict] = []
        self.training_seconds = 0.0
        self.training_steps = 0
        self.training_examples = 0
        self.training_tokens = 0
        self.stream_seconds = 0.0
        self.eval_seconds = 0.0
        self.prediction_log: list[dict] = []

    # -- model lifecycle -------------------------------------------------------

    def _ensure_model(self):
        if self._model is None:
            self.log("loading pretrained base model (single resident copy)")
            self._tokenizer = load_tokenizer(self.lm_config.model_id)
            model = load_base_model(self.lm_config)
            model.to(self.device)
            model.eval()
            if self.initial_adapter is not None:
                self.log(f"continuing from shared bootstrap adapter {self.initial_adapter.name}")
                model = load_adapter_for_training(
                    model, self.initial_adapter, self.lm_config, self.device
                )
            self._model = model
        return self._model, self._tokenizer

    def release(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
            self._tokenizer = None
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

    # -- regime B signal: prequential prediction error --------------------------

    @torch.no_grad()
    def prequential_error(self, row: dict, position: int) -> float:
        """Greedy prediction FIRST, then the 0-1 error once the answer arrives.

        The generated prediction never sees the answer; the answer enters
        only the post-feedback error computation (exact match). Feedback
        delay is 0 windows in this simulation (recorded in the config).
        """
        model, tokenizer = self._ensure_model()
        started = time.perf_counter()
        was_training = model.training
        model.eval()
        raw = generate_answer(model, tokenizer, str(row["question"]), self.lm_config)
        if was_training:
            model.train()
        prediction = extract_answer(raw)
        error = 1.0 - exact_match(prediction, str(row["object"]))
        self.stream_seconds += time.perf_counter() - started
        self.prediction_log.append(
            {
                "position": position,
                "prediction": prediction,
                "reference": str(row["object"]),
                "error": error,
            }
        )
        self.tracker.sample()
        return error

    # -- adaptation ---------------------------------------------------------------

    def adapt(self, rows: list[dict], trigger_info: dict) -> dict:
        """Adapt on the buffered rows, continuing the current adapter."""
        model, tokenizer = self._ensure_model()
        if not self.adapter_attached:
            self.log("first adaptation: attaching fresh LoRA adapter")
            model = attach_lora(model, self.lm_config, self.device)
            self._model = model
            self.adapter_attached = True
        else:
            self.log(
                f"adaptation {self.n_adaptations + 1}: continuing the same adapter"
            )

        set_seed(self.lm_config.seed)
        self.n_adaptations += 1
        pre_training_fingerprint = _adapter_fingerprint(model)
        stats = train_event(
            model, tokenizer, rows, self.lm_config, log=self.log,
            on_step=lambda _s, _l: self.tracker.sample(),
        )
        self.training_seconds += stats["training_seconds"]
        self.training_steps += stats["optimizer_steps"]
        self.training_examples += stats["examples_scheduled"] - stats["examples_skipped"]
        self.training_tokens += stats["total_tokens_trained"]

        checkpoint_dir = self.checkpoints_dir / f"adapt_{self.n_adaptations}"
        model.save_pretrained(str(checkpoint_dir))
        tokenizer.save_pretrained(str(checkpoint_dir))
        continued_from = self.last_checkpoint
        write_json(
            checkpoint_dir / "adaptation_provenance.json",
            {
                "strategy": self.checkpoints_dir.parent.name,
                "adaptation_index": self.n_adaptations,
                "continued_from": (
                    str(continued_from) if continued_from else None
                ),
                "same_adapter_continued": continued_from is not None,
                "trigger": trigger_info,
                "buffer_size": len(rows),
                "buffer_positions": trigger_info.get("buffer_positions"),
                "init_device": str(self.device),
                "initialization_order": INITIALIZATION_ORDER_E3,
                "init_fingerprint_pre_training": pre_training_fingerprint,
                "lm_config": self.lm_config.to_dict(),
                "training_stats": {
                    key: value
                    for key, value in stats.items()
                    if key != "losses_per_step"
                },
                "software": software_versions(),
            },
        )
        checkpoint_bytes = sum(
            f.stat().st_size for f in checkpoint_dir.iterdir() if f.is_file()
        )
        self.last_checkpoint = checkpoint_dir
        record = {
            "adaptation_index": self.n_adaptations,
            **trigger_info,
            "buffer_size": len(rows),
            "optimizer_steps": stats["optimizer_steps"],
            "examples_processed": stats["examples_scheduled"] - stats["examples_skipped"],
            "total_tokens": stats["total_tokens_trained"],
            "training_seconds": stats["training_seconds"],
            "average_loss": stats["average_loss"],
            "checkpoint_dir": str(checkpoint_dir),
            "checkpoint_size_bytes": checkpoint_bytes,
            "continued_from": str(continued_from) if continued_from else None,
            "memory": self.tracker.snapshot(),
        }
        self.adaptation_records.append(record)
        return record

    # -- evaluation ------------------------------------------------------------------

    def evaluate_state(
        self, events: list[PreparedEvent], state_name: str
    ) -> tuple[dict[str, dict], list[dict]]:
        model, tokenizer = self._ensure_model()
        started = time.perf_counter()
        per_event: dict[str, dict] = {}
        records: list[dict] = []
        for event in events:
            evaluation = evaluate_event(
                model, tokenizer, event.eval_rows, self.lm_config, log=self.log
            )
            per_event[str(event.event_id)] = evaluation["metrics"]
            for record in evaluation["records"]:
                records.append(
                    {"state": state_name, "test_event": event.event_id, **record}
                )
        self.eval_seconds += time.perf_counter() - started
        self.tracker.sample()
        return per_event, records


def compute_question_embeddings(
    lm_config: LMRuntimeConfig,
    stream_rows: list[dict],
    device: torch.device,
    log: Any,
    batch_size: int = 8,
) -> np.ndarray:
    """L2-normalised mean-pooled last-hidden-state embeddings of questions.

    Regime-A detectors receive ONLY these question-derived vectors; the
    answer column is never read by this function.
    """
    from transformers import AutoTokenizer

    log(f"computing {len(stream_rows)} question embeddings (base model)")
    tokenizer = AutoTokenizer.from_pretrained(
        lm_config.model_id, local_files_only=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_base_model(lm_config)
    model.to(device)
    model.eval()

    all_embeddings: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(stream_rows), batch_size):
            batch = stream_rows[start : start + batch_size]
            encoded = tokenizer(
                [str(row["question"]) for row in batch],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=lm_config.max_length,
                add_special_tokens=False,
            ).to(device)
            hidden = model(
                **encoded, output_hidden_states=True, use_cache=False
            ).hidden_states[-1]
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            all_embeddings.append(pooled.to("cpu").to(torch.float32).numpy())
            del hidden, pooled, encoded
    del model
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    embeddings = np.concatenate(all_embeddings, axis=0)
    log(f"embeddings ready: {embeddings.shape}")
    return embeddings


@dataclass
class StrategyResult:
    """Everything measured for one trigger strategy."""

    strategy: str
    status: str = "running"
    regime: str = ""
    adaptation_positions: list[int] = field(default_factory=list)
    n_adaptations: int = 0
    states: list[str] = field(default_factory=list)
    accuracy_matrix: list[list[float]] = field(default_factory=list)
    f1_matrix: list[list[float]] = field(default_factory=list)
    ppl_matrix: list[list[float]] = field(default_factory=list)
    eval_points: dict[str, dict] = field(default_factory=dict)
    adaptation_records: list[dict] = field(default_factory=list)
    detector_log: list[dict] = field(default_factory=list)
    window_log: list[dict] = field(default_factory=list)
    error_curve: list[tuple[int, float]] = field(default_factory=list)
    mmd_curve: list[tuple[int, float]] = field(default_factory=list)
    prediction_log: list[dict] = field(default_factory=list)
    resources: dict = field(default_factory=dict)
    detection: dict = field(default_factory=dict)
    cl_metrics: dict = field(default_factory=dict)
    efficiency: dict = field(default_factory=dict)
    error: str | None = None


def build_triggers(
    strategy: str,
    config: E3Config,
    calibration_errors: list[list[float]],
    calibration_embeddings: np.ndarray,
    n_windows: int,
) -> dict:
    """Instantiate the detector stack for one strategy.

    Calibration inputs are shared across strategies (same bootstrap-model
    error stream, same seed, never final-evaluation data).
    """
    triggers: dict = {}
    if strategy.startswith("periodic"):
        k = int(strategy.rsplit("_k", 1)[1])
        triggers["periodic"] = PeriodicTrigger(k, n_windows)
    elif strategy == "loss_threshold":
        calibration = calibrate_loss_threshold(calibration_errors, z=config.loss_z)
        triggers["calibration"] = calibration
        triggers["loss_threshold"] = LossThresholdTrigger(
            [float(np.mean(errors)) for errors in calibration_errors],
            z=config.loss_z,
            refractory_windows=config.loss_refractory_windows,
        )
    elif strategy == "adwin":
        triggers["adwin"] = AdwinTrigger(
            delta=config.adwin_delta,
            grace_period=config.adwin_grace_period,
            clock=config.adwin_clock,
            min_window_length=config.adwin_min_window_length,
            cooldown_windows=config.adwin_cooldown_windows,
        )
    elif strategy == "mmd":
        calibration = calibrate_mmd_threshold(
            calibration_embeddings,
            window_size=config.window_size,
            alpha=config.mmd_alpha,
            n_permutations=config.mmd_n_permutations,
            seed=config.seed,
        )
        triggers["calibration"] = calibration
        triggers["mmd"] = MMDTrigger(
            calibration_embeddings,
            threshold=calibration.threshold,
            bandwidth=calibration.bandwidth,
            refractory_windows=config.mmd_refractory_windows,
        )
    elif strategy == "combined":
        error_calibration = calibrate_loss_threshold(
            calibration_errors, z=config.loss_z
        )
        mmd_calibration = calibrate_mmd_threshold(
            calibration_embeddings,
            window_size=config.window_size,
            alpha=config.mmd_alpha,
            n_permutations=config.mmd_n_permutations,
            seed=config.seed,
        )
        triggers["calibration"] = {
            "loss": error_calibration.as_dict(),
            "mmd": mmd_calibration.as_dict(),
        }
        triggers["adwin"] = AdwinTrigger(
            delta=config.adwin_delta,
            grace_period=config.adwin_grace_period,
            clock=config.adwin_clock,
            min_window_length=config.adwin_min_window_length,
            cooldown_windows=config.adwin_cooldown_windows,
        )
        triggers["mmd"] = MMDTrigger(
            calibration_embeddings,
            threshold=mmd_calibration.threshold,
            bandwidth=mmd_calibration.bandwidth,
            refractory_windows=config.mmd_refractory_windows,
        )
        triggers["combined"] = CombinedTrigger(
            triggers["adwin"],
            triggers["mmd"],
            min_gap_windows=config.combined_min_gap_windows,
        )
    return triggers


def run_strategy(
    strategy: str,
    config: E3Config,
    lm_config: LMRuntimeConfig,
    stream,
    events: list[PreparedEvent],
    baseline_rows: tuple[list[float], list[float], list[float]],
    embeddings: np.ndarray | None,
    base_error_stream: dict[int, float],
    bootstrap_error_stream: dict[int, float],
    shared_calibration_errors: list[list[float]],
    bootstrap_checkpoint: Path | None,
    bootstrap_seconds: float,
    results_dir: Path,
    checkpoints_dir: Path,
    log: Any,
) -> StrategyResult:
    """Execute one trigger strategy end-to-end on the stream."""
    result = StrategyResult(strategy=strategy, regime=STRATEGY_REGIMES[strategy])
    started = time.perf_counter()
    tracker = ResourceTracker()
    tracker.sample()

    adapting = strategy != "no_adapt"
    controller = ModelController(
        lm_config=lm_config,
        device=resolve_device(lm_config.device),
        checkpoints_dir=checkpoints_dir / strategy,
        tracker=tracker,
        log=log,
        initial_adapter=bootstrap_checkpoint if adapting else None,
    )

    n_windows = stream.n_windows
    triggers = (
        build_triggers(
            strategy,
            config,
            shared_calibration_errors,
            embeddings[: config.window_size * 3],
            n_windows,
        )
        if strategy not in ("no_adapt", "always_adapt")
        else {}
    )

    warmup_examples = config.warmup_examples
    segment_ends = [
        (segment + 1) * EXAMPLES_PER_EVENT - 1 for segment in range(N_SEGMENTS)
    ]
    for end in segment_ends:
        assert (end + 1) % config.window_size == 0, "segment end must close a window"

    accuracy_rows = [list(baseline_rows[0])]
    f1_rows = [list(baseline_rows[1])]
    ppl_rows = [list(baseline_rows[2])]
    states = ["base"]
    buffer: list[dict] = []
    window_errors: list[float] = []
    last_adaptation_position = 124 if adapting else -(10**9)

    def _fire(trigger_info: dict, position: int, rows: list[dict]) -> None:
        trigger_info["buffer_positions"] = {
            "start": position - len(rows) + 1,
            "end": position,
        }
        record = controller.adapt(rows, trigger_info)
        record["position"] = position
        result.adaptation_positions.append(position)
        result.adaptation_records.append(record)

    try:
        for position, row in enumerate(stream.rows):
            # 1. Regime-B signal: prequential error BEFORE any training here.
            #    Shared caches cover the base (0-124) and bootstrap
            #    (125-299) segments; afterwards the strategy computes its
            #    own online errors (its model may have diverged).
            #    no_adapt records the shared base curve as the
            #    unmitigated-drift reference (it never adapts).
            if strategy in ERROR_STREAM_STRATEGIES or strategy == "no_adapt":
                if strategy == "no_adapt" or position < config.adapt_budget:
                    error = base_error_stream[position]
                elif position < warmup_examples:
                    error = bootstrap_error_stream[position]
                else:
                    error = controller.prequential_error(row, position)
                window_errors.append(error)
                result.error_curve.append((position, error))
                if "adwin" in triggers:
                    triggers["adwin"].observe_example(error, position)

            # 2. Buffer recent examples for trigger-driven adaptation.
            buffer.append(row)
            if len(buffer) > config.adapt_budget:
                buffer.pop(0)

            # 3. Reference policy: oracle adaptation at every event start,
            #    using the first adapt_budget examples of the new event.
            #    (The bootstrap covers event 1.)
            if strategy == "always_adapt":
                event_start = (position // EXAMPLES_PER_EVENT) * EXAMPLES_PER_EVENT
                if (
                    event_start > 0
                    and position - event_start + 1 == config.adapt_budget
                ):
                    _fire(
                        {
                            "trigger": "always_adapt_oracle_boundary",
                            "window_index": position // config.window_size,
                            "detail": (
                                "oracle event boundary; adaptation uses the "
                                "first adapt_budget examples of the new event"
                            ),
                        },
                        position,
                        list(buffer),
                    )

            # 4. Window close: aggregate signals, evaluate triggers.
            if (position + 1) % config.window_size == 0:
                window_index = (position + 1) // config.window_size - 1
                finite_errors = [
                    value
                    for value in window_errors[-config.window_size :]
                    if np.isfinite(value)
                ]
                mean_error = float(np.mean(finite_errors)) if finite_errors else None
                mmd_statistic = None
                if strategy in MMD_STRATEGIES:
                    mmd_statistic = mmd2_unbiased(
                        triggers["mmd"].reference,
                        embeddings[stream.window(window_index)],
                        triggers["mmd"].bandwidth,
                    )
                    result.mmd_curve.append((position, mmd_statistic))
                result.window_log.append(
                    {
                        "strategy": strategy,
                        "window_index": window_index,
                        "position": position,
                        "mean_error": mean_error,
                        "mmd2_vs_reference": mmd_statistic,
                    }
                )

                fired = False
                trigger_info: dict = {}
                in_warmup = position + 1 <= warmup_examples
                if strategy.startswith("periodic"):
                    fired = triggers["periodic"].observe_window(window_index, position)
                    trigger_info = {
                        "trigger": (
                            f"periodic_k{triggers['periodic'].every_windows}"
                        ),
                        "window_index": window_index,
                    }
                elif strategy == "loss_threshold":
                    fired = triggers["loss_threshold"].observe_window(
                        FeedbackSignal(window_index, tuple(finite_errors)),
                        position,
                    )
                    trigger_info = {
                        "trigger": "loss_threshold",
                        "window_index": window_index,
                        "statistic": mean_error,
                        "threshold": triggers["loss_threshold"].threshold,
                    }
                elif strategy == "adwin":
                    fired = triggers["adwin"].observe_window(window_index, position)
                    trigger_info = {
                        "trigger": "adwin",
                        "window_index": window_index,
                        "alarm_positions": [
                            alarm
                            for alarm in triggers["adwin"].alarms
                            if alarm > position - config.window_size
                        ],
                    }
                elif strategy == "mmd":
                    fired = triggers["mmd"].observe_window(
                        UnlabeledSignal(
                            window_index,
                            embeddings[stream.window(window_index)],
                        ),
                        position,
                    )
                    trigger_info = {
                        "trigger": "mmd",
                        "window_index": window_index,
                        "statistic": triggers["mmd"].log[-1].statistic,
                        "threshold": triggers["mmd"].threshold,
                    }
                elif strategy == "combined":
                    fired = triggers["combined"].observe_window(
                        window_index,
                        position,
                        FeedbackSignal(window_index, tuple(finite_errors)),
                        UnlabeledSignal(
                            window_index,
                            embeddings[stream.window(window_index)],
                        ),
                    )
                    trigger_info = {
                        "trigger": "combined_or",
                        "window_index": window_index,
                        "detail": triggers["combined"].log[-1].reason,
                    }

                if fired and in_warmup:
                    log(
                        f"window {window_index}: trigger suppressed "
                        "(calibration/warm-up period)"
                    )
                    result.window_log[-1]["suppressed_warmup"] = True
                    fired = False
                if (
                    fired
                    and position - last_adaptation_position < config.min_gap_examples
                ):
                    log(
                        f"window {window_index}: trigger suppressed "
                        "(minimum adaptation gap)"
                    )
                    result.window_log[-1]["suppressed_min_gap"] = True
                    fired = False
                if fired:
                    _fire(trigger_info, position, list(buffer))
                    last_adaptation_position = position

                # 5. Segment boundary: evaluate the current state.
                if position in segment_ends:
                    segment = segment_ends.index(position) + 1
                    state_name = f"after_segment_{segment}"
                    log(
                        f"evaluating state {state_name} on all held-out test sets"
                    )
                    if strategy == "no_adapt":
                        # Deterministic identical protocol on the identical
                        # state: reuse the baseline row (measured by E1).
                        per_event = json.loads(
                            json.dumps(
                                _baseline_per_event(baseline_rows, events)
                            )
                        )
                        records = []
                    else:
                        per_event, records = controller.evaluate_state(
                            events, state_name
                        )
                    result.eval_points[state_name] = per_event
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
                    if records:
                        _write_predictions(
                            results_dir / "predictions" / strategy,
                            state_name,
                            records,
                        )

        # -- finalise ----------------------------------------------------------
        result.states = states
        result.accuracy_matrix = accuracy_rows
        result.f1_matrix = f1_rows
        result.ppl_matrix = ppl_rows
        result.n_adaptations = controller.n_adaptations
        result.prediction_log = controller.prediction_log
        result.resources = {
            "n_adaptations": controller.n_adaptations,
            "training_seconds": controller.training_seconds,
            "training_steps": controller.training_steps,
            "training_examples": controller.training_examples,
            "training_tokens": controller.training_tokens,
            "stream_seconds": controller.stream_seconds,
            "eval_seconds": controller.eval_seconds,
            "shared_bootstrap_seconds": bootstrap_seconds if adapting else 0.0,
            "wall_seconds_including_eval": time.perf_counter() - started,
            "peak_mps_allocated_gb": tracker.snapshot()["peak_mps_allocated_gb"],
            "peak_rss_gb": tracker.snapshot()["peak_rss_gb"],
            "adapter_checkpoint_bytes": sum(
                record["checkpoint_size_bytes"]
                for record in result.adaptation_records
            ),
        }

        # Detection accounting. The bootstrap adaptation (position
        # adapt_budget-1) is a documented deployment convention, not a
        # detection, and is excluded. For the oracle reference, the moment
        # a boundary is "known" is the moment its first example arrives, so
        # each oracle adaptation is attributed to position
        # (position - adapt_budget + 1).
        if strategy == "always_adapt":
            detection_positions = [
                position - config.adapt_budget + 1
                for position in result.adaptation_positions
                if position - config.adapt_budget + 1 > 0
            ]
        else:
            detection_positions = [
                position
                for position in result.adaptation_positions
                if position >= config.adapt_budget
            ]
        detection = detection_accounting(
            detection_positions,
            stream.boundaries,
            config.tolerance_examples,
            config.window_size,
        )
        result.detection = detection.as_dict()
        result.detection["fp_rate_per_monitored_window"] = (
            false_positive_rate_per_window(
                detection.false_positives,
                n_windows,
                config.warmup_windows,
            )
        )
        result.detection["attribution_note"] = (
            "accounting uses actual adaptation positions (bootstrap excluded)"
            if strategy != "always_adapt"
            else "oracle adaptations attributed to the boundary they respond to"
        )
        result.cl_metrics = _cl_metrics(accuracy_rows)
        baseline_avg = average_accuracy(accuracy_rows, 0)
        final_avg = average_accuracy(accuracy_rows, len(accuracy_rows) - 1)
        result.efficiency = adaptation_efficiency(
            final_avg,
            baseline_avg,
            controller.training_seconds,
            controller.n_adaptations,
            controller.training_tokens,
        )
        result.status = "complete"

        for detector_name, detector in triggers.items():
            if detector_name == "calibration":
                continue
            for decision in detector.log:
                result.detector_log.append(
                    {
                        "strategy": strategy,
                        "detector": decision.detector,
                        "regime": decision.regime,
                        "window_index": decision.window_index,
                        "position": decision.position,
                        "fired": decision.fired,
                        "statistic": decision.statistic,
                        "threshold": decision.threshold,
                        "reason": decision.reason,
                    }
                )
        _write_strategy_artifacts(results_dir, config, strategy, result)
        controller.release()

    except Exception as error:  # noqa: BLE001 - one failed strategy must not kill the run
        import traceback

        result.status = "failed"
        result.error = f"{type(error).__name__}: {error}"
        result.resources = {"wall_seconds_including_eval": time.perf_counter() - started}
        log(f"STRATEGY FAILED {strategy}: {result.error}")
        log(traceback.format_exc())
        write_json(
            results_dir / f"{strategy}_status.json",
            {"strategy": strategy, "status": result.status, "error": result.error},
        )
        controller.release()
    return result


def _baseline_per_event(
    baseline_rows: tuple[list[float], list[float], list[float]],
    events: list[PreparedEvent],
) -> dict[str, dict]:
    per_event = {}
    for index, event in enumerate(events):
        per_event[str(event.event_id)] = {
            "exact_match": baseline_rows[0][index],
            "token_f1": baseline_rows[1][index],
            "perplexity": baseline_rows[2][index],
        }
    return per_event


def _write_predictions(prediction_dir: Path, state: str, records: list[dict]) -> None:
    prediction_dir.mkdir(parents=True, exist_ok=True)
    path = prediction_dir / f"{state}.jsonl"
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record) + "\n")


def _cl_metrics(matrix: list[list[float]]) -> dict:
    t_final = len(matrix) - 1
    return {
        "average_accuracy_per_state": {
            f"state_{index}": average_accuracy(matrix, index)
            for index in range(len(matrix))
        },
        "forgetting_per_event": {
            f"event_{j}": forgetting(matrix, j) for j in range(1, t_final + 1)
        },
        "average_forgetting": average_forgetting(matrix),
        "backward_transfer_mean_all_events": backward_transfer(matrix),
        "backward_transfer_conventional": backward_transfer_conventional(matrix),
    }


def _write_strategy_artifacts(
    results_dir: Path,
    config: E3Config,
    strategy: str,
    result: StrategyResult,
) -> None:
    out_dir = results_dir / strategy
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        out_dir / "strategy_result.json",
        {
            "strategy": strategy,
            "status": result.status,
            "regime": result.regime,
            "states": result.states,
            "accuracy_matrix": result.accuracy_matrix,
            "token_f1_matrix": result.f1_matrix,
            "perplexity_matrix": result.ppl_matrix,
            "continual_learning": result.cl_metrics,
            "detection": result.detection,
            "adaptation_positions": result.adaptation_positions,
            "adaptations": result.adaptation_records,
            "resources": result.resources,
            "efficiency": result.efficiency,
        },
    )
    write_json(
        out_dir / "window_log.json",
        {"strategy": strategy, "windows": result.window_log},
    )
    write_json(
        out_dir / "prediction_log.json",
        {"strategy": strategy, "predictions": result.prediction_log},
    )
    with (out_dir / "detector_log.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "strategy", "detector", "regime", "window_index", "position",
                "fired", "statistic", "threshold", "reason",
            ]
        )
        for entry in result.detector_log:
            writer.writerow(
                [
                    entry["strategy"], entry["detector"], entry["regime"],
                    entry["window_index"], entry["position"], entry["fired"],
                    entry["statistic"], entry["threshold"], entry["reason"],
                ]
            )


def load_baseline_rows(baseline_path: Path) -> tuple[list[float], list[float], list[float], dict]:
    """Reuse the verified E1 baseline for matrix row 0 (same protocol/sets)."""
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    per_event = baseline["per_event"]
    test_events = sorted(int(key) for key in per_event)
    row = [per_event[str(event)]["exact_match"] for event in test_events]
    f1_row = [per_event[str(event)]["token_f1"] for event in test_events]
    ppl_row = [per_event[str(event)]["perplexity"] for event in test_events]
    return row, f1_row, ppl_row, baseline


class BootstrapRunner:
    """Performs the shared bootstrap adaptation once per run."""

    def __init__(
        self,
        lm_config: LMRuntimeConfig,
        stream,
        budget: int,
        checkpoints_dir: Path,
        log: Any,
    ):
        self.lm_config = lm_config
        self.stream = stream
        self.budget = budget
        self.checkpoints_dir = checkpoints_dir
        self.log = log

    def run(self, tracker: ResourceTracker) -> tuple[Path, dict]:
        self.log(
            f"bootstrap adaptation: fresh rank-{self.lm_config.lora_r} LoRA on the "
            f"first {self.budget} stream examples (documented deployment convention)"
        )
        tokenizer = load_tokenizer(self.lm_config.model_id)
        model = load_base_model(self.lm_config)
        device = resolve_device(self.lm_config.device)
        model.to(device)
        model = attach_lora(model, self.lm_config, device)
        set_seed(self.lm_config.seed)
        rows = self.stream.rows[: self.budget]
        pre_training_fingerprint = _adapter_fingerprint(model)
        stats = train_event(
            model, tokenizer, rows, self.lm_config, log=self.log,
            on_step=lambda _s, _l: tracker.sample(),
        )
        checkpoint_dir = self.checkpoints_dir / "bootstrap"
        model.save_pretrained(str(checkpoint_dir))
        tokenizer.save_pretrained(str(checkpoint_dir))
        write_json(
            checkpoint_dir / "adaptation_provenance.json",
            {
                "strategy": "bootstrap (shared by all adapting strategies)",
                "adaptation_index": 1,
                "continued_from": None,
                "same_adapter_continued": False,
                "trigger": {
                    "trigger": "bootstrap_deployment_convention",
                    "detail": (
                        "documented initial adaptation on the first "
                        "adapt_budget stream examples; excluded from "
                        "detection accounting"
                    ),
                    "buffer_positions": {
                        "start": 0,
                        "end": len(rows) - 1,
                    },
                },
                "buffer_size": len(rows),
                "buffer_positions": {"start": 0, "end": len(rows) - 1},
                "init_device": str(device),
                "initialization_order": INITIALIZATION_ORDER_E3,
                "init_fingerprint_pre_training": pre_training_fingerprint,
                "lm_config": self.lm_config.to_dict(),
                "training_stats": {
                    key: value
                    for key, value in stats.items()
                    if key != "losses_per_step"
                },
                "software": software_versions(),
            },
        )
        record = {
            "adaptation_index": 0,
            "trigger": "bootstrap_deployment_convention",
            "position": len(rows) - 1,
            "buffer_size": len(rows),
            "optimizer_steps": stats["optimizer_steps"],
            "examples_processed": stats["examples_scheduled"] - stats["examples_skipped"],
            "total_tokens": stats["total_tokens_trained"],
            "training_seconds": stats["training_seconds"],
            "average_loss": stats["average_loss"],
            "checkpoint_dir": str(checkpoint_dir),
            "continued_from": None,
            "memory": tracker.snapshot(),
        }
        del model
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        return checkpoint_dir, record


def compute_base_error_stream(
    lm_config: LMRuntimeConfig,
    stream_rows: list[dict],
    log: Any,
) -> dict[int, float]:
    """Prequential 0-1 error of the BASE model for every stream position.

    Computed once per run and shared: it feeds the pre-bootstrap segment of
    every delayed-feedback strategy, the full no-adapt reference curve, and
    the run-level drift diagnostic. The answer never influences the
    prediction being scored.
    """
    device = resolve_device(lm_config.device)
    tokenizer = load_tokenizer(lm_config.model_id)
    model = load_base_model(lm_config)
    model.to(device)
    model.eval()
    errors: dict[int, float] = {}
    with torch.no_grad():
        for position, row in enumerate(stream_rows):
            raw = generate_answer(model, tokenizer, str(row["question"]), lm_config)
            prediction = extract_answer(raw)
            errors[position] = 1.0 - exact_match(prediction, str(row["object"]))
            if (position + 1) % 100 == 0:
                log(f"base error stream: {position + 1}/{len(stream_rows)}")
    del model
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return errors


def compute_bootstrap_calibration_errors(
    lm_config: LMRuntimeConfig,
    stream_rows: list[dict],
    bootstrap_checkpoint: Path,
    start: int,
    end: int,
    log: Any,
) -> dict[int, float]:
    """Prequential errors of the shared bootstrap model for positions
    ``start..end-1`` (calibration segment), computed once and shared."""
    device = resolve_device(lm_config.device)
    tokenizer = load_tokenizer(lm_config.model_id)
    model = load_base_model(lm_config)
    model.to(device)
    model = load_adapter_for_training(model, bootstrap_checkpoint, lm_config, device)
    model.eval()
    errors: dict[int, float] = {}
    with torch.no_grad():
        for position in range(start, end):
            row = stream_rows[position]
            raw = generate_answer(model, tokenizer, str(row["question"]), lm_config)
            prediction = extract_answer(raw)
            errors[position] = 1.0 - exact_match(prediction, str(row["object"]))
            if (position - start + 1) % 50 == 0:
                log(
                    f"bootstrap error stream: {position - start + 1}"
                    f"/{end - start}"
                )
    del model
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return errors


def run_e3(
    config_path: str | Path,
    run_id: str,
    strategies: tuple[str, ...] | None = None,
    pilot_budget: int | None = None,
    seed: int | None = None,
    device: str | None = None,
) -> dict[str, StrategyResult]:
    """Execute the E3 comparison sequentially (never concurrently)."""
    import yaml

    from cycleft.config import load_config, validate_stream_config
    from cycleft.e1.pipeline import resolve_lm_config

    if strategies is None:
        strategies = FULL_STRATEGIES
    raw_config = load_config(config_path)
    validate_stream_config(raw_config)
    lm_config, _train_limit, _resolved = resolve_lm_config(
        raw_config, mode="full", train_examples=None, seed=seed, device=device
    )

    config = E3Config(run_id=run_id, seed=lm_config.seed)
    if pilot_budget is not None:
        config = config.with_pilot_budget(pilot_budget)
    lm_config.lora_r = config.adapt_rank

    def log(message: str) -> None:
        print(f"[e3:{run_id}] {message}", flush=True)

    results_dir = RESULTS_ROOT / run_id
    checkpoints_dir = CHECKPOINTS_ROOT / run_id
    results_dir.mkdir(parents=True, exist_ok=True)

    # -- temporal data audit ----------------------------------------------------
    stream, events, manifest = build_stream(seed=config.seed)
    audit = dataset_audit(stream, events, manifest)
    order_check = load_slice_order_check(stream)
    (results_dir / "dataset_audit.md").write_text(
        summarise_audit(audit, order_check), encoding="utf-8"
    )
    log(
        f"stream ready: {stream.n_examples} examples, {stream.n_windows} windows, "
        f"boundaries at {stream.boundaries}"
    )

    # -- configuration artifacts ---------------------------------------------------
    config_payload = {
        **config.as_dict(),
        "lm_config": lm_config.to_dict(),
        "strategy_regimes": STRATEGY_REGIMES,
        "stream": {
            "n_examples": stream.n_examples,
            "n_windows": stream.n_windows,
            "window_size": stream.window_size,
            "boundaries": stream.boundaries,
            "order": audit["stream_order"],
        },
        "adaptation_action": (
            f"E2 Pareto cell rank{config.adapt_rank} with <= {config.adapt_budget} "
            "optimizer steps per adaptation (buffer = most recent answered examples)"
        ),
        "bootstrap": (
            "shared initial adaptation on stream positions 0..adapt_budget-1; "
            "all adapting strategies continue the same bootstrap adapter; "
            "excluded from detection accounting"
        ),
        "feedback_delay_windows": 0,
        "pilot_budget_active": pilot_budget is not None,
    }
    (results_dir / "detector_config.yaml").write_text(
        yaml.safe_dump(config_payload, sort_keys=False), encoding="utf-8"
    )
    write_json(
        results_dir / "drift_events.json",
        {
            "dataset_id": audit["dataset_id"],
            "known_boundaries": stream.boundaries,
            "boundary_windows": stream.boundary_windows,
            "boundary_dates": [
                {"position": boundary, "slice_transition": transition}
                for boundary, transition in zip(
                    stream.boundaries,
                    ["Nov→Dec → Dec→Jan", "Dec→Jan → Jan→Feb"],
                )
            ],
            "n_known_transitions": len(stream.boundaries),
            "note": (
                "Only these two transitions are treated as ground truth; "
                "within-event windows are used for false alarms under an "
                "explicitly stated stationarity assumption."
            ),
        },
    )
    write_json(
        results_dir / "run_config.json",
        {
            "run_id": run_id,
            "seed": config.seed,
            "strategies": list(strategies),
            "pilot": config.pilot,
            "adapt_budget": config.adapt_budget,
            "adapt_rank": config.adapt_rank,
            "lm_config": lm_config.to_dict(),
            "baseline_source": str(
                REPO_ROOT / "results" / "e1" / "full_20261113"
                / "baseline_metrics.json"
            ),
            "execution": "sequential; one resident model per stage",
        },
    )

    device = resolve_device(lm_config.device)

    # -- shared regime-A signal ---------------------------------------------------
    embeddings = None
    if any(strategy in MMD_STRATEGIES for strategy in strategies):
        embeddings_path = results_dir / "question_embeddings.npy"
        if embeddings_path.exists():
            embeddings = np.load(embeddings_path)
            log(f"loaded cached question embeddings {embeddings.shape}")
        else:
            embeddings = compute_question_embeddings(
                lm_config, stream.rows, device, log
            )
            np.save(embeddings_path, embeddings)

    # -- shared bootstrap adaptation ------------------------------------------------
    tracker = ResourceTracker()
    tracker.sample()
    bootstrap_checkpoint = None
    bootstrap_record = None
    bootstrap_seconds = 0.0
    if any(strategy != "no_adapt" for strategy in strategies):
        bootstrap_path = checkpoints_dir / "bootstrap"
        if (bootstrap_path / "adaptation_provenance.json").exists():
            provenance = json.loads(
                (bootstrap_path / "adaptation_provenance.json").read_text(
                    encoding="utf-8"
                )
            )
            bootstrap_record = {
                "adaptation_index": 0,
                **provenance["training_stats"],
                "checkpoint_dir": str(bootstrap_path),
                "continued_from": None,
            }
            bootstrap_seconds = provenance["training_stats"]["training_seconds"]
            bootstrap_checkpoint = bootstrap_path
            log(
                f"reusing existing bootstrap checkpoint ({bootstrap_seconds:.1f}s, "
                f"{bootstrap_record['optimizer_steps']} steps)"
            )
        else:
            bootstrap_runner = BootstrapRunner(
                lm_config, stream, config.adapt_budget, checkpoints_dir, log
            )
            bootstrap_checkpoint, bootstrap_record = bootstrap_runner.run(tracker)
            bootstrap_seconds = bootstrap_record["training_seconds"]

    # -- shared regime-B caches -------------------------------------------------------
    base_error_stream: dict[int, float] = {}
    bootstrap_error_stream: dict[int, float] = {}
    calibration_errors: list[list[float]] = []
    if any(
        strategy in ERROR_STREAM_STRATEGIES or strategy == "no_adapt"
        for strategy in strategies
    ):
        base_error_path = results_dir / "base_error_stream.json"
        if base_error_path.exists():
            base_error_stream = {
                int(key): value
                for key, value in json.loads(base_error_path.read_text())["errors"].items()
            }
            log(f"loaded cached base error stream ({len(base_error_stream)} positions)")
        else:
            base_error_stream = compute_base_error_stream(
                lm_config, stream.rows, log
            )
            write_json(
                results_dir / "base_error_stream.json",
                {
                    "model_state": "pretrained base (no adapter)",
                    "signal": "prequential 0-1 prediction error",
                    "feedback_delay_windows": 0,
                    "errors": {str(k): v for k, v in base_error_stream.items()},
                },
            )

    if any(strategy in ERROR_STREAM_STRATEGIES for strategy in strategies):
        calibration_path = results_dir / "bootstrap_calibration_errors.json"
        if calibration_path.exists():
            payload = json.loads(calibration_path.read_text())
            bootstrap_error_stream = {
                int(key): value for key, value in payload["errors"].items()
            }
            if "calibration_windows" in payload:
                calibration_errors = payload["calibration_windows"]
            else:
                calibration_errors = [
                    [
                        bootstrap_error_stream[position]
                        for position in range(
                            window * config.window_size,
                            (window + 1) * config.window_size,
                        )
                    ]
                    for window in config.calibration_window_indices
                ]
            log("loaded cached bootstrap calibration errors")
        else:
            bootstrap_error_stream = compute_bootstrap_calibration_errors(
                lm_config,
                stream.rows,
                bootstrap_checkpoint,
                start=config.adapt_budget,
                end=config.warmup_examples,
                log=log,
            )
            calibration_errors = [
                [
                    bootstrap_error_stream[position]
                    for position in range(
                        window * config.window_size,
                        (window + 1) * config.window_size,
                    )
                ]
                for window in config.calibration_window_indices
            ]
            write_json(
                results_dir / "bootstrap_calibration_errors.json",
                {
                    "model_state": "shared bootstrap adapter",
                    "signal": "prequential 0-1 prediction error",
                    "positions": [config.adapt_budget, config.warmup_examples],
                    "calibration_window_indices": list(
                        config.calibration_window_indices
                    ),
                    "errors": {
                        str(k): v for k, v in bootstrap_error_stream.items()
                    },
                },
            )
        write_json(
            results_dir / "calibration_errors.json",
            {
                "n_calibration_windows": config.n_calibration_windows,
                "window_size": config.window_size,
                "windows": calibration_errors,
                "note": (
                    "shared bootstrap-model prequential errors; used by "
                    "every delayed-feedback strategy; never final-evaluation data"
                ),
            },
        )

    baseline_rows = load_baseline_rows(
        REPO_ROOT / "results" / "e1" / "full_20261113" / "baseline_metrics.json"
    )
    log(
        f"reusing verified E1 baseline row {baseline_rows[0]} "
        "(identical protocol and held-out sets)"
    )

    # -- sequential strategy execution ---------------------------------------------
    results: dict[str, StrategyResult] = {}
    for strategy in strategies:
        log(f"=== strategy {strategy} ({STRATEGY_REGIMES[strategy]}) ===")
        results[strategy] = run_strategy(
            strategy=strategy,
            config=config,
            lm_config=lm_config,
            stream=stream,
            events=events,
            baseline_rows=(
                baseline_rows[0], baseline_rows[1], baseline_rows[2]
            ),
            embeddings=embeddings,
            base_error_stream=base_error_stream,
            bootstrap_error_stream=bootstrap_error_stream,
            shared_calibration_errors=calibration_errors,
            bootstrap_checkpoint=bootstrap_checkpoint,
            bootstrap_seconds=bootstrap_seconds,
            results_dir=results_dir,
            checkpoints_dir=checkpoints_dir,
            log=log,
        )

    write_json(
        results_dir / "e3_status.json",
        {
            "run_id": run_id,
            "pilot": config.pilot,
            "adapt_budget": config.adapt_budget,
            "bootstrap": bootstrap_record,
            "completed": [
                strategy
                for strategy, result in results.items()
                if result.status == "complete"
            ],
            "failed": [
                {"strategy": strategy, "error": result.error}
                for strategy, result in results.items()
                if result.status != "complete"
            ],
        },
    )
    return results


if __name__ == "__main__":  # pragma: no cover
    run_e3("configs/e1_stream.yaml", run_id="e3_manual")
