"""E2 grid runner: sequential LoRA adaptation for every (rank, budget) cell.

Per configuration (fresh pretrained base every time):

    event 1: attach fresh LoRA(rank) -> train on the first
             ``budget_examples`` of the shared seed-42 training order ->
             checkpoint -> evaluate all three test sets
    event k: load event k-1 adapter of THIS configuration -> continue
             training -> checkpoint -> evaluate

The base model is reloaded from the pretrained checkpoint for every event
stage and released afterwards, so only one model copy is ever resident.
The baseline (pretrained, no adapter) is evaluated once per E2 run and
shared by all configurations.
"""

from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from cycleft.e1.data import PreparedEvent, prepare_event_datasets
from cycleft.e1.lm import (
    LMRuntimeConfig,
    attach_lora,
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
    backward_transfer,
    backward_transfer_conventional,
    forgetting,
)
from cycleft.e1.pipeline import (
    REPO_ROOT,
    software_versions,
    write_json,
)
from cycleft.e2.grid import (
    E2Configuration,
    assert_budget_semantics,
    build_grid,
    configuration_row,
)
from cycleft.e2.resources import (
    EventResourceRecord,
    ResourceTracker,
    directory_size_bytes,
    event_resource_record,
    total_optimizer_steps,
    total_training_seconds,
    total_training_tokens,
)

RESULTS_ROOT = REPO_ROOT / "results" / "e2"
CHECKPOINTS_ROOT = REPO_ROOT / "checkpoints" / "e2"
DEFAULT_BASELINE = REPO_ROOT / "results" / "e1" / "full_20261113" / "baseline_metrics.json"


@dataclass
class ConfigurationResult:
    """Everything measured for one grid cell."""

    config: E2Configuration
    status: str
    states: list[str] = field(default_factory=list)
    test_events: list[int] = field(default_factory=list)
    accuracy_matrix: list[list[float]] = field(default_factory=list)
    f1_matrix: list[list[float]] = field(default_factory=list)
    ppl_matrix: list[list[float]] = field(default_factory=list)
    training_entries: dict[str, dict] = field(default_factory=dict)
    resource_records: list[EventResourceRecord] = field(default_factory=list)
    total_wall_seconds: float = 0.0
    eval_seconds: float = 0.0
    checkpoint_dirs: dict[str, str] = field(default_factory=dict)
    error: str | None = None

    @property
    def final_avg_exact_match(self) -> float:
        return average_accuracy(self.accuracy_matrix, len(self.accuracy_matrix) - 1)

    @property
    def base_avg_exact_match(self) -> float:
        return average_accuracy(self.accuracy_matrix, 0)


class E2Runner:
    def __init__(
        self,
        run_id: str,
        base_lm_config: LMRuntimeConfig,
        events: list[PreparedEvent],
        baseline: dict,
        baseline_source: str,
        log: Any | None = None,
    ):
        self.run_id = run_id
        self.base_lm_config = base_lm_config
        self.events = events
        self.baseline = baseline
        self.baseline_source = baseline_source
        self.device = resolve_device(base_lm_config.device)
        self.results_dir = RESULTS_ROOT / run_id
        self.checkpoints_dir = CHECKPOINTS_ROOT / run_id
        self._log = log or (
            lambda message: print(f"[e2:{run_id}] {message}", flush=True)
        )
        self.baseline_row = [
            baseline["per_event"][str(event.event_id)]["exact_match"]
            for event in events
        ]
        self.baseline_f1_row = [
            baseline["per_event"][str(event.event_id)]["token_f1"]
            for event in events
        ]
        self.baseline_ppl_row = [
            baseline["per_event"][str(event.event_id)]["perplexity"]
            for event in events
        ]

    def log(self, message: str) -> None:
        self._log(message)

    # -- one configuration ------------------------------------------------

    def configuration_lm_config(self, config: E2Configuration) -> LMRuntimeConfig:
        lm_config = copy.deepcopy(self.base_lm_config)
        lm_config.lora_r = config.lora_rank
        return lm_config

    def run_configuration(self, config: E2Configuration) -> ConfigurationResult:
        result = ConfigurationResult(config=config, status="running")
        started = time.perf_counter()
        lm_config = self.configuration_lm_config(config)
        tracker = ResourceTracker()
        tracker.sample()

        try:
            self.log(
                f"=== {config.config_id}: rank={config.lora_rank} "
                f"budget={config.budget_examples} examples/event "
                f"({config.budget_fraction:.0%}) ==="
            )
            assert_budget_semantics(lm_config)

            previous_checkpoint: Path | None = None
            accuracy_rows = [list(self.baseline_row)]
            f1_rows = [list(self.baseline_f1_row)]
            ppl_rows = [list(self.baseline_ppl_row)]
            states = ["base"]

            for event in self.events:
                set_seed(lm_config.seed)
                tokenizer = load_tokenizer(lm_config.model_id)
                base_model = load_base_model(lm_config)

                event_dir = (
                    self.checkpoints_dir / config.config_id / f"event_{event.event_id}"
                )
                if previous_checkpoint is None:
                    self.log(f"{config.config_id}: fresh LoRA adapter (rank {config.lora_rank})")
                    model = attach_lora(base_model, lm_config, self.device)
                else:
                    self.log(
                        f"{config.config_id}: continuing event {event.event_id} "
                        f"from {previous_checkpoint.name}"
                    )
                    model = load_adapter_for_training(
                        base_model, previous_checkpoint, lm_config, self.device
                    )

                train_rows = event.train_rows[: config.budget_examples]
                stats = train_event(
                    model,
                    tokenizer,
                    train_rows,
                    lm_config,
                    log=self.log,
                    on_step=lambda _step, _loss: tracker.sample(),
                )

                model.save_pretrained(str(event_dir))
                tokenizer.save_pretrained(str(event_dir))
                write_json(
                    event_dir / "training_config.json",
                    {
                        "run_id": self.run_id,
                        "config_id": config.config_id,
                        "event_id": event.event_id,
                        "model_id": lm_config.model_id,
                        "lora_rank": config.lora_rank,
                        "budget_examples": config.budget_examples,
                        "budget_id": config.budget_id,
                        "continued_from": (
                            str(previous_checkpoint) if previous_checkpoint else None
                        ),
                        "fresh_base_load": True,
                        "seed": lm_config.seed,
                        "device": str(self.device),
                        "precision": lm_config.dtype,
                        "lora_precision": lm_config.lora_precision,
                        "lm_config": lm_config.to_dict(),
                        "software": software_versions(),
                        "training_stats": {
                            k: v for k, v in stats.items() if k != "losses_per_step"
                        },
                    },
                )

                resource_record = event_resource_record(
                    config_id=config.config_id,
                    lora_rank=config.lora_rank,
                    budget_examples=config.budget_examples,
                    training_stats=stats,
                    checkpoint_dir=event_dir,
                    continued_from=(
                        str(previous_checkpoint) if previous_checkpoint else None
                    ),
                    tracker=tracker,
                )
                resource_record.event_id = event.event_id
                result.resource_records.append(resource_record)

                eval_started = time.perf_counter()
                per_event: dict[str, dict] = {}
                records_by_test: dict[int, list[dict]] = {}
                for eval_event in self.events:
                    evaluation = evaluate_event(
                        model,
                        tokenizer,
                        eval_event.eval_rows,
                        lm_config,
                        log=self.log,
                    )
                    per_event[str(eval_event.event_id)] = evaluation["metrics"]
                    records_by_test[eval_event.event_id] = evaluation["records"]
                    tracker.sample()
                result.eval_seconds += time.perf_counter() - eval_started

                self._write_predictions(
                    config.config_id,
                    f"after_event_{event.event_id}",
                    event.event_id,
                    records_by_test,
                )
                self.log(
                    f"{config.config_id} after event {event.event_id}: "
                    + " ".join(
                        f"test{j}={per_event[str(j)]['exact_match']:.3f}"
                        for j in sorted(self._test_event_ids())
                    )
                )

                accuracy_rows.append(
                    [per_event[str(e.event_id)]["exact_match"] for e in self.events]
                )
                f1_rows.append(
                    [per_event[str(e.event_id)]["token_f1"] for e in self.events]
                )
                ppl_rows.append(
                    [per_event[str(e.event_id)]["perplexity"] for e in self.events]
                )
                states.append(f"after_event_{event.event_id}")

                result.training_entries[str(event.event_id)] = {
                    "n_train_examples": len(train_rows),
                    "optimizer_steps": stats["optimizer_steps"],
                    "average_loss": stats["average_loss"],
                    "final_loss": stats["final_loss"],
                    "training_seconds": stats["training_seconds"],
                    "trainable_parameters": stats["trainable_parameters"],
                    "total_parameters": stats["total_parameters"],
                    "trainable_percentage": stats["trainable_percentage"],
                    "answer_tokens_trained": stats["answer_tokens_trained"],
                    "total_tokens_trained": stats["total_tokens_trained"],
                    "examples_skipped": stats["examples_skipped"],
                    "checkpoint_dir": str(event_dir),
                    "continued_from": (
                        str(previous_checkpoint) if previous_checkpoint else None
                    ),
                }
                result.checkpoint_dirs[str(event.event_id)] = str(event_dir)

                previous_checkpoint = event_dir
                del model
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()

            result.states = states
            result.test_events = [e.event_id for e in self.events]
            result.accuracy_matrix = accuracy_rows
            result.f1_matrix = f1_rows
            result.ppl_matrix = ppl_rows
            result.total_wall_seconds = time.perf_counter() - started
            result.status = "complete"
            self._write_configuration_artifacts(config, result, lm_config)

        except Exception as error:  # noqa: BLE001 - one failed cell must not kill the grid
            import traceback

            result.status = "failed"
            result.error = f"{type(error).__name__}: {error}"
            result.total_wall_seconds = time.perf_counter() - started
            self.log(f"CONFIGURATION FAILED {config.config_id}: {result.error}")
            self.log(traceback.format_exc())
            write_json(
                self.results_dir / config.config_id / "status.json",
                {
                    "config_id": config.config_id,
                    "status": result.status,
                    "error": result.error,
                    "wall_seconds": result.total_wall_seconds,
                },
            )

        return result

    # -- helpers ------------------------------------------------------------

    def _test_event_ids(self) -> list[int]:
        return [event.event_id for event in self.events]

    def _write_predictions(
        self,
        config_id: str,
        state: str,
        training_event: int,
        records_by_test: dict[int, list[dict]],
    ) -> None:
        pred_dir = self.results_dir / "predictions" / config_id
        pred_dir.mkdir(parents=True, exist_ok=True)
        path = pred_dir / f"{state}.jsonl"
        with path.open("w", encoding="utf-8") as file:
            for test_event, records in sorted(records_by_test.items()):
                for record in records:
                    file.write(
                        json.dumps(
                            {
                                "state": state,
                                "trained_through_event": training_event,
                                "test_event": test_event,
                                **record,
                            }
                        )
                        + "\n"
                    )

    def _cl_metrics(self, matrix: list[list[float]]) -> dict:
        t_final = len(matrix) - 1
        return {
            "average_accuracy_per_state": {
                state: average_accuracy(matrix, index)
                for index, state in enumerate(
                    ["base"] + [f"after_event_{i}" for i in range(1, t_final + 1)]
                )
            },
            "forgetting_per_event": {
                f"event_{j}": forgetting(matrix, j) for j in range(1, t_final + 1)
            },
            "average_forgetting": average_forgetting(matrix),
            "backward_transfer_mean_all_events": backward_transfer(matrix),
            "backward_transfer_conventional": backward_transfer_conventional(matrix),
            "definitions": {
                "backward_transfer_mean_all_events": (
                    "(1/T) * sum_j (A(T,j) - A(j,j)); divisor T includes the "
                    "last event's trivially-zero delta (E1 convention)"
                ),
                "backward_transfer_conventional": (
                    "(1/(T-1)) * sum_{j=1}^{T-1} (A(T,j) - A(j,j)); "
                    "Lopez-Paz & Ranzato (2017) convention"
                ),
            },
        }

    def _write_configuration_artifacts(
        self,
        config: E2Configuration,
        result: ConfigurationResult,
        lm_config: LMRuntimeConfig,
    ) -> None:
        out_dir = self.results_dir / config.config_id
        out_dir.mkdir(parents=True, exist_ok=True)

        cl = self._cl_metrics(result.accuracy_matrix)
        matrix_rows = [
            {
                "config_id": config.config_id,
                "test_event": j,
                "base": result.accuracy_matrix[0][j - 1],
                **{
                    state: result.accuracy_matrix[index][j - 1]
                    for index, state in enumerate(result.states[1:], start=1)
                },
            }
            for j in result.test_events
        ]

        write_json(
            out_dir / "event_metrics.json",
            {
                "config_id": config.config_id,
                "status": result.status,
                "states": result.states,
                "test_events": result.test_events,
                "accuracy_matrix": result.accuracy_matrix,
                "token_f1_matrix": result.f1_matrix,
                "perplexity_matrix": result.ppl_matrix,
                "continual_learning": cl,
                "training": result.training_entries,
                "total_wall_seconds": result.total_wall_seconds,
                "eval_seconds": result.eval_seconds,
            },
        )
        write_json(
            out_dir / "resource_metrics.json",
            {
                "config_id": config.config_id,
                "lora_rank": config.lora_rank,
                "budget_examples_per_event": config.budget_examples,
                "events": [
                    {
                        "event_id": record.event_id,
                        "optimizer_steps": record.optimizer_steps,
                        "examples_processed": record.examples_processed,
                        "examples_skipped": record.examples_skipped,
                        "total_tokens": record.total_tokens,
                        "answer_tokens": record.answer_tokens,
                        "training_seconds": record.training_seconds,
                        "tokens_per_second": record.tokens_per_second,
                        "checkpoint_size_bytes": record.checkpoint_size_bytes,
                        "continued_from": record.continued_from,
                        "memory": record.memory,
                    }
                    for record in result.resource_records
                ],
                "totals": {
                    "optimizer_steps": total_optimizer_steps(result.resource_records),
                    "examples_processed": sum(
                        r.examples_processed for r in result.resource_records
                    ),
                    "total_tokens": total_training_tokens(result.resource_records),
                    "training_seconds": total_training_seconds(
                        result.resource_records
                    ),
                    "wall_seconds_including_eval": result.total_wall_seconds,
                    "eval_seconds": result.eval_seconds,
                    "peak_mps_allocated_gb": max(
                        (r.memory.get("peak_mps_allocated_gb", 0.0)
                         for r in result.resource_records),
                        default=0.0,
                    ),
                    "peak_rss_gb": max(
                        (r.memory.get("peak_rss_gb", 0.0)
                         for r in result.resource_records),
                        default=0.0,
                    ),
                    "adapter_checkpoint_bytes": sum(
                        r.checkpoint_size_bytes for r in result.resource_records
                    ),
                },
                "note": "All values measured on the executing machine; no modeled costs.",
            },
        )
        write_json(
            out_dir / "continual_learning_matrix.csv.json",
            {
                "config_id": config.config_id,
                "rows": matrix_rows,
                "continual_learning": cl,
            },
        )


def load_or_compute_baseline(
    runner_assets_dir: Path,
    baseline_from: Path | None,
    events: list[PreparedEvent],
    lm_config: LMRuntimeConfig,
    device: torch.device,
    log: Any,
) -> tuple[dict, str]:
    """Reuse a compatible baseline or compute a fresh one.

    The baseline (pretrained model, no adapter) is identical for every E2
    configuration, so it is evaluated once. When reusing an E1 baseline
    the model id and evaluation settings are verified first.
    """
    candidates = (
        [baseline_from] if baseline_from else [DEFAULT_BASELINE]
    )
    for candidate in candidates:
        if not candidate.exists():
            continue
        baseline = json.loads(candidate.read_text(encoding="utf-8"))
        if baseline.get("model_id") != lm_config.model_id:
            log(f"Baseline at {candidate} has model_id {baseline.get('model_id')}; ignoring.")
            continue
        missing = [
            str(event.event_id)
            for event in events
            if str(event.event_id) not in baseline.get("per_event", {})
        ]
        if missing:
            log(f"Baseline at {candidate} is missing test events {missing}; ignoring.")
            continue
        log(f"Reusing verified baseline from {candidate}")
        return baseline, str(candidate)

    log("Computing fresh baseline (pretrained model, no adapter)")
    set_seed(lm_config.seed)
    tokenizer = load_tokenizer(lm_config.model_id)
    model = load_base_model(lm_config)
    model.to(device)
    model.eval()

    per_event: dict[str, dict] = {}
    for event in events:
        evaluation = evaluate_event(
            model, tokenizer, event.eval_rows, lm_config, log=log
        )
        per_event[str(event.event_id)] = evaluation["metrics"]
        log(
            f"  baseline test {event.event_id}: "
            f"EM={evaluation['metrics']['exact_match']:.3f}"
        )

    baseline = {
        "state": "base",
        "state_index": 0,
        "device": str(device),
        "precision": lm_config.dtype,
        "model_id": lm_config.model_id,
        "per_event": per_event,
        "mean_exact_match": sum(
            m["exact_match"] for m in per_event.values()
        ) / len(per_event),
        "mean_token_f1": sum(m["token_f1"] for m in per_event.values()) / len(per_event),
    }
    write_json(runner_assets_dir / "baseline_metrics.json", baseline)
    del model
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return baseline, "computed_fresh"


def run_e2(
    config_path: str | Path,
    run_id: str,
    ranks: tuple[int, ...] = (4, 8, 16, 32),
    budget_ids: tuple[str, ...] = ("b125", "b250", "b500"),
    configs: list[tuple[int, str]] | None = None,
    seed: int | None = None,
    device: str | None = None,
    baseline_from: Path | None = None,
    alpha: int | None = None,
) -> dict[str, ConfigurationResult]:
    """Execute the E2 grid sequentially (never concurrently).

    ``alpha`` overrides ``lora_alpha`` for this run (remediation Block A:
    alpha-controlled rank sweep). ``None`` keeps the configured value.
    """
    from cycleft.config import load_config, validate_stream_config
    from cycleft.e1.pipeline import resolve_lm_config

    raw_config = load_config(config_path)
    validate_stream_config(raw_config)
    lm_config, train_limit_unused, resolved = resolve_lm_config(
        raw_config, mode="full", train_examples=None, seed=seed, device=device
    )
    if alpha is not None:
        lm_config.lora_alpha = int(alpha)
        resolved["alpha_override"] = int(alpha)
    assert_budget_semantics(lm_config)

    if train_limit_unused is not None:
        raise ValueError("E2 always trains on prefixes of the full ordered pool.")

    def log(message: str) -> None:
        print(f"[e2:{run_id}] {message}", flush=True)

    results_dir = RESULTS_ROOT / run_id
    results_dir.mkdir(parents=True, exist_ok=True)

    log(f"Preparing datasets (seed={lm_config.seed})")
    events, manifest = prepare_event_datasets(train_limit=None, seed=lm_config.seed)
    manifest["e2_note"] = (
        "Training order is one seed-shuffled permutation of each 500-row pool; "
        "every budget uses a PREFIX of this exact order, so smaller budgets are "
        "strict initial segments of larger ones (E1 full used the same order)."
    )
    write_json(results_dir / "dataset_manifest.json", manifest)

    device = resolve_device(lm_config.device)
    baseline, baseline_source = load_or_compute_baseline(
        results_dir,
        baseline_from,
        events,
        lm_config,
        device,
        log,
    )

    runner = E2Runner(
        run_id=run_id,
        base_lm_config=lm_config,
        events=events,
        baseline=baseline,
        baseline_source=baseline_source,
        log=log,
    )

    write_json(
        runner.results_dir / "run_config.json",
        {
            **resolved,
            "run_id": run_id,
            "ranks": list(ranks),
            "budget_ids": list(budget_ids),
            "baseline_source": baseline_source,
            "execution": "sequential; one model copy in memory",
        },
    )

    if configs is None:
        grid = build_grid(tuple(ranks), tuple(budget_ids))
    else:
        from cycleft.e2.grid import make_configuration

        grid = [make_configuration(rank, budget) for rank, budget in configs]

    results: dict[str, ConfigurationResult] = {}
    for configuration in grid:
        results[configuration.config_id] = runner.run_configuration(configuration)

    write_json(
        runner.results_dir / "grid_status.json",
        {
            run_id: run_id,
            "completed": [
                cid for cid, r in results.items() if r.status == "complete"
            ],
            "failed": [
                {"config_id": cid, "error": r.error}
                for cid, r in results.items()
                if r.status == "failed"
            ],
        },
    )
    return results


if __name__ == "__main__":  # pragma: no cover
    run_e2("configs/e1_stream.yaml", run_id="e2_manual")
