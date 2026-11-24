"""Sequential E1 pipeline: baseline -> event 1 -> event 2 -> event 3.

Stage model (one base-model copy in memory at a time):

    baseline: base model -> eval on all test sets
    event t:  load base -> attach fresh LoRA (t=1) or load event t-1
              adapter (t>1, continuing the same adapter) -> train on the
              event's deterministic training subset -> save checkpoint ->
              eval on all test sets -> release model

All artifacts are written under ``results/e1/<run_id>/`` and checkpoints
under ``checkpoints/e1/<run_id>/event_<t>/``.
"""

from __future__ import annotations

import collections
import json
import platform
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Any

import torch

from cycleft.config import load_config, validate_stream_config
from cycleft.data.stream import StreamEvent
from cycleft.e1.data import PreparedEvent, prepare_event_datasets
from cycleft.e1.lm import (
    DEFAULT_SYSTEM_PROMPT,
    LMRuntimeConfig,
    attach_lora,
    evaluate_event,
    gpu_metadata,
    load_adapter_for_training,
    load_base_model,
    load_tokenizer,
    memory_status,
    release_device_memory,
    resolve_device,
    set_seed,
    train_event,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_ROOT = REPO_ROOT / "results" / "e1"
CHECKPOINTS_ROOT = REPO_ROOT / "checkpoints" / "e1"


def _git_commit() -> str | None:
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            ).stdout.strip()
        )
    except (OSError, subprocess.SubprocessError):
        return None


def software_versions() -> dict:
    def version(name: str) -> str | None:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            return None

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": version("torch"),
        "transformers": version("transformers"),
        "peft": version("peft"),
        "accelerate": version("accelerate"),
        "datasets": version("datasets"),
        "git_commit": _git_commit(),
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload) + "\n")


def resolve_lm_config(
    raw_config: dict,
    mode: str,
    train_examples: int | None,
    seed: int | None,
    device: str | None,
) -> tuple[LMRuntimeConfig, int | None, dict]:
    """Merge the ``e1`` config section with CLI overrides.

    Returns the runtime config, the effective per-event training limit
    (``None`` = all examples), and the resolved experiment config snapshot.
    """
    e1_section = raw_config.get("e1")
    if not isinstance(e1_section, dict):
        raise ValueError("Config is missing the 'e1' experiment section.")

    modes = e1_section.get("modes", {})
    if mode not in modes:
        raise ValueError(f"Unknown mode '{mode}'. Available: {sorted(modes)}")
    mode_settings = modes[mode] or {}

    train_limit = train_examples
    if train_limit is None:
        configured = mode_settings.get("train_examples_per_event")
        train_limit = None if configured in (None, "null") else int(configured)

    effective_seed = seed if seed is not None else int(e1_section.get("seed", 42))
    effective_device = device or e1_section.get("device")

    training = e1_section.get("training", {})
    evaluation = e1_section.get("evaluation", {})

    lm_config = LMRuntimeConfig(
        model_id=e1_section["model_id"],
        max_length=int(e1_section.get("max_length", 128)),
        max_new_tokens=int(evaluation.get("max_new_tokens", 24)),
        system_prompt=e1_section.get("system_prompt", DEFAULT_SYSTEM_PROMPT),
        chat_date_string=e1_section.get("chat_date_string", "01 Jan 2026"),
        device=effective_device,
        seed=effective_seed,
        learning_rate=float(training.get("learning_rate", 1e-4)),
        batch_size=int(training.get("batch_size", 1)),
        gradient_accumulation_steps=int(
            training.get("gradient_accumulation_steps", 1)
        ),
        max_grad_norm=float(training.get("max_grad_norm", 1.0)),
        epochs=int(training.get("epochs", 1)),
        lora_r=int(training.get("lora_r", 8)),
        lora_alpha=int(training.get("lora_alpha", 16)),
        lora_dropout=float(training.get("lora_dropout", 0.05)),
        lora_target_modules=tuple(training.get("lora_target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj"])),
    )

    resolved = {
        "mode": mode,
        "run_id": None,
        "model_id": lm_config.model_id,
        "train_examples_per_event": train_limit,
        "seed": effective_seed,
        "device_requested": effective_device,
        "lm_config": lm_config.to_dict(),
        "stream": raw_config["stream"],
    }
    return lm_config, train_limit, resolved


def _stream_events(raw_config: dict) -> list[StreamEvent]:
    stream = raw_config["stream"]
    events = []
    for event in stream["events"]:
        if event["train_source"] is None:
            continue
        events.append(
            StreamEvent(
                event_id=int(event["event_id"]),
                name=str(event["name"]),
                train_source=str(event["train_source"]),
                eval_source=str(event["eval_source"]),
            )
        )
    if not events:
        raise ValueError("Stream config contains no trainable events.")
    return events


class E1Run:
    """Stateful runner for one E1 execution."""

    def __init__(
        self,
        run_id: str,
        lm_config: LMRuntimeConfig,
        train_limit: int | None,
        resolved_config: dict,
    ):
        self.run_id = run_id
        self.lm_config = lm_config
        self.train_limit = train_limit
        self.resolved_config = {**resolved_config, "run_id": run_id}
        self.results_dir = RESULTS_ROOT / run_id
        self.checkpoints_dir = CHECKPOINTS_ROOT / run_id
        self.device = resolve_device(lm_config.device)
        self.predictions_path = self.results_dir / "predictions.jsonl"
        self.training_log_path = self.results_dir / "training_log.jsonl"
        self.sweep_populations: dict[str, list[dict]] = {}
        self.sweep_selection: dict[str, dict] = {}
        self.sweep_metrics: dict[str, dict[str, dict]] = {}
        self.validation_metrics: dict[str, dict] = {}

    def log(self, message: str) -> None:
        print(f"[e1:{self.run_id}] {message}", flush=True)

    def prepare(self) -> tuple[list[PreparedEvent], dict]:
        dataset_name = (
            self.resolved_config.get("stream", {}).get("dataset", {}).get("name")
            or "temporalwiki"
        )
        dataset_config = (
            self.resolved_config.get("stream", {}).get("dataset", {}).get("adapter", {})
        )
        self.log(
            f"Preparing datasets (dataset={dataset_name}, "
            f"train_limit={self.train_limit}, seed={self.lm_config.seed})"
        )
        events, manifest = prepare_event_datasets(
            train_limit=self.train_limit,
            seed=self.lm_config.seed,
            dataset=dataset_name,
            dataset_config=dataset_config,
            mode=self.resolved_config.get("mode"),
        )
        for event in events:
            self.log(
                f"  event {event.event_id} ({event.slice_tag}): "
                f"{len(event.train_rows)}/{event.train_pool_size} train, "
                f"{event.eval_pool_size} eval"
            )
        manifest["device"] = str(self.device)
        manifest["precision"] = self.lm_config.dtype
        write_json(self.results_dir / "dataset_manifest.json", manifest)
        self.sweep_populations = manifest.get("sweep_populations") or {}
        self.sweep_selection = manifest.get("sweep_selection") or {}
        return events, manifest

    def write_config(self) -> None:
        write_json(self.results_dir / "config.yaml", self.resolved_config)
        write_json(
            self.results_dir / "runtime_metadata.json",
            {
                **software_versions(),
                "device": str(self.device),
                "device_requested": self.lm_config.device,
                "mps_available": torch.backends.mps.is_available(),
                "gpu": gpu_metadata(),
                "memory": memory_status(),
                "precision": self.lm_config.dtype,
            },
        )

    # -- model stages ---------------------------------------------------

    def _release(self, model, tokenizer) -> None:
        del model
        del tokenizer
        release_device_memory(self.device)

    def run_baseline(self, events: list[PreparedEvent]) -> dict:
        self.log("Stage 0: baseline evaluation of the pretrained model")
        set_seed(self.lm_config.seed)
        tokenizer = load_tokenizer(self.lm_config.model_id)
        model = load_base_model(self.lm_config)
        model.to(self.device)
        model.eval()

        per_event: dict[str, dict] = {}
        for event in events:
            self.log(f"  baseline eval on test event {event.event_id}")
            result = evaluate_event(
                model, tokenizer, event.eval_rows, self.lm_config, log=self.log
            )
            per_event[str(event.event_id)] = result["metrics"]
            self._write_predictions("base", 0, event.event_id, result["records"])
            self.log(
                f"    EM={result['metrics']['exact_match']:.3f} "
                f"F1={result['metrics']['token_f1']:.3f} "
                f"ppl={result['metrics']['perplexity']:.2f}"
            )

        baseline = {
            "state": "base",
            "state_index": 0,
            "device": str(self.device),
            "precision": self.lm_config.dtype,
            "model_id": self.lm_config.model_id,
            "per_event": per_event,
            "mean_exact_match": sum(m["exact_match"] for m in per_event.values())
            / len(per_event),
            "mean_token_f1": sum(m["token_f1"] for m in per_event.values())
            / len(per_event),
            "memory_after": memory_status(),
        }
        write_json(self.results_dir / "baseline_metrics.json", baseline)
        if any(e.validation_rows for e in events):
            self._evaluate_validation(
                model, tokenizer, events, state="base", state_index=0)
        if self.sweep_populations:
            self._evaluate_sweeps(model, tokenizer, state="base", state_index=0)
        self._release(model, tokenizer)
        return baseline

    def run_training_event(
        self,
        event: PreparedEvent,
        all_events: list[PreparedEvent],
        previous_checkpoint: Path | None,
    ) -> dict:
        """Train on one event, save the checkpoint, evaluate all test sets."""
        set_seed(self.lm_config.seed)
        event_dir = self.checkpoints_dir / f"event_{event.event_id}"
        tokenizer = load_tokenizer(self.lm_config.model_id)
        base_model = load_base_model(self.lm_config)

        if previous_checkpoint is None:
            self.log(f"Event {event.event_id}: attaching fresh LoRA adapter")
            model = attach_lora(base_model, self.lm_config, self.device)
        else:
            self.log(
                f"Event {event.event_id}: continuing from adapter "
                f"{previous_checkpoint}"
            )
            model = load_adapter_for_training(
                base_model, previous_checkpoint, self.lm_config, self.device
            )

        model.print_trainable_parameters()

        stats = train_event(
            model, tokenizer, event.train_rows, self.lm_config, log=self.log
        )
        self.log(
            f"Event {event.event_id} trained: steps={stats['optimizer_steps']} "
            f"avg_loss={stats['average_loss']:.4f} "
            f"time={stats['training_seconds']:.1f}s"
        )

        # Checkpoint: adapter weights + tokenizer + full provenance.
        model.save_pretrained(str(event_dir))
        tokenizer.save_pretrained(str(event_dir))
        write_json(
            event_dir / "training_config.json",
            {
                "event_id": event.event_id,
                "event_name": event.name,
                "slice_tag": event.slice_tag,
                "model_id": self.lm_config.model_id,
                "continued_from": str(previous_checkpoint) if previous_checkpoint else None,
                "seed": self.lm_config.seed,
                "device": str(self.device),
                "precision": self.lm_config.dtype,
                "lm_config": self.lm_config.to_dict(),
                "software": software_versions(),
                "training_stats": {
                    k: v for k, v in stats.items() if k != "losses_per_step"
                },
                "train_indices_used": event.train_indices_used,
            },
        )
        adapter_size = sum(
            f.stat().st_size for f in event_dir.iterdir() if f.is_file()
        )

        self.log(f"Evaluating after event {event.event_id} on all test sets")
        per_event: dict[str, dict] = {}
        for eval_event in all_events:
            result = evaluate_event(
                model, tokenizer, eval_event.eval_rows, self.lm_config, log=self.log
            )
            per_event[str(eval_event.event_id)] = result["metrics"]
            self._write_predictions(
                f"after_event_{event.event_id}",
                event.event_id,
                eval_event.event_id,
                result["records"],
                checkpoint_dir=event_dir,
            )
            self.log(
                f"    test {eval_event.event_id}: "
                f"EM={result['metrics']['exact_match']:.3f} "
                f"F1={result['metrics']['token_f1']:.3f}"
            )

        if any(e.validation_rows for e in all_events):
            self._evaluate_validation(
                model, tokenizer, all_events,
                state=f"after_event_{event.event_id}",
                state_index=event.event_id)
        if self.sweep_populations:
            self._evaluate_sweeps(
                model, tokenizer,
                state=f"after_event_{event.event_id}",
                state_index=event.event_id,
                checkpoint_dir=event_dir)
        self._release(model, tokenizer)

        entry = {
            "event_id": event.event_id,
            "event_name": event.name,
            "slice_tag": event.slice_tag,
            "continued_from": str(previous_checkpoint) if previous_checkpoint else None,
            "n_train_examples": len(event.train_rows),
            "train_pool_size": event.train_pool_size,
            "checkpoint_dir": str(event_dir),
            "checkpoint_size_bytes": adapter_size,
            **stats,
        }
        append_jsonl(self.training_log_path, entry)
        write_json(
            self.results_dir / f"event_{event.event_id}_eval.json",
            per_event,
        )
        return entry

    def _evaluate_validation(
        self,
        model,
        tokenizer,
        events: list[PreparedEvent],
        state: str,
        state_index: int,
    ) -> None:
        """Evaluate the strictly separate validation population.

        Validation metrics are written to their own file so
        hyperparameter selection can never read final-test metrics.
        """
        per_event: dict[str, dict] = {}
        for event in events:
            if not event.validation_rows:
                continue
            result = evaluate_event(
                model, tokenizer, event.validation_rows, self.lm_config,
                log=self.log)
            per_event[str(event.event_id)] = result["metrics"]
            self._write_predictions(
                state, state_index, event.event_id, result["records"],
                population="validation")
        self.validation_metrics[state] = per_event
        write_json(self.results_dir / "validation_metrics.json",
                   self.validation_metrics)
        self.log(f"  validation evaluated ({state})")

    def _evaluate_sweeps(
        self,
        model,
        tokenizer,
        state: str,
        state_index: int,
        checkpoint_dir: Path | None = None,
    ) -> None:
        """Evaluate the deterministic M1/M3 sweep populations.

        Populations are fixed by the dataset adapter (sizes, sample
        method, seed and exclusions recorded in the dataset manifest).
        Records keep full provenance (fact key, eval family, prompt
        kind, partition).  Perplexity is disabled by the adapter rows.
        """
        from cycleft.e1.wfd_metrics import answer_rates
        for family, rows in sorted(self.sweep_populations.items()):
            if not rows:
                continue
            result = evaluate_event(
                model, tokenizer, rows, self.lm_config, log=self.log)
            rates = answer_rates(result["records"])
            entry = {
                "state": state,
                "state_index": state_index,
                "checkpoint_dir": str(checkpoint_dir) if checkpoint_dir else None,
                "population_size": len(rows),
                "sample_method": self.sweep_selection.get(family, {}).get(
                    "method", "full"),
                "sample_seed": self.sweep_selection.get(family, {}).get("seed"),
                "standard_metrics": result["metrics"],
                "answer_rates": rates,
            }
            self.sweep_metrics.setdefault(family, {})[state] = entry
            for record in result["records"]:
                append_jsonl(
                    self.results_dir / f"sweep_predictions_{family}.jsonl",
                    {"state": state, "state_index": state_index, **record})
            write_json(self.results_dir / f"sweep_metrics_{family}.json",
                       self.sweep_metrics[family])
        self.log(f"  sweep populations evaluated ({state})")

    def build_wfd_summary(self, events: list[PreparedEvent]) -> dict | None:
        """Combined M1/M2/M3 summary from the per-state artifacts."""
        if not self.sweep_populations:
            return None
        from cycleft.e1.wfd_metrics import (
            m1_update_efficacy,
            m2_heldout_transfer,
            m3_neighborhood_retention,
            m3_persistence_rate,
        )
        summary: dict[str, Any] = {"families": {}}
        for family, states in self.sweep_metrics.items():
            summary["families"][family] = states
        if "m1_trained_fact_paraphrases" in self.sweep_populations:
            final_state = sorted(self.sweep_metrics.get(
                "m1_trained_fact_paraphrases", {}))
            if final_state:
                final_records = [
                    json.loads(line) for line in (
                        self.results_dir
                        / "sweep_predictions_m1_trained_fact_paraphrases.jsonl"
                    ).read_text().splitlines() if line.strip()
                ]
                summary["m1_update_efficacy_final"] = m1_update_efficacy(
                    final_records)
        # M2: held-out records live in predictions.jsonl (base vs final)
        by_state: dict[str, list[dict]] = {}
        for line in self.predictions_path.read_text().splitlines():
            record = json.loads(line)
            if record.get("eval_family") in ("heldout_update",
                                             "heldout_paraphrase"):
                by_state.setdefault(record["state"], []).append(record)
        base_records = by_state.get("base", [])
        final_state = max(
            (s for s in by_state if s.startswith("after_event_")),
            key=lambda s: int(s.rsplit("_", 1)[-1]), default=None)
        summary["m2_heldout_transfer"] = m2_heldout_transfer(
            by_state.get(final_state, []) if final_state else [],
            base_records or None)
        # M3
        if "m3_neighborhood_probes" in self.sweep_populations:
            base_records_n = [
                json.loads(line) for line in (
                    self.results_dir
                    / "sweep_predictions_m3_neighborhood_probes.jsonl"
                ).read_text().splitlines() if line.strip()
                and json.loads(line)["state"] == "base"]
            later = []
            for line in (
                self.results_dir
                / "sweep_predictions_m3_neighborhood_probes.jsonl"
            ).read_text().splitlines():
                record = json.loads(line)
                if record["state"].startswith("after_event_"):
                    later.append(record)
            later_by_state = collections.defaultdict(list)
            for record in later:
                later_by_state[record["state"]].append(record)
            summary["m3_neighborhood_retention"] = m3_neighborhood_retention(
                base_records_n,
                *[later_by_state[k] for k in sorted(later_by_state)])
        if "m3_persistence_probes" in self.sweep_populations:
            final_records_p = []
            for line in (
                self.results_dir
                / "sweep_predictions_m3_persistence_probes.jsonl"
            ).read_text().splitlines():
                record = json.loads(line)
                if record["state"].startswith("after_event_"):
                    final_records_p.append(record)
            summary["m3_persistence"] = m3_persistence_rate(final_records_p)
        summary["sweep_selection"] = self.sweep_selection
        return summary

    def _write_predictions(
        self,
        state: str,
        state_index: int,
        test_event: int,
        records: list[dict],
        population: str = "test",
        checkpoint_dir: Path | None = None,
    ) -> None:
        for record in records:
            append_jsonl(
                self.predictions_path,
                {
                    "state": state,
                    "state_index": state_index,
                    "test_event": test_event,
                    "population": population,
                    **(
                        {"checkpoint_dir": str(checkpoint_dir)}
                        if checkpoint_dir else {}
                    ),
                    **record,
                },
            )

    # -- matrix assembly -------------------------------------------------

    def build_event_metrics(self) -> dict:
        baseline = json.loads(
            (self.results_dir / "baseline_metrics.json").read_text(encoding="utf-8")
        )
        test_event_ids = sorted(
            int(key) for key in baseline["per_event"]
        )

        states = ["base"]
        accuracy_matrix = [
            [baseline["per_event"][str(j)]["exact_match"] for j in test_event_ids]
        ]
        f1_matrix = [
            [baseline["per_event"][str(j)]["token_f1"] for j in test_event_ids]
        ]
        ppl_matrix = [
            [baseline["per_event"][str(j)]["perplexity"] for j in test_event_ids]
        ]

        training_entries: dict[str, dict] = {}
        for path in sorted(self.results_dir.glob("event_*_eval.json")):
            event_id = int(path.name.split("_")[1])
            per_event = json.loads(path.read_text(encoding="utf-8"))
            states.append(f"after_event_{event_id}")
            accuracy_matrix.append(
                [per_event[str(j)]["exact_match"] for j in test_event_ids]
            )
            f1_matrix.append([per_event[str(j)]["token_f1"] for j in test_event_ids])
            ppl_matrix.append([per_event[str(j)]["perplexity"] for j in test_event_ids])

            log_entry = self._training_log_entry(event_id)
            if log_entry is not None:
                training_entries[str(event_id)] = log_entry

        matrix_size_ok = len(states) == len(test_event_ids) + 1
        if not matrix_size_ok:
            raise RuntimeError(
                f"Incomplete run: {len(states)} states for "
                f"{len(test_event_ids)} test events. Rerun missing stages."
            )

        return {
            "states": states,
            "test_events": test_event_ids,
            "accuracy_matrix": accuracy_matrix,
            "token_f1_matrix": f1_matrix,
            "perplexity_matrix": ppl_matrix,
            "training": training_entries,
        }

    def _training_log_entry(self, event_id: int) -> dict | None:
        if not self.training_log_path.exists():
            return None
        for line in self.training_log_path.read_text(encoding="utf-8").splitlines():
            entry = json.loads(line)
            if entry.get("event_id") == event_id:
                return entry
        return None


def run_e1(
    config_path: str | Path,
    mode: str = "pilot",
    run_id: str | None = None,
    train_examples: int | None = None,
    seed: int | None = None,
    device: str | None = None,
    start_event: int = 1,
) -> dict:
    """Execute the full E1 sequential experiment (or resume from an event)."""
    raw_config = load_config(config_path)
    validate_stream_config(raw_config)

    lm_config, train_limit, resolved = resolve_lm_config(
        raw_config, mode, train_examples, seed, device
    )

    if run_id is None:
        run_id = f"{mode}_{time.strftime('%Y%m%d_%H%M%S')}"
    if start_event not in (1, 2, 3):
        raise ValueError(f"start_event must be in 1..3, got {start_event}")

    runner = E1Run(run_id, lm_config, train_limit, resolved)
    runner.results_dir.mkdir(parents=True, exist_ok=True)
    runner.log(f"results -> {runner.results_dir}")
    runner.log(f"checkpoints -> {runner.checkpoints_dir}")
    runner.log(f"device={runner.device} precision={lm_config.dtype}")

    events, _ = runner.prepare()
    runner.write_config()

    if start_event == 1:
        runner.run_baseline(events)
    else:
        baseline_path = runner.results_dir / "baseline_metrics.json"
        if not baseline_path.exists():
            raise RuntimeError(
                f"--start-event {start_event} requires an existing baseline "
                f"at {baseline_path}"
            )

    previous_checkpoint: Path | None = None
    for event in events:
        if event.event_id < start_event:
            candidate = runner.checkpoints_dir / f"event_{event.event_id}"
            if not candidate.exists():
                raise RuntimeError(
                    f"Cannot resume at event {start_event}: missing checkpoint "
                    f"{candidate}"
                )
            previous_checkpoint = candidate
            continue

        runner.run_training_event(event, events, previous_checkpoint)
        previous_checkpoint = runner.checkpoints_dir / f"event_{event.event_id}"

    event_metrics = runner.build_event_metrics()
    write_json(runner.results_dir / "event_metrics.json", event_metrics)
    runner.log(f"Wrote {runner.results_dir / 'event_metrics.json'}")
    wfd_summary = runner.build_wfd_summary(events)
    if wfd_summary is not None:
        write_json(runner.results_dir / "wfd_summary.json", wfd_summary)
        runner.log("Wrote wfd_summary.json (M1/M2/M3)")
    return event_metrics


if __name__ == "__main__":  # pragma: no cover
    run_e1("configs/e1_stream.yaml")
