"""Post-run analysis for E1: CL metrics, matrix CSV, plots, summary, README.

Everything here reads the artifacts written by :mod:`cycleft.e1.pipeline`
and never fabricates measurements; missing data raises or is reported as
``null`` in the summary.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from cycleft.e1.metrics import (  # noqa: E402
    average_accuracy,
    average_forgetting,
    backward_transfer,
    forgetting,
)
from cycleft.e1.pipeline import write_json  # noqa: E402


def load_run_artifacts(results_dir: Path) -> dict[str, Any]:
    """Load the artifacts required for analysis."""
    results_dir = Path(results_dir)
    required = {
        "config": "config.yaml",
        "baseline": "baseline_metrics.json",
        "events": "event_metrics.json",
    }
    artifacts: dict[str, Any] = {}
    for key, name in required.items():
        path = results_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Missing artifact for analysis: {path}")
        if name.endswith(".yaml"):
            import yaml

            artifacts[key] = yaml.safe_load(path.read_text(encoding="utf-8"))
        else:
            artifacts[key] = json.loads(path.read_text(encoding="utf-8"))

    training_log = results_dir / "training_log.jsonl"
    artifacts["training_log"] = (
        [
            json.loads(line)
            for line in training_log.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if training_log.exists()
        else []
    )
    return artifacts


def continual_learning_summary(event_metrics: dict) -> dict:
    """Compute average accuracy, forgetting and BWT from the accuracy matrix."""
    matrix = event_metrics["accuracy_matrix"]
    t_final = len(matrix) - 1

    return {
        "average_accuracy_per_state": {
            state: average_accuracy(matrix, index)
            for index, state in enumerate(event_metrics["states"])
        },
        "forgetting_per_event": {
            f"event_{event}": forgetting(matrix, event)
            for event in range(1, t_final + 1)
        },
        "average_forgetting": average_forgetting(matrix),
        "backward_transfer": backward_transfer(matrix),
        "formulas": {
            "forgetting": "forgetting(j) = max over k >= j of A(k, j) - A(T, j); "
            "positive = accuracy on event j dropped by the end of the stream",
            "backward_transfer": "BWT = (1/T) * sum_j ( A(T, j) - A(j, j) ); "
            "negative = net forgetting, positive = later training helped earlier events",
        },
    }


def write_matrix_csv(results_dir: Path, event_metrics: dict) -> Path:
    """Write the continual-learning accuracy matrix as CSV."""
    matrix = event_metrics["accuracy_matrix"]
    test_events = event_metrics["test_events"]
    t_final = len(matrix) - 1

    lines = ["model_state," + ",".join(f"test_event_{j}" for j in test_events)
             + ",average_accuracy"]
    for index, state in enumerate(event_metrics["states"]):
        values = ",".join(f"{value:.6f}" for value in matrix[index])
        lines.append(f"{state},{values},{average_accuracy(matrix, index):.6f}")
    for name, value in (
        ("forgetting", None),
        ("average_forgetting", average_forgetting(matrix)),
        ("backward_transfer", backward_transfer(matrix)),
    ):
        if name == "forgetting":
            lines.append(
                "forgetting_per_event,"
                + ",".join(f"{forgetting(matrix, j):.6f}" for j in range(1, t_final + 1))
                + ","
            )
        else:
            lines.append(f"{name},,,,,{value:.6f}")

    path = results_dir / "continual_learning_matrix.csv"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def generate_plots(results_dir: Path, event_metrics: dict) -> list[Path]:
    """Generate all E1 plots from saved metrics (never from estimates)."""
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    matrix = event_metrics["accuracy_matrix"]
    states = event_metrics["states"]
    test_events = event_metrics["test_events"]
    state_labels = ["base"] + [s.replace("after_event_", "after E") for s in states[1:]]

    generated: list[Path] = []

    # 1. Accuracy across training events.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for column, event in enumerate(test_events):
        ax.plot(
            range(len(states)),
            [row[column] for row in matrix],
            marker="o",
            label=f"test event {event}",
        )
    ax.set_xticks(range(len(states)), state_labels)
    ax.set_xlabel("Model state")
    ax.set_ylabel("Exact match accuracy")
    ax.set_title("E1: accuracy on each test set across sequential training")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = plots_dir / "accuracy_curves.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    generated.append(path)

    # 2. Accuracy matrix heatmap.
    fig, ax = plt.subplots(figsize=(6, 4.5))
    image = ax.imshow(matrix, cmap="viridis", vmin=0.0, aspect="auto")
    ax.set_xticks(range(len(test_events)), [f"test {j}" for j in test_events])
    ax.set_yticks(range(len(states)), state_labels)
    for i in range(len(states)):
        for j in range(len(test_events)):
            ax.text(
                j, i, f"{matrix[i][j]:.2f}",
                ha="center", va="center",
                color="white" if matrix[i][j] < 0.5 else "black",
            )
    ax.set_title("E1 accuracy matrix A(k, j)")
    fig.colorbar(image, ax=ax, label="exact match")
    fig.tight_layout()
    path = plots_dir / "accuracy_heatmap.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    generated.append(path)

    # 3. Forgetting by event.
    t_final = len(matrix) - 1
    forgettings = [forgetting(matrix, event) for event in range(1, t_final + 1)]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([f"event {j}" for j in range(1, t_final + 1)], forgettings)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_ylabel("Forgetting (positive = accuracy dropped)")
    ax.set_title("E1: forgetting by event")
    fig.tight_layout()
    path = plots_dir / "forgetting_by_event.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    generated.append(path)

    # 4. Training loss by event (from training log).
    log_entries = sorted(
        (p for p in [results_dir / "training_log.jsonl"] if p.exists()),
        key=lambda p: p.stat().st_mtime,
    )
    if log_entries:
        entries = [
            json.loads(line)
            for line in log_entries[0].read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        event_ids = [str(entry["event_id"]) for entry in entries]
        avg_losses = [entry["average_loss"] for entry in entries]
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar([f"event {e}" for e in event_ids], avg_losses)
        for bar, loss in zip(bars, avg_losses):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{loss:.3f}", ha="center", va="bottom",
            )
        ax.set_ylabel("Average training loss")
        ax.set_title("E1: training loss by event")
        fig.tight_layout()
        path = plots_dir / "training_loss_by_event.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        generated.append(path)

        # 5. Training time and adapter size by event.
        times = [entry["training_seconds"] for entry in entries]
        sizes_gb = [entry["checkpoint_size_bytes"] / 1e9 for entry in entries]
        fig, ax1 = plt.subplots(figsize=(6.5, 4))
        x = range(len(event_ids))
        ax1.bar([i - 0.2 for i in x], times, width=0.4, label="training time (s)")
        ax1.set_ylabel("Training time (s)")
        ax2 = ax1.twinx()
        ax2.bar([i + 0.2 for i in x], sizes_gb, width=0.4,
                color="tab:orange", label="adapter checkpoint (GB)")
        ax2.set_ylabel("Checkpoint size (GB)")
        ax1.set_xticks(list(x), [f"event {e}" for e in event_ids])
        ax1.set_title("E1: training time and checkpoint size by event")
        fig.tight_layout()
        path = plots_dir / "time_and_size_by_event.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        generated.append(path)

    return generated


def build_summary(results_dir: Path, event_metrics: dict) -> dict:
    """Assemble summary.json: matrix, CL metrics, timing, sizes."""
    cl = continual_learning_summary(event_metrics)
    training = event_metrics.get("training", {})

    return {
        "run_id": results_dir.name,
        "scale": "see config.yaml 'mode' (pilot vs full); never report pilot numbers as full-scale",
        "states": event_metrics["states"],
        "test_events": event_metrics["test_events"],
        "accuracy_matrix": event_metrics["accuracy_matrix"],
        "token_f1_matrix": event_metrics["token_f1_matrix"],
        "perplexity_matrix": event_metrics["perplexity_matrix"],
        "continual_learning": cl,
        "training_summary": {
            event_id: {
                "n_train_examples": entry.get("n_train_examples"),
                "optimizer_steps": entry.get("optimizer_steps"),
                "average_loss": entry.get("average_loss"),
                "training_seconds": entry.get("training_seconds"),
                "checkpoint_size_bytes": entry.get("checkpoint_size_bytes"),
                "trainable_parameters": entry.get("trainable_parameters"),
                "total_parameters": entry.get("total_parameters"),
                "trainable_percentage": entry.get("trainable_percentage"),
                "answer_tokens_trained": entry.get("answer_tokens_trained"),
                "total_tokens_trained": entry.get("total_tokens_trained"),
            }
            for event_id, entry in training.items()
        },
        "total_training_seconds": sum(
            entry.get("training_seconds", 0.0) for entry in training.values()
        ),
        "total_training_examples": sum(
            entry.get("n_train_examples", 0) for entry in training.values()
        ),
        "total_optimizer_steps": sum(
            entry.get("optimizer_steps", 0) for entry in training.values()
        ),
    }


def write_readme(results_dir: Path, summary: dict) -> Path:
    """Write a README documenting the run, formulas, and rerun commands."""
    cl = summary["continual_learning"]
    matrix = summary["accuracy_matrix"]
    states = summary["states"]
    test_events = summary["test_events"]

    header = "| Model state | " + " | ".join(f"Test Event {j}" for j in test_events) + " |"
    separator = "|---" * (len(test_events) + 1) + "|"
    rows = []
    for index, state in enumerate(states):
        cells = " | ".join(f"{value:.4f}" for value in matrix[index])
        rows.append(f"| {state} | {cells} |")

    training_rows = "\n".join(
        f"- Event {event_id}: {entry['n_train_examples']} examples, "
        f"{entry['optimizer_steps']} optimizer steps, "
        f"avg loss {entry['average_loss']:.4f}, "
        f"{entry['training_seconds']:.1f} s, "
        f"checkpoint {entry['checkpoint_size_bytes'] / 1e6:.2f} MB "
        f"({entry['trainable_percentage']:.4f}% params trainable)"
        for event_id, entry in sorted(summary["training_summary"].items())
    )

    content = f"""# E1 Run: {summary['run_id']}

Sequential LoRA continual adaptation of a pretrained LLM on TemporalWiki.
All numbers below are **measured** values read from this run's artifacts.
Scale (pilot vs full) is recorded in `config.yaml` under `mode`.

## Accuracy matrix (exact match)

{header}
{separator}
{chr(10).join(rows)}

## Continual-learning metrics

- Average accuracy per state: `{json.dumps(cl['average_accuracy_per_state'], indent=None)}`
- Forgetting per event: `{json.dumps(cl['forgetting_per_event'])}`
- Average forgetting: `{cl['average_forgetting']:.4f}` (positive = forgetting)
- Backward transfer: `{cl['backward_transfer']:.4f}` (negative = net forgetting)

Formulas:
- `{cl['formulas']['forgetting']}`
- `{cl['formulas']['backward_transfer']}`

## Training summary

{training_rows}

Total: {summary['total_training_examples']} training examples, \
{summary['total_optimizer_steps']} optimizer steps, \
{summary['total_training_seconds']:.1f} s training time.

## Artifacts

- `config.yaml` — resolved run configuration
- `dataset_manifest.json` — per-event dataset stats, seeds, overlaps
- `baseline_metrics.json` — pretrained-model evaluation
- `event_metrics.json` — full evaluation matrices + training stats
- `continual_learning_matrix.csv` — the matrix above in CSV form
- `predictions.jsonl` — every prediction with raw/extracted text and scores
- `training_log.jsonl` — one record per training event
- `summary.json` — machine-readable digest
- `plots/` — figures generated from saved metrics only

## Rerun

```bash
uv run python scripts/e1_run.py --mode pilot --run-id {summary['run_id']}
uv run python scripts/e1_analyze.py --results-dir results/e1/{summary['run_id']}
```

Checkpoints: `checkpoints/e1/{summary['run_id']}/event_1..3/` \
(each contains adapter weights, tokenizer, training config, provenance).
"""
    path = results_dir / "README.md"
    path.write_text(content, encoding="utf-8")
    return path


def analyze_run(results_dir: str | Path) -> dict:
    """Full post-run analysis entry point."""
    results_dir = Path(results_dir)
    artifacts = load_run_artifacts(results_dir)
    event_metrics = artifacts["events"]

    csv_path = write_matrix_csv(results_dir, event_metrics)
    plots = generate_plots(results_dir, event_metrics)
    summary = build_summary(results_dir, event_metrics)
    write_json(results_dir / "summary.json", summary)
    readme_path = write_readme(results_dir, summary)

    return {
        "summary": summary,
        "matrix_csv": str(csv_path),
        "plots": [str(p) for p in plots],
        "readme": str(readme_path),
    }
