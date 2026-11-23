"""E2 analysis: aggregate all configuration results, Pareto frontier, plots.

Everything is derived from the measured artifacts written by the runner;
missing configurations are reported as failed/incomplete, never estimated.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from cycleft.e1.pipeline import write_json  # noqa: E402
from cycleft.e1.metrics import average_accuracy  # noqa: E402
from cycleft.e2.pareto import (  # noqa: E402
    ParetoPoint,
    dominated_set,
    nondominated_set,
    quality_gain_per_resource,
)


def load_configuration_results(results_dir: Path) -> dict[str, dict]:
    """Load every configuration's artifacts found under the run directory."""
    results: dict[str, dict] = {}
    for path in sorted(results_dir.glob("*/event_metrics.json")):
        config_id = path.parent.name
        event_metrics = json.loads(path.read_text(encoding="utf-8"))
        resource_path = path.parent / "resource_metrics.json"
        resource = (
            json.loads(resource_path.read_text(encoding="utf-8"))
            if resource_path.exists()
            else None
        )
        results[config_id] = {
            "event_metrics": event_metrics,
            "resource": resource,
        }
    if not results:
        raise FileNotFoundError(f"No configuration results under {results_dir}")
    return results


def configuration_table(results_dir: Path, results: dict[str, dict]) -> list[dict]:
    """One row per configuration: identity, quality, resources, efficiency."""
    rows: list[dict] = []
    baseline_avg = None
    for config_id, payload in results.items():
        event_metrics = payload["event_metrics"]
        resource = payload["resource"]
        matrix = event_metrics["accuracy_matrix"]
        final_avg = average_accuracy(matrix, len(matrix) - 1)
        base_avg = average_accuracy(matrix, 0)
        if baseline_avg is None:
            baseline_avg = base_avg
        totals = (resource or {}).get("totals", {})
        training_seconds = totals.get("training_seconds")
        total_tokens = totals.get("total_tokens")
        rank = event_metrics.get("config_id", "").split("_")[0].replace("rank", "")
        budget = totals.get("examples_processed")
        params = next(iter(event_metrics.get("training", {}).values()), {}).get(
            "trainable_parameters"
        )
        rows.append(
            {
                "config_id": config_id,
                "lora_rank": int(rank) if rank.isdigit() else None,
                "examples_processed": budget,
                "trainable_parameters": params,
                "final_avg_exact_match": final_avg,
                "base_avg_exact_match": base_avg,
                "quality_gain_over_base": final_avg - base_avg,
                "final_avg_token_f1": _final_avg(result := event_metrics["token_f1_matrix"]),
                "final_avg_perplexity": _final_avg(event_metrics["perplexity_matrix"]),
                "average_forgetting": event_metrics["continual_learning"][
                    "average_forgetting"
                ],
                "backward_transfer_mean_all_events": event_metrics[
                    "continual_learning"
                ]["backward_transfer_mean_all_events"],
                "backward_transfer_conventional": event_metrics["continual_learning"][
                    "backward_transfer_conventional"
                ],
                "optimizer_steps": totals.get("optimizer_steps"),
                "training_seconds": training_seconds,
                "total_tokens": total_tokens,
                "wall_seconds_including_eval": totals.get(
                    "wall_seconds_including_eval"
                ),
                "peak_mps_allocated_gb": totals.get("peak_mps_allocated_gb"),
                "peak_rss_gb": totals.get("peak_rss_gb"),
                "adapter_checkpoint_bytes": totals.get("adapter_checkpoint_bytes"),
                "quality_gain_per_training_second": quality_gain_per_resource(
                    final_avg, base_avg, training_seconds
                ),
                "quality_gain_per_million_tokens": quality_gain_per_resource(
                    final_avg, base_avg, total_tokens / 1e6 if total_tokens else None
                ),
                "status": event_metrics.get("status", "unknown"),
            }
        )
    return rows


def _final_avg(matrix: list[list[float]]) -> float:
    """Mean over the last row (final model state) of a quality matrix."""
    return sum(matrix[-1]) / len(matrix[-1])


def write_csv(path: Path, rows: list[dict]) -> Path:
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    lines = [",".join(columns)]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column)
            if value is None:
                values.append("")
            elif isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value).replace(",", ";"))
        lines.append(",".join(values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def pareto_points(
    rows: list[dict],
    resource_key: str,
) -> list[ParetoPoint]:
    return [
        ParetoPoint(
            label=row["config_id"],
            resource=float(row[resource_key]),
            quality=float(row["final_avg_exact_match"]),
            payload=row,
        )
        for row in rows
        if row.get(resource_key) is not None
    ]


def pareto_table(
    rows: list[dict],
    resource_key: str,
) -> list[dict]:
    """Build a frontier table: per configuration, dominance and frontier flag."""
    points = pareto_points(rows, resource_key)
    front = {p.label for p in nondominated_set(points)}
    dominated = {p.label for p in dominated_set(points)}
    table = []
    for row in rows:
        if row.get(resource_key) is None:
            continue
        table.append(
            {
                "config_id": row["config_id"],
                "resource_axis": resource_key,
                "resource_value": row[resource_key],
                "final_avg_exact_match": row["final_avg_exact_match"],
                "on_pareto_frontier": row["config_id"] in front,
                "pareto_dominated": row["config_id"] in dominated,
            }
        )
    return sorted(table, key=lambda r: r["resource_value"])


def metrics_rows(results: dict[str, dict]) -> list[dict]:
    """Long-format quality metrics for every configuration x state x test event."""
    rows = []
    for config_id, payload in results.items():
        event_metrics = payload["event_metrics"]
        for state_index, state in enumerate(event_metrics["states"]):
            for column, test_event in enumerate(event_metrics["test_events"]):
                rows.append(
                    {
                        "config_id": config_id,
                        "state": state,
                        "state_index": state_index,
                        "test_event": test_event,
                        "exact_match": event_metrics["accuracy_matrix"][state_index][column],
                        "token_f1": event_metrics["token_f1_matrix"][state_index][column],
                        "perplexity": event_metrics["perplexity_matrix"][state_index][column],
                    }
                )
    return rows


def resource_rows(results: dict[str, dict]) -> list[dict]:
    """Long-format resource metrics for every configuration x event."""
    rows = []
    for config_id, payload in results.items():
        resource = payload["resource"]
        if resource is None:
            continue
        for event in resource["events"]:
            rows.append({"config_id": config_id, **event})
        totals = resource["totals"]
        rows.append(
            {
                "config_id": config_id,
                "event_id": "TOTAL",
                "optimizer_steps": totals.get("optimizer_steps"),
                "examples_processed": totals.get("examples_processed"),
                "total_tokens": totals.get("total_tokens"),
                "training_seconds": totals.get("training_seconds"),
                "checkpoint_size_bytes": totals.get("adapter_checkpoint_bytes"),
                "peak_mps_allocated_gb": totals.get("peak_mps_allocated_gb"),
                "peak_rss_gb": totals.get("peak_rss_gb"),
            }
        )
    return rows


def generate_plots(results_dir: Path, rows: list[dict]) -> list[Path]:
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    complete = [row for row in rows if row.get("training_seconds")]

    def scatter_with_frontier(
        ax, x_values, y_values, labels, front_labels, x_label, y_label, title
    ):
        ax.scatter(x_values, y_values, zorder=3)
        for x, y, label in zip(x_values, y_values, labels):
            ax.annotate(label, (x, y), fontsize=6, xytext=(3, 3),
                        textcoords="offset points")
        front = sorted(
            [(x, y) for x, y, label in zip(x_values, y_values, labels)
             if label in front_labels]
        )
        if front:
            ax.plot([p[0] for p in front], [p[1] for p in front],
                    "r--", alpha=0.7, zorder=2, label="Pareto frontier")
            ax.legend()
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.grid(alpha=0.3)

    # 1. Quality vs training time (+ frontier).
    fig, ax = plt.subplots(figsize=(8, 5))
    scatter_with_frontier(
        ax,
        [row["training_seconds"] for row in complete],
        [row["final_avg_exact_match"] for row in complete],
        [row["config_id"] for row in complete],
        {p.label for p in nondominated_set(pareto_points(complete, "training_seconds"))},
        "Measured cumulative training time (s)",
        "Final average exact match",
        "E2 Pareto frontier: quality vs training time",
    )
    fig.tight_layout()
    path = plots_dir / "pareto_time.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    generated.append(path)

    # 2. Quality vs training tokens (+ frontier).
    fig, ax = plt.subplots(figsize=(8, 5))
    scatter_with_frontier(
        ax,
        [row["total_tokens"] / 1e3 for row in complete],
        [row["final_avg_exact_match"] for row in complete],
        [row["config_id"] for row in complete],
        {p.label for p in nondominated_set(pareto_points(complete, "total_tokens"))},
        "Measured training tokens (thousands)",
        "Final average exact match",
        "E2 Pareto frontier: quality vs token budget",
    )
    fig.tight_layout()
    path = plots_dir / "pareto_tokens.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    generated.append(path)

    # 3. Quality vs LoRA rank (per budget).
    fig, ax = plt.subplots(figsize=(7, 5))
    budgets = sorted({row["examples_processed"] for row in complete if row["examples_processed"]})
    for budget in budgets:
        subset = sorted(
            [row for row in complete if row["examples_processed"] == budget],
            key=lambda row: row["lora_rank"],
        )
        ax.plot(
            [row["lora_rank"] for row in subset],
            [row["final_avg_exact_match"] for row in subset],
            marker="o",
            label=f"budget {budget}/event",
        )
    ax.set_xlabel("LoRA rank")
    ax.set_ylabel("Final average exact match")
    ax.set_title("E2: quality vs LoRA rank")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = plots_dir / "quality_vs_rank.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    generated.append(path)

    # 4. Quality vs training budget (per rank).
    fig, ax = plt.subplots(figsize=(7, 5))
    ranks = sorted({row["lora_rank"] for row in complete})
    for rank in ranks:
        subset = sorted(
            [row for row in complete if row["lora_rank"] == rank],
            key=lambda row: row["examples_processed"],
        )
        ax.plot(
            [row["examples_processed"] for row in subset],
            [row["final_avg_exact_match"] for row in subset],
            marker="o",
            label=f"rank {rank}",
        )
    ax.set_xlabel("Training examples per event")
    ax.set_ylabel("Final average exact match")
    ax.set_title("E2: quality vs training budget")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = plots_dir / "quality_vs_budget.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    generated.append(path)

    # 5. Quality gain per training second (efficiency).
    with_efficiency = [row for row in complete
                       if row.get("quality_gain_per_training_second") is not None]
    if with_efficiency:
        fig, ax = plt.subplots(figsize=(8, 5))
        labels = [row["config_id"] for row in with_efficiency]
        values = [row["quality_gain_per_training_second"] * 1000 for row in with_efficiency]
        ax.bar(labels, values)
        ax.set_ylabel("EM gain per 1000 training seconds")
        ax.set_title("E2: quality improvement per unit of measured compute")
        ax.tick_params(axis="x", rotation=60)
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        path = plots_dir / "efficiency.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        generated.append(path)

    # 6. Forgetting vs resource consumption.
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(
        [row["training_seconds"] for row in complete],
        [row["average_forgetting"] for row in complete],
    )
    for row in complete:
        ax.annotate(
            row["config_id"],
            (row["training_seconds"], row["average_forgetting"]),
            fontsize=6, xytext=(3, 3), textcoords="offset points",
        )
    ax.set_xlabel("Measured cumulative training time (s)")
    ax.set_ylabel("Average forgetting")
    ax.set_title("E2: forgetting vs resource consumption")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = plots_dir / "forgetting_vs_resources.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    generated.append(path)

    return generated


def build_summary(results_dir: Path, rows: list[dict]) -> dict:
    complete = [row for row in rows if row.get("training_seconds")]
    time_points = pareto_points(complete, "training_seconds")
    token_points = pareto_points(complete, "total_tokens")
    return {
        "n_configurations_completed": len(complete),
        "n_configurations_total": len(rows),
        "pareto_frontier_by_training_time": [
            p.label for p in nondominated_set(time_points)
        ],
        "pareto_frontier_by_tokens": [p.label for p in nondominated_set(token_points)],
        "dominated_configurations": sorted(
            {p.label for p in dominated_set(time_points)}
            & {p.label for p in dominated_set(token_points)}
        ),
        "best_final_avg_exact_match": (
            max(complete, key=lambda row: row["final_avg_exact_match"])["config_id"]
            if complete
            else None
        ),
        "most_efficient_per_second": (
            max(
                (row for row in complete
                 if row.get("quality_gain_per_training_second") is not None),
                key=lambda row: row["quality_gain_per_training_second"],
                default=None,
            )["config_id"]
            if complete
            else None
        ),
        "cost_note": (
            "All resource numbers are wall-clock seconds and token counts "
            "measured locally on Apple M3/MPS. No cloud dollar cost is "
            "implied; monetary conversion requires an explicit hardware and "
            "price assumption and would be a modeled estimate."
        ),
    }


def write_readme(results_dir: Path, rows: list[dict], summary: dict) -> Path:
    complete = sorted(
        [row for row in rows if row.get("training_seconds")],
        key=lambda row: row["training_seconds"],
    )
    table_lines = [
        "| config | rank | examples/event | trainable params | final avg EM | EM gain/s | avg forgetting | BWT (conv.) | train time (s) | tokens | peak MPS (GB) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in complete:
        gain_per_s = row.get("quality_gain_per_training_second")
        gain_display = (
            f"{gain_per_s * 1000:.4f}" if gain_per_s is not None else "n/a"
        )
        table_lines.append(
            f"| {row['config_id']} | {row['lora_rank']} | {row['examples_processed']} | "
            f"{row['trainable_parameters']} | {row['final_avg_exact_match']:.4f} | "
            f"{gain_display} | "
            f"{row['average_forgetting']:.4f} | {row['backward_transfer_conventional']:.4f} | "
            f"{row['training_seconds']:.1f} | {row['total_tokens']} | "
            f"{row['peak_mps_allocated_gb']:.2f} |"
        )

    content = f"""# E2 Run: {results_dir.name}

Cost-quality Pareto frontier over LoRA rank (4/8/16/32) x per-event
training budget (125/250/500 examples = optimizer steps; batch size 1,
gradient accumulation 1, verified by `assert_budget_semantics`).

All numbers are **measured**. Every configuration started from a fresh
pretrained base; within a configuration, event k>1 continued the
configuration's own event k-1 adapter. The shared baseline row is the
pretrained model without an adapter.

## Results

{chr(10).join(table_lines)}

## Pareto frontiers (computed from measured points)

- By training time: `{summary['pareto_frontier_by_training_time']}`
- By training tokens: `{summary['pareto_frontier_by_tokens']}`
- Dominated (by both axes): `{summary['dominated_configurations']}`

{summary['cost_note']}

## Artifacts

- `configurations.csv` — grid definition per configuration
- `metrics.csv` — long-format quality per configuration x state x test event
- `continual_learning_matrix.csv` — per-configuration accuracy matrices + CL metrics
- `resource_metrics.csv` — per-event and total resource measurements
- `pareto_frontier.csv` — dominance flags on both resource axes
- `summary.json`, `predictions/<config>/`, `plots/`
- `../e1_audit.md` — E1 audit (BWT definitions, leakage analysis, consistency checks)

## Rerun

```bash
uv run python scripts/e2_run.py --run-id {results_dir.name}  # full grid
uv run python scripts/e2_analyze.py --results-dir {results_dir}
```
"""
    path = results_dir / "README.md"
    path.write_text(content, encoding="utf-8")
    return path


def analyze_run(results_dir: str | Path) -> dict:
    results_dir = Path(results_dir)
    results = load_configuration_results(results_dir)
    rows = configuration_table(results_dir, results)

    configurations_path = write_csv(results_dir / "configurations.csv", rows)
    metrics_path = write_csv(results_dir / "metrics.csv", metrics_rows(results))
    resource_path = write_csv(results_dir / "resource_metrics.csv", resource_rows(results))

    cl_rows = []
    for config_id, payload in results.items():
        event_metrics = payload["event_metrics"]
        cl = event_metrics["continual_learning"]
        for state_index, state in enumerate(event_metrics["states"]):
            cl_rows.append(
                {
                    "config_id": config_id,
                    "state": state,
                    **{
                        f"test_event_{j}": event_metrics["accuracy_matrix"][state_index][k]
                        for k, j in enumerate(event_metrics["test_events"])
                    },
                    "average_accuracy": average_accuracy(
                        event_metrics["accuracy_matrix"], state_index
                    ),
                }
            )
        cl_rows.append(
            {
                "config_id": config_id,
                "state": "forgetting_per_event",
                **cl["forgetting_per_event"],
                "average_accuracy": cl["average_forgetting"],
            }
        )
        cl_rows.append(
            {
                "config_id": config_id,
                "state": "backward_transfer",
                "test_event_1": cl["backward_transfer_mean_all_events"],
                "test_event_2": cl["backward_transfer_conventional"],
                "average_accuracy": None,
            }
        )
    cl_path = write_csv(results_dir / "continual_learning_matrix.csv", cl_rows)

    pareto_time = pareto_table(rows, "training_seconds")
    pareto_tokens = pareto_table(rows, "total_tokens")
    pareto_path = write_csv(results_dir / "pareto_frontier.csv", pareto_time + pareto_tokens)

    summary = build_summary(results_dir, rows)
    write_json(results_dir / "summary.json", summary)
    plots = generate_plots(results_dir, rows)
    readme = write_readme(results_dir, rows, summary)

    return {
        "rows": rows,
        "summary": summary,
        "files": {
            "configurations": str(configurations_path),
            "metrics": str(metrics_path),
            "continual_learning": str(cl_path),
            "resource_metrics": str(resource_path),
            "pareto_frontier": str(pareto_path),
            "summary": str(results_dir / "summary.json"),
            "readme": str(readme),
            "plots": [str(p) for p in plots],
        },
    }
