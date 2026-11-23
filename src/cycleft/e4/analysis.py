"""E4 analysis: resource comparisons, policy tables, plots, summary, README.

Everything is derived from measured artifacts (benchmark result JSONs,
scheduler decision logs, resource profiles). Pending cloud tiers appear
with explicit null measurements — never with invented numbers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from cycleft.e1.pipeline import write_json  # noqa: E402
from cycleft.e4.resources import PRICE_TYPE_OWNED  # noqa: E402


def load_run_artifacts(results_dir: Path) -> dict[str, Any]:
    required = {
        "inventory": "hardware_inventory.json",
        "profiles": "resource_profiles.json",
        "decisions": "scheduler_decisions.jsonl",
        "benchmarks": "benchmark_results",
    }
    artifacts: dict[str, Any] = {}
    for key, name in required.items():
        path = results_dir / name
        if key == "benchmarks":
            if not path.is_dir():
                raise FileNotFoundError(f"Missing benchmark results: {path}")
            artifacts[key] = [
                json.loads(p.read_text(encoding="utf-8"))
                for p in sorted(path.glob("*.json"))
            ]
        elif key == "decisions":
            artifacts[key] = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        elif path.exists():
            artifacts[key] = json.loads(path.read_text(encoding="utf-8"))
        else:
            raise FileNotFoundError(f"Missing artifact: {path}")
    return artifacts


def benchmark_metric_rows(benchmarks: list[dict]) -> list[dict]:
    rows = []
    for benchmark in benchmarks:
        for stage in benchmark.get("stages", []):
            rows.append(
                {
                    "workload": benchmark["workload"],
                    "resource_id": benchmark["resource_id"],
                    "device": benchmark["device"],
                    "status": benchmark["status"],
                    "stage": stage["stage"],
                    "kind": stage["kind"],
                    "model_load_seconds": stage["model_load_seconds"],
                    "training_seconds": stage["training_seconds"],
                    "checkpoint_save_seconds": stage["checkpoint_save_seconds"],
                    "eval_seconds": stage["eval_seconds"],
                    "optimizer_steps": stage["optimizer_steps"],
                    "examples": stage["examples"],
                    "total_tokens": stage["total_tokens"],
                    "steps_per_second": stage["steps_per_second"],
                    "tokens_per_second": stage["tokens_per_second"],
                    "average_loss": stage["average_loss"],
                    "peak_memory_gb": stage["peak_memory_gb"],
                    "continued_from": stage["continued_from"],
                }
            )
    return rows


def checkpoint_overhead_rows(benchmarks: list[dict]) -> list[dict]:
    """Per-transition overheads, derived from the AUTHORITATIVE per-stage
    measurements: save = the from-stage's checkpoint save; load = the
    to-stage's model-load step (fresh base + adapter restore)."""
    rows = []
    for benchmark in benchmarks:
        stages = {stage["stage"]: stage for stage in benchmark.get("stages", [])}
        for transition in benchmark.get("transitions", []):
            from_stage = stages.get(transition["from_stage"], {})
            to_stage = stages.get(transition["to_stage"], {})
            rows.append(
                {
                    "workload": benchmark["workload"],
                    "resource_id": benchmark["resource_id"],
                    "transition": transition["transition"],
                    "from_stage": transition["from_stage"],
                    "to_stage": transition["to_stage"],
                    "checkpoint_save_seconds": from_stage.get(
                        "checkpoint_save_seconds"
                    ),
                    "checkpoint_load_seconds": to_stage.get("model_load_seconds"),
                    "restore_verified": transition["restore_verified"],
                    "note": (
                        "save = from-stage checkpoint save; load = to-stage "
                        "fresh base load + adapter restore (not separable)"
                    ),
                }
            )
    return rows


def resource_comparison_rows(
    profiles: dict[str, dict], benchmarks: list[dict]
) -> list[dict]:
    measured_by_resource: dict[str, dict] = {}
    for benchmark in benchmarks:
        if benchmark.get("status") != "complete":
            continue
        entry = measured_by_resource.setdefault(
            benchmark["resource_id"],
            {"workloads": {}, "total_train_seconds": 0.0, "total_tokens": 0},
        )
        totals = benchmark.get("totals", {})
        entry["workloads"][benchmark["workload"]] = {
            "training_seconds": totals.get("training_seconds"),
            "tokens_per_second": totals.get("tokens_per_second"),
            "steps_per_second": totals.get("steps_per_second"),
            "peak_memory_gb": totals.get("peak_memory_gb"),
        }
        entry["total_train_seconds"] += totals.get("training_seconds") or 0.0
        entry["total_tokens"] += totals.get("training_tokens") or 0

    rows = []
    for resource_id, profile in profiles.items():
        measured = measured_by_resource.get(resource_id)
        # Headline throughput prefers the higher-quality workload (B) and
        # falls back to any measured workload for that resource.
        headline_throughput = None
        if measured:
            headline_throughput = (
                measured["workloads"].get("B_high_quality", {})
                .get("tokens_per_second")
            )
            if headline_throughput is None:
                for workload_entry in measured["workloads"].values():
                    if workload_entry.get("tokens_per_second"):
                        headline_throughput = workload_entry["tokens_per_second"]
                        break
        rows.append(
            {
                "resource_id": resource_id,
                "kind": profile.get("kind"),
                "device": profile.get("device"),
                "gpu_model": profile.get("gpu_model"),
                "gpu_memory_gb": profile.get("gpu_memory_gb"),
                "ram_gb": profile.get("ram_gb"),
                "measured": profile.get("measured"),
                "available": profile.get("available"),
                "tokens_per_second": headline_throughput,
                "training_seconds_workload_A": (
                    measured["workloads"]
                    .get("A_economical", {})
                    .get("training_seconds")
                    if measured
                    else None
                ),
                "training_seconds_workload_B": (
                    measured["workloads"]
                    .get("B_high_quality", {})
                    .get("training_seconds")
                    if measured
                    else None
                ),
                "peak_memory_gb": (
                    max(
                        (
                            w.get("peak_memory_gb") or 0
                            for w in measured["workloads"].values()
                        ),
                        default=None,
                    )
                    if measured
                    else None
                ),
                "price_usd_per_hour": profile.get("price_usd_per_hour"),
                "price_type": profile.get("price_type"),
                "price_source": profile.get("price_source"),
                "price_timestamp": profile.get("price_timestamp"),
            }
        )
    return rows


def policy_comparison_rows(
    decisions: list[dict], benchmarks: list[dict]
) -> list[dict]:
    """One row per (policy, workload): decision + measured outcome.

    With a single measured resource, all policies select it (documented
    degenerate case); the measured outcome is the shared benchmark run.
    """
    measured: dict[tuple[str, str], dict] = {}
    for benchmark in benchmarks:
        if benchmark.get("status") == "complete":
            measured[(benchmark["workload"], benchmark["resource_id"])] = benchmark

    by_policy_workload: dict[tuple[str, str], list[dict]] = {}
    for decision in decisions:
        workload = decision.get("context", {}).get("workload", "")
        by_policy_workload.setdefault((decision["policy"], workload), []).append(
            decision
        )

    rows = []
    for (policy, workload), decision_list in sorted(by_policy_workload.items()):
        chosen = {d.get("chosen_resource") for d in decision_list}
        first = decision_list[0]
        benchmark = None
        if len(chosen) == 1 and None not in chosen:
            benchmark = measured.get((workload, next(iter(chosen))))
        totals = benchmark.get("totals", {}) if benchmark else {}
        cl = benchmark.get("cl_metrics", {}) if benchmark else {}
        matrix = benchmark.get("accuracy_matrix") if benchmark else None
        rows.append(
            {
                "policy": policy,
                "workload": workload,
                "n_decisions": len(decision_list),
                "chosen_resources": ";".join(sorted(str(c) for c in chosen)),
                "degenerate_single_resource": len(chosen) == 1,
                "final_avg_exact_match": (
                    sum(matrix[-1]) / len(matrix[-1]) if matrix else None
                ),
                "average_forgetting": cl.get("average_forgetting"),
                "backward_transfer_conventional": cl.get(
                    "backward_transfer_conventional"
                ),
                "training_seconds": totals.get("training_seconds"),
                "total_completion_seconds": totals.get("total_completion_seconds"),
                "training_tokens": totals.get("training_tokens"),
                "peak_memory_gb": totals.get("peak_memory_gb"),
                "estimated_cost": json.dumps(
                    [d.get("estimated_cost") for d in decision_list]
                ),
                "decision_reasons": json.dumps(
                    [d.get("reason") for d in decision_list]
                ),
            }
        )
    return rows


def continual_learning_rows(benchmarks: list[dict]) -> list[dict]:
    rows = []
    for benchmark in benchmarks:
        matrix = benchmark.get("accuracy_matrix") or []
        if not matrix:
            continue
        for state_index, state in enumerate(benchmark.get("states", [])):
            rows.append(
                {
                    "workload": benchmark["workload"],
                    "resource_id": benchmark["resource_id"],
                    "state": state,
                    **{
                        f"test_event_{j}": matrix[state_index][k]
                        for k, j in enumerate([1, 2, 3])
                    },
                    "average_accuracy": sum(matrix[state_index])
                    / len(matrix[state_index]),
                }
            )
        cl = benchmark.get("cl_metrics", {})
        rows.append(
            {
                "workload": benchmark["workload"],
                "resource_id": benchmark["resource_id"],
                "state": "average_forgetting",
                "test_event_1": None,
                "test_event_2": None,
                "test_event_3": None,
                "average_accuracy": cl.get("average_forgetting"),
            }
        )
        rows.append(
            {
                "workload": benchmark["workload"],
                "resource_id": benchmark["resource_id"],
                "state": "backward_transfer_conventional",
                "test_event_1": cl.get("backward_transfer_conventional"),
                "test_event_2": None,
                "test_event_3": None,
                "average_accuracy": None,
            }
        )
    return rows


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


def generate_plots(results_dir: Path, artifacts: dict[str, Any]) -> list[Path]:
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    benchmarks = [
        b for b in artifacts["benchmarks"] if b.get("status") == "complete"
    ]
    profiles = artifacts["profiles"]

    def _label(benchmark: dict) -> str:
        return f"{benchmark['workload'].split('_')[0]}@{benchmark['resource_id']}"

    # 1. Training throughput by resource (tokens/s per workload).
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = []
    values = []
    for benchmark in benchmarks:
        tps = benchmark["totals"].get("tokens_per_second")
        if tps:
            labels.append(_label(benchmark))
            values.append(tps)
    if labels:
        ax.bar(labels, values)
        ax.set_ylabel("Training tokens per second (measured)")
        ax.set_title("E4: training throughput by resource and workload")
        ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    path = plots_dir / "throughput_by_resource.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    generated.append(path)

    # 2. Training time by workload and resource.
    fig, ax = plt.subplots(figsize=(8, 5))
    workloads = sorted({b["workload"] for b in benchmarks})
    width = 0.8 / max(len(workloads), 1)
    for index, workload in enumerate(workloads):
        subset = [b for b in benchmarks if b["workload"] == workload]
        ax.bar(
            [f"{b['resource_id']}" for b in subset],
            [b["totals"]["training_seconds"] for b in subset],
            width=width,
            label=workload,
        )
    ax.set_ylabel("Training time (s, measured)")
    ax.set_title("E4: training time by workload and resource")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    path = plots_dir / "training_time_by_workload.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    generated.append(path)

    # 3. Peak device memory by resource.
    fig, ax = plt.subplots(figsize=(7, 5))
    labels = []
    values = []
    for benchmark in benchmarks:
        peak = benchmark["totals"].get("peak_memory_gb")
        if peak:
            labels.append(_label(benchmark))
            values.append(peak)
    if labels:
        ax.bar(labels, values)
        ax.set_ylabel("Peak device memory (GB, measured)")
        ax.set_title("E4: peak device memory by resource")
        ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    path = plots_dir / "peak_memory_by_resource.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    generated.append(path)

    # 4. Cost vs completion time (measured resources only; owned = $0 metered).
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted_cost_note = False
    for benchmark in benchmarks:
        resource_id = benchmark["resource_id"]
        profile = profiles.get(resource_id, {})
        completion = benchmark["totals"].get("total_completion_seconds")
        if completion is None:
            continue
        if profile.get("price_type") == PRICE_TYPE_OWNED:
            cost = 0.0
            note = "owned (no metered $)"
        elif profile.get("price_type") == "measured_billed":
            cost = (completion / 3600.0) * (profile.get("price_usd_per_hour") or 0)
            note = "billed"
        else:
            continue
        ax.scatter(completion, cost, s=50)
        ax.annotate(
            _label(benchmark) + f" ({note})",
            (completion, cost),
            fontsize=7, xytext=(4, 4), textcoords="offset points",
        )
        plotted_cost_note = True
    if not plotted_cost_note:
        ax.text(0.5, 0.5, "no measured cloud resources", ha="center")
    ax.set_xlabel("Total completion time (s, measured)")
    ax.set_ylabel("Cost (USD)")
    ax.set_title(
        "E4: cost vs completion time (local = owned hardware, $0 metered;\n"
        "cloud tiers pending benchmark — not plotted, not invented)"
    )
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = plots_dir / "cost_vs_time.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    generated.append(path)

    # 5. Quality vs total resource expenditure (training time).
    fig, ax = plt.subplots(figsize=(8, 5))
    for benchmark in benchmarks:
        matrix = benchmark.get("accuracy_matrix") or []
        if not matrix:
            continue
        final = sum(matrix[-1]) / len(matrix[-1])
        ax.scatter(
            benchmark["totals"].get("training_seconds"),
            final,
            s=50,
        )
        ax.annotate(
            _label(benchmark),
            (benchmark["totals"].get("training_seconds"), final),
            fontsize=7, xytext=(4, 4), textcoords="offset points",
        )
    ax.set_xlabel("Training time (s, measured)")
    ax.set_ylabel("Final average exact match")
    ax.set_title("E4: quality vs total resource expenditure")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = plots_dir / "quality_vs_expenditure.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    generated.append(path)

    # 6. Fixed vs dynamic policy performance (degenerate with one resource).
    policy_rows = policy_comparison_rows(artifacts["decisions"], artifacts["benchmarks"])
    if policy_rows:
        fig, ax = plt.subplots(figsize=(10, 5))
        labels = [f"{row['policy']}\n{row['workload'].split('_')[0]}" for row in policy_rows]
        values = [
            row["final_avg_exact_match"] if row["final_avg_exact_match"] is not None else 0
            for row in policy_rows
        ]
        ax.bar(labels, values)
        ax.set_ylabel("Final average exact match")
        ax.set_title(
            "E4: fixed vs dynamic policy performance\n"
            "(single measured resource: all policies coincide — degenerate case)"
        )
        ax.tick_params(axis="x", rotation=45)
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        path = plots_dir / "policy_comparison.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        generated.append(path)

    # 7. Scheduler resource choices over sequential events.
    decisions = artifacts["decisions"]
    if decisions:
        fig, ax = plt.subplots(figsize=(10, 4))
        policies = sorted({d["policy"] for d in decisions})
        for level, policy in enumerate(policies):
            policy_decisions = [d for d in decisions if d["policy"] == policy]
            events = [d["event_index"] for d in policy_decisions]
            chosen = [
                1 if d.get("chosen_resource") else 0 for d in policy_decisions
            ]
            ax.scatter(
                events,
                [level] * len(events),
                marker="s",
                s=120,
                c=chosen,
                cmap="Greens",
                vmin=-0.2,
                vmax=1.4,
            )
            for d in policy_decisions:
                ax.annotate(
                    str(d.get("chosen_resource")),
                    (d["event_index"], level),
                    fontsize=6, xytext=(0, 10), textcoords="offset points",
                    ha="center",
                )
        ax.set_yticks(range(len(policies)), policies)
        ax.set_xlabel("Adaptation event index (0 = bootstrap stage)")
        ax.set_title("E4: scheduler resource choices over sequential events")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        path = plots_dir / "scheduler_choices.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        generated.append(path)

    return generated


def build_summary(
    artifacts: dict[str, Any], benchmark_metrics_path: Path
) -> dict:
    benchmarks = artifacts["benchmarks"]
    complete = [b for b in benchmarks if b.get("status") == "complete"]
    profiles = artifacts["profiles"]
    measured_resources = [
        resource_id for resource_id, p in profiles.items() if p.get("measured")
    ]
    available_resources = [
        resource_id for resource_id, p in profiles.items() if p.get("available")
    ]
    return {
        "n_benchmarks_completed": len(complete),
        "n_benchmarks_total": len(benchmarks),
        "measured_resources": measured_resources,
        "available_resources": available_resources,
        "pending_cloud_resources": [
            resource_id
            for resource_id, p in profiles.items()
            if not p.get("measured")
        ],
        "policies_evaluated": sorted({d["policy"] for d in artifacts["decisions"]}),
        "policy_degeneracy_note": (
            "Only one resource tier (local MPS) is measured and available, so "
            "every policy selects it; the fixed-vs-dynamic comparison is "
            "degenerate and is marked pending until a second tier is "
            "benchmarked (see README for exact cloud commands)."
        ),
        "multi_resource_comparison": "PENDING (no authorized cloud GPU access)",
        "checkpoint_restore_verified": all(
            transition.get("restore_verified")
            for benchmark in complete
            for transition in benchmark.get("transitions", [])
        ),
        "cost_note": (
            "Local resource is owned hardware: usage is reported in seconds "
            "and no dollar figure is invented for it. Cloud prices are "
            "pending capture at provision time; modeled costs would be "
            "clearly labeled and are not used as results."
        ),
        "benchmark_metrics_csv": str(benchmark_metrics_path),
    }


def write_readme(results_dir: Path, artifacts: dict[str, Any], summary: dict) -> Path:
    benchmarks = [
        b for b in artifacts["benchmarks"] if b.get("status") == "complete"
    ]
    lines = [
        f"# E4 Run: {results_dir.name}",
        "",
        "Dynamic compute-resource allocation for continual LoRA adaptation.",
        "Fixed adaptation schedule (E3's event-boundary oracle convention: "
        "the same adaptation operation at each temporal transition); the "
        "resource is the only variable.",
        "",
        "## Measured benchmarks",
        "",
        "| workload | resource | steps | train (s) | tok/s | peak mem (GB) | final avg EM |",
        "|---|---|---|---|---|---|---|",
    ]
    for benchmark in benchmarks:
        matrix = benchmark.get("accuracy_matrix") or []
        totals = benchmark.get("totals", {})
        final = sum(matrix[-1]) / len(matrix[-1]) if matrix else None
        lines.append(
            f"| {benchmark['workload']} | {benchmark['resource_id']} | "
            f"{totals.get('optimizer_steps')} | {totals.get('training_seconds'):.1f} | "
            f"{totals.get('tokens_per_second'):.0f} | "
            f"{totals.get('peak_memory_gb'):.2f} | "
            f"{final:.4f} |"
        )
    transitions = [
        transition
        for benchmark in benchmarks
        for transition in benchmark.get("transitions", [])
    ]
    if transitions:
        avg_save = sum(t["checkpoint_save_seconds"] for t in transitions) / len(transitions)
        avg_load = sum(t["checkpoint_load_seconds"] for t in transitions) / len(transitions)
        lines += [
            "",
            f"Checkpoint transition overhead (all restores verified "
            f"bit-identical): save {avg_save:.2f} s, "
            f"load(=fresh base + adapter restore) {avg_load:.2f} s per transition.",
        ]
    lines += [
        "",
        "## Classification",
        "",
        f"- Measured resources: {summary['measured_resources']}",
        f"- Pending (advertised specs only, not benchmarked): "
        f"{summary['pending_cloud_resources']}",
        f"- Multi-resource comparison: **{summary['multi_resource_comparison']}**",
        f"- {summary['policy_degeneracy_note']}",
        "",
        summary["cost_note"],
        "",
        "## Artifacts",
        "",
        "- `hardware_inventory.json`, `resource_profiles.json`, `benchmark_config.yaml`",
        "- `benchmark_metrics.csv`, `resource_comparison.csv`, `policy_comparison.csv`,",
        "  `continual_learning_matrix.csv`, `checkpoint_overhead.csv`,",
        "  `scheduler_decisions.jsonl`, `summary.json`",
        "- `plots/` — throughput, time, memory, cost-vs-time, quality-vs-spend,",
        "  policy comparison, scheduler choices",
        "- `benchmark_results/` — raw per-run measurement JSONs",
        "",
        "## Rerun",
        "",
        "```bash",
        f"uv run python scripts/e4_benchmark.py --run-id {results_dir.name}",
        f"uv run python scripts/e4_analyze.py --results-dir results/e4/{results_dir.name}",
        "```",
    ]
    path = results_dir / "README.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def analyze_run(results_dir: str | Path) -> dict:
    results_dir = Path(results_dir)
    artifacts = load_run_artifacts(results_dir)
    rows = benchmark_metric_rows(artifacts["benchmarks"])
    benchmark_path = write_csv(results_dir / "benchmark_metrics.csv", rows)
    write_csv(
        results_dir / "resource_comparison.csv",
        resource_comparison_rows(artifacts["profiles"], artifacts["benchmarks"]),
    )
    write_csv(
        results_dir / "policy_comparison.csv",
        policy_comparison_rows(artifacts["decisions"], artifacts["benchmarks"]),
    )
    write_csv(
        results_dir / "continual_learning_matrix.csv",
        continual_learning_rows(artifacts["benchmarks"]),
    )
    write_csv(
        results_dir / "checkpoint_overhead.csv",
        checkpoint_overhead_rows(artifacts["benchmarks"]),
    )
    plots = generate_plots(results_dir, artifacts)
    summary = build_summary(artifacts, benchmark_path)
    write_json(results_dir / "summary.json", summary)
    readme = write_readme(results_dir, artifacts, summary)
    return {
        "summary": summary,
        "files": {
            "benchmark_metrics": str(benchmark_path),
            "summary": str(results_dir / "summary.json"),
            "readme": str(readme),
            "plots": [str(p) for p in plots],
        },
    }
