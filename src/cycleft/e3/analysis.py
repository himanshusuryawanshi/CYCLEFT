"""E3 analysis: aggregate strategy results into tables, plots, summary, README.

Everything is derived from measured artifacts written by the runner; missing
strategies are reported as failed, never estimated. With only two known
drift boundaries, detection comparisons are reported with confusion counts
and explicitly qualified.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from cycleft.e1.metrics import average_accuracy  # noqa: E402
from cycleft.e1.pipeline import write_json  # noqa: E402
from cycleft.e3.stream import EXAMPLES_PER_EVENT  # noqa: E402


def load_strategy_results(results_dir: Path) -> dict[str, dict]:
    results: dict[str, dict] = {}
    for path in sorted(results_dir.glob("*/strategy_result.json")):
        strategy = path.parent.name
        results[strategy] = json.loads(path.read_text(encoding="utf-8"))
    if not results:
        raise FileNotFoundError(f"No strategy results under {results_dir}")
    # Attach per-window monitoring curves from the window logs.
    for strategy, payload in results.items():
        window_log_path = results_dir / strategy / "window_log.json"
        if window_log_path.exists():
            windows = json.loads(window_log_path.read_text(encoding="utf-8"))[
                "windows"
            ]
            payload["window_log"] = windows
            payload["error_curve"] = [
                (w["position"], w["mean_error"])
                for w in windows
                if w.get("mean_error") is not None
            ]
            payload["mmd_curve"] = [
                (w["position"], w["mmd2_vs_reference"])
                for w in windows
                if w.get("mmd2_vs_reference") is not None
            ]
    return results


def trigger_metric_rows(results: dict[str, dict], tolerance: int) -> list[dict]:
    rows = []
    for strategy, payload in results.items():
        detection = payload.get("detection", {})
        rows.append(
            {
                "strategy": strategy,
                "regime": payload.get("regime", ""),
                "status": payload.get("status", ""),
                "n_adaptations": detection.get("n_triggers_total"),
                "true_positives": detection.get("true_positives"),
                "false_positives": detection.get("false_positives"),
                "missed": detection.get("missed"),
                "recall": detection.get("recall"),
                "detection_delays_examples": json.dumps(
                    detection.get("detection_delays_examples", {})
                ),
                "fp_rate_per_monitored_window": detection.get(
                    "fp_rate_per_monitored_window"
                ),
                "early_warning_positions": json.dumps(
                    detection.get("early_warning_positions", [])
                ),
                "false_positive_positions": json.dumps(
                    detection.get("false_positive_positions", [])
                ),
                "tolerance_examples": tolerance,
            }
        )
    return rows


def adaptation_metric_rows(results: dict[str, dict]) -> list[dict]:
    always = results.get("always_adapt", {})
    always_n = always.get("resources", {}).get("n_adaptations") or 0
    rows = []
    for strategy, payload in results.items():
        resources = payload.get("resources", {})
        if not resources:
            rows.append({"strategy": strategy, "status": payload.get("status")})
            continue
        rows.append(
            {
                "strategy": strategy,
                "status": payload.get("status", ""),
                "n_adaptations": resources.get("n_adaptations"),
                "optimizer_steps": resources.get("training_steps"),
                "training_examples": resources.get("training_examples"),
                "training_tokens": resources.get("training_tokens"),
                "training_seconds": resources.get("training_seconds"),
                "stream_seconds": resources.get("stream_seconds"),
                "eval_seconds": resources.get("eval_seconds"),
                "wall_seconds_including_eval": resources.get(
                    "wall_seconds_including_eval"
                ),
                "peak_mps_allocated_gb": resources.get("peak_mps_allocated_gb"),
                "peak_rss_gb": resources.get("peak_rss_gb"),
                "adapter_checkpoint_bytes": resources.get(
                    "adapter_checkpoint_bytes"
                ),
                "avoided_adaptations_vs_always_adapt": (
                    always_n - resources.get("n_adaptations", 0)
                    if always_n
                    else None
                ),
            }
        )
    return rows


def quality_rows(results: dict[str, dict]) -> list[dict]:
    rows = []
    for strategy, payload in results.items():
        matrix = payload.get("accuracy_matrix") or []
        if not matrix:
            continue
        cl = payload.get("continual_learning", {})
        final_avg = average_accuracy(matrix, len(matrix) - 1)
        final_f1 = sum(payload["token_f1_matrix"][-1]) / len(payload["token_f1_matrix"][-1])
        rows.append(
            {
                "strategy": strategy,
                "final_avg_exact_match": final_avg,
                "final_avg_token_f1": final_f1,
                "accuracy_matrix": json.dumps(matrix),
                "average_forgetting": cl.get("average_forgetting"),
                "backward_transfer_conventional": cl.get(
                    "backward_transfer_conventional"
                ),
                "quality_gain_over_no_adapt": payload.get("efficiency", {}).get(
                    "quality_gain_over_no_adapt"
                ),
                "quality_gain_per_training_second": payload.get("efficiency", {}).get(
                    "quality_gain_per_training_second"
                ),
                "quality_gain_per_adaptation": payload.get("efficiency", {}).get(
                    "quality_gain_per_adaptation"
                ),
            }
        )
    return rows


def continual_learning_rows(results: dict[str, dict]) -> list[dict]:
    rows = []
    for strategy, payload in results.items():
        matrix = payload.get("accuracy_matrix") or []
        if not matrix:
            continue
        for state_index, state in enumerate(payload.get("states", [])):
            rows.append(
                {
                    "strategy": strategy,
                    "state": state,
                    **{
                        f"test_event_{j}": matrix[state_index][k]
                        for k, j in enumerate([1, 2, 3])
                    },
                    "average_accuracy": average_accuracy(matrix, state_index),
                }
            )
        cl = payload.get("continual_learning", {})
        rows.append(
            {
                "strategy": strategy,
                "state": "forgetting_per_event",
                **cl.get("forgetting_per_event", {}),
                "average_accuracy": cl.get("average_forgetting"),
            }
        )
        rows.append(
            {
                "strategy": strategy,
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


def generate_plots(
    results_dir: Path,
    results: dict[str, dict],
    stream_boundaries: list[int],
    window_size: int,
) -> list[Path]:
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    def _boundary_lines(ax):
        for boundary in stream_boundaries:
            ax.axvline(
                boundary, color="black", linestyle=":", alpha=0.7,
                label="known boundary" if boundary == stream_boundaries[0] else None,
            )

    # 1. Prequential error signal over stream time.
    error_strategies = [
        (strategy, payload)
        for strategy, payload in results.items()
        if payload.get("error_curve")
    ]
    if error_strategies:
        fig, ax = plt.subplots(figsize=(11, 5))
        all_errors = [
            value
            for _, payload in error_strategies
            for _, value in payload["error_curve"]
        ]
        lo, hi = min(all_errors), max(all_errors)
        marker_y = lo - 0.08 * (hi - lo + 1e-9)
        for strategy, payload in error_strategies:
            positions = [p for p, _ in payload["error_curve"]]
            values = [v for _, v in payload["error_curve"]]
            ax.plot(positions, values, alpha=0.35, linewidth=0.5, label=strategy)
            adaptations = payload.get("adaptation_positions", [])
            if adaptations:
                ax.plot(
                    adaptations,
                    [marker_y] * len(adaptations),
                    "|",
                    markersize=12,
                    label=f"{strategy} adaptations",
                )
        _boundary_lines(ax)
        ax.set_xlabel("Stream position (examples)")
        ax.set_ylabel("Prequential 0-1 prediction error")
        ax.set_title("E3: monitoring signal (prequential error) vs stream time")
        ax.legend(fontsize=6, ncol=2)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        path = plots_dir / "error_signal.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        generated.append(path)

    # 2. MMD statistic over time.
    mmd_payloads = [
        (strategy, payload)
        for strategy, payload in results.items()
        if payload.get("mmd_curve")
    ]
    if mmd_payloads:
        fig, ax = plt.subplots(figsize=(11, 5))
        for strategy, payload in mmd_payloads:
            positions = [p for p, _ in payload["mmd_curve"]]
            values = [v for _, v in payload["mmd_curve"]]
            ax.plot(positions, values, marker="o", markersize=3, label=strategy)
        _boundary_lines(ax)
        ax.set_xlabel("Stream position (examples)")
        ax.set_ylabel("MMD^2 (recent window vs reference)")
        ax.set_title("E3: MMD statistic over stream time")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        path = plots_dir / "mmd_signal.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        generated.append(path)

    # 3. ADWIN alarms over time.
    adwin_rows = [
        (strategy, payload)
        for strategy, payload in results.items()
        if strategy in ("adwin", "combined")
    ]
    if adwin_rows:
        fig, ax = plt.subplots(figsize=(11, 3))
        for level, (strategy, payload) in enumerate(adwin_rows):
            adaptations = payload.get("adaptation_positions", [])
            ax.plot(
                adaptations,
                [level] * len(adaptations),
                "|",
                markersize=14,
                label=f"{strategy} adaptations",
            )
        _boundary_lines(ax)
        ax.set_yticks(range(len(adwin_rows)), [s for s, _ in adwin_rows])
        ax.set_xlabel("Stream position (examples)")
        ax.set_title("E3: ADWIN-based adaptation events over stream time")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        path = plots_dir / "adwin_alarms.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        generated.append(path)

    # 4. Detection delay per method.
    strategies = [s for s, p in results.items() if p.get("detection")]
    delays = []
    for strategy in strategies:
        delay_map = results[strategy]["detection"].get(
            "detection_delays_examples", {}
        )
        delays.append(
            max(delay_map.values(), default=0) if delay_map else 0
        )
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(strategies, delays)
    ax.set_ylabel("Max detection delay (examples)")
    ax.set_title("E3: detection delay per method (tolerance-window detections)")
    ax.tick_params(axis="x", rotation=60)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    path = plots_dir / "detection_delay.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    generated.append(path)

    # 5. False positives per method.
    fig, ax = plt.subplots(figsize=(8, 4))
    fps = [
        results[s]["detection"].get("false_positives", 0) for s in strategies
    ]
    ax.bar(strategies, fps)
    ax.set_ylabel("False positives (stationarity assumption)")
    ax.set_title("E3: false positives per method")
    ax.tick_params(axis="x", rotation=60)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    path = plots_dir / "false_positives.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    generated.append(path)

    # 6. Final quality vs adaptation time.
    fig, ax = plt.subplots(figsize=(8, 5))
    for strategy, payload in results.items():
        resources = payload.get("resources", {})
        matrix = payload.get("accuracy_matrix") or []
        if not resources or not matrix:
            continue
        final_avg = average_accuracy(matrix, len(matrix) - 1)
        ax.scatter(
            resources.get("training_seconds", 0),
            final_avg,
            label=strategy,
            s=42,
        )
        ax.annotate(strategy, (
            resources.get("training_seconds", 0), final_avg
        ), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Total adaptation training time (s, measured)")
    ax.set_ylabel("Final average exact match")
    ax.set_title("E3: final quality vs adaptation expenditure")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = plots_dir / "quality_vs_time.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    generated.append(path)

    # 7. Quality-cost comparison.
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ordered = sorted(results.keys())
    em_values = [
        average_accuracy(results[s]["accuracy_matrix"], len(results[s]["accuracy_matrix"]) - 1)
        if results[s].get("accuracy_matrix") else 0
        for s in ordered
    ]
    n_adaptations = [
        results[s].get("resources", {}).get("n_adaptations", 0) for s in ordered
    ]
    ax1.bar(ordered, em_values, color="tab:blue", label="final avg EM")
    ax1.set_ylabel("Final average exact match")
    ax1.tick_params(axis="x", rotation=60)
    ax2 = ax1.twinx()
    ax2.plot(
        ordered, n_adaptations, "o--", color="tab:red", label="# adaptations"
    )
    ax2.set_ylabel("Number of adaptations")
    ax1.set_title("E3: quality and adaptation count per strategy")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
    ax1.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    path = plots_dir / "quality_cost.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    generated.append(path)

    return generated


def build_summary(
    results: dict[str, dict],
    tolerance: int,
    n_boundaries: int,
    n_windows: int,
) -> dict:
    complete = {
        strategy: payload
        for strategy, payload in results.items()
        if payload.get("status") == "complete"
    }
    best = None
    if complete:
        best = max(
            complete.items(),
            key=lambda item: average_accuracy(
                item[1]["accuracy_matrix"], len(item[1]["accuracy_matrix"]) - 1
            ),
        )[0]
    return {
        "n_strategies_completed": len(complete),
        "n_strategies_total": len(results),
        "n_known_boundaries": n_boundaries,
        "detection_tolerance_examples": tolerance,
        "best_final_avg_exact_match": best,
        "quality_by_strategy": {
            strategy: average_accuracy(
                payload["accuracy_matrix"], len(payload["accuracy_matrix"]) - 1
            )
            for strategy, payload in complete.items()
        },
        "adaptations_by_strategy": {
            strategy: payload["resources"].get("n_adaptations")
            for strategy, payload in complete.items()
        },
        "statistical_caveat": (
            "Only two genuine drift boundaries exist; detection recall and "
            "delay comparisons rest on two positive cases and must not be "
            "read as statistically significant detector superiority. False "
            "positives inside events assume within-event stationarity, which "
            "the dataset does not verify."
        ),
        "n_windows": n_windows,
    }


def write_readme(
    results_dir: Path,
    results: dict[str, dict],
    summary: dict,
    config: dict,
) -> Path:
    lines = [
        f"# E3 Run: {results_dir.name}",
        "",
        "Trigger-based adaptation control on the TemporalWiki 3-event stream.",
        "All numbers are **measured**; the stream order is the E1/E2 seed-42",
        "training order; evaluation reuses the exact E1/E2 held-out protocol.",
        "",
        f"- Known drift boundaries: {summary['n_known_boundaries']} "
        f"(stream positions {EXAMPLES_PER_EVENT} and {2 * EXAMPLES_PER_EVENT}).",
        f"- Detection tolerance: {summary['detection_tolerance_examples']} examples "
        "after a boundary (fixed a priori).",
        f"- Adaptation action: {config.get('adaptation_action')}.",
        f"- Strategies completed: {summary['n_strategies_completed']}/"
        f"{summary['n_strategies_total']}.",
        "",
        "## Integrity notes",
        "",
        f"- {summary['statistical_caveat']}",
        "- Regime-A detectors (MMD) receive question embeddings only; answers",
        "  never enter their inputs (enforced by the signal types and tested).",
        "- Regime-B detectors receive prequential losses computed before any",
        "  training on the scored example (predict-then-score order).",
        "",
        "## Artifacts",
        "",
        "- `dataset_audit.md` — temporal data audit",
        "- `detector_config.yaml` — full detector + adaptation configuration",
        "- `drift_events.json` — ground-truth boundaries",
        "- `trigger_metrics.csv`, `adaptation_metrics.csv`,",
        "  `continual_learning_matrix.csv`, `summary.json`",
        "- `<strategy>/` — strategy_result.json, window_log.json, detector_log.csv",
        "- `predictions/<strategy>/` — held-out predictions per segment state",
        "- `plots/` — monitoring signals, delays, false positives, quality-cost",
        "",
        "## Rerun",
        "",
        "```bash",
        f"uv run python scripts/e3_run.py --run-id {results_dir.name}  # full comparison",
        "uv run python scripts/e3_analyze.py --results-dir "
        f"results/e3/{results_dir.name}",
        "```",
    ]
    path = results_dir / "README.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def analyze_run(results_dir: str | Path, tolerance: int = 100) -> dict:
    results_dir = Path(results_dir)
    results = load_strategy_results(results_dir)
    config = json.loads(
        (results_dir / "run_config.json").read_text(encoding="utf-8")
    ) if (results_dir / "run_config.json").exists() else {}
    drift = json.loads((results_dir / "drift_events.json").read_text(encoding="utf-8"))
    n_boundaries = drift.get("n_known_transitions", 2)
    n_windows = drift.get("n_windows", 30)
    if not n_windows:
        n_windows = 30

    trigger_path = write_csv(
        results_dir / "trigger_metrics.csv",
        trigger_metric_rows(results, tolerance),
    )
    adaptation_path = write_csv(
        results_dir / "adaptation_metrics.csv", adaptation_metric_rows(results)
    )
    quality_path = write_csv(results_dir / "quality_metrics.csv", quality_rows(results))
    cl_path = write_csv(
        results_dir / "continual_learning_matrix.csv",
        continual_learning_rows(results),
    )

    # Run-level detection log: every detector decision across strategies.
    detection_log_rows = []
    for strategy, payload in results.items():
        for entry in payload.get("detector_log", []):
            detection_log_rows.append(entry)
        for window in payload.get("window_log", []):
            detection_log_rows.append(
                {
                    "strategy": strategy,
                    "detector": "window_signal",
                    "regime": payload.get("regime", ""),
                    "window_index": window["window_index"],
                    "position": window["position"],
                    "fired": any(
                        key in window for key in ("suppressed_warmup", "suppressed_min_gap")
                    ),
                    "statistic": window.get("mean_error"),
                    "threshold": None,
                    "reason": (
                        "mean_error="
                        f"{window.get('mean_error')}; mmd2="
                        f"{window.get('mmd2_vs_reference')}"
                    ),
                }
            )
    detection_log_path = write_csv(
        results_dir / "detection_log.csv", detection_log_rows
    )

    window_size = 50
    plots = generate_plots(
        results_dir, results, drift.get("known_boundaries", [500, 1000]), window_size
    )
    summary = build_summary(results, tolerance, n_boundaries, n_windows)
    write_json(results_dir / "summary.json", summary)
    readme = write_readme(results_dir, results, summary, config)

    return {
        "summary": summary,
        "files": {
            "trigger_metrics": str(trigger_path),
            "adaptation_metrics": str(adaptation_path),
            "quality_metrics": str(quality_path),
            "continual_learning": str(cl_path),
            "detection_log": str(detection_log_path),
            "summary": str(results_dir / "summary.json"),
            "readme": str(readme),
            "plots": [str(plot) for plot in plots],
        },
    }
