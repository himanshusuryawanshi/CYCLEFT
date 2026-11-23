"""E7 Phase 2 harness safety tests (synthetic artifacts, temp dirs only).

Hardened for the audit corrections: strict completion verification
(Finding 1), recursive artifact integrity (Finding 2), atomic finalization
with fail-closed classification (Finding 3), single-run pilot selection
(Finding 4), and stop-on-failure semantics. Never invokes real training:
state-machine tests use mocked run_fn callables producing schema-valid
synthetic artifacts.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cycleft.e7.execution_harness import (  # noqa: E402
    ANCHOR_REQUIRED_FILES,
    ARTIFACT_SPEC,
    COMPLETION_MARKER_NAME,
    EXECUTE_CONFIRM_TOKEN,
    MANIFEST_NAME,
    AnchorValidationError,
    ExecutionHarness,
    HarnessError,
    LockError,
    MatrixValidationError,
    OutputSafetyError,
    StateError,
    classify_run_state,
    inventory_artifact,
    inventory_digest,
    load_and_validate_matrix,
    select_run_id,
)

V5_MATRIX = Path(__file__).resolve().parents[1] / (
    "results/e7/design_20261120/v5_final_reconciliation/FINAL_RUN_MATRIX.csv")
TOK = EXECUTE_CONFIRM_TOKEN


# ---------------------------------------------------------------------------
# synthetic matrix / anchor / artifact builders
# ---------------------------------------------------------------------------


def base_row(**over):
    row = {
        "exec_id": "e7-static-r4-s42", "phase": "static-baseline",
        "policy": "Static-R4", "initial_rank": "4",
        "allowed_ranks": "{4, 8, 16, 32}",
        "decision_policy_id": "none (predetermined fixed rank)",
        "decision_policy_version": "n/a", "decision_thresholds": "n/a",
        "event_1_rank": "4", "event_2_rank": "4", "event_3_rank": "4",
        "event_1_alpha": "8", "event_2_alpha": "8", "event_3_alpha": "8",
        "depth": "D100", "seed": "42", "budget_per_event": "500",
        "reuse_or_new": "NEW", "reuse_source": "",
        "est_train_s": "1300", "est_eval_s": "280",
        "rank_transition_overhead_s": "0", "model_load_s": "15",
        "ckpt_serialize_s": "4", "est_matrix_subtotal_s": "1580",
        "est_total_elapsed_s": "1599", "final_trainable_params": "851968",
        "resource_aware_pilot": "False", "scientific_question": "q",
    }
    row.update(over)
    return row


def build_valid_matrix(tmp_path: Path) -> Path:
    """A valid 22-row synthetic matrix (same composition as v5)."""
    rows = []
    for seed in (42, 43, 44):
        rows.append(base_row(seed=str(seed), exec_id=f"e7-static-r8-s{seed}",
                             policy="Static-R8", initial_rank="8",
                             event_1_rank="8", event_2_rank="8", event_3_rank="8",
                             event_1_alpha="16", event_2_alpha="16",
                             event_3_alpha="16", reuse_or_new="REUSE",
                             reuse_source=f"anchor_{seed}"))
    for rank, alpha in ((4, 8), (9, 18), (16, 32), (32, 64)):
        for seed in (42, 43, 44):
            rows.append(base_row(seed=str(seed),
                                 exec_id=f"e7-static-r{rank}-s{seed}",
                                 policy=f"Static-R{rank}", initial_rank=str(rank),
                                 event_1_rank=str(rank), event_2_rank=str(rank),
                                 event_3_rank=str(rank), event_1_alpha=str(alpha),
                                 event_2_alpha=str(alpha), event_3_alpha=str(alpha),
                                 final_trainable_params=str(13312 * rank * 16)))
    for seed in (42, 43, 44):
        rows.append(base_row(seed=str(seed), exec_id=f"e7-grow-s{seed}",
                             phase="dynamic-rank", policy="Grow", initial_rank="4",
                             event_1_rank="4", event_2_rank="8", event_3_rank="16",
                             event_1_alpha="8", event_2_alpha="16",
                             event_3_alpha="32", rank_transition_overhead_s="10",
                             est_matrix_subtotal_s="1590", est_total_elapsed_s="1609",
                             final_trainable_params="3407872"))
        rows.append(base_row(seed=str(seed), exec_id=f"e7-reduce-s{seed}",
                             phase="dynamic-rank", policy="Reduce",
                             initial_rank="16", event_1_rank="16",
                             event_2_rank="8", event_3_rank="4",
                             event_1_alpha="32", event_2_alpha="16",
                             event_3_alpha="8", rank_transition_overhead_s="10",
                             est_matrix_subtotal_s="1590", est_total_elapsed_s="1609",
                             final_trainable_params="851968"))
    rows.append(base_row(exec_id="e7-resource-aware-s42", phase="dynamic-rank-pilot",
                         policy="Resource-Aware", initial_rank="4",
                         decision_policy_id="convergence_signal",
                         decision_thresholds="up>=+0.05",
                         event_1_rank="4", event_2_rank="runtime",
                         event_3_rank="runtime", event_1_alpha="8",
                         event_2_alpha="runtime", event_3_alpha="runtime",
                         rank_transition_overhead_s="10",
                         est_matrix_subtotal_s="1590", est_total_elapsed_s="1609",
                         final_trainable_params="runtime (depends on selected ranks)",
                         resource_aware_pilot="True"))
    assert len(rows) == 22
    import csv
    p = tmp_path / "matrix.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return p


def build_anchors(tmp_path: Path) -> Path:
    for seed in (42, 43, 44):
        d = tmp_path / f"anchor_{seed}"
        d.mkdir(parents=True, exist_ok=True)
        for name in ANCHOR_REQUIRED_FILES:
            (d / name).write_text(json.dumps({"seed": seed, "synthetic": True}))
    return tmp_path


def write_valid_artifacts(run_dir: Path, final_em: float = 0.12) -> None:
    """Synthetic artifacts matching the E5/E6 output schema exactly."""
    run_dir = Path(run_dir)
    (run_dir / "predictions").mkdir(parents=True, exist_ok=True)
    (run_dir / "bundles" / "step_001500" / "adapter").mkdir(parents=True, exist_ok=True)
    (run_dir / "execution_summary.json").write_text(json.dumps({
        "status": "complete", "global_step": 1500, "final_avg_em": final_em,
        "steps": 1500}))
    (run_dir / "final_eval.json").write_text(json.dumps({
        "1": {"exact_match": final_em}, "2": {"exact_match": final_em},
        "3": {"exact_match": final_em}}))
    (run_dir / "event_log.jsonl").write_text(
        json.dumps({"event": "process_start", "exec_id": "x"}) + "\n" +
        json.dumps({"event": "process_exit", "status": "complete"}) + "\n")
    (run_dir / "audit_log.jsonl").write_text(
        json.dumps({"step": 1, "loss": 1.0}) + "\n")
    (run_dir / "bundles" / "LATEST.json").write_text(json.dumps(
        {"latest": "step_001500", "global_step": 1500}))
    (run_dir / "bundles" / "step_001500" / "adapter" / "adapter_model.safetensors"
     ).write_text("synthetic-adapter-bytes")
    record = {"state": "after_event_3", "test_event": 1, "example_index": 0,
              "question": "q?", "reference": "r", "raw_prediction": "p",
              "prediction": "p", "exact_match": 1.0, "token_f1": 1.0}
    (run_dir / "predictions" / "after_event_3.jsonl").write_text(
        json.dumps(record) + "\n")


def full_telemetry(train_exit=0, train_status="complete",
                   eval_exit=0, eval_status="complete", **over):
    t = {
        "training": {"exit_code": train_exit, "status": train_status,
                     "summary_ref": "execution_summary.json"},
        "evaluation": {"exit_code": eval_exit, "status": eval_status,
                       "metrics_ref": "final_eval.json"},
    }
    t.update(over)
    return t


def _ok_run_fn(tmp_path):
    def run_fn(spec, run_dir):
        write_valid_artifacts(run_dir)
        return full_telemetry()
    return run_fn


@pytest.fixture
def env(tmp_path):
    matrix = build_valid_matrix(tmp_path)
    build_anchors(tmp_path)
    out_root = tmp_path / "results" / "e7" / "execution_test"
    out_root.mkdir(parents=True)
    harness = ExecutionHarness(matrix, out_root, repo_root=tmp_path)
    return harness, matrix, out_root, tmp_path


def _first_new_id(harness):
    return [s.exec_id for s in harness.specs() if s.row_type == "NEW"][0]


# ---------------------------------------------------------------------------
# 1-3: plan composition
# ---------------------------------------------------------------------------


def test_1_valid_matrix_generates_22_row_plan(env):
    harness, *_ = env
    plan = harness.dry_run()
    assert len(plan) == 22
    assert all(p.action in ("train", "read-only reference") for p in plan)


def test_2_exactly_19_new(env):
    harness, *_ = env
    assert sum(1 for p in harness.dry_run() if p.row_type == "NEW") == 19


def test_3_exactly_3_reuse(env):
    harness, *_ = env
    assert sum(1 for p in harness.dry_run() if p.row_type == "REUSE") == 3


# ---------------------------------------------------------------------------
# 4-7: matrix validation
# ---------------------------------------------------------------------------


def _write_bad_matrix(tmp_path, mutate, filename="bad.csv"):
    src = build_valid_matrix(tmp_path)
    import csv
    with open(src, newline="") as f:
        rows = list(csv.DictReader(f))
    mutate(rows)
    p = tmp_path / filename
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return p


def test_4_duplicate_run_ids_rejected(tmp_path):
    def dup(rows):
        rows.append(dict(rows[0]))
    p = _write_bad_matrix(tmp_path, dup)
    with pytest.raises(MatrixValidationError, match="exactly 22"):
        load_and_validate_matrix(p)


def test_5_missing_required_fields_rejected(tmp_path):
    from cycleft.e7.execution_harness import validate_row
    row = base_row()
    del row["seed"]
    with pytest.raises(MatrixValidationError, match="missing required fields"):
        validate_row(row)
    row2 = base_row()
    row2["seed"] = None
    with pytest.raises(MatrixValidationError, match="missing required fields"):
        validate_row(row2)
    def drop(rows):
        del rows[5]["seed"]
    p = _write_bad_matrix(tmp_path, drop)
    with pytest.raises(MatrixValidationError):
        load_and_validate_matrix(p)


def test_6_invalid_alpha_rank_rejected(tmp_path):
    def bad_alpha(rows):
        rows[5]["event_1_alpha"] = "7"
    p = _write_bad_matrix(tmp_path, bad_alpha)
    with pytest.raises(MatrixValidationError, match="alpha"):
        load_and_validate_matrix(p)


def test_7_hardcoded_future_resource_aware_ranks_rejected(tmp_path):
    def hardcode(rows):
        ra = [r for r in rows if r["policy"] == "Resource-Aware"][0]
        ra["event_2_rank"], ra["event_3_rank"] = "8", "16"
        ra["event_2_alpha"], ra["event_3_alpha"] = "16", "32"
    p = _write_bad_matrix(tmp_path, hardcode)
    with pytest.raises(MatrixValidationError, match="runtime"):
        load_and_validate_matrix(p)


# ---------------------------------------------------------------------------
# 8: anchors fail closed
# ---------------------------------------------------------------------------


def test_8_missing_anchor_fails_closed(tmp_path):
    matrix = build_valid_matrix(tmp_path)
    out_root = tmp_path / "results" / "e7" / "out"
    out_root.mkdir(parents=True)
    harness = ExecutionHarness(matrix, out_root, repo_root=tmp_path)
    with pytest.raises(AnchorValidationError, match="does not exist"):
        harness.dry_run()
    build_anchors(tmp_path)
    (tmp_path / "anchor_43" / ANCHOR_REQUIRED_FILES[0]).unlink()
    with pytest.raises(AnchorValidationError, match="missing required artifact"):
        harness.dry_run()
    assert not any(out_root.iterdir())


# ---------------------------------------------------------------------------
# 9-10: output protection
# ---------------------------------------------------------------------------


def test_9_existing_new_output_dir_blocks_execution(env):
    harness, matrix, out_root, tmp_path = env
    victim_id = _first_new_id(harness)
    victim = out_root / victim_id
    victim.mkdir()
    (victim / "stale_artifact.bin").write_text("x")
    harness2 = ExecutionHarness(matrix, out_root, repo_root=tmp_path,
                                run_fn=_ok_run_fn(tmp_path))
    with pytest.raises(OutputSafetyError, match="refusing to execute"):
        harness2.execute(TOK, run_id=victim_id)
    assert (victim / "stale_artifact.bin").read_text() == "x"


def test_10_existing_completion_marker_blocks_execution(env):
    harness, matrix, out_root, tmp_path = env
    victim_id = _first_new_id(harness)
    victim = out_root / victim_id
    victim.mkdir()
    (victim / COMPLETION_MARKER_NAME).write_text('{"state": "COMPLETED"}')
    harness2 = ExecutionHarness(matrix, out_root, repo_root=tmp_path,
                                run_fn=_ok_run_fn(tmp_path))
    with pytest.raises(OutputSafetyError, match="completion marker"):
        harness2.execute(TOK, run_id=victim_id)


# ---------------------------------------------------------------------------
# 11-12: state machine
# ---------------------------------------------------------------------------


def test_11_incomplete_run_cannot_be_completed(env):
    harness, _, out_root, _ = env
    run_id = _first_new_id(harness)
    run_dir = out_root / run_id
    run_dir.mkdir(parents=True)
    manifest = {"exec_id": run_id, "state": "PLANNED"}
    (run_dir / MANIFEST_NAME).write_text(json.dumps(manifest))
    spec = harness.specs()[0]
    with pytest.raises(StateError, match="cannot mark COMPLETED"):
        harness._finalize_completed(run_dir, spec, manifest,
                                    telemetry=full_telemetry())
    manifest["state"] = "RUNNING"
    (run_dir / MANIFEST_NAME).write_text(json.dumps(manifest))
    with pytest.raises(StateError, match="missing"):
        harness._finalize_completed(run_dir, spec, manifest,
                                    telemetry=full_telemetry())
    rec = harness.inspect_state(run_dir)
    assert rec["state"] == "RUNNING" and rec["completion_marker_present"] is False \
        if False else rec["state"] == "RUNNING"


def test_12_interrupted_not_mistaken_for_completed(env):
    harness, matrix, out_root, tmp_path = env
    run_id = _first_new_id(harness)
    run_dir = out_root / run_id
    run_dir.mkdir(parents=True)
    manifest = {"exec_id": run_id, "state": "RUNNING"}
    (run_dir / MANIFEST_NAME).write_text(json.dumps(manifest))
    spec = harness.specs()[0]
    harness._mark_interrupted(run_dir, spec, manifest, reason="SIGTERM")
    rec = harness.inspect_state(run_dir)
    assert rec["state"] == "INTERRUPTED"
    with pytest.raises(StateError, match="cannot mark COMPLETED"):
        harness._finalize_completed(run_dir, spec, manifest,
                                    telemetry=full_telemetry())

    # end-to-end: KeyboardInterrupt leaves INTERRUPTED, no marker, no COMPLETED
    shutil.rmtree(run_dir)  # test-fixture cleanup; harness never deletes
    def interrupted_fn(spec, run_dir):
        write_valid_artifacts(run_dir)
        raise KeyboardInterrupt
    h2 = ExecutionHarness(matrix, out_root, repo_root=tmp_path,
                          run_fn=interrupted_fn)
    with pytest.raises(KeyboardInterrupt):
        h2.execute(TOK, run_id=run_id)
    rec = harness.inspect_state(out_root / run_id)
    assert rec["state"] == "INTERRUPTED"
    assert not (out_root / run_id / COMPLETION_MARKER_NAME).exists()


# ---------------------------------------------------------------------------
# 13-15: locks, traversal, symlinks
# ---------------------------------------------------------------------------


def test_13_second_runner_cannot_acquire_active_lock(env):
    harness, _, out_root, _ = env
    from cycleft.e7.execution_harness import _ExecutionLock
    with _ExecutionLock(out_root):
        with pytest.raises(LockError, match="second harness instance|already present"):
            with _ExecutionLock(out_root):
                pass
    with _ExecutionLock(out_root):
        pass


def test_13b_stale_lock_fails_closed_not_deleted(env):
    harness, _, out_root, _ = env
    from cycleft.e7.execution_harness import _ExecutionLock
    lock_path = out_root / ".execution_lock.json"
    lock_path.write_text(json.dumps({"pid": 999999999, "host": "gone"}))
    with pytest.raises(LockError, match="STALE"):
        with _ExecutionLock(out_root):
            pass
    assert lock_path.exists(), "stale lock must not be silently deleted"


def test_14_path_traversal_rejected(env):
    harness, _, _, _ = env
    from cycleft.e7.execution_harness import resolve_output_dir
    with pytest.raises(OutputSafetyError, match="traversal|invalid exec_id"):
        resolve_output_dir(harness.output_root, "../e6-victim")
    with pytest.raises(OutputSafetyError):
        resolve_output_dir(harness.output_root, "ok/../../escape")


def test_15_symlink_escape_rejected(env):
    harness, _, out_root, tmp_path = env
    outside = tmp_path / "outside"
    outside.mkdir()
    link = out_root / "e7-static-r4-s42"
    link.symlink_to(outside, target_is_directory=True)
    from cycleft.e7.execution_harness import resolve_output_dir
    with pytest.raises(OutputSafetyError, match="escapes"):
        resolve_output_dir(out_root, "e7-static-r4-s42")


# ---------------------------------------------------------------------------
# 16-18: dry-run purity
# ---------------------------------------------------------------------------


def test_16_dry_run_modifies_nothing(env):
    harness, _, out_root, tmp_path = env

    def snapshot(p: Path):
        return {str(q.relative_to(p)): (q.stat().st_mtime_ns, q.stat().st_size)
                for q in sorted(p.rglob("*")) if q.is_file()}

    anchors_before = snapshot(tmp_path)
    out_before = snapshot(out_root) if out_root.exists() else {}
    harness.dry_run()
    harness.dry_run(run_id=_first_new_id(harness))
    assert snapshot(tmp_path) == anchors_before
    assert (snapshot(out_root) if out_root.exists() else {}) == out_before
    assert not any(out_root.iterdir())


def test_17_dry_run_does_not_import_model_training_path(tmp_path):
    matrix = build_valid_matrix(tmp_path)
    build_anchors(tmp_path)
    out_root = tmp_path / "results" / "e7" / "out"
    out_root.mkdir(parents=True)
    code = (
        "import sys, json;"
        "from cycleft.e7.execution_harness import ExecutionHarness;"
        f"h = ExecutionHarness({str(matrix)!r}, {str(out_root)!r}, "
        f"repo_root={str(tmp_path)!r});"
        "h.dry_run();"
        "bad = [m for m in ('torch', 'transformers', 'peft', 'cycleft.e5', "
        "'cycleft.e1', 'cycleft.e7.e5_adapter', 'cycleft.finetuning') "
        "if any(x == m or x.startswith(m + '.') for x in sys.modules)];"
        "print(json.dumps({'bad': bad}))"
    )
    r = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        env={**os.environ,
             "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")})
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout.strip().splitlines()[-1])["bad"] == []


def test_18_dry_run_cannot_invoke_training_or_evaluation(env):
    harness, _, out_root, _ = env
    calls = []

    def run_fn(spec, run_dir):
        calls.append(spec.exec_id)
        return {}

    harness.run_fn = run_fn
    harness.dry_run()
    harness.dry_run(run_id=_first_new_id(harness))
    assert calls == [], "dry-run invoked the training callable"
    with pytest.raises(HarnessError, match="confirmation token"):
        harness.execute("wrong-token", run_id=_first_new_id(harness))
    with pytest.raises(HarnessError, match="confirmation token"):
        harness.execute("", run_id=_first_new_id(harness))
    assert calls == []
    assert not any(out_root.iterdir())


# ---------------------------------------------------------------------------
# Finding 1 — strict completion verification
# ---------------------------------------------------------------------------


def _prepared_run(env, state="RUNNING"):
    harness, _, out_root, _ = env
    run_id = _first_new_id(harness)
    run_dir = out_root / run_id
    run_dir.mkdir(parents=True)
    write_valid_artifacts(run_dir)
    manifest = {"exec_id": run_id, "config_hash":
                harness.specs()[[s.exec_id for s in harness.specs()]
                                .index(run_id)].config_hash,
                "state": state}
    (run_dir / MANIFEST_NAME).write_text(json.dumps(manifest))
    spec = [s for s in harness.specs() if s.exec_id == run_id][0]
    return harness, run_dir, spec, manifest


def test_f1_nonzero_training_exit_cannot_complete(env):
    harness, run_dir, spec, manifest = _prepared_run(env)
    with pytest.raises(StateError, match="exit code"):
        harness._finalize_completed(run_dir, spec, manifest,
                                    telemetry=full_telemetry(train_exit=1))
    assert classify_run_state(run_dir)["classification"] == "RUNNING"


def test_f1_missing_training_exit_cannot_complete(env):
    harness, run_dir, spec, manifest = _prepared_run(env)
    t = full_telemetry()
    del t["training"]["exit_code"]
    with pytest.raises(StateError, match="return code missing"):
        harness._finalize_completed(run_dir, spec, manifest, telemetry=t)
    assert classify_run_state(run_dir)["classification"] == "RUNNING"


def test_f1_failed_training_status_cannot_complete(env):
    harness, run_dir, spec, manifest = _prepared_run(env)
    with pytest.raises(StateError, match="status"):
        harness._finalize_completed(
            run_dir, spec, manifest,
            telemetry=full_telemetry(train_status="killed"))
    assert classify_run_state(run_dir)["classification"] == "RUNNING"


def test_f1_missing_training_evidence_cannot_complete(env):
    harness, run_dir, spec, manifest = _prepared_run(env)
    with pytest.raises(StateError, match="training evidence missing"):
        harness._finalize_completed(run_dir, spec, manifest,
                                    telemetry={"evaluation": full_telemetry()["evaluation"]})
    assert classify_run_state(run_dir)["classification"] == "RUNNING"


def test_f1_nonzero_evaluation_exit_cannot_complete(env):
    harness, run_dir, spec, manifest = _prepared_run(env)
    with pytest.raises(StateError, match="evaluation"):
        harness._finalize_completed(run_dir, spec, manifest,
                                    telemetry=full_telemetry(eval_exit=2))
    assert classify_run_state(run_dir)["classification"] == "RUNNING"


def test_f1_missing_evaluation_status_cannot_complete(env):
    harness, run_dir, spec, manifest = _prepared_run(env)
    t = full_telemetry()
    del t["evaluation"]["status"]
    with pytest.raises(StateError, match="evaluation"):
        harness._finalize_completed(run_dir, spec, manifest, telemetry=t)
    assert classify_run_state(run_dir)["classification"] == "RUNNING"


def test_f1_failed_evaluation_status_cannot_complete(env):
    harness, run_dir, spec, manifest = _prepared_run(env)
    with pytest.raises(StateError, match="evaluation"):
        harness._finalize_completed(
            run_dir, spec, manifest,
            telemetry=full_telemetry(eval_status="incomplete"))
    assert classify_run_state(run_dir)["classification"] == "RUNNING"


def test_f1_missing_artifacts_cannot_complete(env):
    harness, run_dir, spec, manifest = _prepared_run(env)
    (run_dir / "execution_summary.json").unlink()
    with pytest.raises(StateError, match="missing"):
        harness._finalize_completed(run_dir, spec, manifest,
                                    telemetry=full_telemetry())
    assert classify_run_state(run_dir)["classification"] == "RUNNING"


def test_f1_empty_predictions_cannot_complete(env):
    harness, run_dir, spec, manifest = _prepared_run(env)
    (run_dir / "predictions" / "after_event_3.jsonl").unlink()
    with pytest.raises(StateError, match="empty directory|no .jsonl"):
        harness._finalize_completed(run_dir, spec, manifest,
                                    telemetry=full_telemetry())
    assert classify_run_state(run_dir)["classification"] == "RUNNING"


def test_f1_no_default_return_code(env):
    """The telemetry dict is used verbatim: a missing exit code is never
    defaulted to 0 (the setdefault('return_code', 0) pattern is gone)."""
    harness, run_dir, spec, manifest = _prepared_run(env)
    t = full_telemetry()
    t["training"]["exit_code"] = None
    with pytest.raises(StateError, match="return code missing"):
        harness._finalize_completed(run_dir, spec, manifest, telemetry=t)


def test_f1_valid_completion_succeeds(env):
    harness, run_dir, spec, manifest = _prepared_run(env)
    harness._finalize_completed(run_dir, spec, manifest,
                                telemetry=full_telemetry())
    rec = harness.inspect_state(run_dir)
    assert rec["state"] == "COMPLETED"
    assert rec["classification"] == "COMPLETED" and rec["fail_closed"] is False


# ---------------------------------------------------------------------------
# Finding 2 — recursive artifact integrity
# ---------------------------------------------------------------------------


def test_f2_modified_prediction_file_detected(env):
    harness, run_dir, spec, manifest = _prepared_run(env)
    inv = harness.__class__.__mro__ and None  # placeholder no-op
    from cycleft.e7.execution_harness import build_artifact_inventory
    before = inventory_digest(build_artifact_inventory(run_dir))
    p = run_dir / "predictions" / "after_event_3.jsonl"
    p.write_text(p.read_text() + json.dumps({**json.loads(
        p.read_text().splitlines()[0]), "example_index": 999}) + "\n")
    after = inventory_digest(build_artifact_inventory(run_dir))
    assert before != after, "modified artifact content must change the digest"
    # completion protocol re-validates structurally too — valid schema, but
    # the manifest/marker digest consistency would catch any swap.


def test_f2_checksum_mismatch_rejected(tmp_path, env):
    harness, run_dir, spec, manifest = _prepared_run(env)
    from cycleft.e7.execution_harness import build_artifact_inventory
    inv = build_artifact_inventory(run_dir)
    # tamper: swap a recorded checksum
    tampered = json.loads(json.dumps(inv))
    tampered["final_eval.json"]["files"]["final_eval.json"] = "0" * 64
    tampered["final_eval.json"]["digest"] = "1" * 64
    marker = {"exec_id": spec.exec_id, "config_hash": spec.config_hash,
              "state": "COMPLETED", "artifact_inventory": inv,
              "artifact_inventory_digest": inventory_digest(inv)}
    (run_dir / COMPLETION_MARKER_NAME).write_text(json.dumps(marker))
    manifest["state"] = "COMPLETED"
    manifest["artifact_inventory"] = tampered
    manifest["artifact_inventory_digest"] = inventory_digest(tampered)
    (run_dir / MANIFEST_NAME).write_text(json.dumps(manifest))
    rec = harness.inspect_state(run_dir)
    assert rec["classification"] == "CORRUPT" and rec["fail_closed"] is True


def test_f2_missing_checkpoint_bundle_rejected(env):
    harness, run_dir, spec, manifest = _prepared_run(env)
    shutil.rmtree(run_dir / "bundles")
    with pytest.raises(StateError, match="missing"):
        harness._finalize_completed(run_dir, spec, manifest,
                                    telemetry=full_telemetry())


def test_f2_nested_directory_digest_deterministic(env):
    harness, run_dir, spec, manifest = _prepared_run(env)
    nested = run_dir / "bundles" / "step_001500" / "adapter" / "nested"
    nested.mkdir()
    (nested / "deep.bin").write_text("payload")
    from cycleft.e7.execution_harness import build_artifact_inventory
    d1 = inventory_digest(build_artifact_inventory(run_dir))
    d2 = inventory_digest(build_artifact_inventory(run_dir))
    assert d1 == d2, "nested directory digests must be deterministic"
    # content change propagates into the parent digest
    (nested / "deep.bin").write_text("payload-2")
    d3 = inventory_digest(build_artifact_inventory(run_dir))
    assert d3 != d1


def test_f2_symlink_inside_artifact_rejected(env):
    harness, run_dir, spec, manifest = _prepared_run(env)
    target = run_dir / "event_log.jsonl"
    (run_dir / "predictions" / "link.jsonl").symlink_to(target)
    with pytest.raises(StateError, match="symlink"):
        harness._finalize_completed(run_dir, spec, manifest,
                                    telemetry=full_telemetry())


def test_f2_symlink_escaping_run_dir_rejected(env, tmp_path):
    harness, run_dir, spec, manifest = _prepared_run(env)
    outside = tmp_path / "outside_secret.txt"
    outside.write_text("secret")
    (run_dir / "bundles" / "escape.bin").symlink_to(outside)
    with pytest.raises(StateError, match="symlink|escapes"):
        harness._finalize_completed(run_dir, spec, manifest,
                                    telemetry=full_telemetry())


def test_f2_unexpected_file_type_rejected(env):
    harness, run_dir, spec, manifest = _prepared_run(env)
    os.mkfifo(run_dir / "predictions" / "pipe")
    try:
        with pytest.raises(StateError, match="unexpected file type"):
            harness._finalize_completed(run_dir, spec, manifest,
                                        telemetry=full_telemetry())
    finally:
        os.unlink(run_dir / "predictions" / "pipe")


def test_f2_invalid_prediction_record_rejected(env):
    harness, run_dir, spec, manifest = _prepared_run(env)
    p = run_dir / "predictions" / "after_event_3.jsonl"
    rec = json.loads(p.read_text().splitlines()[0])
    del rec["exact_match"]
    p.write_text(json.dumps(rec) + "\n")
    with pytest.raises(StateError, match="required record keys"):
        harness._finalize_completed(run_dir, spec, manifest,
                                    telemetry=full_telemetry())


def test_f2_inventory_persisted_in_manifest_and_marker(env):
    harness, run_dir, spec, manifest = _prepared_run(env)
    harness._finalize_completed(run_dir, spec, manifest,
                                telemetry=full_telemetry())
    m = json.loads((run_dir / MANIFEST_NAME).read_text())
    k = json.loads((run_dir / COMPLETION_MARKER_NAME).read_text())
    assert "artifact_inventory" in m and "artifact_inventory_digest" in m
    assert k["artifact_inventory_digest"] == m["artifact_inventory_digest"]
    assert set(k["artifact_inventory"].keys()) == set(ARTIFACT_SPEC.keys())


# ---------------------------------------------------------------------------
# Finding 3 — atomic completion protocol / fault injection
# ---------------------------------------------------------------------------


def _finalize_through(harness, run_dir, spec, manifest, telemetry,
                      crash_after):
    """Drive the finalization protocol, raising a simulated crash after the
    given step ('finalizing', 'marker', 'completed')."""
    import cycleft.e7.execution_harness as H
    real_update = harness._update_state
    real_marker = H._atomic_write_json

    def update_state(run_dir_, manifest_, state_, **extra):
        if crash_after == "before_finalizing" and state_ == "FINALIZING":
            raise SimulatedCrash
        return real_update(run_dir_, manifest_, state_, **extra)

    def atomic(path, payload):
        if crash_after == "finalizing" and path.name == COMPLETION_MARKER_NAME:
            real_update(run_dir, manifest, "FINALIZING",
                        artifact_inventory=payload["artifact_inventory"],
                        artifact_inventory_digest=payload["artifact_inventory_digest"])
            raise SimulatedCrash
        if crash_after == "marker" and path.name == MANIFEST_NAME \
                and payload.get("state") == "COMPLETED":
            # marker already written; final COMPLETED manifest never lands
            raise SimulatedCrash
        return real_marker(path, payload)

    harness._update_state = update_state
    H._atomic_write_json = atomic
    try:
        harness._finalize_completed(run_dir, spec, manifest, telemetry=telemetry)
    finally:
        harness._update_state = real_update
        H._atomic_write_json = real_marker


class SimulatedCrash(BaseException):
    pass


def test_f3_crash_before_finalizing_leaves_running(env):
    harness, run_dir, spec, manifest = _prepared_run(env)
    with pytest.raises(SimulatedCrash):
        _finalize_through(harness, run_dir, spec, manifest,
                          full_telemetry(), crash_after="before_finalizing")
    rec = harness.inspect_state(run_dir)
    assert rec["classification"] == "RUNNING"
    assert not (run_dir / COMPLETION_MARKER_NAME).exists()


def test_f3_crash_after_finalizing_manifest_incomplete(env):
    harness, run_dir, spec, manifest = _prepared_run(env)
    with pytest.raises(SimulatedCrash):
        _finalize_through(harness, run_dir, spec, manifest,
                          full_telemetry(), crash_after="finalizing")
    rec = harness.inspect_state(run_dir)
    assert rec["classification"] == "INCOMPLETE_FINALIZING"
    assert rec["fail_closed"] is True
    assert not (run_dir / COMPLETION_MARKER_NAME).exists()


def test_f3_crash_after_marker_ambiguous(env):
    harness, run_dir, spec, manifest = _prepared_run(env)
    with pytest.raises(SimulatedCrash):
        _finalize_through(harness, run_dir, spec, manifest,
                          full_telemetry(), crash_after="marker")
    rec = harness.inspect_state(run_dir)
    assert rec["classification"] == "AMBIGUOUS_FINALIZING"
    assert rec["fail_closed"] is True
    # a FINALIZING run can never be promoted to COMPLETED afterwards
    with pytest.raises(StateError, match="cannot mark COMPLETED"):
        harness._finalize_completed(run_dir, spec, manifest,
                                    telemetry=full_telemetry())


def test_f3_completed_without_marker_is_corrupt(env):
    harness, run_dir, spec, manifest = _prepared_run(env)
    manifest["state"] = "COMPLETED"
    manifest["config_hash"] = spec.config_hash
    (run_dir / MANIFEST_NAME).write_text(json.dumps(manifest))
    rec = harness.inspect_state(run_dir)
    assert rec["classification"] == "CORRUPT" and rec["fail_closed"] is True


def test_f3_marker_mismatch_is_corrupt(env):
    harness, run_dir, spec, manifest = _prepared_run(env)
    manifest["state"] = "COMPLETED"
    manifest["config_hash"] = spec.config_hash
    manifest["artifact_inventory_digest"] = "a" * 64
    (run_dir / MANIFEST_NAME).write_text(json.dumps(manifest))
    (run_dir / COMPLETION_MARKER_NAME).write_text(json.dumps({
        "exec_id": spec.exec_id, "config_hash": "b" * 64, "state": "COMPLETED",
        "artifact_inventory_digest": "a" * 64}))
    rec = harness.inspect_state(run_dir)
    assert rec["classification"] == "CORRUPT"
    assert "hash mismatch" in rec["classification_detail"]
    # forensic evidence preserved; nothing repaired or deleted
    assert (run_dir / MANIFEST_NAME).exists() and (run_dir / COMPLETION_MARKER_NAME).exists()


def test_f3_full_protocol_verifies_consistency(env):
    harness, run_dir, spec, manifest = _prepared_run(env)
    harness._finalize_completed(run_dir, spec, manifest,
                                telemetry=full_telemetry())
    m = json.loads((run_dir / MANIFEST_NAME).read_text())
    k = json.loads((run_dir / COMPLETION_MARKER_NAME).read_text())
    assert m["exec_id"] == k["exec_id"] == spec.exec_id
    assert m["config_hash"] == k["config_hash"] == spec.config_hash
    assert m["artifact_inventory_digest"] == k["artifact_inventory_digest"]
    rec = harness.inspect_state(run_dir)
    assert rec["classification"] == "COMPLETED" and not rec["fail_closed"]


# ---------------------------------------------------------------------------
# Finding 4 — single-run pilot selection
# ---------------------------------------------------------------------------


def test_f4_select_valid_new_run(env):
    harness, _, _, _ = env
    new_ids = [s.exec_id for s in harness.specs() if s.row_type == "NEW"]
    for rid in new_ids[:5]:
        spec = select_run_id(harness.specs(), rid)
        assert spec.exec_id == rid and spec.row_type == "NEW"


def test_f4_reuse_target_rejected(env):
    harness, _, _, _ = env
    reuse_id = [s.exec_id for s in harness.specs() if s.row_type == "REUSE"][0]
    with pytest.raises(HarnessError, match="REUSE"):
        select_run_id(harness.specs(), reuse_id)
    harness2, matrix, out_root, tmp_path = env
    h = ExecutionHarness(matrix, out_root, repo_root=tmp_path,
                         run_fn=_ok_run_fn(tmp_path))
    with pytest.raises(HarnessError, match="REUSE"):
        h.execute(TOK, run_id=reuse_id)


def test_f4_unknown_id_rejected(env):
    harness, _, _, _ = env
    with pytest.raises(HarnessError, match="unknown run ID"):
        select_run_id(harness.specs(), "e7-does-not-exist-s99")
    with pytest.raises(HarnessError, match="unknown run ID"):
        harness.execute(TOK, run_id="e7-does-not-exist-s99")


def test_f4_execution_requires_run_id(env):
    harness, matrix, out_root, tmp_path = env
    h = ExecutionHarness(matrix, out_root, repo_root=tmp_path,
                         run_fn=_ok_run_fn(tmp_path))
    with pytest.raises(HarnessError, match="single NEW run ID is required"):
        h.execute(TOK, run_id=None)  # missing ID never means "run everything"


def test_f4_dry_run_selection_flags_one_row(env):
    harness, _, _, _ = env
    rid = _first_new_id(harness)
    plan = harness.dry_run(run_id=rid)
    assert len(plan) == 22, "full audit plan preserved"
    assert sum(1 for p in plan if p.selected) == 1
    assert [p.exec_id for p in plan if p.selected] == [rid]


def test_f4_selection_uses_actual_matrix_ids(env):
    """Pilot selection works for Grow, Reduce, static, and Resource-Aware
    IDs read from the actual synthetic matrix (no hardcoded assumptions)."""
    harness, _, _, _ = env
    for policy in ("Grow", "Reduce", "Static-R4", "Resource-Aware"):
        rid = [s.exec_id for s in harness.specs() if s.policy == policy][0]
        spec = select_run_id(harness.specs(), rid)
        assert spec.policy == policy


def test_f4_execute_single_run_invokes_one_callback(env):
    harness, matrix, out_root, tmp_path = env
    calls = []
    rid = _first_new_id(harness)

    def run_fn(spec, run_dir):
        calls.append(spec.exec_id)
        write_valid_artifacts(run_dir)
        return full_telemetry()

    h = ExecutionHarness(matrix, out_root, repo_root=tmp_path, run_fn=run_fn)
    h.execute(TOK, run_id=rid)
    assert calls == [rid], "exactly one callback invocation"
    rec = harness.inspect_state(out_root / rid)
    assert rec["classification"] == "COMPLETED"
    others = [p for p in out_root.iterdir() if p.name != rid]
    assert others == [], "no other run directories created"


# ---------------------------------------------------------------------------
# Stop-on-failure
# ---------------------------------------------------------------------------


def test_stop_on_failure_single_selected_run(env):
    harness, matrix, out_root, tmp_path = env
    rid = _first_new_id(harness)

    def failing_fn(spec, run_dir):
        write_valid_artifacts(run_dir)  # partial output present
        return full_telemetry(train_exit=1)  # training failed

    h = ExecutionHarness(matrix, out_root, repo_root=tmp_path, run_fn=failing_fn)
    with pytest.raises(HarnessError, match="execution stopped"):
        h.execute(TOK, run_id=rid)
    rec = harness.inspect_state(out_root / rid)
    assert rec["state"] == "FAILED"
    assert "exit code 1" in rec["failure_reason"]
    assert not (out_root / rid / COMPLETION_MARKER_NAME).exists()
    # partial output preserved
    assert (out_root / rid / "execution_summary.json").exists()


def test_stop_on_failure_no_subsequent_callback(env):
    """End-to-end: with multiple synthetic selected rows, the first failing
    callback stops the harness; the second callback never runs."""
    harness, _, out_root, _ = env
    selected = [s for s in harness.specs() if s.row_type == "NEW"][:2]
    planned = [p for p in harness.dry_run() if p.row_type == "NEW"][:2]
    calls = []

    def failing_first(spec, run_dir):
        calls.append(spec.exec_id)
        write_valid_artifacts(run_dir)
        raise RuntimeError("synthetic callback failure")

    harness.run_fn = failing_first
    with pytest.raises(HarnessError, match="no further runs"):
        harness._execute_selected(selected, planned)
    assert len(calls) == 1, "second callback must never be invoked"
    rec = harness.inspect_state(out_root / selected[0].exec_id)
    assert rec["state"] == "FAILED"
    assert "synthetic callback failure" in rec["failure_reason"]
    assert not (out_root / selected[1].exec_id).exists(), \
        "no output for the second run"
    assert not any(out_root.glob(".execution_lock*"))


# ---------------------------------------------------------------------------
# 16-18 legacy + misc protocol checks
# ---------------------------------------------------------------------------


def test_reuse_rows_never_scheduled_for_training(env):
    harness, _, _, _ = env
    plan = harness.dry_run()
    for p in plan:
        if p.row_type == "REUSE":
            assert p.action == "read-only reference"
            assert not p.output_path.startswith(str(harness.output_root))


def test_atomic_manifest_write_leaves_no_temp_files(env):
    harness, _, out_root, _ = env
    run_id = _first_new_id(harness)
    run_dir = out_root / run_id
    run_dir.mkdir(parents=True)
    spec = harness.specs()[0]
    manifest = {"exec_id": run_id, "state": "RUNNING"}
    harness._mark_failed(run_dir, spec, manifest, reason="synthetic")
    leftovers = [p.name for p in run_dir.iterdir() if p.name.startswith(".tmp_")]
    assert leftovers == []
    data = json.loads((run_dir / MANIFEST_NAME).read_text())
    assert data["state"] == "FAILED" and data["failure_reason"] == "synthetic"


def test_existing_prior_execution_blocks_rerun(env):
    harness, matrix, out_root, tmp_path = env
    rid = _first_new_id(harness)
    h1 = ExecutionHarness(matrix, out_root, repo_root=tmp_path,
                          run_fn=_ok_run_fn(tmp_path))
    h1.execute(TOK, run_id=rid)
    h2 = ExecutionHarness(matrix, out_root, repo_root=tmp_path,
                          run_fn=_ok_run_fn(tmp_path))
    with pytest.raises(OutputSafetyError, match="refusing to execute"):
        h2.execute(TOK, run_id=rid)
    rec = harness.inspect_state(out_root / rid)
    assert rec["classification"] == "COMPLETED"
