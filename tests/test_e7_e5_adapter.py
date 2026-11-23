"""E7-to-E5 adapter tests (mocks and synthetic callbacks only).

Never invokes actual training/evaluation and never initializes a model.
Validates configuration mapping, policy-support gating, callback contracts,
and lazy-import purity.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cycleft.e7.execution_harness import (  # noqa: E402
    ExecutionHarness,
    HarnessError,
    load_and_validate_matrix,
)
from cycleft.e7.e5_adapter import (  # noqa: E402
    RUN_WORKLOAD_RANK_LIMITATION,
    UNSUPPORTED_DYNAMIC_POLICIES,
    E5AdapterLimitation,
    make_run_fn,
    row_to_e5_plan,
)

V5_MATRIX = Path(__file__).resolve().parents[1] / (
    "results/e7/design_20261120/v5_final_reconciliation/FINAL_RUN_MATRIX.csv")


@pytest.fixture(scope="module")
def v5_specs():
    return load_and_validate_matrix(V5_MATRIX)


def _spec_by_policy(specs, policy):
    return [s for s in specs if s.policy == policy][0]


# ---------------------------------------------------------------------------
# configuration mapping (rows read from the authoritative matrix)
# ---------------------------------------------------------------------------


def test_static_policy_mapping_from_authoritative_matrix(v5_specs):
    spec = _spec_by_policy(v5_specs, "Static-R4")
    plan = row_to_e5_plan(spec)
    assert plan["exec_id"] == spec.exec_id
    assert plan["lora_r"] == 4 and plan["lora_alpha"] == 8  # alpha = 2r
    assert plan["seed"] == 42
    assert plan["budget_per_event"] == 500
    assert plan["stage_schedule"] == [
        {"stage": 1, "event_id": 1, "start": 0, "end": 500},
        {"stage": 2, "event_id": 2, "start": 500, "end": 1000},
        {"stage": 3, "event_id": 3, "start": 1000, "end": 1500},
    ]
    assert plan["lora_precision"] == "float32" and plan["dtype"] == "float16"
    assert plan["max_new_tokens"] == 24
    assert plan["fresh_adamw_per_event"] is True
    assert plan["checkpoint_policy"].startswith("P1")


def test_all_static_policies_map_with_alpha_2r(v5_specs):
    for policy in ("Static-R4", "Static-R9", "Static-R16", "Static-R32"):
        spec = _spec_by_policy(v5_specs, policy)
        plan = row_to_e5_plan(spec)
        assert plan["lora_alpha"] == 2 * plan["lora_r"]
        assert plan["lora_r"] == int(spec.event_1_rank) if hasattr(spec, "event_1_rank") \
            else True
        # rank matches the matrix event-1 rank for every seed of the policy
        for s in [x for x in v5_specs if x.policy == policy]:
            p2 = row_to_e5_plan(s)
            assert p2["lora_r"] == int(s.event_ranks[0])
            assert p2["seed"] == s.seed


def test_dynamic_policies_raise_limitation(v5_specs):
    for policy in sorted(UNSUPPORTED_DYNAMIC_POLICIES):
        spec = _spec_by_policy(v5_specs, policy)
        with pytest.raises(E5AdapterLimitation, match="stopped for review"):
            row_to_e5_plan(spec)


def test_limitation_text_is_precise(v5_specs):
    spec = _spec_by_policy(v5_specs, "Grow")
    with pytest.raises(E5AdapterLimitation) as ei:
        row_to_e5_plan(spec)
    msg = str(ei.value)
    assert "run_workload" in msg and "single fixed" in msg
    assert "not authorized" in msg and "prohibited" in msg
    assert msg == f"{spec.exec_id} (Grow): {RUN_WORKLOAD_RANK_LIMITATION}"


def test_reuse_rows_never_mapped(v5_specs):
    spec = _spec_by_policy(v5_specs, "Static-R8")  # REUSE anchor row
    with pytest.raises(E5AdapterLimitation, match="read-only"):
        row_to_e5_plan(spec)


# ---------------------------------------------------------------------------
# make_run_fn: mocked callbacks (never real training)
# ---------------------------------------------------------------------------


def test_adapter_mocked_invocation_contract(tmp_path, v5_specs):
    spec = _spec_by_policy(v5_specs, "Static-R9")
    calls = []
    run_dir = tmp_path / "run"

    def train_fn(spec_, run_dir_, plan):
        calls.append(("train", spec_.exec_id, plan["lora_r"], run_dir_))
        return {"exit_code": 0, "status": "complete"}

    def predict_fn(spec_, run_dir_, plan):
        calls.append(("eval", spec_.exec_id, run_dir_))
        return {"exit_code": 0, "status": "complete"}

    run_fn = make_run_fn(tmp_path, train_fn=train_fn, predict_fn=predict_fn)
    run_dir.mkdir()
    telemetry = run_fn(spec, run_dir)
    assert [c[0] for c in calls] == ["train", "eval"]
    assert calls[0][1] == calls[1][1] == spec.exec_id
    assert calls[0][2] == 9  # Static-R9
    assert calls[0][3] == run_dir and calls[1][2] == run_dir
    assert telemetry["training"]["exit_code"] == 0
    assert telemetry["evaluation"]["status"] == "complete"
    assert telemetry["e5_plan"]["lora_r"] == 9
    assert set(telemetry["artifact_refs"]) == {
        "execution_summary", "final_eval", "event_log", "audit_log",
        "bundles", "predictions"}


def test_adapter_training_failure_propagates(tmp_path, v5_specs):
    spec = _spec_by_policy(v5_specs, "Static-R16")

    def train_fn(spec_, run_dir_, plan):
        return {"exit_code": 1, "status": "killed"}

    run_fn = make_run_fn(tmp_path, train_fn=train_fn,
                         predict_fn=lambda *a: {"exit_code": 0, "status": "complete"})
    telemetry = run_fn(spec, tmp_path)
    assert telemetry["training"]["exit_code"] == 1
    assert telemetry["training"]["status"] == "killed"
    # the harness must refuse COMPLETED with this telemetry (fail closed)
    from cycleft.e7.execution_harness import StateError
    run_dir = tmp_path / "rd"
    run_dir.mkdir()
    manifest = {"exec_id": spec.exec_id, "state": "RUNNING"}
    h = ExecutionHarness(V5_MATRIX, tmp_path / "out", repo_root=tmp_path)
    with pytest.raises(StateError, match="exit code"):
        h._finalize_completed(run_dir, spec, manifest, telemetry=telemetry)


def test_adapter_evaluation_failure_propagates(tmp_path, v5_specs):
    spec = _spec_by_policy(v5_specs, "Static-R32")

    def predict_fn(spec_, run_dir_, plan):
        return {"exit_code": 3, "status": "error"}

    run_fn = make_run_fn(tmp_path, train_fn=lambda *a: {"exit_code": 0, "status": "complete"},
                         predict_fn=predict_fn)
    telemetry = run_fn(spec, tmp_path)
    assert telemetry["evaluation"]["exit_code"] == 3
    from cycleft.e7.execution_harness import StateError
    run_dir = tmp_path / "rd"
    run_dir.mkdir()
    manifest = {"exec_id": spec.exec_id, "state": "RUNNING"}
    h = ExecutionHarness(V5_MATRIX, tmp_path / "out", repo_root=tmp_path)
    with pytest.raises(StateError, match="evaluation"):
        h._finalize_completed(run_dir, spec, manifest, telemetry=telemetry)


def test_adapter_output_path_isolation(tmp_path, v5_specs):
    seen = []
    spec_a = _spec_by_policy(v5_specs, "Static-R4")
    spec_b = [s for s in v5_specs if s.policy == "Static-R4" and s.seed == 43][0]

    def train_fn(spec_, run_dir_, plan):
        seen.append((spec_.exec_id, Path(run_dir_).name))
        return {"exit_code": 0, "status": "complete"}

    run_fn = make_run_fn(tmp_path, train_fn=train_fn,
                         predict_fn=lambda *a: {"exit_code": 0, "status": "complete"})
    run_fn(spec_a, tmp_path / "run_a")
    run_fn(spec_b, tmp_path / "run_b")
    assert seen[0][0] != seen[1][0]
    assert seen[0][1] != seen[1][1], "each run must receive an isolated output path"


def test_adapter_validates_before_any_model_work(tmp_path, v5_specs):
    """row_to_e5_plan raises BEFORE callbacks would run (validation precedes
    model loading by construction)."""
    spec = _spec_by_policy(v5_specs, "Reduce")
    invoked = []

    def train_fn(spec_, run_dir_, plan):
        invoked.append("train")
        return {"exit_code": 0, "status": "complete"}

    run_fn = make_run_fn(tmp_path, train_fn=train_fn,
                         predict_fn=lambda *a: {"exit_code": 0, "status": "complete"})
    with pytest.raises(E5AdapterLimitation):
        run_fn(spec, tmp_path)
    assert invoked == [], "callbacks must not run when plan validation fails"


# ---------------------------------------------------------------------------
# lazy-import purity
# ---------------------------------------------------------------------------


def test_adapter_module_and_planning_import_no_training_dependencies():
    """Subprocess: importing the adapter and calling row_to_e5_plan must not
    import torch/transformers/peft/E5/E1 modules."""
    code = (
        "import sys, json;"
        "from cycleft.e7.e5_adapter import row_to_e5_plan;"
        "from cycleft.e7.execution_harness import load_and_validate_matrix;"
        f"specs = load_and_validate_matrix({str(V5_MATRIX)!r});"
        "static = [s for s in specs if s.policy == 'Static-R4'][0];"
        "row_to_e5_plan(static);"
        "bad = [m for m in ('torch', 'transformers', 'peft', 'cycleft.e5', "
        "'cycleft.e1', 'cycleft.finetuning') "
        "if any(x == m or x.startswith(m + '.') for x in sys.modules)];"
        "print(json.dumps({'bad': bad}))"
    )
    r = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        env={**os.environ,
             "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")})
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout.strip().splitlines()[-1])["bad"] == []


def test_default_callbacks_are_lazy_not_invoked_here(tmp_path, v5_specs):
    """make_run_fn with defaults (real E5-backed callbacks) wires lazily:
    nothing is imported at wiring time, and the dynamic-policy limitation
    fires before any training dependency could load. (Runtime laziness is
    proven hermetically by the subprocess tests; this in-process check is a
    static AST audit because pytest collection imports sibling test modules
    that legitimately import torch.)"""
    import ast
    from cycleft.e7 import e5_adapter
    tree = ast.parse(Path(e5_adapter.__file__).read_text())
    top_level = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_level.add(node.module.split(".")[0])
    assert top_level <= {"__future__", "json", "pathlib", "typing"}, \
        f"adapter top-level imports must be stdlib-only; found {sorted(top_level)}"
    run_fn = make_run_fn(tmp_path)  # defaults wired; nothing imported yet
    spec = _spec_by_policy(v5_specs, "Grow")
    with pytest.raises(E5AdapterLimitation):
        run_fn(spec, tmp_path)  # limitation fires pre-import


# ---------------------------------------------------------------------------
# harness + adapter end-to-end (mocked callbacks, real protocol)
# ---------------------------------------------------------------------------


def test_harness_with_mocked_e5_run_fn_completes(tmp_path, v5_specs):
    """Full protocol: harness + adapter-shaped run_fn producing schema-valid
    artifacts -> COMPLETED with consistent evidence."""
    from tests.test_e7_harness import full_telemetry, write_valid_artifacts  # noqa
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
    from test_e7_harness import full_telemetry, write_valid_artifacts

    matrix = V5_MATRIX
    out_root = tmp_path / "results" / "e7" / "exec"
    out_root.mkdir(parents=True)
    # synthetic anchors at the real relative paths (repo_root is the temp dir)
    for anchor_rel in ("results/e2/full_20261115/rank8_b500",
                       "results/e2/remediation_B_r8_b500_s43/rank8_b500",
                       "results/e2/remediation_B_r8_b500_s44/rank8_b500"):
        d = tmp_path / anchor_rel
        d.mkdir(parents=True, exist_ok=True)
        for fname in ("event_metrics.json", "resource_metrics.json",
                      "continual_learning_matrix.csv.json"):
            (d / fname).write_text(json.dumps({"synthetic": True}))

    def run_fn(spec, run_dir):
        plan = row_to_e5_plan(spec)
        write_valid_artifacts(run_dir)
        return {
            "training": {"exit_code": 0, "status": "complete",
                         "global_step": 1500,
                         "final_avg_em": 0.1},
            "evaluation": {"exit_code": 0, "status": "complete"},
            "e5_plan": plan,
        }

    harness = ExecutionHarness(matrix, out_root, repo_root=tmp_path, run_fn=run_fn)
    grow_id = [s.exec_id for s in v5_specs if s.policy == "Grow"][0]
    token = "i-authorize-e7-phase2-execution"
    # Dynamic policy: the adapter's callback raises E5AdapterLimitation; the
    # harness correctly wraps it in stop-on-failure semantics (FAILED state,
    # preserved reason, no further execution).
    from cycleft.e7.e5_adapter import RUN_WORKLOAD_RANK_LIMITATION
    with pytest.raises(HarnessError, match="E5AdapterLimitation"):
        harness.execute(token, run_id=grow_id)
    rec = harness.inspect_state(out_root / grow_id)
    assert rec["state"] == "FAILED"
    assert "single fixed lora_r/lora_alpha" in rec["failure_reason"]
    assert "stopped for review" in rec["failure_reason"]
    assert not (out_root / grow_id / "completion_marker.json").exists()

    static_id = [s.exec_id for s in v5_specs if s.policy == "Static-R9"][0]
    harness.execute(token, run_id=static_id)
    rec = harness.inspect_state(out_root / static_id)
    assert rec["classification"] == "COMPLETED" and not rec["fail_closed"]
    marker = json.loads((out_root / static_id / "completion_marker.json").read_text())
    assert marker["config_hash"] == \
        [s for s in v5_specs if s.exec_id == static_id][0].config_hash
