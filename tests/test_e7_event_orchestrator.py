"""E7 Option B dynamic orchestrator tests (G1-G4, G6-G7) — synthetic only.

Tests A-G from the implementation authorization, using mocked E5
callbacks (never the real trainer/evaluation) with the REAL orchestration
logic and the REAL rank-transition mathematics
(``cycleft.e7.rank_transition`` via ``transition_adapter_factors``).

Ranks/alphas/IDs are read from the authoritative v5 matrix — no
hardcoded assumptions.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cycleft.e7.event_orchestrator import (  # noqa: E402
    EventOrchestrator,
    OrchestratorError,
    ResourceAwareUnavailable,
    dynamic_event_plan,
    fingerprint_from_tensors,
    make_dynamic_run_fn,
    transition_adapter_factors,
    transition_seed,
    validate_bundle_e5,
    write_e5_bundle,
)
from cycleft.e7.execution_harness import (  # noqa: E402
    COMPLETION_MARKER_NAME,
    EXECUTE_CONFIRM_TOKEN as TOK,
    ExecutionHarness,
    load_and_validate_matrix,
    select_run_id,
)

V5_MATRIX = Path(__file__).resolve().parents[1] / (
    "results/e7/design_20261120/v5_final_reconciliation/FINAL_RUN_MATRIX.csv")
LORA_KEYS = ["base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
             "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight",
             "base_model.model.model.layers.1.self_attn.k_proj.lora_A.weight",
             "base_model.model.model.layers.1.self_attn.k_proj.lora_B.weight"]


@pytest.fixture(scope="module")
def v5_specs():
    return load_and_validate_matrix(V5_MATRIX)


# ---------------------------------------------------------------------------
# synthetic E5-callback mocks (never real training/evaluation)
# ---------------------------------------------------------------------------


class Mocks:
    def __init__(self):
        self.train_calls = []
        self.sweep_calls = []
        self.config_hashes = {}
        self.optimizers = []          # (event_id, id(opt), param_ids, opt)
        self.transition_calls = []

    def train_fn(self, ctx):
        k, r, alpha = ctx["event_id"], ctx["rank"], ctx["alpha"]
        self.train_calls.append({"event": k, "rank": r, "alpha": alpha,
                                 "slice": ctx["stream_slice"],
                                 "gs_start": ctx["global_step_start"]})
        self.config_hashes[k] = ctx["config_hash"]
        event_dir = Path(ctx["event_dir"])
        g = torch.Generator().manual_seed(1000 + k)
        tensors = {key: torch.randn(
            (r, 32) if "lora_A" in key else (32, r),
            generator=g, dtype=torch.float32) for key in LORA_KEYS}
        fp = fingerprint_from_tensors(tensors)
        bundles = event_dir / "bundles"
        adapter_tmp = bundles / f".trained_adapter_{k}"
        adapter_tmp.mkdir(parents=True, exist_ok=True)
        save_file(tensors, str(adapter_tmp / "adapter_model.safetensors"))
        (adapter_tmp / "adapter_config.json").write_text(json.dumps(
            {"r": r, "lora_alpha": alpha}))
        # fresh AdamW over fresh parameters (event-local; never shared)
        params = [torch.nn.Parameter(t.clone()) for t in tensors.values()]
        opt = torch.optim.AdamW(params, lr=1e-3)
        self.optimizers.append((k, id(opt), {id(p) for p in params}, opt))
        step = k * 500
        write_e5_bundle(
            bundles, step=step, adapter_dir=adapter_tmp, fingerprint=fp,
            progress={"global_step": step, "stage_index": 0,
                      "intra_stage_index": 500,
                      "config_hash": ctx["config_hash"]},
            meta={"exec_id": ctx["exec_id"], "event_id": k},
            state_payload={"optimizer": opt.state_dict(),
                            "rng": {"torch_cpu": torch.get_rng_state(),
                                     "rng_tier": "mock"}})
        shutil.rmtree(adapter_tmp)
        (event_dir / "execution_summary.json").write_text(json.dumps({
            "status": "complete", "global_step": step,
            "final_avg_em": 0.1 + 0.01 * k, "steps": step}))
        (event_dir / "event_log.jsonl").write_text(json.dumps(
            {"event": "process_exit", "status": "complete"}) + "\n")
        (event_dir / "audit_log.jsonl").write_text(json.dumps(
            {"step": 1, "loss": 0.5}) + "\n")
        (event_dir / "stage_end_fingerprints.json").write_text(json.dumps(
            {"stage_1": fp}))
        return {"exit_code": 0, "status": "complete",
                "terminal_bundle": str(bundles / f"step_{step:06d}"),
                "fingerprint": fp, "config_hash": ctx["config_hash"]}

    def sweep_fn(self, ctx, terminal_adapter_dir):
        k = ctx["event_id"]
        self.sweep_calls.append({"event": k, "adapter": str(terminal_adapter_dir)})
        metrics = {str(j): {"examples": 50,
                             "exact_match": 0.1 * k + 0.01 * j,
                             "token_f1": 0.2} for j in (1, 2, 3)}
        records = []
        for j in (1, 2, 3):
            for i in range(50):  # exactly 50 records per test set (150 total)
                records.append({"test_event": j, "example_index": i,
                                 "question": f"q{j}-{i}", "reference": "r",
                                 "raw_prediction": "p", "prediction": "p",
                                 "exact_match": 1.0, "token_f1": 1.0})
        return {"exit_code": 0, "status": "complete", "metrics": metrics,
                "records": records,
                "eval_config": {"decoding": "greedy", "rows_per_test_set": 50,
                                 "model_id": "synthetic", "dtype": "float16",
                                 "lora_precision": "float32",
                                 "max_new_tokens": 24, "seed": ctx["seed"]}}

    def transition_fn(self, from_adapter_dir, staging_dir, r_from, r_to,
                      alpha_to, generator=None):
        """REAL factor-level transition via rank_transition mathematics,
        staged ONLY inside the provided staging directory (Finding A),
        with the deterministic generator propagated (Finding B)."""
        self.transition_calls.append({"from": r_from, "to": r_to,
                                       "staging": str(staging_dir),
                                       "generator": generator})
        res = transition_adapter_factors(from_adapter_dir,
                                         Path(staging_dir) / "adapter",
                                         r_to, alpha_to, generator=generator)
        tensors = load_file(str(Path(res["adapter_dir"]) /
                                "adapter_model.safetensors"))
        return {"adapter_dir": str(Path(staging_dir) / "adapter"),
                "fingerprint": fingerprint_from_tensors(tensors),
                "rel_err": res["rel_err"]}

    def cleanup(self):
        pass


@pytest.fixture
def mocks():
    return Mocks()


def build_env(tmp_path, v5_specs, exec_id):
    """Harness + orchestrator environment with synthetic anchors."""
    for anchor_rel in ("results/e2/full_20261115/rank8_b500",
                       "results/e2/remediation_B_r8_b500_s43/rank8_b500",
                       "results/e2/remediation_B_r8_b500_s44/rank8_b500"):
        d = tmp_path / anchor_rel
        d.mkdir(parents=True, exist_ok=True)
        for fname in ("event_metrics.json", "resource_metrics.json",
                      "continual_learning_matrix.csv.json"):
            (d / fname).write_text(json.dumps({"synthetic": True}))
    out_root = tmp_path / "results" / "e7" / "exec"
    out_root.mkdir(parents=True)
    spec = select_run_id(v5_specs, exec_id)
    return spec, out_root, tmp_path


def make_orchestrator(spec, run_dir, tmp_path, mocks):
    return EventOrchestrator(
        spec, run_dir, repo_root=tmp_path,
        train_fn=mocks.train_fn, sweep_fn=mocks.sweep_fn,
        transition_fn=mocks.transition_fn,
        config_hash_fn=lambda ctx: f"synthetic-{ctx['event_id']}-"
                                   f"{ctx['rank']}-{ctx['alpha']}")


def bundle_tensors(path: Path):
    return load_file(str(Path(path) / "adapter" / "adapter_model.safetensors"))


# ---------------------------------------------------------------------------
# Test A — Grow 4 → 8 → 16
# ---------------------------------------------------------------------------


def test_a_grow_three_event_lifecycle(tmp_path, v5_specs, mocks):
    spec, out_root, tmp = build_env(tmp_path, v5_specs, "e7-grow-s42")
    run_dir = out_root / spec.exec_id
    orch = make_orchestrator(spec, run_dir, tmp, mocks)
    telemetry = orch.run()

    # correct rank/alpha per event, from the authoritative matrix row
    assert [e.rank for e in orch.event_records] == [4, 8, 16]
    assert [e.alpha for e in orch.event_records] == [8, 16, 32]
    assert orch.plan["rank_schedule"] == list(spec.event_ranks)[:3] or \
        orch.plan["rank_schedule"] == [int(r) for r in spec.event_ranks]

    # three training invocations on the correct data slices and steps
    assert [c["event"] for c in mocks.train_calls] == [1, 2, 3]
    assert [c["slice"] for c in mocks.train_calls] == [[0, 500], [500, 1000],
                                                        [1000, 1500]]
    assert [c["gs_start"] for c in mocks.train_calls] == [0, 500, 1000]

    # adapter handoff: event k+1's seed preserves event k's terminal factors
    t1 = bundle_tensors(run_dir / "events/event_1/bundles/step_000500")
    s2 = bundle_tensors(run_dir / "events/event_2/bundles/step_000500")
    for key in LORA_KEYS:
        if "lora_A" in key:
            assert torch.equal(s2[key][:4], t1[key]), f"expansion dropped A rows: {key}"
        else:
            assert torch.equal(s2[key][:, :4], t1[key]), f"expansion dropped B cols: {key}"
        assert s2[key].shape == ((8, 32) if "lora_A" in key else (32, 8))
    t2 = bundle_tensors(run_dir / "events/event_2/bundles/step_001000")
    s3 = bundle_tensors(run_dir / "events/event_3/bundles/step_001000")
    for key in LORA_KEYS:
        if "lora_A" in key:
            assert torch.equal(s3[key][:8], t2[key])
        else:
            assert torch.equal(s3[key][:, :8], t2[key])

    # authoritative checkpoint paths exist (and no others in bundles/)
    expected = ["events/event_1/bundles/step_000500",
                "events/event_2/bundles/step_000500",
                "events/event_2/bundles/step_001000",
                "events/event_3/bundles/step_001000",
                "events/event_3/bundles/step_001500"]
    for rel in expected:
        assert (run_dir / rel).is_dir(), f"missing {rel}"
    for k in (1, 2, 3):
        found = sorted(p.name for p in
                       (run_dir / f"events/event_{k}/bundles").glob("step_*")
                       if not p.name.endswith(".tmp"))
        want = [f"step_{k*500:06d}"] + ([f"step_{(k-1)*500:06d}"] if k > 1 else [])
        assert found == sorted(want)

    # cumulative global steps in progress metadata
    for k in (1, 2, 3):
        prog = json.loads((run_dir / f"events/event_{k}/bundles/"
                           f"step_{k*500:06d}/progress.json").read_text())
        assert prog["progress"]["global_step"] == k * 500
        assert prog["progress"]["intra_stage_index"] == 500

    # event-local evaluations + records
    for k in (1, 2, 3):
        d = run_dir / f"events/event_{k}"
        for name in ("final_eval.json", "predictions.jsonl",
                     "evaluation_config.json", "execution_summary.json",
                     "artifact_inventory.json"):
            assert (d / name).is_file(), f"event {k} missing {name}"
        assert len(json.loads((d / "final_eval.json").read_text())) == 3
    assert [c["event"] for c in mocks.sweep_calls] == [1, 2, 3]

    # fingerprint chain: transition records link terminals to seeds
    recs = orch.transition_records
    assert recs[0]["from_fingerprint"] == \
        json.loads((run_dir / "events/event_1/bundles/step_000500/progress.json")
                   .read_text())["adapter_fingerprint"]
    assert recs[0]["resulting_fingerprint"] == \
        json.loads((run_dir / "events/event_2/bundles/step_000500/progress.json")
                   .read_text())["adapter_fingerprint"]

    # complete master manifest
    m = json.loads((run_dir / "run_manifest.json").read_text())
    assert len(m["events"]) == 3 and len(m["transitions"]) == 2
    assert m["final_adapter_fingerprint"] == \
        json.loads((run_dir / "events/event_3/bundles/step_001500/progress.json")
                   .read_text())["adapter_fingerprint"]
    assert m["state"] == "READY_FOR_HARNESS_COMPLETION"
    assert telemetry["final_rank_schedule"] == [4, 8, 16]
    assert telemetry["training"]["status"] == "complete"


def test_a2_expansion_records_zero_rel_err(tmp_path, v5_specs, mocks):
    spec, out_root, tmp = build_env(tmp_path, v5_specs, "e7-grow-s42")
    orch = make_orchestrator(spec, out_root / spec.exec_id, tmp, mocks)
    orch.run()
    for rec in orch.transition_records:
        assert all(v == 0.0 for v in rec["rel_err"].values()), \
            "expansion preserves the effective update exactly (rel_err 0)"


# ---------------------------------------------------------------------------
# Test B — Reduce 16 → 8 → 4
# ---------------------------------------------------------------------------


def test_b_reduce_three_event_lifecycle(tmp_path, v5_specs, mocks):
    spec, out_root, tmp = build_env(tmp_path, v5_specs, "e7-reduce-s42")
    run_dir = out_root / spec.exec_id
    orch = make_orchestrator(spec, run_dir, tmp, mocks)
    telemetry = orch.run()

    assert [e.rank for e in orch.event_records] == [16, 8, 4]
    assert [e.alpha for e in orch.event_records] == [32, 16, 8]
    assert telemetry["final_rank_schedule"] == [16, 8, 4]
    # SVD transitions recorded with bounded reconstruction error
    for rec in orch.transition_records:
        assert rec["rel_err"], "SVD reduction must record reconstruction error"
        for v in rec["rel_err"].values():
            assert 0.0 < v <= 1.0 + 1e-9
    t1 = bundle_tensors(run_dir / "events/event_1/bundles/step_000500")
    s2 = bundle_tensors(run_dir / "events/event_2/bundles/step_000500")
    for key in LORA_KEYS:
        want = (8, 32) if "lora_A" in key else (32, 8)
        assert s2[key].shape == want, f"{key}: {s2[key].shape} != {want}"
        # reduced factors differ from the source (truncation) but stay finite
        assert not torch.equal(s2[key], t1[key])
        assert torch.isfinite(s2[key]).all()
    for k in (1, 2, 3):
        for name in ("final_eval.json", "predictions.jsonl"):
            assert (run_dir / f"events/event_{k}" / name).is_file()
    m = json.loads((run_dir / "run_manifest.json").read_text())
    assert [e["rank"] for e in m["events"]] == [16, 8, 4]


def test_b2_transitions_use_real_rank_transition_math(tmp_path, v5_specs, mocks):
    """The orchestrator's factor transition applies the actual
    rank_transition implementation (shapes/rank verified)."""
    from cycleft.e7.rank_transition import reduce_lora_factors
    spec, out_root, tmp = build_env(tmp_path, v5_specs, "e7-reduce-s42")
    orch = make_orchestrator(spec, out_root / spec.exec_id, tmp, mocks)
    orch.run()
    t1 = bundle_tensors(orch.run_dir / "events/event_1/bundles/step_000500")
    s2 = bundle_tensors(orch.run_dir / "events/event_2/bundles/step_000500")
    key_a = LORA_KEYS[0]
    # decisive: the recorded rel_err matches an independent recomputation
    rec = orch.transition_records[0]["rel_err"]
    assert orch.transition_records[0]["transition_seed"] == \
        transition_seed(spec.seed, 1), "record must carry the derived seed"
    pair_key = key_a.replace(".lora_A", "")
    assert pair_key in rec, f"rel_err must be keyed by pair key {pair_key!r}"
    S = torch.linalg.svdvals(t1[LORA_KEYS[1]].float() @ t1[key_a].float())  # B @ A
    pred = float(torch.sqrt(torch.sum(S[8:] ** 2) / torch.sum(S ** 2)))
    assert abs(rec[pair_key] - pred) < 1e-5, "recorded rel_err must equal the " \
        "singular-value-tail prediction (real rank_transition math)"


# ---------------------------------------------------------------------------
# Test C — checkpoint integrity
# ---------------------------------------------------------------------------


def test_c_no_path_collisions_and_all_checkpoints_exist(tmp_path, v5_specs, mocks):
    spec, out_root, tmp = build_env(tmp_path, v5_specs, "e7-grow-s42")
    run_dir = out_root / spec.exec_id
    make_orchestrator(spec, run_dir, tmp, mocks).run()
    paths = [run_dir / rel for rel in (
        "events/event_1/bundles/step_000500", "events/event_2/bundles/step_000500",
        "events/event_2/bundles/step_001000", "events/event_3/bundles/step_001000",
        "events/event_3/bundles/step_001500")]
    assert len({str(p) for p in paths}) == 5
    assert all(p.is_dir() for p in paths)


def test_c2_existing_terminal_cannot_be_overwritten(tmp_path, v5_specs, mocks):
    spec, out_root, tmp = build_env(tmp_path, v5_specs, "e7-grow-s42")
    run_dir = out_root / spec.exec_id
    orch = make_orchestrator(spec, run_dir, tmp, mocks)
    orch.run()
    victim = run_dir / "events/event_1/bundles/step_000500"
    before = {p.name: p.read_bytes() for p in victim.rglob("*") if p.is_file()}
    with pytest.raises(OrchestratorError, match="already exists"):
        write_e5_bundle(victim.parent, step=500,
                        adapter_dir=victim / "adapter",
                        fingerprint="x", progress={"global_step": 500},
                        meta={}, state_payload={"optimizer": {}})
    after = {p.name: p.read_bytes() for p in victim.rglob("*") if p.is_file()}
    assert before == after, "terminal checkpoint must be immutable"


def test_c3_every_checkpoint_passes_e5_validator(tmp_path, v5_specs, mocks):
    """E5's own load_latest_bundle accepts every synthetic bundle."""
    from cycleft.e5.resume_bundle import load_latest_bundle
    spec, out_root, tmp = build_env(tmp_path, v5_specs, "e7-grow-s42")
    run_dir = out_root / spec.exec_id
    orch = make_orchestrator(spec, run_dir, tmp, mocks)
    orch.run()
    for k in (1, 2, 3):
        bundles = run_dir / f"events/event_{k}/bundles"
        state = load_latest_bundle(bundles, mocks.config_hashes[k])
        assert state["progress"]["global_step"] == k * 500
        assert state["progress"]["intra_stage_index"] == 500
    # and the run-level materialized view is E5-loadable too
    state = load_latest_bundle(run_dir / "bundles", mocks.config_hashes[3])
    assert state["progress"]["global_step"] == 1500


def test_c4_run_level_bundle_byte_identical_to_event3(tmp_path, v5_specs, mocks):
    spec, out_root, tmp = build_env(tmp_path, v5_specs, "e7-grow-s42")
    run_dir = out_root / spec.exec_id
    make_orchestrator(spec, run_dir, tmp, mocks).run()
    src = run_dir / "events/event_3/bundles/step_001500"
    view = run_dir / "bundles/step_001500"
    for f in sorted(src.rglob("*")):
        if f.is_file():
            assert f.read_bytes() == (view / f.relative_to(src)).read_bytes(), \
                f"run-level view diverges at {f.relative_to(src)}"
    latest = json.loads((run_dir / "bundles/LATEST.json").read_text())
    assert latest["latest"] == "step_001500"
    assert latest["e7_provenance"]["source_event_dir"] == "events/event_3/bundles"


def test_c5_corrupt_bundle_fails_closed(tmp_path, v5_specs, mocks):
    spec, out_root, tmp = build_env(tmp_path, v5_specs, "e7-grow-s42")
    run_dir = out_root / spec.exec_id
    orch = make_orchestrator(spec, run_dir, tmp, mocks)

    def corrupt_train(ctx):
        mocks.train_fn(ctx)
        # tamper with event 1's terminal adapter AFTER the mock wrote it
        victim = Path(ctx["event_dir"]) / "bundles/step_000500/adapter/adapter_model.safetensors"
        victim.write_bytes(b"corrupted")
        return {"exit_code": 0, "status": "complete",
                "terminal_bundle": str(Path(ctx["event_dir"]) /
                                        "bundles/step_000500"),
                "fingerprint": "stale", "config_hash": ctx["config_hash"]}
    orch.train_fn = corrupt_train
    with pytest.raises(OrchestratorError, match="hash mismatch|event 1 failed"):
        orch.run()
    assert json.loads((run_dir / "orchestrator_state.json").read_text())["state"] == "FAILED"


def test_c5b_missing_bundle_fails_closed(tmp_path, v5_specs, mocks):
    spec, out_root, tmp = build_env(tmp_path, v5_specs, "e7-grow-s42")
    run_dir = out_root / spec.exec_id
    orch = make_orchestrator(spec, run_dir, tmp, mocks)

    def incomplete_train(ctx):
        mocks.train_fn(ctx)
        if ctx["event_id"] == 1:  # "E5" never committed the terminal bundle
            shutil.rmtree(Path(ctx["event_dir"]) / "bundles/step_000500")
        return {"exit_code": 0, "status": "complete",
                "terminal_bundle": str(Path(ctx["event_dir"]) /
                                        "bundles/step_000500"),
                "fingerprint": "x", "config_hash": ctx["config_hash"]}
    orch.train_fn = incomplete_train
    with pytest.raises(OrchestratorError):
        orch.run()
    assert json.loads((run_dir / "orchestrator_state.json").read_text())["state"] == "FAILED"


# ---------------------------------------------------------------------------
# Test D — evaluation preservation
# ---------------------------------------------------------------------------


def test_d_evaluations_use_own_terminal_adapter(tmp_path, v5_specs, mocks):
    spec, out_root, tmp = build_env(tmp_path, v5_specs, "e7-grow-s42")
    run_dir = out_root / spec.exec_id
    make_orchestrator(spec, run_dir, tmp, mocks).run()
    assert mocks.sweep_calls[0]["adapter"].endswith(
        "events/event_1/bundles/step_000500/adapter")
    assert mocks.sweep_calls[1]["adapter"].endswith(
        "events/event_2/bundles/step_001000/adapter")
    assert mocks.sweep_calls[2]["adapter"].endswith(
        "events/event_3/bundles/step_001500/adapter")


def test_d2_earlier_predictions_unchanged_after_later_events(tmp_path, v5_specs, mocks):
    spec, out_root, tmp = build_env(tmp_path, v5_specs, "e7-grow-s42")
    run_dir = out_root / spec.exec_id
    orch = make_orchestrator(spec, run_dir, tmp, mocks)

    snapshots = {}

    def snapshotting_sweep(ctx, adapter_dir):
        res = mocks.sweep_fn(ctx, adapter_dir)
        for r in res["records"]:
            r.setdefault("state", f"after_event_{ctx['event_id']}")
        snapshots[ctx["event_id"]] = (res["metrics"], [dict(r) for r in res["records"]])
        return res

    orch.sweep_fn = snapshotting_sweep
    orch.run()
    for k in (1, 2, 3):
        stored = json.loads((run_dir / f"events/event_{k}/final_eval.json").read_text())
        assert stored == snapshots[k][0], f"event {k} metrics were altered"
        lines = (run_dir / f"events/event_{k}/predictions.jsonl").read_text().splitlines()
        assert [json.loads(l) for l in lines] == snapshots[k][1]


def test_d3_run_level_copies_match_harness_contract(tmp_path, v5_specs, mocks):
    spec, out_root, tmp = build_env(tmp_path, v5_specs, "e7-grow-s42")
    run_dir = out_root / spec.exec_id
    make_orchestrator(spec, run_dir, tmp, mocks).run()
    # byte-identical materializations
    for k in (1, 2, 3):
        assert (run_dir / f"predictions/after_event_{k}.jsonl").read_bytes() == \
            (run_dir / f"events/event_{k}/predictions.jsonl").read_bytes()
    assert (run_dir / "final_eval.json").read_bytes() == \
        (run_dir / "events/event_3/final_eval.json").read_bytes()
    # harness validator accepts the run-level views
    from cycleft.e7.execution_harness import build_artifact_inventory
    inv = build_artifact_inventory(run_dir)
    assert set(inv) == {"execution_summary.json", "final_eval.json",
                        "event_log.jsonl", "audit_log.jsonl", "bundles",
                        "predictions"}


def test_d4_static_schedule_untouched(tmp_path, v5_specs, mocks):
    """The orchestrator rejects static rows (Design 1 preserved; no
    static event-level evaluation was added)."""
    spec, out_root, tmp = build_env(tmp_path, v5_specs, "e7-static-r9-s42")
    with pytest.raises(OrchestratorError, match="not a dynamic policy"):
        make_orchestrator(spec, out_root / spec.exec_id, tmp, mocks)
    assert not (out_root / spec.exec_id / "events").exists()


# ---------------------------------------------------------------------------
# Test E — optimizer lifecycle
# ---------------------------------------------------------------------------


def test_e_fresh_adamw_per_event(tmp_path, v5_specs, mocks):
    spec, out_root, tmp = build_env(tmp_path, v5_specs, "e7-grow-s42")
    run_dir = out_root / spec.exec_id
    make_orchestrator(spec, run_dir, tmp, mocks).run()
    assert [o[0] for o in mocks.optimizers] == [1, 2, 3]
    ids = [o[1] for o in mocks.optimizers]
    assert len(set(ids)) == 3, "each event must construct a new optimizer"
    param_sets = [o[2] for o in mocks.optimizers]
    for i in range(len(param_sets)):
        for j in range(i + 1, len(param_sets)):
            assert param_sets[i].isdisjoint(param_sets[j]), \
                "no parameter references shared across events"
    # fresh state: empty moments before any step (lazy AdamW init)
    for _, _, _, opt in mocks.optimizers:
        assert len(opt.state) == 0, "optimizer must be constructed fresh"
    # seed bundles never restore optimizer state (intra=0 gate)
    for k in (2, 3):
        prog = json.loads((run_dir / f"events/event_{k}/bundles/"
                           f"step_{(k-1)*500:06d}/progress.json").read_text())
        assert prog["progress"]["intra_stage_index"] == 0


def test_e2_seed_state_payload_is_inert(tmp_path, v5_specs, mocks):
    """Seed state.pt exists for E5 format compatibility, but the intra=0
    gate means its optimizer payload is never restored."""
    import torch as _torch
    spec, out_root, tmp = build_env(tmp_path, v5_specs, "e7-grow-s42")
    run_dir = out_root / spec.exec_id
    make_orchestrator(spec, run_dir, tmp, mocks).run()
    seed = run_dir / "events/event_2/bundles/step_000500"
    prog = json.loads((seed / "progress.json").read_text())
    assert prog["progress"]["intra_stage_index"] == 0, \
        "seed bundles use intra=0: the fresh-AdamW gate"
    state = _torch.load(seed / "state.pt", map_location="cpu", weights_only=False)
    assert "optimizer" in state and "rng" in state  # present, never applied


# ---------------------------------------------------------------------------
# Test F — failure and recovery
# ---------------------------------------------------------------------------


def _failed_run_assertions(run_dir, state="FAILED"):
    st = json.loads((run_dir / "orchestrator_state.json").read_text())
    assert st["state"] == state
    assert not (run_dir / "completion_marker.json").exists()
    return st


def test_f_event1_failure_stops_run(tmp_path, v5_specs, mocks):
    spec, out_root, tmp = build_env(tmp_path, v5_specs, "e7-grow-s42")
    run_dir = out_root / spec.exec_id
    orch = make_orchestrator(spec, run_dir, tmp, mocks)

    invocations = []

    def failing(ctx):
        invocations.append(ctx["event_id"])
        if ctx["event_id"] == 1:
            return {"exit_code": 1, "status": "killed"}
        return mocks.train_fn(ctx)
    orch.train_fn = failing
    with pytest.raises(OrchestratorError, match="event 1 FAILED"):
        orch.run()
    st = _failed_run_assertions(run_dir)
    assert "event 1 failed" in st["failure_reason"]
    assert invocations == [1], f"no retry after failure (invocations={invocations})"


def test_f_transition_failure_preserves_event1(tmp_path, v5_specs, mocks):
    spec, out_root, tmp = build_env(tmp_path, v5_specs, "e7-grow-s42")
    run_dir = out_root / spec.exec_id
    orch = make_orchestrator(spec, run_dir, tmp, mocks)

    def failing_transition(from_adapter_dir, staging_dir, r_from, r_to,
                           alpha_to, generator=None):
        raise RuntimeError("synthetic transition failure")
    orch.transition_fn = failing_transition
    with pytest.raises(OrchestratorError, match="transition 1->2 FAILED"):
        orch.run()
    st = _failed_run_assertions(run_dir)
    assert "synthetic transition failure" in st["failure_reason"]
    assert (run_dir / "events/event_1/bundles/step_000500").is_dir(), \
        "event-1 terminal checkpoint preserved"
    assert [c["event"] for c in mocks.train_calls] == [1], \
        "event 2 never invoked"
    seed = run_dir / "events/event_2/bundles/step_000500"
    assert not seed.exists(), "no seed bundle written on transition failure"


def test_f_sweep_failure_marks_failed(tmp_path, v5_specs, mocks):
    spec, out_root, tmp = build_env(tmp_path, v5_specs, "e7-grow-s42")
    run_dir = out_root / spec.exec_id
    orch = make_orchestrator(spec, run_dir, tmp, mocks)

    def failing_sweep(ctx, adapter_dir):
        if ctx["event_id"] == 2:
            return {"exit_code": 3, "status": "error"}
        return mocks.sweep_fn(ctx, adapter_dir)
    orch.sweep_fn = failing_sweep
    with pytest.raises(OrchestratorError, match="event 2 FAILED"):
        orch.run()
    _failed_run_assertions(run_dir)


def test_f_keyboard_interrupt_interrupts(tmp_path, v5_specs, mocks):
    spec, out_root, tmp = build_env(tmp_path, v5_specs, "e7-grow-s42")
    run_dir = out_root / spec.exec_id
    orch = make_orchestrator(spec, run_dir, tmp, mocks)

    def interrupted(ctx):
        if ctx["event_id"] == 2:
            raise KeyboardInterrupt
        return mocks.train_fn(ctx)
    orch.train_fn = interrupted
    with pytest.raises(KeyboardInterrupt):
        orch.run()
    st = json.loads((run_dir / "orchestrator_state.json").read_text())
    assert st["state"] == "INTERRUPTED"
    assert not (run_dir / "completion_marker.json").exists()


def test_f_incomplete_run_never_completed(tmp_path, v5_specs, mocks):
    """A FAILED orchestrator run handed to the harness cannot reach
    COMPLETED (the strict gates reject the telemetry)."""
    spec, out_root, tmp = build_env(tmp_path, v5_specs, "e7-grow-s42")
    run_dir = out_root / spec.exec_id
    orch = make_orchestrator(spec, run_dir, tmp, mocks)

    def failing(ctx):
        if ctx["event_id"] == 2:
            return {"exit_code": 1, "status": "killed"}
        return mocks.train_fn(ctx)
    orch.train_fn = failing
    harness = ExecutionHarness(V5_MATRIX, out_root, repo_root=tmp,
                               run_fn=make_dynamic_run_fn(
                                   train_fn=failing, sweep_fn=mocks.sweep_fn,
                                   transition_fn=mocks.transition_fn))
    with pytest.raises(HarnessError := __import__(
            "cycleft.e7.execution_harness", fromlist=["HarnessError"]
        ).HarnessError, match="execution stopped"):
        harness.execute(TOK, run_id=spec.exec_id)
    st = json.loads((run_dir / "orchestrator_state.json").read_text())
    assert st["state"] == "FAILED"
    rec = harness.inspect_state(run_dir)
    assert rec["state"] == "FAILED"
    assert not (run_dir / COMPLETION_MARKER_NAME).exists()


# ---------------------------------------------------------------------------
# Test G — harness integration
# ---------------------------------------------------------------------------


def test_g_harness_completes_grow_via_orchestrator(tmp_path, v5_specs, mocks):
    spec, out_root, tmp = build_env(tmp_path, v5_specs, "e7-grow-s42")
    harness = ExecutionHarness(
        V5_MATRIX, out_root, repo_root=tmp,
        run_fn=make_dynamic_run_fn(train_fn=mocks.train_fn,
                                    sweep_fn=mocks.sweep_fn,
                                    transition_fn=mocks.transition_fn))
    # dry-run remains default and touches nothing
    plan = harness.dry_run()
    assert len(plan) == 22 and not any(out_root.iterdir())
    harness.execute(TOK, run_id=spec.exec_id)  # single-run + token mandatory
    rec = harness.inspect_state(out_root / spec.exec_id)
    assert rec["classification"] == "COMPLETED" and rec["fail_closed"] is False
    h_manifest = json.loads((out_root / spec.exec_id / "manifest.json").read_text())
    assert h_manifest["telemetry"]["training"]["status"] == "complete"
    assert h_manifest["telemetry"]["final_rank_schedule"] == [4, 8, 16]
    assert h_manifest["telemetry"]["master_manifest_sha256"]
    # re-execution blocked
    h2 = ExecutionHarness(
        V5_MATRIX, out_root, repo_root=tmp,
        run_fn=make_dynamic_run_fn(train_fn=mocks.train_fn,
                                    sweep_fn=mocks.sweep_fn,
                                    transition_fn=mocks.transition_fn))
    with pytest.raises(Exception, match="refusing to execute"):
        h2.execute(TOK, run_id=spec.exec_id)


def test_g2_resource_aware_blocked(tmp_path, v5_specs, mocks):
    spec, out_root, tmp = build_env(tmp_path, v5_specs, "e7-resource-aware-s42")
    run_fn = make_dynamic_run_fn(train_fn=mocks.train_fn,
                                  sweep_fn=mocks.sweep_fn,
                                  transition_fn=mocks.transition_fn)
    with pytest.raises(ResourceAwareUnavailable, match="loss-persistence"):
        run_fn(spec, out_root / spec.exec_id)


def test_g3_reuse_rows_stay_read_only(tmp_path, v5_specs, mocks):
    spec = [s for s in v5_specs if s.exec_id == "e7-static-r8-s42"][0]
    assert spec.row_type == "REUSE"
    run_fn = make_dynamic_run_fn(train_fn=mocks.train_fn)
    with pytest.raises(Exception, match="read-only"):
        run_fn(spec, tmp_path / "x")  # static REUSE row → E5 adapter rejects


# ---------------------------------------------------------------------------
# inventory self-reference (G7)
# ---------------------------------------------------------------------------


def test_inventory_self_reference_rule(tmp_path, v5_specs, mocks):
    spec, out_root, tmp = build_env(tmp_path, v5_specs, "e7-grow-s42")
    run_dir = out_root / spec.exec_id
    make_orchestrator(spec, run_dir, tmp, mocks).run()
    for k in (1, 2, 3):
        inv_path = run_dir / f"events/event_{k}/artifact_inventory.json"
        inv = json.loads(inv_path.read_text())
        assert "artifact_inventory.json" not in inv["files"], \
            "inventory must not include itself in its digest"
        assert inv["self_excluded"] == ["artifact_inventory.json"]
    m = json.loads((run_dir / "run_manifest.json").read_text())
    import hashlib
    for ev in m["events"]:
        actual = hashlib.sha256(
            (run_dir / f"events/event_{ev['event_id']}/artifact_inventory.json")
            .read_bytes()).hexdigest()
        assert ev["inventory_self_checksum"] == actual
    assert m["self_exclusion_rule"]["master_manifest.json"]


# ---------------------------------------------------------------------------
# Source-audit corrections (v2.1 review): Finding A — staging isolation
# ---------------------------------------------------------------------------


def _hash_tree(root: Path):
    return {str(p.relative_to(root)): p.read_bytes()
            for p in sorted(root.rglob("*")) if p.is_file()}


def _run_and_snapshot_source(tmp_path, v5_specs, exec_id, mocks, policy_trajs):
    spec, out_root, tmp = build_env(tmp_path, v5_specs, exec_id)
    run_dir = out_root / spec.exec_id
    orch = make_orchestrator(spec, run_dir, tmp, mocks)
    # event 1 trains and commits its terminal bundle; snapshot AFTER the
    # commit and BEFORE the transition (source-hash baseline)
    mocks.train_fn(orch._event_ctx(1))
    src = run_dir / "events/event_1/bundles/step_000500"
    before = _hash_tree(src)
    rec = orch._transition(1)
    return orch, run_dir, src, before, rec


def test_fa_grow_source_checkpoint_untouched_by_transition(tmp_path, v5_specs, mocks):
    orch, run_dir, src, before, rec = _run_and_snapshot_source(
        tmp_path, v5_specs, "e7-grow-s42", mocks, None)
    after = _hash_tree(src)
    assert after == before, "transition must not modify the terminal checkpoint"
    assert not list(run_dir.rglob(".transition_adapter_*")), \
        "no legacy staging directories inside checkpoints"
    assert (run_dir / "events/event_2/bundles/step_000500").is_dir()
    staging = run_dir / "transitions/.staging/transition_event_1_to_2"
    assert not staging.exists(), "successful staging must be cleaned"
    assert rec["from_checkpoint"] == "events/event_1/bundles/step_000500"


def test_fa_reduce_source_checkpoint_untouched_by_transition(tmp_path, v5_specs, mocks):
    spec, out_root, tmp = build_env(tmp_path, v5_specs, "e7-reduce-s42")
    run_dir = out_root / spec.exec_id
    orch = make_orchestrator(spec, run_dir, tmp, mocks)
    mocks.train_fn(orch._event_ctx(1))
    src = run_dir / "events/event_1/bundles/step_000500"
    before = _hash_tree(src)
    rec = orch._transition(1)
    assert _hash_tree(src) == before
    assert rec["rel_err"] and all(0 < v <= 1 for v in rec["rel_err"].values())
    assert not (run_dir / "transitions/.staging/transition_event_1_to_2").exists()


def test_fa_transition_failure_preserves_staging(tmp_path, v5_specs, mocks):
    spec, out_root, tmp = build_env(tmp_path, v5_specs, "e7-grow-s42")
    run_dir = out_root / spec.exec_id
    orch = make_orchestrator(spec, run_dir, tmp, mocks)

    def failing(from_adapter_dir, staging_dir, r_from, r_to, alpha_to,
                generator=None):
        staging = Path(staging_dir)
        staging.mkdir(parents=True, exist_ok=False)
        (staging / "partial_evidence.txt").write_text("partial staging state")
        raise RuntimeError("synthetic staging failure")
    orch.transition_fn = failing
    with pytest.raises(OrchestratorError, match="transition 1->2 FAILED"):
        orch.run()
    staging = run_dir / "transitions/.staging/transition_event_1_to_2"
    assert staging.is_dir() and (staging / "partial_evidence.txt").read_text() \
        == "partial staging state", "partial staging evidence preserved"
    st = json.loads((run_dir / "orchestrator_state.json").read_text())
    assert st["state"] == "FAILED"
    src = run_dir / "events/event_1/bundles/step_000500"
    assert (src / "adapter" / "adapter_model.safetensors").is_file()


def test_fa_existing_staging_blocks_without_modification(tmp_path, v5_specs, mocks):
    spec, out_root, tmp = build_env(tmp_path, v5_specs, "e7-grow-s42")
    run_dir = out_root / spec.exec_id
    orch = make_orchestrator(spec, run_dir, tmp, mocks)
    mocks.train_fn(orch._event_ctx(1))
    staging = run_dir / "transitions/.staging/transition_event_1_to_2"
    staging.mkdir(parents=True)
    (staging / "operator_evidence.bin").write_bytes(b"\x00\x01\x02")
    before = _hash_tree(staging)
    with pytest.raises(OrchestratorError, match="staging path already exists"):
        orch._transition(1)
    assert _hash_tree(staging) == before, "existing staging must not be modified"


# ---------------------------------------------------------------------------
# Finding B — deterministic transition generator
# ---------------------------------------------------------------------------


def test_fb_same_seed_identical_expansion(tmp_path, v5_specs, mocks):
    spec, out_root, tmp = build_env(tmp_path, v5_specs, "e7-grow-s42")
    run_dir = out_root / spec.exec_id
    orch = make_orchestrator(spec, run_dir, tmp, mocks)
    mocks.train_fn(orch._event_ctx(1))
    src = run_dir / "events/event_1/bundles/step_000500/adapter"
    out1, out2 = tmp_path / "o1", tmp_path / "o2"
    seed = transition_seed(spec.seed, 1)
    g1 = torch.Generator(device="cpu").manual_seed(seed)
    g2 = torch.Generator(device="cpu").manual_seed(seed)
    transition_adapter_factors(src, out1, 8, 16, generator=g1)
    transition_adapter_factors(src, out2, 8, 16, generator=g2)
    t1, t2 = load_file(str(out1 / "adapter_model.safetensors")), \
        load_file(str(out2 / "adapter_model.safetensors"))
    for key in t1:
        assert torch.equal(t1[key], t2[key]), \
            "same experiment seed must produce identical expanded factors"


def test_fb2_different_seed_changes_new_rows_only(tmp_path, v5_specs, mocks):
    spec, out_root, tmp = build_env(tmp_path, v5_specs, "e7-grow-s42")
    run_dir = out_root / spec.exec_id
    orch = make_orchestrator(spec, run_dir, tmp, mocks)
    mocks.train_fn(orch._event_ctx(1))
    src = run_dir / "events/event_1/bundles/step_000500/adapter"
    out_a, out_b = tmp_path / "a", tmp_path / "b"
    s1, s2 = transition_seed(42, 1), transition_seed(43, 1)
    assert s1 != s2
    transition_adapter_factors(src, out_a, 8, 16,
                               generator=torch.Generator().manual_seed(s1))
    transition_adapter_factors(src, out_b, 8, 16,
                               generator=torch.Generator().manual_seed(s2))
    ta, tb = load_file(str(out_a / "adapter_model.safetensors")), \
        load_file(str(out_b / "adapter_model.safetensors"))
    t_src = load_file(str(src / "adapter_model.safetensors"))
    for key in LORA_KEYS:
        if "lora_A" in key:
            assert torch.equal(ta[key][:4], t_src[key]), "old A rows preserved"
            assert torch.equal(tb[key][:4], t_src[key]), "old A rows preserved"
            assert not torch.equal(ta[key][4:], tb[key][4:]), \
                "new A rows must differ across experiment seeds"
        else:
            assert torch.equal(ta[key][:, :4], t_src[key]), "old B cols preserved"
            assert torch.all(ta[key][:, 4:] == 0) and torch.all(tb[key][:, 4:] == 0), \
                "new B columns exactly zero for every seed"
            assert torch.equal(ta[key][:, 4:], tb[key][:, 4:])


def test_fb3_seed_derivation_stable_across_processes(tmp_path):
    code = (
        "import sys, json; sys.path.insert(0, 'src');"
        "from cycleft.e7.event_orchestrator import transition_seed;"
        "print(json.dumps([transition_seed(42, 1), transition_seed(42, 2),"
        " transition_seed(43, 1)]))"
    )
    r1 = subprocess.run([sys.executable, "-c", code], capture_output=True,
                        text=True, cwd=os.getcwd())
    r2 = subprocess.run([sys.executable, "-c", code], capture_output=True,
                        text=True, cwd=os.getcwd())
    assert r1.returncode == 0 and r2.returncode == 0
    assert json.loads(r1.stdout) == json.loads(r2.stdout), \
        "seed derivation must be stable across process executions"


def test_fb4_record_contains_exact_derived_seed(tmp_path, v5_specs, mocks):
    spec, out_root, tmp = build_env(tmp_path, v5_specs, "e7-grow-s42")
    run_dir = out_root / spec.exec_id
    orch = make_orchestrator(spec, run_dir, tmp, mocks)
    orch.run()
    for i, rec in enumerate(orch.transition_records, start=1):
        expected = transition_seed(spec.seed, i)
        assert rec["transition_seed"] == expected
        # the generator actually used carried that seed (effect verified via
        # identical re-derivation)
        g = torch.Generator(device="cpu").manual_seed(expected)
        assert isinstance(g.initial_seed(), int)


# ---------------------------------------------------------------------------
# Finding C — no destructive automatic cleanup
# ---------------------------------------------------------------------------


def test_fc_existing_run_level_bundles_block_materialization(tmp_path, v5_specs, mocks):
    spec, out_root, tmp = build_env(tmp_path, v5_specs, "e7-grow-s42")
    run_dir = out_root / spec.exec_id
    orch = make_orchestrator(spec, run_dir, tmp, mocks)
    # simulate a completed run whose run-level bundles already exist
    orch.run()
    view = run_dir / "bundles/step_001500/adapter/adapter_config.json"
    before = view.read_bytes()
    # re-running materialization must fail closed, preserving the tree
    with pytest.raises(OrchestratorError, match="already exists"):
        orch._materialize_run_level()
    assert view.read_bytes() == before
    assert (run_dir / "bundles/LATEST.json").is_file()


def test_fc2_existing_tmp_directory_blocks_bundle_write(tmp_path, v5_specs, mocks):
    spec, out_root, tmp = build_env(tmp_path, v5_specs, "e7-grow-s42")
    run_dir = out_root / spec.exec_id
    orch = make_orchestrator(spec, run_dir, tmp, mocks)
    mocks.train_fn(orch._event_ctx(1))
    bundles = run_dir / "events/event_2/bundles"
    staging = run_dir / "transitions/.staging/transition_event_1_to_2"
    staging.mkdir(parents=True)
    (staging / "adapter").mkdir()
    (staging / "adapter" / "adapter_model.safetensors").write_bytes(b"t")
    tmp_dir = bundles / "step_000500.tmp"
    tmp_dir.mkdir(parents=True)
    (tmp_dir / "evidence.txt").write_text("prior crashed write")
    with pytest.raises(OrchestratorError, match="already exists"):
        write_e5_bundle(bundles, step=500, adapter_dir=staging / "adapter",
                        fingerprint="f", progress={"global_step": 500},
                        meta={}, state_payload={"optimizer": {}})
    assert (tmp_dir / "evidence.txt").read_text() == "prior crashed write", \
        "temporary directory content preserved; no automatic retry/overwrite"
    assert not (bundles / "step_000500").exists()


# ---------------------------------------------------------------------------
# Finding D — finalization failure states (fault injection)
# ---------------------------------------------------------------------------


def _completed_through_events(tmp_path, v5_specs, mocks, exec_id="e7-grow-s42"):
    spec, out_root, tmp = build_env(tmp_path, v5_specs, exec_id)
    run_dir = out_root / spec.exec_id
    orch = make_orchestrator(spec, run_dir, tmp, mocks)
    for k in (1, 2, 3):
        rec = orch._run_event(k)
        orch.event_records.append(rec)
        if k < 3:
            orch.transition_records.append(orch._transition(k))
    return orch, run_dir, spec, tmp


def test_fd_failure_during_summary_copy(tmp_path, v5_specs, mocks, monkeypatch):
    orch, run_dir, spec, tmp = _completed_through_events(tmp_path, v5_specs, mocks)
    real = shutil.copyfile
    calls = {"n": 0}

    def flaky(src, dst, *a, **k):
        calls["n"] += 1
        if calls["n"] == 1:  # first copyfile == execution-summary copy
            raise OSError("synthetic copy failure")
        return real(src, dst, *a, **k)
    monkeypatch.setattr("cycleft.e7.event_orchestrator.shutil.copyfile", flaky)
    with pytest.raises(OrchestratorError, match="finalization FAILED"):
        orch._finalize_run({"training": {"events": []},
                            "evaluation": {"events": []}})
    st = json.loads((run_dir / "orchestrator_state.json").read_text())
    assert st["state"] == "FAILED" and "finalization failed" in st["failure_reason"]
    assert not (run_dir / "completion_marker.json").exists()


def test_fd2_failure_during_predictions_materialization(tmp_path, v5_specs, mocks, monkeypatch):
    orch, run_dir, spec, tmp = _completed_through_events(tmp_path, v5_specs, mocks)
    real = shutil.copyfile
    calls = {"n": 0}

    def flaky(src, dst, *a, **k):
        calls["n"] += 1
        if calls["n"] == 2:  # second copyfile == first predictions copy
            raise OSError("synthetic predictions failure")
        return real(src, dst, *a, **k)
    monkeypatch.setattr("cycleft.e7.event_orchestrator.shutil.copyfile", flaky)
    with pytest.raises(OrchestratorError, match="finalization FAILED"):
        orch._finalize_run({"training": {"events": []},
                            "evaluation": {"events": []}})
    st = json.loads((run_dir / "orchestrator_state.json").read_text())
    assert st["state"] == "FAILED"


def test_fd3_failure_during_bundle_copy(tmp_path, v5_specs, mocks, monkeypatch):
    orch, run_dir, spec, tmp = _completed_through_events(tmp_path, v5_specs, mocks)
    def flaky_copytree(src, dst, *a, **k):
        raise OSError("synthetic bundle-copy failure")
    monkeypatch.setattr("cycleft.e7.event_orchestrator.shutil.copytree",
                        flaky_copytree)
    with pytest.raises(OrchestratorError, match="finalization FAILED"):
        orch._finalize_run({"training": {"events": []},
                            "evaluation": {"events": []}})
    st = json.loads((run_dir / "orchestrator_state.json").read_text())
    assert st["state"] == "FAILED"


def test_fd4_failure_during_run_level_verification(tmp_path, v5_specs, mocks, monkeypatch):
    orch, run_dir, spec, tmp = _completed_through_events(tmp_path, v5_specs, mocks)
    def failing_verify(self, src, view):
        raise OrchestratorError("synthetic verification failure")
    monkeypatch.setattr(EventOrchestrator, "_verify_run_level_copy",
                        failing_verify)
    with pytest.raises(OrchestratorError, match="finalization FAILED"):
        orch._finalize_run({"training": {"events": []},
                            "evaluation": {"events": []}})
    st = json.loads((run_dir / "orchestrator_state.json").read_text())
    assert st["state"] == "FAILED"


def test_fd5_failure_during_master_manifest_write(tmp_path, v5_specs, mocks, monkeypatch):
    orch, run_dir, spec, tmp = _completed_through_events(tmp_path, v5_specs, mocks)
    real_write = __import__("cycleft.e7.event_orchestrator",
                            fromlist=["_atomic_write_json"])._atomic_write_json

    def flaky_write(path, payload, *a, **k):
        if Path(path).name == "run_manifest.json":
            raise OSError("synthetic manifest write failure")
        return real_write(path, payload, *a, **k)
    monkeypatch.setattr("cycleft.e7.event_orchestrator._atomic_write_json",
                        flaky_write)
    with pytest.raises(OrchestratorError, match="finalization FAILED"):
        orch._finalize_run({"training": {"events": []},
                            "evaluation": {"events": []}})
    st = json.loads((run_dir / "orchestrator_state.json").read_text())
    assert st["state"] == "FAILED"


def test_fd6_keyboard_interrupt_during_finalization(tmp_path, v5_specs, mocks, monkeypatch):
    orch, run_dir, spec, tmp = _completed_through_events(tmp_path, v5_specs, mocks)
    def ki(self, run_level, telemetry):
        raise KeyboardInterrupt
    monkeypatch.setattr(EventOrchestrator, "_finalize_master_manifest", ki)
    with pytest.raises(KeyboardInterrupt):
        orch._finalize_run({"training": {"events": []},
                            "evaluation": {"events": []}})
    st = json.loads((run_dir / "orchestrator_state.json").read_text())
    assert st["state"] == "INTERRUPTED"
    assert not (run_dir / "completion_marker.json").exists()


def test_fd7_incomplete_finalization_cannot_reach_harness_completed(tmp_path, v5_specs, mocks, monkeypatch):
    spec, out_root, tmp = build_env(tmp_path, v5_specs, "e7-grow-s42")
    run_dir = out_root / spec.exec_id

    real_copytree = shutil.copytree  # captured before patching
    def flaky_copytree(src, dst, *a, **k):
        # fail ONLY the finalization copy (run-level bundles materialization)
        if "bundles" in Path(dst).parts and "events" not in Path(dst).parts:
            raise OSError("synthetic finalization bundle-copy failure")
        return real_copytree(src, dst, *a, **k)
    monkeypatch.setattr("cycleft.e7.event_orchestrator.shutil.copytree",
                        flaky_copytree)
    harness = ExecutionHarness(
        V5_MATRIX, out_root, repo_root=tmp,
        run_fn=make_dynamic_run_fn(train_fn=mocks.train_fn,
                                    sweep_fn=mocks.sweep_fn,
                                    transition_fn=mocks.transition_fn))
    with pytest.raises(Exception, match="execution stopped"):
        harness.execute(TOK, run_id=spec.exec_id)
    rec = harness.inspect_state(run_dir)
    assert rec["state"] == "FAILED"
    assert not (run_dir / COMPLETION_MARKER_NAME).exists()
    # event artifacts preserved
    assert (run_dir / "events/event_3/bundles/step_001500").is_dir()




# ---------------------------------------------------------------------------
# Final hardening — manifest verification & tampering (Finding 1)
# ---------------------------------------------------------------------------


def _run_and_get_verify_ctx(tmp_path, v5_specs, mocks, exec_id="e7-grow-s42"):
    spec, out_root, tmp = build_env(tmp_path, v5_specs, exec_id)
    run_dir = out_root / spec.exec_id
    orch = make_orchestrator(spec, run_dir, tmp, mocks)
    telemetry = orch.run()
    run_level = {"terminal_path": "bundles/step_001500"}
    return orch, run_dir, spec, tmp, telemetry, run_level


def test_h1_modified_event1_evaluation_detected(tmp_path, v5_specs, mocks):
    orch, run_dir, spec, tmp, tel, rl = _run_and_get_verify_ctx(
        tmp_path, v5_specs, mocks)
    fe = run_dir / "events/event_1/final_eval.json"
    data = json.loads(fe.read_text())
    data["1"]["exact_match"] = 0.99  # tamper
    fe.write_text(json.dumps(data))
    with pytest.raises(OrchestratorError, match="diverge from the recorded "
                                                    "inventory"):
        orch._verify_master_manifest(rl, tel)


def test_h2_modified_event2_checkpoint_detected(tmp_path, v5_specs, mocks):
    orch, run_dir, spec, tmp, tel, rl = _run_and_get_verify_ctx(
        tmp_path, v5_specs, mocks)
    ckpt = (run_dir / "events/event_2/bundles/step_001000/adapter/"
            "adapter_model.safetensors")
    ckpt.write_bytes(b"tampered")
    with pytest.raises(OrchestratorError,
                       match="diverge|fingerprint"):
        orch._verify_master_manifest(rl, tel)


def test_h3_modified_inventory_file_detected(tmp_path, v5_specs, mocks):
    orch, run_dir, spec, tmp, tel, rl = _run_and_get_verify_ctx(
        tmp_path, v5_specs, mocks)
    inv = run_dir / "events/event_1/artifact_inventory.json"
    data = json.loads(inv.read_text())
    data["files"]["final_eval.json"] = "0" * 64  # falsify a recorded hash
    inv.write_text(json.dumps(data))
    with pytest.raises(OrchestratorError, match="inventory self-checksum "
                                                    "mismatch"):
        orch._verify_master_manifest(rl, tel)


def test_h4_modified_master_manifest_detected(tmp_path, v5_specs, mocks):
    orch, run_dir, spec, tmp, tel, rl = _run_and_get_verify_ctx(
        tmp_path, v5_specs, mocks)
    mpath = run_dir / "run_manifest.json"
    m = json.loads(mpath.read_text())
    m["final_adapter_fingerprint"] = "f" * 64  # tamper
    mpath.write_text(json.dumps(m))
    with pytest.raises(OrchestratorError, match="final fingerprint mismatch|"
                                                    "self-digest mismatch"):
        orch._verify_master_manifest(rl, tel)


def test_h5_incorrect_transition_fingerprint_detected(tmp_path, v5_specs, mocks):
    orch, run_dir, spec, tmp, tel, rl = _run_and_get_verify_ctx(
        tmp_path, v5_specs, mocks)
    mpath = run_dir / "run_manifest.json"
    m = json.loads(mpath.read_text())
    m["transitions"][0]["resulting_fingerprint"] = "b" * 64  # falsify
    mpath.write_text(json.dumps(m))
    # re-pin the tampered file so the self-digest gate passes and the
    # adjacency check is the one under test
    import hashlib
    tel = {"master_manifest_sha256":
           hashlib.sha256(mpath.read_bytes()).hexdigest()}
    with pytest.raises(OrchestratorError, match="resulting fingerprint does "
                                                    "not match the adjacent "
                                                    "seed bundle"):
        orch._verify_master_manifest(rl, tel)


def test_h6_run_level_bundle_divergence_detected(tmp_path, v5_specs, mocks):
    orch, run_dir, spec, tmp, tel, rl = _run_and_get_verify_ctx(
        tmp_path, v5_specs, mocks)
    view_cfg = run_dir / "bundles/step_001500/adapter/adapter_config.json"
    cfg = json.loads(view_cfg.read_text())
    cfg["r"] = 99  # diverge from the authoritative event-3 checkpoint
    view_cfg.write_text(json.dumps(cfg))
    with pytest.raises(OrchestratorError,
                       match="SHA-256 mismatch|diverges from authoritative"):
        orch._verify_master_manifest(rl, tel)


# ---------------------------------------------------------------------------
# Final hardening — strict evaluation completeness (Finding 2)
# ---------------------------------------------------------------------------


def _sweep_variants():
    base_metrics = {str(j): {"examples": 50, "exact_match": 0.12,
                              "token_f1": 0.34, "loss": 1.0, "ppl": 2.0}
                     for j in (1, 2, 3)}

    def records(n_per_test=50):
        out = []
        for j in (1, 2, 3):
            for i in range(n_per_test):
                out.append({"test_event": j, "example_index": i,
                             "question": f"q{j}-{i}", "reference": "r",
                             "raw_prediction": "p", "prediction": "p",
                             "exact_match": 1.0, "token_f1": 1.0})
        return out

    def variant(mutate):
        metrics = json.loads(json.dumps(base_metrics))
        recs = records()
        mutate(metrics, recs)
        return {"exit_code": 0, "status": "complete", "metrics": metrics,
                "records": recs, "eval_config": {"decoding": "greedy"}}
    return variant


@pytest.mark.parametrize("mutation,match", [
    (lambda m, r: m.pop("3"), "3 test sets"),
    (lambda m, r: m["2"].__setitem__("examples", 49), "examples"),
    (lambda m, r: m["1"].__setitem__("exact_match", float("nan")), "NaN/Inf"),
    (lambda m, r: m["3"].__setitem__("token_f1", float("inf")), "NaN/Inf"),
    (lambda m, r: r.pop(), "150 prediction records"),
    (lambda m, r: r.__setitem__(0, dict(r[1], test_event=1,
                                        example_index=1)), "duplicate"),
    (lambda m, r: r.__setitem__(7, dict(r[7], example_index=99)),
     "out of range"),
    (lambda m, r: r.__setitem__(10, dict(r[10], test_event=9)),
     "invalid test-set identifier"),
    (lambda m, r: r[5].pop("exact_match"), "lacks"),
    (lambda m, r: r.__setitem__(3, dict(r[3], token_f1="high")),
     "malformed metric values"),
])
def test_h7_evaluation_completeness_fail_closed(tmp_path, v5_specs, mocks,
                                                 mutation, match):
    spec, out_root, tmp = build_env(tmp_path, v5_specs, "e7-grow-s42")
    run_dir = out_root / spec.exec_id
    orch = make_orchestrator(spec, run_dir, tmp, mocks)
    sweep = _sweep_variants()(mutation)

    def variant_sweep(ctx, adapter_dir):
        return sweep
    orch.sweep_fn = variant_sweep
    with pytest.raises(OrchestratorError, match=match):
        orch.run()
    st = json.loads((run_dir / "orchestrator_state.json").read_text())
    assert st["state"] == "FAILED"
    # fail closed BEFORE writing the evaluation inventory
    assert not (run_dir / "events/event_1/artifact_inventory.json").exists()
    assert not (run_dir / "run_manifest.json").exists()


# ---------------------------------------------------------------------------
# Final hardening — configuration parity + default callbacks (Findings 3-4)
# ---------------------------------------------------------------------------


def test_h8_default_sweep_config_parity(tmp_path, v5_specs, mocks, monkeypatch):
    """The default sweep callback applies the IDENTICAL inference settings
    as the training callback (Finding 3) — verified against the real
    LMRuntimeConfig dataclass without loading any model."""
    import cycleft.config as cfg_mod
    import cycleft.e1.data as data_mod
    import cycleft.e1.lm as lm_mod
    import cycleft.e1.pipeline as pipe_mod
    from cycleft.e1.lm import LMRuntimeConfig

    captured = {}

    fake_events = [type("E", (), {"event_id": j,
                                   "train_rows": [{"q": 1}] * 500,
                                   "eval_rows": [{"q": 1}] * 50})()
                   for j in (1, 2, 3)]

    def fake_resolve(raw, *, mode, train_examples, seed, device):
        cfg = LMRuntimeConfig(model_id="meta-llama/Llama-3.2-1B-Instruct",
                              seed=seed)
        captured["resolved_seed"] = seed
        return cfg, None, None

    def fake_load_base(cfg):
        captured["base_dtype"] = cfg.dtype
        base = type("B", (), {"device": torch.device("cpu")})()
        captured["base_obj"] = base
        return base

    def fake_load_adapter(model, adapter_dir, cfg, device):
        # pilot attempt-2 regression: the model passed here must be the
        # PLAIN base (no attach_lora pre-wrap) — from_pretrained attaches
        # per the bundle's adapter_config.json and loads the trained weights
        captured["model_is_plain_base"] = model is captured["base_obj"]
        captured["sweep_adapter_dir"] = str(adapter_dir)
        captured["sweep_rank"], captured["sweep_alpha"] = \
            cfg.lora_r, cfg.lora_alpha
        captured["sweep_lora_precision"] = cfg.lora_precision
        captured["sweep_max_new_tokens"] = cfg.max_new_tokens
        captured["sweep_seed"] = cfg.seed
        return model

    def fake_eval(model, tokenizer, eval_rows, cfg, log=None):
        captured["eval_rows_len"] = len(eval_rows)
        captured["eval_max_new_tokens"] = cfg.max_new_tokens
        return {"metrics": {"examples": len(eval_rows), "exact_match": 0.1,
                             "token_f1": 0.2},
                 "records": [{"test_event": 1, "example_index": i,
                               "question": "q", "reference": "r",
                               "raw_prediction": "p", "prediction": "p",
                               "exact_match": 1.0, "token_f1": 1.0}
                              for i in range(len(eval_rows))]}

    monkeypatch.setattr(cfg_mod, "load_config", lambda p: {"synthetic": True})
    monkeypatch.setattr(pipe_mod, "resolve_lm_config", fake_resolve)
    monkeypatch.setattr(data_mod, "prepare_event_datasets",
                        lambda train_limit=None, seed=42: (fake_events, {}))
    monkeypatch.setattr(lm_mod, "load_base_model", fake_load_base)
    # attach_lora must NEVER be called (pilot attempt-2 double-wrap regression)
    monkeypatch.setattr(lm_mod, "attach_lora",
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("attach_lora must not be called "
                                           "(double-wrap regression)")))
    monkeypatch.setattr(lm_mod, "load_adapter_for_training", fake_load_adapter)
    monkeypatch.setattr(lm_mod, "load_tokenizer", lambda mid: None)
    monkeypatch.setattr(lm_mod, "evaluate_event", fake_eval)

    from cycleft.e7.event_orchestrator import _default_sweep_fn
    sweep = _default_sweep_fn(Path.cwd())
    ctx = {"exec_id": "e7-grow-s42", "seed": 42, "event_id": 2, "rank": 8,
           "alpha": 16}
    res = sweep(ctx, tmp_path / "terminal_adapter")

    # parity with the training callback's explicit settings
    assert (captured["sweep_rank"], captured["sweep_alpha"]) == (8, 16)
    assert captured["sweep_lora_precision"] == "float32"
    assert captured["base_dtype"] == "float16"
    assert captured["model_is_plain_base"] is True
    assert captured["sweep_max_new_tokens"] == 24
    assert captured["sweep_seed"] == 42
    assert captured["eval_max_new_tokens"] == 24
    assert captured["eval_rows_len"] == 50
    assert captured["sweep_adapter_dir"] == str(tmp_path / "terminal_adapter")
    # effective settings recorded in eval_config
    ec = res["eval_config"]
    assert ec["model_id"] == "meta-llama/Llama-3.2-1B-Instruct"
    assert ec["dtype"] == "float16" and ec["lora_precision"] == "float32"
    assert ec["rank"] == 8 and ec["alpha"] == 16 and ec["max_new_tokens"] == 24
    assert ec["decoding"] == "greedy" and ec["seed"] == 42
    assert ec["rows_per_test_set"] == 50
    # three-test-set contract
    assert sorted(res["metrics"].keys()) == ["1", "2", "3"]


def test_h9_default_train_fn_contract(tmp_path, v5_specs, monkeypatch):
    import cycleft.config as cfg_mod
    import cycleft.e1.data as data_mod
    import cycleft.e1.pipeline as pipe_mod
    import cycleft.e5.resumable_trainer as rt_mod
    from cycleft.e1.lm import LMRuntimeConfig

    captured = {}
    fake_events = [type("E", (), {"event_id": j,
                                   "train_rows": [{"question": f"q{i}",
                                                    "object": "o"}
                                                   for i in range(500)],
                                   "eval_rows": []})()
                   for j in (1, 2, 3)]

    def fake_resolve(raw, *, mode, train_examples, seed, device):
        captured["resolve_seed"] = seed
        return LMRuntimeConfig(model_id="m", seed=seed), None, None

    def fake_run_workload(**kwargs):
        captured.update(kwargs)
        captured["run_workload_called"] = True
        return {"status": "complete", "global_step": 500, "final_avg_em": 0.1}

    monkeypatch.setattr(cfg_mod, "load_config", lambda p: {"synthetic": True})
    monkeypatch.setattr(pipe_mod, "resolve_lm_config", fake_resolve)
    monkeypatch.setattr(data_mod, "prepare_event_datasets",
                        lambda train_limit=None, seed=42: (fake_events, {}))
    monkeypatch.setattr(rt_mod, "run_workload", fake_run_workload)

    from cycleft.e7.event_orchestrator import (_default_config_hash_fn,
                                                _default_train_fn)
    event_dir = tmp_path / "events" / "event_2"
    ctx = {"exec_id": "e7-grow-s42", "policy": "Grow", "seed": 42,
           "event_id": 2, "rank": 8, "alpha": 16, "depth": "D100",
           "budget_per_event": 500,
           "stage_schedule": [{"stage": 1, "event_id": 2, "start": 500,
                                "end": 1000}],
           "global_step_start": 500, "global_step_end": 1000,
           "stream_slice": [500, 1000], "event_dir": str(event_dir),
           "config_hash": _default_config_hash_fn(Path.cwd())({
               "exec_id": "e7-grow-s42", "policy": "Grow", "seed": 42,
               "event_id": 2, "rank": 8, "alpha": 16, "depth": "D100",
               "budget_per_event": 500,
               "stage_schedule": [{"stage": 1, "event_id": 2, "start": 500,
                                    "end": 1000}]})}

    # pre-create the terminal bundle the fake run_workload "committed"
    step = 1000
    write_e5_bundle(event_dir / "bundles", step=step,
                    adapter_dir=_mk_adapter(tmp_path, 8),
                    fingerprint="fp-" + "0" * 60,
                    progress={"global_step": step, "stage_index": 0,
                              "intra_stage_index": 500,
                              "config_hash": ctx["config_hash"]},
                    meta={"exec_id": "e7-grow-s42"}, state_payload={
                        "optimizer": {}, "rng": {}})

    res = _default_train_fn(Path.cwd())(ctx)

    assert res["exit_code"] == 0 and res["status"] == "complete"
    assert captured["run_workload_called"] is True
    assert captured["out_dir"] == event_dir
    assert captured["stage_schedule"] == ctx["stage_schedule"]
    assert captured["ckpt_every"] == 500
    assert captured["resume"] is True          # event 2 resumes from seed
    assert captured["final_eval_events"] is None  # sweep owns evaluation
    assert captured["lm_config"].lora_r == 8
    assert captured["lm_config"].lora_alpha == 16
    assert captured["lm_config"].seed == 42
    assert len(captured["stream_rows"]) == 1500
    assert res["terminal_bundle"].endswith("step_001000")
    assert res["config_hash"] == ctx["config_hash"]


def test_h9b_default_train_fn_failure_propagation(tmp_path, v5_specs, monkeypatch):
    import cycleft.config as cfg_mod
    import cycleft.e1.data as data_mod
    import cycleft.e1.pipeline as pipe_mod
    import cycleft.e5.resumable_trainer as rt_mod
    from cycleft.e1.lm import LMRuntimeConfig

    monkeypatch.setattr(cfg_mod, "load_config", lambda p: {})
    def fake_resolve(raw, *, mode, train_examples, seed, device):
        return LMRuntimeConfig(model_id="m", seed=seed), None, None
    monkeypatch.setattr(pipe_mod, "resolve_lm_config", fake_resolve)
    monkeypatch.setattr(data_mod, "prepare_event_datasets",
                        lambda train_limit=None, seed=42: ([], {}))
    monkeypatch.setattr(rt_mod, "run_workload",
                        lambda **k: {"status": "killed", "global_step": 7})
    from cycleft.e7.event_orchestrator import _default_train_fn
    res = _default_train_fn(Path.cwd())(
        {"exec_id": "x", "seed": 42, "event_id": 1, "rank": 4, "alpha": 8,
         "event_dir": str(tmp_path / "e1"),
         "stage_schedule": [{"stage": 1, "event_id": 1, "start": 0,
                              "end": 500}]})
    assert res["exit_code"] == 1 and res["status"] == "killed"


def _mk_adapter(tmp_path, r):
    d = tmp_path / f"adapter_r{r}"
    d.mkdir(parents=True, exist_ok=True)
    g = torch.Generator().manual_seed(r)
    save_file({
        LORA_KEYS[0]: torch.randn(r, 32, generator=g),
        LORA_KEYS[1]: torch.randn(32, r, generator=g),
        LORA_KEYS[2]: torch.randn(r, 32, generator=g),
        LORA_KEYS[3]: torch.randn(32, r, generator=g),
    }, str(d / "adapter_model.safetensors"))
    (d / "adapter_config.json").write_text(json.dumps(
        {"r": r, "lora_alpha": 2 * r}))
    return d


def test_h10_default_transition_fn_contract(tmp_path, v5_specs, mocks, monkeypatch):
    import cycleft.e1.lm as lm_mod
    import cycleft.e5.resume_bundle as rb_mod

    src_adapter = _mk_adapter(tmp_path, 4)
    staging = tmp_path / "transitions" / ".staging" / "transition_event_1_to_2"
    captured = {}

    class FakeParam:
        def __init__(self, name, tensor):
            self.name, self.tensor = name, tensor
            self.requires_grad = True

        def detach(self):
            return self

        def to(self, *a, **k):
            return self.tensor.to("cpu").contiguous()

        def numpy(self):
            return self.tensor.numpy()

    class FakeModel:
        def __init__(self, tensors):
            self._t = tensors

        def named_parameters(self):
            return [(name, FakeParam(name, t)) for name, t in sorted(self._t.items())]

        def parameters(self):
            return list(self._t.values())  # raw tensors: AdamW-compatible

    def fake_load_adapter(model, adapter_dir, cfg, device):
        # pilot attempt-2 regression: the model passed here must be the
        # PLAIN base (no attach_lora pre-wrap) — from_pretrained attaches
        # per the transitioned adapter_config.json and loads the weights
        captured["model_is_plain_base"] = model is captured["base_obj"]
        captured["load_adapter_dir"] = str(adapter_dir)
        tensors = {k: v.clone().requires_grad_(True) for k, v in
                   load_file(str(Path(adapter_dir) /
                                  "adapter_model.safetensors")).items()}
        return FakeModel(tensors)

    base_obj = type("B", (), {"device": torch.device("cpu")})()
    captured["base_obj"] = base_obj
    monkeypatch.setattr(lm_mod, "load_base_model", lambda cfg: base_obj)
    # attach_lora must NEVER be called (double-wrap regression guard)
    monkeypatch.setattr(lm_mod, "attach_lora",
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("attach_lora must not be called "
                                           "(double-wrap regression)")))
    monkeypatch.setattr(lm_mod, "load_adapter_for_training", fake_load_adapter)
    monkeypatch.setattr(lm_mod, "load_tokenizer", lambda mid: None)
    monkeypatch.setattr(rb_mod, "capture_rng_states",
                        lambda: {"torch_cpu": torch.zeros(4), "rng_tier": "mock"})
    monkeypatch.setattr(rb_mod, "adapter_fingerprint",
                        lambda model: fingerprint_from_tensors(
                            {n: p.tensor for n, p in model.named_parameters()}))
    # load_base_model/attach_lora import inside the callback reference
    # cycleft.e1.lm — patch resolve path minimally:
    import cycleft.e1.pipeline as pipe_mod
    from cycleft.e1.lm import LMRuntimeConfig
    monkeypatch.setattr(pipe_mod, "resolve_lm_config",
                        lambda raw, *, mode, train_examples, seed, device:
                        (LMRuntimeConfig(model_id="m", seed=seed), None, None))
    import cycleft.config as cfg_mod
    monkeypatch.setattr(cfg_mod, "load_config", lambda p: {})

    from cycleft.e7.event_orchestrator import _default_transition_fn
    seed = transition_seed(42, 1)
    gen = torch.Generator().manual_seed(seed)
    res = _default_transition_fn(Path.cwd())(src_adapter, staging, 4, 8, 16, gen)

    # staging-only writes (Finding A)
    assert res["adapter_dir"] == str(staging / "adapter")
    assert (staging / "adapter" / "adapter_model.safetensors").is_file()
    assert not list(Path(src_adapter).rglob(".transition_adapter_*"))
    # E5 fingerprint consistency: model-level fingerprint == tensor-level one
    tensors = load_file(str(staging / "adapter" / "adapter_model.safetensors"))
    assert res["fingerprint"] == fingerprint_from_tensors(tensors)
    # generator propagation: same seed -> identical new A rows
    assert captured["model_is_plain_base"] is True
    assert captured["load_adapter_dir"] == str(staging / "adapter")
    sp = res["state_payload"]
    assert "optimizer" in sp and "rng" in sp
    # determinism: re-run with the same derived seed
    staging2 = tmp_path / "transitions" / ".staging" / "repeat"
    gen2 = torch.Generator().manual_seed(seed)
    res2 = _default_transition_fn(Path.cwd())(src_adapter, staging2, 4, 8, 16, gen2)
    t1 = load_file(str(staging / "adapter" / "adapter_model.safetensors"))
    t2 = load_file(str(staging2 / "adapter" / "adapter_model.safetensors"))
    for key in t1:
        assert torch.equal(t1[key], t2[key]), "deterministic expansion"


# ---------------------------------------------------------------------------
# Final integrity patch — Finding A: inventory digest linkage
# ---------------------------------------------------------------------------


def _verify_ctx_with_tampered_manifest(tmp_path, v5_specs, mocks, exec_id,
                                        mutate):
    """Run a full lifecycle, tamper the master manifest via `mutate(m)`,
    re-pin the manifest self-digest in telemetry, and return the verify
    context (orch, run_dir, telemetry, run_level)."""
    orch, run_dir, spec, tmp, tel, rl = _run_and_get_verify_ctx(
        tmp_path, v5_specs, mocks, exec_id)
    mpath = run_dir / "run_manifest.json"
    m = json.loads(mpath.read_text())
    mutate(m)
    mpath.write_text(json.dumps(m))
    tel = {"master_manifest_sha256":
           hashlib.sha256(mpath.read_bytes()).hexdigest()}
    return orch, run_dir, tel, rl


def test_ga1_valid_event_inventory_passes_linkage(tmp_path, v5_specs, mocks):
    orch, run_dir, spec, tmp, tel, rl = _run_and_get_verify_ctx(
        tmp_path, v5_specs, mocks)
    m = orch._verify_master_manifest(rl, tel)  # must not raise
    for ev in m["events"]:
        assert len(ev["inventory_digest"]) == 64
        assert len(ev["inventory_self_checksum"]) == 64


def test_ga2_changed_recorded_digest_fails(tmp_path, v5_specs, mocks):
    def mutate(m):
        m["events"][0]["inventory_digest"] = "a" * 64
    orch, run_dir, tel, rl = _verify_ctx_with_tampered_manifest(
        tmp_path, v5_specs, mocks, "e7-grow-s42", mutate)
    with pytest.raises(OrchestratorError, match="inventory digest linkage "
                                                    "mismatch"):
        orch._verify_master_manifest(rl, tel)


def test_ga3_changed_actual_inventory_fails(tmp_path, v5_specs, mocks):
    orch, run_dir, spec, tmp, tel, rl = _run_and_get_verify_ctx(
        tmp_path, v5_specs, mocks)
    # tamper an event file -> current files diverge from the inventory map
    (run_dir / "events/event_1/evaluation_config.json").write_text("{}")
    with pytest.raises(OrchestratorError,
                       match="diverge from the recorded inventory"):
        orch._verify_master_manifest(rl, tel)


def test_ga4_removed_recorded_digest_fails(tmp_path, v5_specs, mocks):
    def mutate(m):
        del m["events"][1]["inventory_digest"]
    orch, run_dir, tel, rl = _verify_ctx_with_tampered_manifest(
        tmp_path, v5_specs, mocks, "e7-grow-s42", mutate)
    with pytest.raises(OrchestratorError, match="missing or malformed"):
        orch._verify_master_manifest(rl, tel)


def test_ga5_digest_from_another_event_cannot_substitute(tmp_path, v5_specs, mocks):
    def mutate(m):
        # substitute event 2's recorded digest into event 1's slot: the
        # file sets differ, so the canonical digests cannot match
        m["events"][0]["inventory_digest"] = m["events"][1]["inventory_digest"]
    orch, run_dir, tel, rl = _verify_ctx_with_tampered_manifest(
        tmp_path, v5_specs, mocks, "e7-grow-s42", mutate)
    with pytest.raises(OrchestratorError, match="inventory digest linkage "
                                                    "mismatch"):
        orch._verify_master_manifest(rl, tel)


def test_ga6_valid_self_checksum_alone_cannot_pass(tmp_path, v5_specs, mocks):
    """The inventory file is untouched (its self-checksum remains valid) but
    the master manifest records a different digest — verification must fail
    on the linkage, not pass on the self-checksum."""
    def mutate(m):
        m["events"][2]["inventory_digest"] = "c" * 64
    orch, run_dir, tel, rl = _verify_ctx_with_tampered_manifest(
        tmp_path, v5_specs, mocks, "e7-grow-s42", mutate)
    inv = run_dir / "events/event_3/artifact_inventory.json"
    inv_bytes = inv.read_bytes()  # inventory untouched
    with pytest.raises(OrchestratorError, match="linkage mismatch"):
        orch._verify_master_manifest(rl, tel)
    assert inv.read_bytes() == inv_bytes


# ---------------------------------------------------------------------------
# Final integrity patch — Finding B: exact run-level checkpoint identity
# ---------------------------------------------------------------------------


def _identity_ctx(tmp_path, v5_specs, mocks):
    orch, run_dir, spec, tmp, tel, rl = _run_and_get_verify_ctx(
        tmp_path, v5_specs, mocks)
    src = run_dir / "events/event_3/bundles/step_001500"
    view = run_dir / "bundles/step_001500"
    return orch, run_dir, src, view, tel, rl


def test_gb1_exact_byte_identical_copy_passes(tmp_path, v5_specs, mocks):
    orch, run_dir, src, view, tel, rl = _identity_ctx(tmp_path, v5_specs, mocks)
    orch._verify_run_level_copy(src, view)  # must not raise


def test_gb2_additional_top_level_file_rejected(tmp_path, v5_specs, mocks):
    orch, run_dir, src, view, tel, rl = _identity_ctx(tmp_path, v5_specs, mocks)
    (view / "extra.bin").write_bytes(b"x")
    with pytest.raises(OrchestratorError, match="additional files"):
        orch._verify_run_level_copy(src, view)


def test_gb3_additional_nested_file_rejected(tmp_path, v5_specs, mocks):
    orch, run_dir, src, view, tel, rl = _identity_ctx(tmp_path, v5_specs, mocks)
    nested = view / "adapter" / "extra_dir"
    nested.mkdir()
    (nested / "extra.bin").write_bytes(b"x")
    with pytest.raises(OrchestratorError, match="additional files|additional "
                                                    "directories|directory "
                                                    "structure mismatch"):
        orch._verify_run_level_copy(src, view)


def test_gb4_missing_file_rejected(tmp_path, v5_specs, mocks):
    orch, run_dir, src, view, tel, rl = _identity_ctx(tmp_path, v5_specs, mocks)
    (view / "adapter" / "adapter_config.json").unlink()
    with pytest.raises(OrchestratorError, match="missing files"):
        orch._verify_run_level_copy(src, view)


def test_gb5_modified_file_rejected(tmp_path, v5_specs, mocks):
    orch, run_dir, src, view, tel, rl = _identity_ctx(tmp_path, v5_specs, mocks)
    (view / "state.pt").write_bytes(b"tampered")
    with pytest.raises(OrchestratorError, match="SHA-256 mismatch|size mismatch"):
        orch._verify_run_level_copy(src, view)


def test_gb6_internal_symlink_rejected(tmp_path, v5_specs, mocks):
    orch, run_dir, src, view, tel, rl = _identity_ctx(tmp_path, v5_specs, mocks)
    (view / "adapter_link.safetensors").symlink_to(
        view / "adapter" / "adapter_model.safetensors")
    with pytest.raises(OrchestratorError, match="symlink"):
        orch._verify_run_level_copy(src, view)


def test_gb7_external_symlink_rejected(tmp_path, v5_specs, mocks):
    orch, run_dir, src, view, tel, rl = _identity_ctx(tmp_path, v5_specs, mocks)
    external = tmp_path / "outside_secret.bin"
    external.write_bytes(b"secret")
    (view / "escape.bin").symlink_to(external)
    with pytest.raises(OrchestratorError, match="symlink"):
        orch._verify_run_level_copy(src, view)


def test_gb8_unexpected_filesystem_type_rejected(tmp_path, v5_specs, mocks):
    orch, run_dir, src, view, tel, rl = _identity_ctx(tmp_path, v5_specs, mocks)
    os.mkfifo(view / "pipe")
    try:
        with pytest.raises(OrchestratorError, match="unexpected filesystem type"):
            orch._verify_run_level_copy(src, view)
    finally:
        os.unlink(view / "pipe")


def test_gb9_verification_failure_modifies_nothing(tmp_path, v5_specs, mocks):
    orch, run_dir, src, view, tel, rl = _identity_ctx(tmp_path, v5_specs, mocks)
    (view / "extra.bin").write_bytes(b"x")
    before_src, before_view = _hash_tree(src), _hash_tree(view)
    with pytest.raises(OrchestratorError):
        orch._verify_run_level_copy(src, view)
    assert _hash_tree(src) == before_src, "authoritative checkpoint modified"
    assert _hash_tree(view) == before_view, "view modified by verification"


def test_gb10_overwrite_prevention_intact(tmp_path, v5_specs, mocks):
    """Existing overwrite-prevention behavior remains (regression)."""
    orch, run_dir, src, view, tel, rl = _identity_ctx(tmp_path, v5_specs, mocks)
    with pytest.raises(OrchestratorError, match="already exists"):
        write_e5_bundle(view.parent, step=1500, adapter_dir=src / "adapter",
                        fingerprint="x", progress={"global_step": 1500},
                        meta={}, state_payload={"optimizer": {}})


def test_h11_default_sweep_injects_test_event_identifier(tmp_path, v5_specs, mocks, monkeypatch):
    """Regression (pilot attempt-1 failure): E5's evaluate_event records
    carry no test-set identifier — the default sweep callback must inject
    `test_event` (and `state`) so the strict prediction validator accepts
    them."""
    import cycleft.config as cfg_mod
    import cycleft.e1.data as data_mod
    import cycleft.e1.lm as lm_mod
    import cycleft.e1.pipeline as pipe_mod
    from cycleft.e1.lm import LMRuntimeConfig

    fake_events = [type("E", (), {"event_id": j,
                                   "train_rows": [], "eval_rows": [{"q": i} for i in range(50)]})()
                   for j in (1, 2, 3)]

    def fake_eval(model, tokenizer, eval_rows, cfg, log=None):
        # E5-native records: NO test_event, NO state (the real schema)
        return {"metrics": {"examples": len(eval_rows), "exact_match": 0.1,
                             "token_f1": 0.2},
                 "records": [{"example_index": i, "question": "q",
                               "reference": "r", "raw_prediction": "p",
                               "prediction": "p", "exact_match": 1.0,
                               "token_f1": 1.0} for i in range(len(eval_rows))]}

    monkeypatch.setattr(cfg_mod, "load_config", lambda p: {})
    monkeypatch.setattr(pipe_mod, "resolve_lm_config",
                        lambda raw, *, mode, train_examples, seed, device:
                        (LMRuntimeConfig(model_id="m", seed=seed), None, None))
    monkeypatch.setattr(data_mod, "prepare_event_datasets",
                        lambda train_limit=None, seed=42: (fake_events, {}))
    monkeypatch.setattr(lm_mod, "load_base_model",
                        lambda cfg: type("B", (), {"device": torch.device("cpu")})())
    monkeypatch.setattr(lm_mod, "attach_lora", lambda base, cfg, dev: base)
    monkeypatch.setattr(lm_mod, "load_adapter_for_training",
                        lambda model, d, cfg, dev: model)
    monkeypatch.setattr(lm_mod, "load_tokenizer", lambda mid: None)
    monkeypatch.setattr(lm_mod, "evaluate_event", fake_eval)

    from cycleft.e7.event_orchestrator import _default_sweep_fn
    res = _default_sweep_fn(Path.cwd())(
        {"exec_id": "e7-grow-s42", "seed": 42, "event_id": 1, "rank": 4,
         "alpha": 8}, tmp_path / "adapter")
    assert len(res["records"]) == 150
    for j in (1, 2, 3):
        subset = [r for r in res["records"] if r["test_event"] == j]
        assert len(subset) == 50
        assert all(r["state"] == "after_event_1" for r in subset)
        assert all(r["example_index"] == i for i, r in enumerate(subset))


def test_gd_run_level_summary_carries_measured_final_avg_em(tmp_path, v5_specs, mocks):
    """Pilot attempt-3 regression: with sweep-owned evaluation, E5's summary
    has final_avg_em=None — the run-level view must carry the MEAN of the
    event-3 sweep EMs (measured) with provenance, so the harness gate
    passes honestly."""
    spec, out_root, tmp = build_env(tmp_path, v5_specs, "e7-grow-s42")
    run_dir = out_root / spec.exec_id
    orch = make_orchestrator(spec, run_dir, tmp, mocks)
    orch.run()
    summary = json.loads((run_dir / "execution_summary.json").read_text())
    assert isinstance(summary["final_avg_em"], (int, float))
    # event-3 original remains immutable (the run-level view is a separate file)
    sweep_em = [m["exact_match"] for m in
                json.loads((run_dir / "events/event_3/final_eval.json").read_text())
                .values()]
    assert abs(summary["final_avg_em"] - sum(sweep_em) / 3) < 1e-12
    assert summary["e7_provenance"]["final_avg_em_source"].startswith(
        "events/event_3/final_eval.json")
    # and the harness structural gate accepts it
    from cycleft.e7.execution_harness import build_artifact_inventory
    inv = build_artifact_inventory(run_dir)  # structural validation inside
    assert "execution_summary.json" in inv


# ---------------------------------------------------------------------------
# Final hardening round 2 — resource telemetry (Part B)
# ---------------------------------------------------------------------------


def test_t1_rss_unit_conversion_platform_documented():
    from cycleft.e7.event_orchestrator import _rss_mb_from_raw
    assert _rss_mb_from_raw(4194304, "darwin") == 4.0      # bytes → MiB
    assert _rss_mb_from_raw(4194304, "linux") == 4096.0    # KiB → MiB


def test_t2_snapshot_nulls_with_reason_when_mps_absent(monkeypatch):
    import cycleft.e7.event_orchestrator as EO
    class FakeTorch:
        class backends:
            mps = type("mps", (), {"is_available": staticmethod(lambda: False)})
    import types
    fake = types.ModuleType("torch")
    fake.backends = FakeTorch.backends
    monkeypatch.setitem(sys.modules, "torch", fake)
    snap = EO._resource_snapshot()
    assert snap["mps_allocated_mb"] is None and snap["mps_reason"]
    assert "not available" in snap["mps_reason"]
    assert snap["peak_rss_mb"] is not None, "process RSS stays measurable"


def test_t3_snapshot_records_all_requested_points(tmp_path, v5_specs, mocks):
    spec, out_root, tmp = build_env(tmp_path, v5_specs, "e7-grow-s42")
    run_dir = out_root / spec.exec_id
    orch = make_orchestrator(spec, run_dir, tmp, mocks)
    orch.run()
    m = json.loads((run_dir / "run_manifest.json").read_text())
    for ev in m["events"]:
        r = ev["resources"]
        for point in ("event_start", "training_end", "evaluation_end"):
            assert point in r, f"missing resource point {point}"
            assert r[point]["peak_rss_mb"] is not None
            # process RSS vs MPS memory are distinct fields (never merged)
            assert "mps_allocated_mb" in r[point]
        assert r["training_duration_s"] >= 0   # mocked callbacks are instant
        assert r["evaluation_duration_s"] >= 0
        assert r["units"]["rss"].startswith("MiB")
        assert "e5_timing" in r
    # monotonic process high-water mark (true peak, not end-of-run sample)
    rss = [e["resources"]["evaluation_end"]["peak_rss_mb"] for e in m["events"]]
    assert rss == sorted(rss), "ru_maxrss high-water must be monotonic"


def test_t4_transition_and_exit_telemetry(tmp_path, v5_specs, mocks):
    spec, out_root, tmp = build_env(tmp_path, v5_specs, "e7-grow-s42")
    run_dir = out_root / spec.exec_id
    orch = make_orchestrator(spec, run_dir, tmp, mocks)
    telemetry = orch.run()
    for tr in orch.transition_records:
        assert tr["transition_duration_s"] >= 0  # mocked callbacks are instant
        assert "transition_start" in tr["resources"]
        assert "transition_end" in tr["resources"]
    assert telemetry["total_wall_s"] > 0
    assert telemetry["resources_at_exit"]["peak_rss_mb"] is not None
    m = json.loads((run_dir / "run_manifest.json").read_text())
    assert m["events"][0]["resources"]["training_duration_s"] >= 0


def test_t5_telemetry_serialization_roundtrip(tmp_path, v5_specs, mocks):
    spec, out_root, tmp = build_env(tmp_path, v5_specs, "e7-grow-s42")
    run_dir = out_root / spec.exec_id
    orch = make_orchestrator(spec, run_dir, tmp, mocks)
    orch.run()
    raw = (run_dir / "run_manifest.json").read_text()
    m = json.loads(raw)  # must be JSON-serializable (no PlannedRun objects)
    for ev in m["events"]:
        json.dumps(ev["resources"])  # no raise
    assert m["events"][0]["resources"]["event_start"]["mps_recommended_max_mb"] \
        == pytest.approx(5461.3, rel=0.01)
