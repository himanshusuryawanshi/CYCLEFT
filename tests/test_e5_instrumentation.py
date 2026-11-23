"""Phase-2 instrumentation validation tests.

Unit-level fixtures are deterministic and CPU-based; the real-model
validation execution lives in scripts/e5_instrumentation_validation.py.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch

from cycleft.e5.event_log import (
    EventLog,
    read_events,
    supervisor_termination_record,
)
from cycleft.e5.position_audit import validate_positions
from cycleft.e5.timing import TimingRegistry

ROOT = Path(__file__).resolve().parents[1]


# -- timing registry ---------------------------------------------------------


def test_timing_nonzero_and_categories_separate():
    registry = TimingRegistry(exec_id="t", mps_sync=False)
    t = registry.start("train")
    x = sum(i * i for i in range(200000))  # real CPU work
    registry.stop("train", t)
    t = registry.start("eval_generation")
    y = sum(i for i in range(100000))
    registry.stop("eval_generation", t)
    d = registry.to_dict()["categories"]
    assert d["train"]["total_ns"] > 0
    assert d["eval_generation"]["total_ns"] > 0
    # separate categories: no cross-contamination
    assert d["train"]["count"] == 1 and d["eval_generation"]["count"] == 1
    assert d["train"]["total_s"] < 5.0  # sanity: bounded work


def test_timing_mps_sync_around_train_segments():
    calls = {"n": 0}
    import torch

    original = torch.mps.synchronize

    def counting_sync():
        calls["n"] += 1

    torch.mps.synchronize = counting_sync
    try:
        registry = TimingRegistry(exec_id="t", mps_sync=True)
        token = registry.start("train")
        registry.stop("train", token)
    finally:
        torch.mps.synchronize = original
    assert calls["n"] == 2, "synchronize must run before AND after train intervals"


def test_timing_merge_multiple_processes():
    a = TimingRegistry(exec_id="attempt-1")
    b = TimingRegistry(exec_id="attempt-2")
    a.add("train", 100, 1)
    b.add("train", 250, 2)
    merged = TimingRegistry(exec_id="merged")
    merged.merge(a)
    merged.merge(b)
    assert merged.totals_ns["train"] == 350
    assert merged.counts["train"] == 3
    # per-process records retained
    assert a.to_dict()["exec_id"] == "attempt-1"
    assert b.totals_ns["train"] == 250


def test_timing_unknown_category_rejected():
    registry = TimingRegistry(exec_id="t")
    with pytest.raises(KeyError):
        registry.start("not_a_category")


# -- event log persistence ---------------------------------------------------


def test_event_log_survives_abrupt_termination(tmp_path):
    """Events are flushed per emit: a process that dies without closing the
    log still leaves every emitted event readable."""
    path = tmp_path / "event_log.jsonl"
    log = EventLog(path, exec_id="victim")
    log.emit("process_start", {"attempt": 1}, fsync=True)
    log.emit("step", {"position": 1, "attempt": 1})
    log.emit("step", {"position": 2, "attempt": 1})
    # simulate SIGKILL: no close(), the file object is abandoned
    del log
    events = read_events(path)
    assert [e["event"] for e in events] == ["process_start", "step", "step"]
    assert events[1]["position"] == 1


def test_event_log_rejects_unknown_types(tmp_path):
    log = EventLog(tmp_path / "e.jsonl", exec_id="t")
    with pytest.raises(KeyError):
        log.emit("bogus_event")
    log.close()


# -- supervisor termination records -------------------------------------------


def test_supervisor_records_sigkill_exit_status(tmp_path):
    """Deterministic fixture: the supervisor SIGKILLs a sleeping child and
    records the signal in the termination log."""
    child_code = "import time; print('ready', flush=True); time.sleep(300)"
    child = subprocess.Popen(
        [sys.executable, "-c", child_code],
        stdout=subprocess.PIPE, text=True,
    )
    try:
        child.stdout.readline()  # wait for 'ready'
        child.kill()             # SIGKILL
        child.wait()
        record = supervisor_termination_record(
            tmp_path / "supervisor_log.jsonl",
            exec_id="e5p2-fixture", child_pid=child.pid, mode="killrun",
            returncode=child.returncode,
        )
    finally:
        if child.poll() is None:
            child.kill()
    assert child.returncode == -9
    assert record["exit_status"] == "signal_9"
    assert record["signal"] == 9
    stored = json.loads(
        (tmp_path / "supervisor_log.jsonl").read_text().splitlines()[-1]
    )
    assert stored["exit_status"] == "signal_9"
    assert stored["child_pid"] == child.pid


# -- position audit validator -------------------------------------------------


def step_event(position, attempt, boundary_before=False):
    return {"position": position, "attempt": attempt,
            "boundary_before": boundary_before}


def test_position_validator_complete_run():
    events = [step_event(p, 1) for p in range(1, 376)]
    result = validate_positions(events, target_positions=375)
    assert result["exactly_once_composite"] is True
    assert result["missing_logical_positions"] == []
    assert result["reexecuted_entries"] == 0
    assert result["unique_logical_positions_completed"] == 375
    assert result["physical_step_attempts"] == 375


def test_position_validator_valid_replay_after_fallback():
    """Kill at 62 with the newest bundle at 50: positions 51-62 are attempted
    twice across a restoration boundary — valid replay, not a bug."""
    events = [step_event(p, 1) for p in range(1, 63)]          # pre-kill 1..62
    events += [step_event(p, 2, boundary_before=(p == 51))
               for p in range(51, 376)]                        # resume from 50
    result = validate_positions(events, target_positions=375,
                                interruption_boundaries=[2])
    assert result["reexecuted_entries"] == 12
    assert result["reexecuted_positions"] == list(range(51, 63))
    assert result["valid_replays"] and not result["false_duplicate_completions"]
    assert result["missing_logical_positions"] == []
    assert result["exactly_once_composite"] is True
    assert result["pre_kill_entries"] == 62
    assert result["post_resume_entries"] == 325


def test_position_validator_detects_missing_positions():
    events = [step_event(p, 1) for p in range(1, 100)] + \
             [step_event(p, 1) for p in range(101, 376)]   # position 100 missing
    result = validate_positions(events, target_positions=375)
    assert result["missing_logical_positions"] == [100]
    assert result["exactly_once_composite"] is False


def test_position_validator_flags_false_duplicate_completions():
    """Two completions of the same position WITHIN one attempt (no
    restoration boundary) = a progress-accounting bug, not a valid replay."""
    events = [step_event(p, 1) for p in range(1, 50)]
    events += [step_event(49, 1), step_event(50, 1)]  # 49 completed twice, same attempt
    events += [step_event(p, 1) for p in range(51, 376)]
    result = validate_positions(events, target_positions=375)
    assert result["false_duplicate_completions"], "must be flagged"
    assert not result["valid_replays"]
    assert result["exactly_once_composite"] is False


def test_position_validator_empty_log():
    result = validate_positions([], target_positions=10)
    assert result["missing_logical_positions"] == list(range(1, 11))
    assert result["exactly_once_composite"] is False
    assert result["physical_step_attempts"] == 0


# -- audit logging enabled by default (real model, tiny schedule) --------------


def test_audit_enabled_by_default_and_nonzero_train_timing(tmp_path):
    """Controlled real-model workload: 1 stage x 3 steps. Verifies the audit
    log exists by default, training timing is nonzero and separated from
    eval/checkpoint categories."""
    from cycleft.e1.data import prepare_event_datasets
    from cycleft.e1.pipeline import resolve_lm_config
    from cycleft.e5.resumable_trainer import run_workload

    raw_config = resolve_lm_config(
        __import__("cycleft.config", fromlist=["load_config"]).load_config(
            ROOT / "configs" / "e1_stream.yaml"),
        mode="full", train_examples=None, seed=42, device=None,
    )[0]
    events, _ = prepare_event_datasets(train_limit=None, seed=42)
    rows = [dict(r) for r in events[0].train_rows[:3]]
    schedule = [{"stage": 1, "event_id": 1, "start": 0, "end": 3}]
    out = tmp_path / "instrumented_run"
    summary = run_workload(
        lm_config=raw_config, stream_rows=rows, out_dir=out,
        exec_id="e5p2-instrumentation-validation", ckpt_every=3,
        stage_schedule=schedule, log=lambda m: None,
        final_eval_events=None,  # eval timing validated separately below
    )
    assert (out / "audit_log.jsonl").exists(), "audit logging must be on by default"
    entries = [json.loads(l) for l in (out / "audit_log.jsonl").read_text().splitlines()]
    assert len(entries) == 3
    assert entries[0]["exec_id"] == "e5p2-instrumentation-validation"
    assert entries[0]["attempt"] == 1
    assert entries[0]["dataset_example_position"] == 0
    timing = summary["timing"]["categories"]
    assert timing["train"]["total_s"] > 0, "active training time must be nonzero"
    assert timing["ckpt_serialize"]["total_s"] > 0
    assert timing["ckpt_fsync_commit"]["total_s"] > 0
    assert timing["durable_copy"]["total_s"] > 0
    assert "eval_generation" not in timing or timing["eval_generation"]["total_s"] == 0
    assert summary["process_wall_s"] >= timing["train"]["total_s"]
    # event log: process_start, train segments, steps, checkpoint, exit
    events_logged = read_events(out / "event_log.jsonl")
    kinds = [e["event"] for e in events_logged]
    assert "process_start" in kinds and "process_exit" in kinds
    assert kinds.count("step") == 3
    assert "checkpoint_end" in kinds
