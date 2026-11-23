"""E5 append-only event log (Phase 2).

JSONL events, flushed (and fsynced at critical boundaries) so the log
survives abrupt process termination. The parent supervisor appends
termination records with exit status and signal information — a SIGKILLed
worker never writes its own final summary.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

EVENT_TYPES = (
    "process_start",
    "train_segment_start",
    "train_segment_end",
    "step",
    "checkpoint_start",
    "checkpoint_end",
    "interruption_request",
    "checkpoint_restore",
    "resume_position",
    "eval_start",
    "eval_end",
    "process_exit",
    "supervisor_termination",
)


class EventLog:
    def __init__(self, path: Path, exec_id: str = "", pid: int | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.exec_id = exec_id
        self.pid = pid if pid is not None else os.getpid()
        self._fh = self.path.open("a", encoding="utf-8")

    def emit(self, event_type: str, payload: dict | None = None, fsync: bool = False) -> None:
        if event_type not in EVENT_TYPES:
            raise KeyError(f"unknown event type {event_type!r}")
        record = {
            "ts_ns": time.perf_counter_ns(),
            "ts_epoch": time.time(),
            "exec_id": self.exec_id,
            "pid": self.pid,
            "event": event_type,
        }
        if payload:
            record.update(payload)
        self._fh.write(json.dumps(record, default=str) + "\n")
        self._fh.flush()
        if fsync:
            os.fsync(self._fh.fileno())

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def read_events(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def supervisor_termination_record(
    supervisor_log: Path,
    exec_id: str,
    child_pid: int,
    mode: str,
    returncode: int,
    extra: dict | None = None,
) -> dict:
    """Parent-supervisor record of child termination. Called by the SUPERVISOR
    (which survives the child) — never by the SIGKILLed child itself.

    returncode semantics (subprocess): negative N ⇒ child terminated by
    signal N (e.g. -9 ⇒ SIGKILL); 0 ⇒ clean exit.
    """
    record = {
        "exec_id": exec_id,
        "child_pid": child_pid,
        "mode": mode,
        "returncode": returncode,
        "exit_status": (
            "clean" if returncode == 0
            else f"signal_{-returncode}" if returncode < 0
            else f"exit_code_{returncode}"
        ),
        "signal": -returncode if returncode < 0 else None,
        "ts_epoch": time.time(),
    }
    if extra:
        record.update(extra)
    with Path(supervisor_log).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    return record
