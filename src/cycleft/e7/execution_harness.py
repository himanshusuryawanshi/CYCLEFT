"""E7 Phase 2 guarded execution harness (hardened; dry-run default).

Hardening corrections (2026-11-21, independent-audit findings):

- Finding 1 (strict completion): COMPLETED requires explicit training AND
  evaluation evidence — exit code 0 and status "complete" for BOTH — plus
  full artifact presence and integrity. Missing evidence fails closed.
  The previous ``telemetry.setdefault("return_code", 0)`` pattern is gone.
- Finding 2 (recursive integrity): artifact directories are verified
  recursively (sorted relative paths, SHA-256 per file, deterministic
  digest); symlinks, path escapes, empty directories, and unexpected file
  types are rejected; the machine-readable inventory is persisted in the
  manifest and must match the completion marker.
- Finding 3 (atomic completion): FINALIZING -> marker -> COMPLETED ->
  re-read verification. Inconsistent evidence (COMPLETED without marker,
  FINALIZING with/without marker, hash or run-ID mismatches) fails closed
  and is never auto-deleted or repaired.
- Finding 4 (single-run pilot): execution requires exactly one NEW run ID
  (`--run-id`); REUSE/unknown/multiple IDs rejected; missing run ID never
  falls back to executing the full matrix. Dry-run may select one row for
  display while preserving the complete 22-row audit plan.
- Stop-on-failure: a callback failure marks the run FAILED (preserving all
  partial output), records the reason, and STOPS the harness immediately.
  No retry, no overwrite, no failure-to-success conversion.

Dry-run and matrix-inspection paths import only the Python standard
library; the E5-backed training callable is provided separately by
``cycleft.e7.e5_adapter`` (lazy imports) and injected as ``run_fn``.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as _dt
import hashlib
import json
import os
import re
import socket
import stat as _stat
import sys
import tempfile
from pathlib import Path
from typing import Callable

__all__ = [
    "ARTIFACT_SPEC", "EXECUTE_CONFIRM_TOKEN", "STATES",
    "AnchorRecord", "AnchorValidationError", "ExecutionHarness", "HarnessError",
    "LockError", "MatrixValidationError", "OutputSafetyError", "RunSpec",
    "StateError", "Telemetry", "classify_run_state", "inventory_digest",
    "load_and_validate_matrix", "main", "select_run_id", "validate_row",
]

EXECUTE_CONFIRM_TOKEN = "i-authorize-e7-phase2-execution"
STATES = ("PLANNED", "RUNNING", "FINALIZING", "COMPLETED", "FAILED", "INTERRUPTED")
TERMINAL_OK = "COMPLETED"

ALLOWED_POLICIES = {"Static-R4", "Static-R8", "Static-R9", "Static-R16",
                    "Static-R32", "Grow", "Reduce", "Resource-Aware"}
ALLOWED_PHASES = {"static-baseline", "dynamic-rank", "dynamic-rank-pilot"}
ALLOWED_SEEDS = {42, 43, 44}
EXEC_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
ANCHOR_REQUIRED_FILES = ("event_metrics.json", "resource_metrics.json",
                         "continual_learning_matrix.csv.json")
MANIFEST_NAME = "manifest.json"
COMPLETION_MARKER_NAME = "completion_marker.json"
LOCK_NAME = ".execution_lock.json"

MATRIX_COUNTS = {"total": 22, "new": 19, "reuse": 3}
MATRIX_POLICY_SEEDS = {
    "Static-R4": [42, 43, 44], "Static-R8": [42, 43, 44],
    "Static-R9": [42, 43, 44], "Static-R16": [42, 43, 44],
    "Static-R32": [42, 43, 44], "Grow": [42, 43, 44],
    "Reduce": [42, 43, 44], "Resource-Aware": [42],
}

# Required run artifacts (grounded in the E5 run_workload / E6 output
# schema — see E5_INTEGRATION_SPEC.md; no invented schema).
ARTIFACT_SPEC = {
    "execution_summary.json": "file",
    "final_eval.json": "file",
    "event_log.jsonl": "file",
    "audit_log.jsonl": "file",
    "bundles": "dir",
    "predictions": "dir",
}
PREDICTION_RECORD_KEYS = ("state", "test_event", "example_index", "question",
                          "reference", "raw_prediction", "prediction",
                          "exact_match", "token_f1")


class HarnessError(Exception):
    """Base class for all harness failures (fail closed)."""


class MatrixValidationError(HarnessError):
    pass


class OutputSafetyError(HarnessError):
    pass


class AnchorValidationError(HarnessError):
    pass


class LockError(HarnessError):
    pass


class StateError(HarnessError):
    pass


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Atomic JSON write: temp file in the same directory, fsync, os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp_", suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


@dataclasses.dataclass(frozen=True)
class Telemetry:
    """Logging fields prepared for eventual (authorized) execution."""
    wall_start: str | None = None
    wall_end: str | None = None
    training_s: float | None = None
    evaluation_s: float | None = None
    checkpoint_serialize_s: float | None = None
    rank_transition_s: float | None = None
    return_code: int | None = None
    peak_rss_mb: float | None = None
    final_rank_schedule: tuple | None = None
    final_eval_metrics: dict | None = None
    artifact_checksums: dict | None = None

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["final_rank_schedule"] = list(self.final_rank_schedule) if self.final_rank_schedule else None
        return d


@dataclasses.dataclass(frozen=True)
class RunSpec:
    exec_id: str
    policy: str
    phase: str
    seed: int
    row_type: str                      # NEW | REUSE
    initial_rank: int | str
    event_ranks: tuple                 # ints, or "runtime" for RA events 2-3
    event_alphas: tuple
    depth: str
    budget_per_event: int
    resource_aware_pilot: bool
    reuse_source: str                  # '' for NEW
    decision_policy_id: str
    decision_thresholds: str
    row: dict                          # original matrix row

    @property
    def config_hash(self) -> str:
        canonical = json.dumps(self.row, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()

    @property
    def anchor_path(self) -> Path | None:
        if self.row_type != "REUSE" or not self.reuse_source:
            return None
        raw = self.reuse_source.split(" (")[0].strip()  # strip annotation suffix
        return Path(raw)


@dataclasses.dataclass(frozen=True)
class AnchorRecord:
    exec_id: str
    path: str
    files: tuple
    checksums: dict

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _int_field(row: dict, key: str, exec_id: str) -> int:
    try:
        return int(row[key])
    except (KeyError, TypeError, ValueError) as e:
        raise MatrixValidationError(f"{exec_id}: field {key!r} must be an integer "
                                    f"(got {row.get(key)!r})") from e


def validate_row(row: dict) -> RunSpec:
    """Validate one matrix record; raise MatrixValidationError on any defect."""
    exec_id = row.get("exec_id", "<missing>")
    if "exec_id" not in row or row.get("exec_id") is None or not row["exec_id"]:
        raise MatrixValidationError("malformed record: missing exec_id")
    missing = [k for k in ("phase", "policy", "depth", "seed", "budget_per_event",
                           "reuse_or_new", "decision_policy_id")
               if row.get(k) is None]
    if missing:
        raise MatrixValidationError(f"{exec_id}: missing required fields {missing}")

    if not EXEC_ID_RE.match(row["exec_id"]) or ".." in row["exec_id"]:
        raise MatrixValidationError(f"{exec_id}: invalid exec_id format")

    policy, phase = row["policy"], row["phase"]
    if policy not in ALLOWED_POLICIES:
        raise MatrixValidationError(f"{exec_id}: unknown policy {policy!r}")
    if phase not in ALLOWED_PHASES:
        raise MatrixValidationError(f"{exec_id}: unknown phase {phase!r}")

    row_type = row["reuse_or_new"]
    if row_type not in ("NEW", "REUSE"):
        raise MatrixValidationError(f"{exec_id}: reuse_or_new must be NEW or REUSE")

    seed = _int_field(row, "seed", exec_id)
    if seed not in ALLOWED_SEEDS:
        raise MatrixValidationError(f"{exec_id}: seed {seed} not in {sorted(ALLOWED_SEEDS)}")
    if seed not in MATRIX_POLICY_SEEDS[policy]:
        raise MatrixValidationError(f"{exec_id}: seed {seed} not allowed for policy {policy}")

    budget = _int_field(row, "budget_per_event", exec_id)
    if budget != 500:
        raise MatrixValidationError(f"{exec_id}: budget_per_event must be 500 (b500 protocol)")
    if row["depth"] != "D100":
        raise MatrixValidationError(f"{exec_id}: depth must be D100")

    ranks = [row[f"event_{i}_rank"] for i in (1, 2, 3)]
    alphas = [row[f"event_{i}_alpha"] for i in (1, 2, 3)]

    if policy.startswith("Static-"):
        r = _int_field(row, "initial_rank", exec_id)
        if ranks != [str(r)] * 3:
            raise MatrixValidationError(f"{exec_id}: static ranks must be fixed at {r}")
        for k, j in zip(ranks, alphas):
            if int(j) != 2 * int(k):
                raise MatrixValidationError(f"{exec_id}: alpha {j} != 2 * rank {k}")
    elif policy == "Grow":
        if ranks != ["4", "8", "16"] or alphas != ["8", "16", "32"]:
            raise MatrixValidationError(f"{exec_id}: Grow must be 4->8->16 with alphas 8/16/32")
    elif policy == "Reduce":
        if ranks != ["16", "8", "4"] or alphas != ["32", "16", "8"]:
            raise MatrixValidationError(f"{exec_id}: Reduce must be 16->8->4 with alphas 32/16/8")
    elif policy == "Resource-Aware":
        if ranks[0] != "4" or alphas[0] != "8":
            raise MatrixValidationError(f"{exec_id}: Resource-Aware must start at rank 4 / alpha 8")
        if ranks[1:] != ["runtime", "runtime"] or alphas[1:] != ["runtime", "runtime"]:
            raise MatrixValidationError(
                f"{exec_id}: Resource-Aware future ranks/alphas must be 'runtime' "
                f"(hardcoded trajectories are rejected)")
        if row.get("decision_policy_id") != "convergence_signal":
            raise MatrixValidationError(f"{exec_id}: Resource-Aware decision_policy_id must be "
                                        f"'convergence_signal'")
    else:  # pragma: no cover - policy already restricted to known set
        raise MatrixValidationError(f"{exec_id}: unhandled policy {policy!r}")

    if row_type == "NEW" and row.get("reuse_source", "").strip():
        raise MatrixValidationError(f"{exec_id}: NEW row must have empty reuse_source")
    if row_type == "REUSE" and not row.get("reuse_source", "").strip():
        raise MatrixValidationError(f"{exec_id}: REUSE row must reference its anchor artifact")

    return RunSpec(
        exec_id=row["exec_id"], policy=policy, phase=phase, seed=seed,
        row_type=row_type, initial_rank=row["initial_rank"],
        event_ranks=tuple(ranks), event_alphas=tuple(alphas),
        depth=row["depth"], budget_per_event=budget,
        resource_aware_pilot=policy == "Resource-Aware",
        reuse_source=row.get("reuse_source", ""),
        decision_policy_id=row["decision_policy_id"],
        decision_thresholds=row.get("decision_thresholds", ""),
        row=dict(row),
    )


def load_and_validate_matrix(matrix_path: Path) -> list[RunSpec]:
    """Parse and fully validate the authoritative matrix; fail closed."""
    with open(matrix_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if len(rows) != MATRIX_COUNTS["total"]:
        raise MatrixValidationError(
            f"matrix must have exactly {MATRIX_COUNTS['total']} outcome rows, got {len(rows)}")
    specs = [validate_row(r) for r in rows]
    ids = [s.exec_id for s in specs]
    if len(set(ids)) != len(ids):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        raise MatrixValidationError(f"duplicate run IDs: {dupes}")
    new = [s for s in specs if s.row_type == "NEW"]
    reuse = [s for s in specs if s.row_type == "REUSE"]
    if len(new) != MATRIX_COUNTS["new"] or len(reuse) != MATRIX_COUNTS["reuse"]:
        raise MatrixValidationError(
            f"matrix composition must be {MATRIX_COUNTS['new']} NEW + "
            f"{MATRIX_COUNTS['reuse']} REUSE, got {len(new)} + {len(reuse)}")
    for policy, seeds in MATRIX_POLICY_SEEDS.items():
        got = sorted(s.seed for s in specs if s.policy == policy)
        if got != sorted(seeds):
            raise MatrixValidationError(f"policy {policy}: expected seeds {seeds}, got {got}")
    return specs


def select_run_id(specs: list[RunSpec], run_id: str) -> RunSpec:
    """Select exactly one NEW run by ID (Finding 4); fail closed otherwise."""
    if not isinstance(run_id, str) or not run_id.strip():
        raise HarnessError("run selection requires a non-empty run ID")
    matches = [s for s in specs if s.exec_id == run_id]
    if not matches:
        raise HarnessError(
            f"unknown run ID {run_id!r}: not present in the validated matrix")
    if len(matches) > 1:
        raise HarnessError(f"run ID {run_id!r} matches {len(matches)} rows; "
                           f"IDs must be unique")
    spec = matches[0]
    if spec.row_type != "NEW":
        raise HarnessError(
            f"{run_id!r} is a REUSE row: REUSE anchors are read-only references "
            f"and are never execution targets")
    return spec


def validate_anchor(spec: RunSpec, repo_root: Path) -> AnchorRecord:
    """Validate a REUSE anchor read-only: existence, required files, checksums."""
    anchor = spec.anchor_path
    if anchor is None:
        raise AnchorValidationError(f"{spec.exec_id}: REUSE row has no anchor path")
    full = (repo_root / anchor)
    if not full.is_dir():
        raise AnchorValidationError(f"{spec.exec_id}: anchor {anchor} does not exist")
    files, checksums = [], {}
    for name in ANCHOR_REQUIRED_FILES:
        p = full / name
        if not p.is_file():
            raise AnchorValidationError(f"{spec.exec_id}: anchor {anchor} missing required "
                                        f"artifact {name!r}")
        files.append(name)
        checksums[name] = _sha256_file(p)
    return AnchorRecord(exec_id=spec.exec_id, path=str(anchor), files=tuple(files),
                        checksums=checksums)


def _resolve_within(root: Path, candidate: Path) -> Path:
    """Resolve candidate and verify it stays inside root (no traversal/escape)."""
    root_res = root.resolve()
    cand_res = candidate.resolve()
    if cand_res != root_res and root_res not in cand_res.parents:
        raise OutputSafetyError(
            f"resolved path {cand_res} escapes authorized output root {root_res}")
    return cand_res


def resolve_output_dir(output_root: Path, exec_id: str) -> Path:
    """Safe per-run directory path under the authorized root."""
    if not EXEC_ID_RE.match(exec_id) or ".." in exec_id:
        raise OutputSafetyError(f"invalid exec_id {exec_id!r} (path traversal rejected)")
    return _resolve_within(output_root, output_root / exec_id)


class _ExecutionLock:
    """Single-instance lock for an output root (O_CREAT|O_EXCL, PID recorded).

    Stale locks (owning PID no longer alive) are DETECTED and REPORTED but
    never silently deleted — the operator removes them explicitly.
    """

    def __init__(self, output_root: Path):
        self.path = output_root / LOCK_NAME
        self._held = False

    def __enter__(self) -> "_ExecutionLock":
        output_root = self.path.parent
        output_root.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            try:
                info = json.loads(self.path.read_text())
            except (OSError, json.JSONDecodeError):
                info = {"note": "unreadable lock"}
            pid = info.get("pid")
            alive = False
            if isinstance(pid, int) and pid > 0:
                try:
                    os.kill(pid, 0)
                    alive = True
                except ProcessLookupError:
                    alive = False
                except PermissionError:
                    alive = True  # exists, owned by another user
            kind = "active" if alive else "STALE"
            raise LockError(
                f"{kind} execution lock present at {self.path} (info={info}); "
                f"a second harness instance may not use this output root. "
                f"Stale locks must be removed manually by the operator "
                f"(never auto-deleted).")
        payload = {"pid": os.getpid(), "host": socket.gethostname(),
                   "acquired_at": _now(), "harness": "e7-phase2"}
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as e:
            raise LockError(f"execution lock already present at {self.path}") from e
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f)
        self._held = True
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._held and self.path.exists():
            self.path.unlink()
        self._held = False


@dataclasses.dataclass
class PlannedRun:
    exec_id: str
    policy: str
    seed: int
    row_type: str
    event_ranks: tuple
    event_alphas: tuple
    output_path: str            # per-run dir for NEW; anchor path for REUSE
    config_hash: str
    reuse_source: str
    action: str                 # "train" for NEW; "read-only reference" for REUSE
    selected: bool = False

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["event_ranks"] = list(self.event_ranks)
        d["event_alphas"] = list(self.event_alphas)
        return d


# ---------------------------------------------------------------------------
# Finding 2 — recursive artifact integrity
# ---------------------------------------------------------------------------


def _walk_artifact_dir(base: Path, run_dir_resolved: Path) -> list[tuple[str, Path]]:
    """Deterministic recursive walk rejecting symlinks, escapes, and any
    entry that is neither a regular file nor a directory."""
    entries: list[tuple[str, Path]] = []

    def walk(dir_path: Path, prefix: str) -> None:
        for child in sorted(dir_path.iterdir(), key=lambda p: p.name):
            rel = f"{prefix}{child.name}"
            st = child.lstat()
            if _stat.S_ISLNK(st.st_mode):
                raise StateError(
                    f"artifact integrity: symlink rejected at {child} "
                    f"(relative {rel!r})")
            resolved = child.resolve()
            if resolved != run_dir_resolved and run_dir_resolved not in resolved.parents:
                raise StateError(
                    f"artifact integrity: path escapes the run directory at "
                    f"{child} (resolved {resolved})")
            if _stat.S_ISDIR(st.st_mode):
                walk(child, f"{rel}/")
            elif _stat.S_ISREG(st.st_mode):
                entries.append((rel, child))
            else:
                raise StateError(
                    f"artifact integrity: unexpected file type at {child} "
                    f"(relative {rel!r}, mode {st.st_mode:#x})")

    walk(base, "")
    return entries


def inventory_artifact(run_dir: Path, name: str) -> dict:
    """Build the integrity inventory for one required artifact.

    Files hash their contents; directories hash every contained regular
    file (deterministic order: sorted relative POSIX paths) and produce a
    deterministic digest. Empty directories are invalid.
    """
    run_dir_resolved = run_dir.resolve()
    target = run_dir / name
    if not target.exists():
        raise StateError(f"artifact integrity: required artifact {name!r} missing")
    if target.is_symlink():
        raise StateError(f"artifact integrity: {name!r} must not be a symlink")
    if target.is_file():
        digest = _sha256_file(target)
        return {"type": "file", "size": target.stat().st_size,
                "files": {name: digest}, "digest": digest}
    if not target.is_dir():
        raise StateError(f"artifact integrity: {name!r} is neither file nor directory")
    walked = _walk_artifact_dir(target, run_dir_resolved)
    if not walked:
        raise StateError(
            f"artifact integrity: {name!r} is an empty directory; an empty "
            f"directory is not a valid artifact")
    files = {rel: _sha256_file(p) for rel, p in walked}
    canonical = "".join(f"{rel}:{files[rel]}\n" for rel in sorted(files))
    return {"type": "dir", "size": sum((run_dir / name / rel).stat().st_size
                                       for rel in files),
            "files": files,
            "digest": hashlib.sha256(canonical.encode()).hexdigest()}


def build_artifact_inventory(run_dir: Path) -> dict:
    """Full inventory: structural validation + integrity for every required
    artifact. Raises StateError (fail closed) on any defect."""
    inventory = {}
    for name in sorted(ARTIFACT_SPEC):
        inv = inventory_artifact(run_dir, name)
        _validate_artifact_structure(run_dir, name, inv)
        inventory[name] = inv
    return inventory


def inventory_digest(inventory: dict) -> str:
    """Deterministic digest over the whole inventory."""
    canonical = json.dumps(inventory, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _validate_artifact_structure(run_dir: Path, name: str, inv: dict) -> None:
    """Structural validation grounded in the E5/E6 output schema."""
    if name == "execution_summary.json":
        data = json.loads((run_dir / name).read_text())
        if not isinstance(data, dict):
            raise StateError("execution_summary.json must be a JSON object")
        if data.get("status") != "complete":
            raise StateError(f"execution_summary.json status is "
                             f"{data.get('status')!r}, expected 'complete'")
        if int(data.get("global_step", -1)) != 1500:
            raise StateError(f"execution_summary.json global_step "
                             f"{data.get('global_step')!r} != 1500 (b500 protocol)")
        if not isinstance(data.get("final_avg_em"), (int, float)):
            raise StateError("execution_summary.json final_avg_em missing/not numeric")
    elif name == "final_eval.json":
        data = json.loads((run_dir / name).read_text())
        if not (isinstance(data, dict) and data):
            raise StateError("final_eval.json must be a non-empty JSON object")
        for event, metrics in data.items():
            if not isinstance(metrics, dict) or \
                    not isinstance(metrics.get("exact_match"), (int, float)):
                raise StateError(f"final_eval.json event {event!r} lacks numeric "
                                 f"exact_match")
    elif name == "event_log.jsonl":
        lines = [l for l in (run_dir / name).read_text().splitlines() if l.strip()]
        if not lines:
            raise StateError("event_log.jsonl is empty")
        for i, line in enumerate(lines):
            rec = json.loads(line)
            if not isinstance(rec, dict) or "event" not in rec:
                raise StateError(f"event_log.jsonl line {i+1} is not an event record")
    elif name == "audit_log.jsonl":
        lines = [l for l in (run_dir / name).read_text().splitlines() if l.strip()]
        if not lines:
            raise StateError("audit_log.jsonl is empty")
        for i, line in enumerate(lines):
            if not isinstance(json.loads(line), dict):
                raise StateError(f"audit_log.jsonl line {i+1} is not a JSON object")
    elif name == "bundles":
        latest = run_dir / "bundles" / "LATEST.json"
        if not latest.is_file():
            raise StateError("bundles/LATEST.json missing")
        data = json.loads(latest.read_text())
        latest_name = data.get("latest")
        if not isinstance(latest_name, str) or not latest_name:
            raise StateError("bundles/LATEST.json lacks a 'latest' entry")
        adapter = run_dir / "bundles" / latest_name / "adapter"
        if not adapter.is_dir() or not any(adapter.iterdir()):
            raise StateError(f"bundle adapter directory {latest_name}/adapter "
                             f"missing or empty")
    elif name == "predictions":
        jsonl = sorted(p for p in (run_dir / "predictions").glob("*.jsonl"))
        if not jsonl:
            raise StateError("predictions/ contains no .jsonl prediction files")
        for p in jsonl:
            text = p.read_text()
            if not text.strip():
                raise StateError(f"predictions/{p.name} is empty")
            for i, line in enumerate(text.splitlines()):
                if not line.strip():
                    continue
                rec = json.loads(line)
                missing = [k for k in PREDICTION_RECORD_KEYS if k not in rec]
                if missing:
                    raise StateError(f"predictions/{p.name} line {i+1} lacks "
                                     f"required record keys {missing}")


# ---------------------------------------------------------------------------
# Finding 3 — run-state classification (fail closed)
# ---------------------------------------------------------------------------


def classify_run_state(run_dir: Path) -> dict:
    """Classify recorded completion evidence; NEVER repairs or deletes."""
    manifest_path = run_dir / MANIFEST_NAME
    marker_path = run_dir / COMPLETION_MARKER_NAME
    if not manifest_path.is_file():
        return {"classification": "MANIFEST_MISSING", "fail_closed": True,
                "detail": f"no manifest at {manifest_path}"}
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as e:
        return {"classification": "CORRUPT", "fail_closed": True,
                "detail": f"manifest unreadable: {e}"}
    state = manifest.get("state")
    marker = None
    if marker_path.is_file():
        try:
            marker = json.loads(marker_path.read_text())
        except json.JSONDecodeError as e:
            return {"classification": "CORRUPT", "fail_closed": True,
                    "detail": f"completion marker unreadable: {e}"}

    if state == "COMPLETED":
        if marker is None:
            return {"classification": "CORRUPT", "fail_closed": True,
                    "detail": "manifest COMPLETED without a completion marker"}
        problems = []
        if marker.get("exec_id") != manifest.get("exec_id"):
            problems.append("run ID mismatch between manifest and marker")
        if marker.get("config_hash") != manifest.get("config_hash"):
            problems.append("configuration hash mismatch between manifest and marker")
        if marker.get("artifact_inventory_digest") != \
                manifest.get("artifact_inventory_digest"):
            problems.append("artifact inventory digest mismatch between manifest "
                            "and marker")
        if problems:
            return {"classification": "CORRUPT", "fail_closed": True,
                    "detail": "; ".join(problems)}
        return {"classification": "COMPLETED", "fail_closed": False,
                "detail": "manifest and marker agree"}
    if state == "FINALIZING":
        if marker is not None:
            return {"classification": "AMBIGUOUS_FINALIZING", "fail_closed": True,
                    "detail": "FINALIZING manifest with a completion marker: "
                              "finalization did not verify; operator review required"}
        return {"classification": "INCOMPLETE_FINALIZING", "fail_closed": True,
                "detail": "FINALIZING manifest without a completion marker"}
    if state in ("RUNNING", "FAILED", "INTERRUPTED", "PLANNED"):
        return {"classification": state, "fail_closed": state == "RUNNING",
                "detail": f"recorded state {state!r}"}
    return {"classification": "CORRUPT", "fail_closed": True,
            "detail": f"unknown recorded state {state!r}"}


# ---------------------------------------------------------------------------
# harness
# ---------------------------------------------------------------------------


class ExecutionHarness:
    """Guarded, sequential executor for the E7 v5 matrix.

    ``dry_run()`` is side-effect free for experiment outputs and never
    imports the model-training path. ``execute()`` requires the explicit
    confirmation token, an injected ``run_fn`` training callable, and
    exactly one NEW run ID; it is NEVER invoked by design-only tasks.
    """

    def __init__(self, matrix_path: Path, output_root: Path,
                 repo_root: Path | None = None,
                 run_fn: Callable[[RunSpec, Path], dict] | None = None):
        self.matrix_path = Path(matrix_path)
        self.output_root = Path(output_root)
        self.repo_root = Path(repo_root) if repo_root else Path.cwd()
        self.run_fn = run_fn
        self._specs: list[RunSpec] | None = None
        self._anchors: dict[str, AnchorRecord] = {}
        self._interrupted = False

    # ------------------------------------------------------------------ #
    # matrix + anchors                                                    #
    # ------------------------------------------------------------------ #

    def specs(self) -> list[RunSpec]:
        if self._specs is None:
            self._specs = load_and_validate_matrix(self.matrix_path)
        return self._specs

    def validate_anchors(self) -> dict[str, AnchorRecord]:
        """Fail closed BEFORE any NEW run if any anchor is missing/invalid."""
        records = {}
        for spec in self.specs():
            if spec.row_type == "REUSE":
                records[spec.exec_id] = validate_anchor(spec, self.repo_root)
        self._anchors = records
        return records

    def _validate_output_root(self) -> Path:
        authorized = (self.repo_root / "results" / "e7").resolve()
        root_res = self.output_root.resolve()
        if root_res != authorized and authorized not in root_res.parents:
            raise OutputSafetyError(
                f"output root {self.output_root} resolves outside the authorized "
                f"E7 experiment tree {authorized}")
        if self.output_root.exists() and self.output_root.is_symlink():
            raise OutputSafetyError("output root must not be a symlink")
        return self.output_root

    # ------------------------------------------------------------------ #
    # dry run                                                             #
    # ------------------------------------------------------------------ #

    def dry_run(self, run_id: str | None = None) -> list[PlannedRun]:
        """Build the ordered execution plan. Creates/modify NOTHING in the
        output root; never touches anchors; never imports the training path.

        With ``run_id``, the full 22-row matrix is still validated and the
        complete plan preserved; only the display/selection flag changes.
        """
        self._validate_output_root()
        specs = self.specs()
        self.validate_anchors()  # read-only existence + checksums
        selected_id = None
        if run_id is not None:
            selected_id = select_run_id(specs, run_id).exec_id
        plan = []
        for spec in specs:
            if spec.row_type == "REUSE":
                out = str(spec.anchor_path)
                action = "read-only reference"
            else:
                out = str(resolve_output_dir(self.output_root, spec.exec_id))
                action = "train"
            plan.append(PlannedRun(
                exec_id=spec.exec_id, policy=spec.policy, seed=spec.seed,
                row_type=spec.row_type, event_ranks=spec.event_ranks,
                event_alphas=spec.event_alphas, output_path=out,
                config_hash=spec.config_hash, reuse_source=spec.reuse_source,
                action=action, selected=spec.exec_id == selected_id))
        return plan

    # ------------------------------------------------------------------ #
    # guarded execution (NOT invoked in design-only tasks)                #
    # ------------------------------------------------------------------ #

    def execute(self, confirm_token: str, run_id: str) -> PlannedRun:
        """Execute exactly one selected NEW run under the execution lock.

        Requires the explicit confirmation token AND a valid single NEW
        run ID. A missing run ID never falls back to executing every NEW
        row (Finding 4).
        """
        if confirm_token != EXECUTE_CONFIRM_TOKEN:
            raise HarnessError(
                "execution refused: missing or incorrect confirmation token "
                f"(expected --confirm-token {EXECUTE_CONFIRM_TOKEN!r})")
        if run_id is None:
            raise HarnessError(
                "execution refused: a single NEW run ID is required "
                "(--run-id); a missing run ID is never treated as permission "
                "to execute the full matrix")
        self._validate_output_root()
        self.validate_anchors()  # fail closed before any NEW run
        spec = select_run_id(self.specs(), run_id)
        if self.run_fn is None:
            raise HarnessError("execution refused: no run_fn training callable injected")
        planned = [p for p in self.dry_run() if p.exec_id == spec.exec_id][0]

        with _ExecutionLock(self.output_root):
            executed = self._execute_selected([spec], [planned])
        return executed[0]

    def _execute_selected(self, selected: list[RunSpec],
                          planned: list[PlannedRun]) -> list[PlannedRun]:
        """Sequential execution of the selected NEW runs (stop-on-failure)."""
        results = []
        for spec, planned_run in zip(selected, planned):
            if spec.row_type == "REUSE":
                results.append(planned_run)  # read-only reference; never trained
                continue
            run_dir = resolve_output_dir(self.output_root, spec.exec_id)
            self._guard_fresh_dir(run_dir, spec)
            run_dir.mkdir(parents=True, exist_ok=False)
            manifest = {
                "exec_id": spec.exec_id, "config_hash": spec.config_hash,
                "state": "RUNNING", "created_at": _now(), "updated_at": _now(),
                "source_row": spec.row, "anchor": None,
                "telemetry": Telemetry(wall_start=_now(),
                                       final_rank_schedule=spec.event_ranks).to_dict(),
            }
            _atomic_write_json(run_dir / MANIFEST_NAME, manifest)
            try:
                telemetry = self.run_fn(spec, run_dir)  # injected callable
                self._finalize_completed(run_dir, spec, manifest, telemetry)
            except KeyboardInterrupt:
                self._mark_interrupted(run_dir, spec, manifest,
                                       reason="KeyboardInterrupt")
                raise
            except BaseException as e:  # noqa: BLE001
                reason = f"{type(e).__name__}: {e}"
                self._mark_failed(run_dir, spec, manifest, reason=reason)
                # Stop-on-failure: preserve partial output, record the
                # exception, stop immediately; no retry, no cleanup, and
                # never convert failure into success.
                raise HarnessError(
                    f"execution stopped: {spec.exec_id} FAILED ({reason}); "
                    f"partial output preserved at {run_dir}; no further runs "
                    f"executed; no automatic retry") from e
            results.append(planned_run)
        return results

    # ------------------------------------------------------------------ #
    # state machine                                                       #
    # ------------------------------------------------------------------ #

    def _guard_fresh_dir(self, run_dir: Path, spec: RunSpec) -> None:
        if run_dir.exists():
            has_marker = (run_dir / COMPLETION_MARKER_NAME).exists()
            has_manifest = (run_dir / MANIFEST_NAME).exists()
            has_content = any(run_dir.iterdir())
            if has_marker or has_manifest or has_content:
                raise OutputSafetyError(
                    f"{spec.exec_id}: refusing to execute — output directory "
                    f"{run_dir} already contains artifacts"
                    + (" and a completion marker" if has_marker else "")
                    + (" and a prior execution manifest" if has_manifest else "")
                    + "; existing run directories are never overwritten, "
                      "cleaned, or reused")

    def _read_manifest(self, run_dir: Path) -> dict:
        p = run_dir / MANIFEST_NAME
        if not p.is_file():
            raise StateError(f"no manifest at {p} (fail closed on ambiguous state)")
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError as e:
            raise StateError(f"corrupt manifest at {p}: {e}") from e

    def _update_state(self, run_dir: Path, manifest: dict, state: str,
                      **extra) -> None:
        if state not in STATES:
            raise StateError(f"unknown state {state!r}")
        manifest["state"] = state
        manifest["updated_at"] = _now()
        manifest.update(extra)
        _atomic_write_json(run_dir / MANIFEST_NAME, manifest)

    def _finalize_completed(self, run_dir: Path, spec: RunSpec,
                            manifest: dict, telemetry: dict) -> None:
        """Crash-consistent completion protocol (Findings 1-3).

        COMPLETED only when training AND evaluation explicitly returned
        exit code 0 with status "complete", every required artifact exists
        and passes recursive integrity + structural validation, and the
        manifest/marker agree. Any missing or conflicting evidence fails
        closed without changing the recorded state.
        """
        if manifest.get("state") != "RUNNING":
            raise StateError(
                f"{spec.exec_id}: cannot mark COMPLETED from state "
                f"{manifest.get('state')!r} (incomplete/failed/interrupted runs "
                f"are never silently promoted)")

        # --- Step 1: explicit execution evidence (Finding 1) -------------
        training = telemetry.get("training") if isinstance(telemetry, dict) else None
        evaluation = telemetry.get("evaluation") if isinstance(telemetry, dict) else None
        if not isinstance(training, dict):
            raise StateError(f"{spec.exec_id}: completion refused — training "
                             f"evidence missing (fail closed)")
        if training.get("exit_code") is None:
            raise StateError(f"{spec.exec_id}: completion refused — training "
                             f"return code missing (fail closed)")
        if training.get("exit_code") != 0:
            raise StateError(f"{spec.exec_id}: completion refused — training "
                             f"exit code {training.get('exit_code')!r} != 0")
        if training.get("status") != "complete":
            raise StateError(f"{spec.exec_id}: completion refused — training "
                             f"status {training.get('status')!r} != 'complete'")
        if not isinstance(evaluation, dict):
            raise StateError(f"{spec.exec_id}: completion refused — evaluation "
                             f"evidence missing (fail closed)")
        if evaluation.get("exit_code") is None:
            raise StateError(f"{spec.exec_id}: completion refused — evaluation "
                             f"return code missing (fail closed)")
        if evaluation.get("exit_code") != 0:
            raise StateError(f"{spec.exec_id}: completion refused — evaluation "
                             f"exit code {evaluation.get('exit_code')!r} != 0")
        if evaluation.get("status") != "complete":
            raise StateError(f"{spec.exec_id}: completion refused — evaluation "
                             f"status {evaluation.get('status')!r} != 'complete'")

        # --- Step 2: artifact inventory (Finding 2) ----------------------
        inventory = build_artifact_inventory(run_dir)
        digest = inventory_digest(inventory)

        # --- Step 3: FINALIZING manifest (atomic) ------------------------
        self._update_state(run_dir, manifest, "FINALIZING",
                           artifact_inventory=inventory,
                           artifact_inventory_digest=digest,
                           telemetry=telemetry)

        # --- Step 4: completion marker (atomic) --------------------------
        _atomic_write_json(run_dir / COMPLETION_MARKER_NAME, {
            "exec_id": spec.exec_id, "config_hash": spec.config_hash,
            "state": TERMINAL_OK,
            "artifact_inventory_digest": digest,
            "artifact_inventory": inventory,
            "completed_at": _now(),
        })

        # --- Step 5: final COMPLETED manifest (atomic) -------------------
        self._update_state(run_dir, manifest, TERMINAL_OK,
                           completed_at=_now(),
                           artifact_inventory=inventory,
                           artifact_inventory_digest=digest)

        # --- Step 6: re-read and verify consistency ----------------------
        m2 = json.loads((run_dir / MANIFEST_NAME).read_text())
        k2 = json.loads((run_dir / COMPLETION_MARKER_NAME).read_text())
        problems = []
        if m2.get("state") != TERMINAL_OK or k2.get("state") != TERMINAL_OK:
            problems.append("state mismatch after finalization")
        if not (m2.get("exec_id") == k2.get("exec_id") == spec.exec_id):
            problems.append("run ID mismatch between manifest and marker")
        if not (m2.get("config_hash") == k2.get("config_hash") == spec.config_hash):
            problems.append("configuration hash mismatch between manifest and marker")
        if m2.get("artifact_inventory_digest") != k2.get("artifact_inventory_digest"):
            problems.append("artifact inventory digest mismatch between manifest "
                            "and marker")
        if problems:
            raise StateError(
                f"{spec.exec_id}: completion evidence inconsistent after "
                f"finalization ({'; '.join(problems)}); fail closed — forensic "
                f"artifacts preserved, operator review required")

    def _mark_failed(self, run_dir: Path, spec: RunSpec, manifest: dict,
                     reason: str) -> None:
        self._update_state(run_dir, manifest, "FAILED",
                           failure_reason=reason, failed_at=_now())

    def _mark_interrupted(self, run_dir: Path, spec: RunSpec, manifest: dict,
                          reason: str) -> None:
        """Interrupted runs stay INTERRUPTED — distinguishable from COMPLETED,
        never auto-restarted (resume safety is not claimed or implemented)."""
        self._update_state(run_dir, manifest, "INTERRUPTED",
                           interruption_reason=reason, interrupted_at=_now())

    def inspect_state(self, run_dir: Path) -> dict:
        """Read-only inspection; classifies completion evidence fail-closed."""
        manifest = self._read_manifest(run_dir)
        classification = classify_run_state(run_dir)
        return {
            "exec_id": manifest.get("exec_id"),
            "state": manifest.get("state"),
            "classification": classification["classification"],
            "fail_closed": classification["fail_closed"],
            "classification_detail": classification["detail"],
            "failure_reason": manifest.get("failure_reason"),
            "interruption_reason": manifest.get("interruption_reason"),
            "updated_at": manifest.get("updated_at"),
        }


def _plan_markdown(plan: list[PlannedRun], totals_note: str,
                   selected_id: str | None) -> str:
    lines = [
        "# E7 Phase 2 — Dry-Run Execution Plan",
        "",
        "Generated by the guarded harness in **dry-run mode** (default).",
        "**No experiment was executed. No output directory was created or",
        "modified. No model was loaded.**",
        "",
    ]
    if selected_id is not None:
        lines += [f"**Single-run pilot selection: `{selected_id}`** (full 22-row",
                  "audit plan preserved below; at most this one row may execute,",
                  "under separate explicit authorization).", ""]
    lines += [
        "| # | exec_id | policy | seed | type | ranks | alphas | output path | action | selected |",
        "|---|---------|--------|------|------|-------|--------|-------------|--------|----------|",
    ]
    for i, p in enumerate(plan, 1):
        ranks = "->".join(str(r) for r in p.event_ranks)
        alphas = "->".join(str(a) for a in p.event_alphas)
        lines.append(f"| {i} | {p.exec_id} | {p.policy} | {p.seed} | {p.row_type} "
                     f"| {ranks} | {alphas} | `{p.output_path}` | {p.action} "
                     f"| {'**YES**' if p.selected else ''} |")
    n_new = sum(1 for p in plan if p.row_type == "NEW")
    n_reuse = sum(1 for p in plan if p.row_type == "REUSE")
    lines += [
        "",
        f"**Plan totals: {len(plan)} rows = {n_new} NEW + {n_reuse} REUSE.** "
        f"{totals_note}",
        "",
        "REUSE rows are read-only references to existing E2 Static-R8 anchor",
        "artifacts; they are never scheduled as training. NEW rows map to",
        "unique per-run directories under the authorized output root and are",
        "executed strictly one at a time (single-run pilot selection) under",
        "the execution lock after explicit authorization",
        "(`--execute --confirm-token ... --run-id <NEW run ID>`).",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="E7 Phase 2 guarded execution harness")
    ap.add_argument("--matrix", type=Path,
                    default=Path("results/e7/design_20261120/v5_final_reconciliation/FINAL_RUN_MATRIX.csv"))
    ap.add_argument("--output-root", type=Path, default=Path("results/e7/execution_20261121"))
    ap.add_argument("--repo-root", type=Path, default=Path.cwd())
    ap.add_argument("--plan-out", type=Path, default=None,
                    help="write machine-readable plan (JSON) + markdown to this directory")
    ap.add_argument("--run-id", action="append", default=None,
                    help="single-run pilot selection (exactly one; repeat flag rejected)")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", default=True,
                      help="(default) plan only; no execution, no outputs")
    mode.add_argument("--execute", action="store_true",
                      help="execute the single selected run (requires --confirm-token "
                           "and --run-id; NOT authorized yet)")
    ap.add_argument("--confirm-token", type=str, default="")
    args = ap.parse_args(argv)

    if args.run_id and len(args.run_id) > 1:
        print("ERROR: multiple --run-id values are rejected; select exactly one "
              "NEW run per authorization", file=sys.stderr)
        return 2
    selected_id = args.run_id[0] if args.run_id else None

    harness = ExecutionHarness(args.matrix, args.output_root, repo_root=args.repo_root)
    plan = harness.dry_run(run_id=selected_id)
    totals_note = ("Validated against the authoritative v5 matrix "
                   "(22 rows = 19 NEW + 3 REUSE; anchors verified read-only).")
    md = _plan_markdown(plan, totals_note, selected_id)
    print(md)
    if args.plan_out is not None:
        args.plan_out.mkdir(parents=True, exist_ok=True)
        (args.plan_out / "DRY_RUN_PLAN.md").write_text(md)
        _atomic_write_json(args.plan_out / "DRY_RUN_PLAN.json", {
            "generated_at": _now(),
            "mode": "DRY-RUN",
            "executed": False,
            "statement": "No experiment was executed. No output directory was "
                         "created or modified. No model was loaded.",
            "matrix": str(args.matrix),
            "output_root_planned": str(args.output_root),
            "selected_run_id": selected_id,
            "totals": {"rows": len(plan),
                       "new": sum(1 for p in plan if p.row_type == "NEW"),
                       "reuse": sum(1 for p in plan if p.row_type == "REUSE")},
            "anchors": {k: v.to_dict() for k, v in harness._anchors.items()},
            "plan": [p.to_dict() for p in plan],
        })
    if args.execute:
        # Guarded path — requires token + single NEW run ID; refused otherwise.
        harness.execute(args.confirm_token, run_id=selected_id)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
