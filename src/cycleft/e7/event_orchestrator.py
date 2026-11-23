"""E7 Option B dynamic rank event orchestrator (G1-G4, G6-G7).

Coordinates three sequential invocations of the EXISTING E5 workflow
(``cycleft.e5.resumable_trainer.run_workload``) — one per continual-learning
event — with LoRA rank transitions at event boundaries performed by the
Phase-1-verified ``cycleft.e7.rank_transition`` mathematics. The
orchestrator is a coordinator, not a trainer: E5 remains responsible for
training, checkpointing, evaluation semantics, and interruption behavior.

Design authority: results/e7/dynamic_orchestration_design_20261121_v2_1/
(event-isolated directories, seed/terminal checkpoint table, per-event
evaluation schema, master manifest, runtime accounting).

Structure per dynamic run (Grow 4→8→16; Reduce 16→8→4):

    run_dir/
      events/event_k/            ← E5 out_dir for invocation k (private)
        bundles/step_000500/     ← event-1 TERMINAL (k=1; no seed)
        bundles/step_000500/     ← event-2 SEED (k=2; transitioned, intra=0)
        bundles/step_001000/     ← event-2 TERMINAL
        ...
        final_eval.json  predictions.jsonl  evaluation_config.json
        execution_summary.json  artifact_inventory.json
        event_log.jsonl  audit_log.jsonl  stage_end_fingerprints.json
      transitions/transition_event_{k}_to_{k+1}.json
      predictions/after_event_{1,2,3}.jsonl     ← run-level materializations
      final_eval.json  execution_summary.json   ← (byte-identical copies)
      event_log.jsonl  audit_log.jsonl          ← (ordered concatenations)
      bundles/step_001500/ + LATEST.json        ← E5-loadable event-3 view
      master_manifest.json                      ← deterministic full index
      orchestrator_state.json                   ← PLANNED/RUNNING/FAILED/
                                                  INTERRUPTED/
                                                  READY_FOR_HARNESS_COMPLETION

Fresh AdamW per event is preserved by E5's own control flow: seed bundles
are written with ``intra_stage_index: 0``, so ``mid_stage_resume`` is
False and ``run_workload`` constructs a fresh optimizer (and re-seeds RNG)
per event; no optimizer state is migrated anywhere.

LAZY IMPORTS: module top level is standard-library only. torch /
safetensors / E5 / E1 modules are imported inside the execution callbacks
and torch-touching helpers, so matrix planning and dry-run paths never
load training dependencies. Resource-Aware is explicitly out of scope
(signal unavailable from E5 — see e5_adapter.RUN_WORKLOAD_RANK_LIMITATION).
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

__all__ = [
    "EVENT_SLICE", "NUM_EVENTS", "ORCHESTRATOR_STATES",
    "EventOrchestrator", "OrchestratorError",
    "ResourceAwareUnavailable", "dynamic_event_plan",
    "event_config_hash", "fingerprint_from_tensors",
    "make_dynamic_run_fn", "transition_adapter_factors",
    "validate_bundle_e5", "write_e5_bundle",
]

EVENT_SLICE = 500
NUM_EVENTS = 3
TERMINAL_STEPS = {k: k * EVENT_SLICE for k in (1, 2, 3)}
SEED_STEPS = {k: (k - 1) * EVENT_SLICE for k in (2, 3)}
ORCHESTRATOR_STATES = ("PLANNED", "RUNNING", "FAILED", "INTERRUPTED",
                       "READY_FOR_HARNESS_COMPLETION")
SUPPORTED_DYNAMIC_POLICIES = {"Grow", "Reduce"}
MANIFEST_NAME = "run_manifest.json"
STATE_NAME = "orchestrator_state.json"
INVENTORY_NAME = "artifact_inventory.json"
SELF_EXCLUDED_FILES = {INVENTORY_NAME}  # documented self-reference rule


class OrchestratorError(Exception):
    """Fail-closed orchestrator failure."""


class DynamicPolicyError(OrchestratorError):
    """Row is not a dynamic policy (static rows belong to the E5
    single-invocation adapter)."""


class ResourceAwareUnavailable(OrchestratorError):
    """Resource-Aware requires a causal training-loss signal that the
    existing E5 workflow does not persist; blocked pending the separate
    E5 loss-persistence authorization (never approximated)."""


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp_", suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _canonical_hash(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def transition_seed(experiment_seed: int, from_event: int) -> int:
    """Deterministic transition-generator seed (Finding B).

    Derivation: ``sha256("e7-transition:{experiment_seed}:{from_event}")``
    first 8 bytes as an unsigned 63-bit integer — stable across processes,
    unique per (experiment seed, boundary), and always accepted by
    ``torch.Generator.manual_seed``.
    """
    digest = hashlib.sha256(
        f"e7-transition:{experiment_seed}:{from_event}".encode()).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFF_FFFF_FFFF_FFFF


def dynamic_event_plan(spec) -> dict:
    """Derive the three-event plan from a validated matrix row.

    Grow 4→8→16 (α 8/16/32); Reduce 16→8→4 (α 32/16/8). Ranks/alphas come
    from the row itself (already α=2r-validated by the harness). Static
    rows are redirected to the E5 single-invocation adapter; Resource-
    Aware is blocked (signal unavailable — never approximated).
    """
    from cycleft.e7.e5_adapter import E5AdapterLimitation

    if spec.policy == "Resource-Aware":
        raise ResourceAwareUnavailable(
            f"{spec.exec_id}: Resource-Aware requires event-window training "
            f"losses that run_workload does not persist; blocked pending the "
            f"separate E5 loss-persistence authorization (no approximation, "
            f"no hardcoded ranks).")
    if spec.policy not in SUPPORTED_DYNAMIC_POLICIES:
        raise DynamicPolicyError(
            f"{spec.exec_id}: policy {spec.policy!r} is not a dynamic policy; "
            f"static rows use the single-invocation E5 adapter")
    if spec.row_type != "NEW":
        raise DynamicPolicyError(
            f"{spec.exec_id}: REUSE anchors are read-only references")
    ranks = [int(spec.event_ranks[i]) for i in range(NUM_EVENTS)]
    alphas = [int(spec.event_alphas[i]) for i in range(NUM_EVENTS)]
    for r, a in zip(ranks, alphas):
        if a != 2 * r:
            raise OrchestratorError(
                f"{spec.exec_id}: alpha {a} != 2 * rank {r}")
    events = []
    for k in (1, 2, 3):
        events.append({
            "event_id": k,
            "rank": ranks[k - 1],
            "alpha": alphas[k - 1],
            "global_step_start": (k - 1) * EVENT_SLICE,
            "global_step_end": k * EVENT_SLICE,
            "stream_slice": [(k - 1) * EVENT_SLICE, k * EVENT_SLICE],
            "stage_schedule": [{"stage": 1, "event_id": k,
                                "start": (k - 1) * EVENT_SLICE,
                                "end": k * EVENT_SLICE}],
        })
    return {"exec_id": spec.exec_id, "policy": spec.policy, "seed": spec.seed,
            "depth": spec.depth, "events": events,
            "rank_schedule": ranks, "alpha_schedule": alphas}


def _rss_mb_from_raw(raw: int, platform: str = sys.platform) -> float:
    """Normalize getrusage ru_maxrss to MiB. macOS reports BYTES; Linux
    reports KiB. Documented conversion — tested directly."""
    return raw / (1024 * 1024) if platform == "darwin" else raw / 1024


def _process_peak_rss_mb() -> float:
    """Kernel-maintained peak RSS high-water mark (true maximum, not an
    end-of-process sample). Requires the resource module (POSIX)."""
    import resource
    return round(_rss_mb_from_raw(resource.getrusage(
        resource.RUSAGE_SELF).ru_maxrss), 1)


def _resource_snapshot() -> dict:
    """Point-in-time resource sample. Every unavailable measurement is
    null WITH an explicit reason — never zero (Finding: telemetry
    honesty). Process RSS (host memory) is reported separately from MPS
    (GPU memory) — different address spaces, never merged."""
    snap = {"peak_rss_mb": None, "rss_reason": None,
            "mps_allocated_mb": None, "mps_driver_mb": None,
            "mps_recommended_max_mb": None, "mps_reason": None}
    try:
        snap["peak_rss_mb"] = _process_peak_rss_mb()
    except Exception as e:  # noqa: BLE001
        snap["rss_reason"] = f"unavailable: {type(e).__name__}: {e}"
    try:
        import torch  # lazy
        if not torch.backends.mps.is_available():
            snap["mps_reason"] = "MPS backend not available on this process"
            return snap
        snap["mps_allocated_mb"] = round(
            torch.mps.current_allocated_memory() / 1048576, 1)
        driver = getattr(torch.mps, "driver_allocated_memory", None)
        snap["mps_driver_mb"] = (round(driver() / 1048576, 1)
                                  if callable(driver) else None)
        if snap["mps_driver_mb"] is None:
            snap["mps_reason"] = ("driver_allocated_memory not supported by "
                                  "this torch version")
        snap["mps_recommended_max_mb"] = round(
            torch.mps.recommended_max_memory() / 1048576, 1)
    except Exception as e:  # noqa: BLE001
        snap["mps_reason"] = f"unavailable: {type(e).__name__}: {e}"
    return snap


def event_config_hash(event_ctx: dict) -> str:
    """Deterministic per-event configuration hash (seed-bundle gate value).

    Execution-time note: the default train callback resolves the real
    ``LMRuntimeConfig`` and MUST use E5's ``config_hash(lm_config.to_dict())``
    so ``load_latest_bundle`` accepts the seed; this function defines the
    orchestrator-level canonical hash used for synthetic/mock flows and is
    the documented fallback when the E1 config is unavailable.
    """
    return _canonical_hash(event_ctx)


def fingerprint_from_tensors(tensors: dict) -> str:
    """E5-mirroring fingerprint over a tensor mapping (sorted name + bytes).

    Mirrors ``cycleft.e5.resume_bundle.adapter_fingerprint``'s algorithm at
    the state-dict level so model-free paths (synthetic tests, factor-level
    transitions) produce self-consistent fingerprints. Execution-time
    default callbacks use E5's own ``adapter_fingerprint(model)``.
    """
    import torch  # lazy

    digest = hashlib.sha256()
    for name in sorted(tensors):
        t = tensors[name]
        digest.update(name.encode())
        digest.update(t.detach().to("cpu").contiguous().numpy().tobytes())
    return digest.hexdigest()


def _lora_pairs(state: dict) -> dict[str, tuple[str, str]]:
    """Group safetensors keys into (pair_key, (A_key, B_key)) pairs."""
    a_keys = {k for k in state if "lora_A" in k}
    b_keys = {k for k in state if "lora_B" in k}
    pairs = {}
    for a in a_keys:
        b = a.replace("lora_A", "lora_B")
        if b not in b_keys:
            raise OrchestratorError(f"unpaired LoRA factor: {a}")
        pairs[a.replace(".lora_A", "")] = (a, b)
        b_keys.discard(b)
    if b_keys:
        raise OrchestratorError(f"unpaired LoRA factors: {sorted(b_keys)}")
    if not pairs:
        raise OrchestratorError("no LoRA factors found in adapter")
    return pairs


def transition_adapter_factors(from_adapter_dir: Path, out_adapter_dir: Path,
                               r_to: int, alpha_to: int, *,
                               generator=None) -> dict:
    """Transition every LoRA factor pair to ``r_to`` (real Phase-1 math).

    Reads ``adapter_model.safetensors`` from ``from_adapter_dir``, applies
    ``expand_lora_factors`` (r_to > r_from; zero-preserving) or
    ``reduce_lora_factors`` (r_to < r_from; thin/core SVD) per layer, and
    writes a new adapter directory (safetensors + adapter_config.json with
    r=r_to, alpha=alpha_to). Model-free and deterministic given the
    generator. Returns per-pair reconstruction errors (0.0 for expansion —
    the effective update is preserved exactly).
    """
    import torch  # lazy
    from safetensors.torch import load_file, save_file

    from cycleft.e7.rank_transition import expand_lora_factors, reduce_lora_factors

    from_dir, out_dir = Path(from_adapter_dir), Path(out_adapter_dir)
    state = load_file(str(from_dir / "adapter_model.safetensors"))
    pairs = _lora_pairs(state)
    r_from_set = {tuple(t.shape)[0 if "lora_A" in k else 1]
                  for k, t in state.items()}
    if len(r_from_set) != 1:
        raise OrchestratorError(f"inconsistent source ranks: {r_from_set}")
    r_from = r_from_set.pop()

    new_state, rel_errs = {}, {}
    for pair_key, (a_key, b_key) in sorted(pairs.items()):
        A, B = state[a_key], state[b_key]
        if r_to > r_from:
            A2, B2 = expand_lora_factors(A, B, r_to, generator=generator)
            rel_errs[pair_key] = 0.0  # effective update preserved exactly
        elif r_to < r_from:
            A2, B2, rel = reduce_lora_factors(A, B, r_to)
            rel_errs[pair_key] = rel
        else:
            raise OrchestratorError(
                f"transition requires r_to != r_from ({r_to} == {r_from})")
        if A2.shape[0] != r_to or B2.shape[1] != r_to:
            raise OrchestratorError(
                f"{pair_key}: transitioned shapes {A2.shape}/{B2.shape} "
                f"do not match r_to={r_to}")
        if not (torch.isfinite(A2).all() and torch.isfinite(B2).all()):
            raise OrchestratorError(f"{pair_key}: non-finite transitioned factors")
        new_state[a_key] = A2.contiguous()
        new_state[b_key] = B2.contiguous()
    # carry over any non-LoRA tensors unchanged (none expected in E7 adapters)
    for k, v in state.items():
        if "lora_A" not in k and "lora_B" not in k:
            new_state[k] = v

    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(new_state, str(out_dir / "adapter_model.safetensors"))
    cfg_path = from_dir / "adapter_config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    cfg["r"] = int(r_to)
    cfg["lora_alpha"] = int(alpha_to)
    (out_dir / "adapter_config.json").write_text(
        json.dumps(cfg, indent=2, sort_keys=True), encoding="utf-8")
    return {"r_from": r_from, "r_to": r_to, "pairs": len(pairs),
            "rel_err": rel_errs, "adapter_dir": out_dir}


def write_e5_bundle(bundles_dir: Path, *, step: int, adapter_dir: Path,
                    fingerprint: str, progress: dict, meta: dict,
                    state_payload: dict, torch_version: str = "n/a",
                    device: str = "") -> Path:
    """Write an E5-format checkpoint bundle atomically (no model required).

    Replicates ``cycleft.e5.resume_bundle.save_bundle``'s on-disk format:
    adapter/ (copied verbatim), state.pt (provided payload — for seed
    bundles a throwaway zero-moment optimizer that intra=0 seeding never
    restores), progress.json (format_version, progress, meta,
    adapter_fingerprint, file_hashes excluding progress.json itself),
    LATEST.json (plus E7 provenance fields E5 ignores). Does NOT prune
    (event namespaces hold at most seed+terminal).
    """
    import torch  # lazy

    bundles_dir = Path(bundles_dir)
    bundles_dir.mkdir(parents=True, exist_ok=True)
    final = bundles_dir / f"step_{step:06d}"
    if final.exists():
        raise OrchestratorError(
            f"refusing to write bundle: {final} already exists "
            f"(checkpoints are immutable)")
    tmp = bundles_dir / f"step_{step:06d}.tmp"
    if tmp.exists():
        # Finding C: a pre-existing staging directory is preserved evidence
        # of an earlier interrupted write — fail closed, never delete.
        raise OrchestratorError(
            f"refusing to write bundle: temporary staging {tmp} already "
            f"exists (prior interrupted write preserved; operator review "
            f"required; nothing was modified)")
    tmp.mkdir(parents=True)
    try:
        shutil.copytree(Path(adapter_dir), tmp / "adapter")
        torch.save(state_payload, tmp / "state.pt")
        payload = {
            "format_version": _e5_format_version(),
            "progress": progress,
            "meta": meta,
            "adapter_fingerprint": fingerprint,
            "torch_version": torch.__version__,
            "device": str(meta.get("device", "")),
        }
        (tmp / "progress.json").write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8")
        for f in sorted(tmp.rglob("*")):
            if f.is_file():
                with open(f, "rb") as fh:
                    os.fsync(fh.fileno())
        file_hashes = {}
        for f in sorted(tmp.rglob("*")):
            if f.is_file() and f.name != "progress.json":
                file_hashes[str(f.relative_to(tmp))] = _sha256_file(f)
        payload["file_hashes"] = file_hashes
        (tmp / "progress.json").write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8")
        if final.exists():
            raise OrchestratorError(f"bundle appeared during write: {final}")
        os.rename(tmp, final)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    latest_tmp = bundles_dir / "LATEST.json.tmp"
    latest_tmp.write_text(json.dumps({
        "latest": final.name, "global_step": step,
        "adapter_fingerprint": fingerprint,
        "e7_provenance": {"written_by": "e7-event-orchestrator",
                          "meta_digest": _canonical_hash(meta)},
    }, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(latest_tmp, bundles_dir / "LATEST.json")
    return final


_FORMAT_VERSION_CACHE: int | None = None


def _e5_format_version() -> int:
    """E5 FORMAT_VERSION, imported lazily (single source of truth)."""
    global _FORMAT_VERSION_CACHE
    if _FORMAT_VERSION_CACHE is None:
        from cycleft.e5.resume_bundle import FORMAT_VERSION
        _FORMAT_VERSION_CACHE = int(FORMAT_VERSION)
    return _FORMAT_VERSION_CACHE


def validate_bundle_e5(bundle_dir: Path, expected_config_hash: str, *,
                       expected_global_step: int | None = None) -> dict:
    """Fail-closed E5 bundle validation (format, config hash, file hashes,
    fingerprint presence). Mirrors ``load_latest_bundle``'s acceptance
    gates without loading state.pt."""
    bundle_dir = Path(bundle_dir)
    ppath = bundle_dir / "progress.json"
    if not ppath.is_file():
        raise OrchestratorError(f"bundle {bundle_dir}: progress.json missing")
    try:
        payload = json.loads(ppath.read_text())
    except json.JSONDecodeError as e:
        raise OrchestratorError(f"bundle {bundle_dir}: corrupt progress.json: {e}")
    if payload.get("format_version") != _e5_format_version():
        raise OrchestratorError(
            f"bundle {bundle_dir}: format_version "
            f"{payload.get('format_version')!r} != {_e5_format_version()}")
    progress = payload.get("progress", {})
    if progress.get("config_hash") != expected_config_hash:
        raise OrchestratorError(
            f"bundle {bundle_dir}: config_hash mismatch "
            f"({progress.get('config_hash')!r} != {expected_config_hash!r})")
    for rel, digest in payload.get("file_hashes", {}).items():
        f = bundle_dir / rel
        if not f.is_file() or _sha256_file(f) != digest:
            raise OrchestratorError(
                f"bundle {bundle_dir}: file hash mismatch for {rel}")
    if expected_global_step is not None and \
            progress.get("global_step") != expected_global_step:
        raise OrchestratorError(
            f"bundle {bundle_dir}: global_step {progress.get('global_step')!r} "
            f"!= {expected_global_step}")
    if not payload.get("adapter_fingerprint"):
        raise OrchestratorError(f"bundle {bundle_dir}: adapter_fingerprint missing")
    if not (bundle_dir / "adapter" / "adapter_config.json").is_file():
        raise OrchestratorError(f"bundle {bundle_dir}: adapter_config.json missing")
    return payload


@dataclasses.dataclass
class EventRecord:
    event_id: int
    rank: int
    alpha: int
    global_step_interval: tuple
    dir: str
    input_bundle: dict | None
    output_bundle: dict
    training: dict
    evaluation: dict
    inventory_digest: str
    inventory_self_checksum: str
    resources: dict | None = None

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["global_step_interval"] = list(self.global_step_interval)
        return d


class EventOrchestrator:
    """Three-event dynamic-rank orchestrator (G1-G4, G6-G7).

    Callbacks (all injectable; defaults are lazy E5-backed implementations
    written for the authorized execution path and never invoked in
    design/test tasks):

    - ``train_fn(event_ctx) -> {"exit_code", "status", "terminal_bundle",
      "fingerprint"}`` — one E5 workload invocation for the event.
    - ``sweep_fn(event_ctx, terminal_adapter_dir) -> {"exit_code",
      "status", "metrics", "records", "eval_config"}`` — per-event
      evaluation from that event's own terminal checkpoint.
    - ``transition_fn(from_adapter_dir, r_from, r_to, alpha_to) ->
      {"adapter_dir", "fingerprint", "rel_err"}`` — boundary rank
      transition (default composes the real factor-level
      ``transition_adapter_factors`` with the model-based fingerprint).
    - ``config_hash_fn(event_ctx) -> str`` — the event's E5 config hash.
    """

    def __init__(self, spec, run_dir: Path, *,
                 train_fn: Callable | None = None,
                 sweep_fn: Callable | None = None,
                 transition_fn: Callable | None = None,
                 config_hash_fn: Callable | None = None,
                 repo_root: Path | None = None):
        self.plan = dynamic_event_plan(spec)
        self.spec = spec
        self.run_dir = Path(run_dir)
        self.repo_root = Path(repo_root) if repo_root else Path.cwd()
        self.train_fn = train_fn
        self.sweep_fn = sweep_fn
        self.transition_fn = transition_fn
        self.config_hash_fn = config_hash_fn or _default_config_hash_fn(
            self.repo_root)
        self.event_records: list[EventRecord] = []
        self.transition_records: list[dict] = []
        self._seed_fingerprints: dict[int, str] = {}
        self.state = "PLANNED"

    # ---------------------------------------------------------------- #
    # state machine (orchestrator-level; harness owns COMPLETED)         #
    # ---------------------------------------------------------------- #

    def _write_state(self, state: str, **extra) -> None:
        if state not in ORCHESTRATOR_STATES:
            raise OrchestratorError(f"unknown orchestrator state {state!r}")
        self.state = state
        _atomic_write_json(self.run_dir / STATE_NAME, {
            "exec_id": self.plan["exec_id"], "state": state,
            "note": ("completion authority is the hardened execution harness; "
                     "READY_FOR_HARNESS_COMPLETION is not COMPLETED"),
            **extra,
        })

    def _fail(self, reason: str) -> None:
        self._write_state("FAILED", failure_reason=reason,
                          failed_at=_now())

    # ---------------------------------------------------------------- #
    # per-event helpers                                                   #
    # ---------------------------------------------------------------- #

    def _event_ctx(self, k: int) -> dict:
        ev = self.plan["events"][k - 1]
        return {
            "exec_id": self.plan["exec_id"], "policy": self.plan["policy"],
            "seed": self.plan["seed"], "event_id": k,
            "rank": ev["rank"], "alpha": ev["alpha"],
            "depth": self.plan["depth"], "budget_per_event": EVENT_SLICE,
            "stage_schedule": ev["stage_schedule"],
            "global_step_start": ev["global_step_start"],
            "global_step_end": ev["global_step_end"],
            "stream_slice": ev["stream_slice"],
            "event_dir": str(self.run_dir / "events" / f"event_{k}"),
            "config_hash": self.config_hash_fn(self._cfg_ctx(k)),
        }

    def _cfg_ctx(self, k: int) -> dict:
        ev = self.plan["events"][k - 1]
        return {"exec_id": self.plan["exec_id"], "policy": self.plan["policy"],
                "seed": self.plan["seed"], "event_id": k, "rank": ev["rank"],
                "alpha": ev["alpha"], "depth": self.plan["depth"],
                "budget_per_event": EVENT_SLICE,
                "stage_schedule": ev["stage_schedule"]}

    @staticmethod
    def _require(ok: bool, msg: str) -> None:
        if not ok:
            raise OrchestratorError(msg)

    def _run_event(self, k: int) -> EventRecord:
        ctx = self._event_ctx(k)
        event_dir = Path(ctx["event_dir"])
        ev = self.plan["events"][k - 1]

        resources = {"event_start": _resource_snapshot()}
        t_train0 = time.perf_counter()
        train = self._callback(self.train_fn, "train")(
            ctx) if self.train_fn else _default_train_fn(self.repo_root)(ctx)
        training_duration_s = round(time.perf_counter() - t_train0, 2)
        resources["training_end"] = _resource_snapshot()
        if train.get("config_hash") is not None and \
                train["config_hash"] != ctx["config_hash"]:
            raise OrchestratorError(
                f"event {k}: train callback config_hash "
                f"{train['config_hash']!r} != orchestrator hash "
                f"{ctx['config_hash']!r} (seed-bundle gate would fail)")
        self._require(train.get("exit_code") == 0,
                      f"event {k}: training exit code "
                      f"{train.get('exit_code')!r} != 0")
        self._require(train.get("status") == "complete",
                      f"event {k}: training status {train.get('status')!r} "
                      f"!= 'complete'")
        terminal = Path(train["terminal_bundle"])
        expected_step = k * EVENT_SLICE
        self._require(terminal == event_dir / "bundles" / f"step_{expected_step:06d}",
                      f"event {k}: terminal bundle {terminal} is not the "
                      f"authoritative path .../step_{expected_step:06d}")
        payload = validate_bundle_e5(
            terminal, ctx["config_hash"], expected_global_step=expected_step)
        self._require(payload["progress"]["intra_stage_index"] == EVENT_SLICE,
                      f"event {k}: terminal bundle intra "
                      f"{payload['progress']['intra_stage_index']} != {EVENT_SLICE}")
        adapter_dir = terminal / "adapter"

        t_sweep0 = time.perf_counter()
        sweep = self._callback(self.sweep_fn, "sweep")(
            ctx, adapter_dir) if self.sweep_fn else _default_sweep_fn(
            self.repo_root)(ctx, adapter_dir)
        evaluation_duration_s = round(time.perf_counter() - t_sweep0, 2)
        resources["evaluation_end"] = _resource_snapshot()
        self._require(sweep.get("exit_code") == 0,
                      f"event {k}: evaluation exit code "
                      f"{sweep.get('exit_code')!r} != 0")
        self._require(sweep.get("status") == "complete",
                      f"event {k}: evaluation status {sweep.get('status')!r}")
        metrics = sweep["metrics"]
        self._require(isinstance(metrics, dict) and len(metrics) == 3,
                      f"event {k}: evaluation must cover 3 test sets")
        self._require(set(metrics.keys()) == {"1", "2", "3"},
                      f"event {k}: metric test-set keys must be exactly "
                      f"{{'1','2','3'}}; got {sorted(metrics.keys())}")
        for tid, m in metrics.items():
            self._require(isinstance(m, dict), f"event {k}: test {tid} metrics "
                                               f"must be an object")
            self._require(m.get("examples") == 50,
                          f"event {k}: test {tid} evaluated "
                          f"{m.get('examples')!r} examples != 50")
            for field in ("exact_match", "token_f1"):
                v = m.get(field)
                self._require(isinstance(v, (int, float)) and
                              not isinstance(v, bool),
                              f"event {k}: test {tid} {field} not numeric")
                import math
                self._require(math.isfinite(v),
                              f"event {k}: test {tid} {field} is NaN/Inf")
            for field in ("loss", "ppl"):
                if field in m:
                    self._require(isinstance(m[field], (int, float)) and
                                  math.isfinite(m[field]),
                                  f"event {k}: test {tid} {field} non-finite")
        records = sweep["records"]
        self._require(isinstance(records, list) and len(records) == 150,
                      f"event {k}: expected exactly 150 prediction records, "
                      f"got {len(records) if isinstance(records, list) else type(records)}")
        seen = set()
        per_test = {1: [], 2: [], 3: []}
        required = ("test_event", "example_index", "question", "reference",
                    "raw_prediction", "prediction", "exact_match", "token_f1")
        for rec in records:
            self._require(isinstance(rec, dict), f"event {k}: malformed "
                                                  f"prediction record")
            for key in required:
                self._require(key in rec,
                              f"event {k}: prediction record lacks {key!r}")
            t, idx = rec["test_event"], rec["example_index"]
            self._require(t in (1, 2, 3),
                          f"event {k}: invalid test-set identifier {t!r}")
            self._require(isinstance(idx, int) and 0 <= idx <= 49,
                          f"event {k}: example_index {idx!r} out of range")
            self._require((t, idx) not in seen,
                          f"event {k}: duplicate prediction record "
                          f"(test {t}, example {idx})")
            seen.add((t, idx))
            per_test[t].append(idx)
            self._require(isinstance(rec["exact_match"], (int, float)) and
                          math.isfinite(rec["exact_match"]) and
                          isinstance(rec["token_f1"], (int, float)) and
                          math.isfinite(rec["token_f1"]),
                          f"event {k}: malformed metric values in record "
                          f"(test {t}, example {idx})")
        for t in (1, 2, 3):
            self._require(sorted(per_test[t]) == list(range(50)),
                          f"event {k}: test {t} must contain example indices "
                          f"0–49 exactly once (got {len(per_test[t])} records)")
        _atomic_write_json(event_dir / "final_eval.json", metrics)
        pred_path = event_dir / "predictions.jsonl"
        with open(pred_path, "w", encoding="utf-8") as f:
            for rec in records:
                rec = dict(rec)
                rec.setdefault("state", f"after_event_{k}")
                f.write(json.dumps(rec) + "\n")
        _atomic_write_json(event_dir / "evaluation_config.json", {
            "model_id": (sweep.get("eval_config") or {}).get("model_id"),
            "dtype": (sweep.get("eval_config") or {}).get("dtype"),
            "lora_precision": (sweep.get("eval_config") or {}).get(
                "lora_precision"),
            "max_new_tokens": (sweep.get("eval_config") or {}).get(
                "max_new_tokens"),
            "decoding": (sweep.get("eval_config") or {}).get("decoding"),
            "seed": ctx["seed"],
            "event_id": k, "rank": ev["rank"], "alpha": ev["alpha"],
            "adapter_checkpoint": str(adapter_dir),
            "adapter_fingerprint": payload["adapter_fingerprint"],
            "test_sets": sorted(metrics.keys(), key=lambda x: int(x)),
            "rows_per_test_set": 50,
        })

        inv = self._event_inventory(event_dir)
        _atomic_write_json(event_dir / INVENTORY_NAME, inv)
        inv_self = _sha256_file(event_dir / INVENTORY_NAME)

        # E5-internal timing (model load / ckpt serialize / train categories)
        e5_timing = {}
        try:
            summary = json.loads((event_dir / "execution_summary.json").read_text())
            e5_timing = summary.get("timing", {})
        except (OSError, json.JSONDecodeError):
            e5_timing = {}
        resources["training_duration_s"] = training_duration_s
        resources["evaluation_duration_s"] = evaluation_duration_s
        resources["e5_timing"] = e5_timing
        resources["units"] = {
            "rss": "MiB (getrusage ru_maxrss high-water; darwin bytes, linux KiB)",
            "mps": "MiB (torch.mps allocated / driver / recommended)",
            "durations": "seconds (wall)"}

        return EventRecord(
            event_id=k, rank=ev["rank"], alpha=ev["alpha"],
            global_step_interval=(ev["global_step_start"], ev["global_step_end"]),
            dir=str(event_dir.relative_to(self.run_dir)),
            input_bundle=None if k == 1 else {
                "path": str(seed_bundle_rel(k)),
                "kind": "seed (transitioned)",
                "fingerprint": self._seed_fingerprints[k],
                "config_hash": ctx["config_hash"]},
            output_bundle={
                "path": str(terminal.relative_to(self.run_dir)),
                "kind": "terminal",
                "fingerprint": payload["adapter_fingerprint"],
                "config_hash": ctx["config_hash"]},
            training={"status": train["status"], "exit_code": train["exit_code"],
                      "summary": str(event_dir / "execution_summary.json")},
            evaluation={"status": sweep["status"], "exit_code": sweep["exit_code"],
                        "metrics_ref": "final_eval.json",
                        "predictions_ref": "predictions.jsonl",
                        "config_ref": "evaluation_config.json",
                        "metrics": metrics},
            inventory_digest=_canonical_hash(inv),
            inventory_self_checksum=inv_self,
            resources=resources)

    @staticmethod
    def _callback(fn, name: str) -> Callable:
        if fn is None:
            raise OrchestratorError(f"{name} callback missing")
        return fn

    def _event_inventory(self, event_dir: Path) -> dict:
        """Recursive SHA-256 inventory of the event directory, excluding the
        inventory file itself (documented self-reference rule: the finalized
        inventory's own checksum is recorded in the master manifest)."""
        files = {}
        for f in sorted(Path(event_dir).rglob("*")):
            if f.is_file() and f.name not in SELF_EXCLUDED_FILES:
                files[str(f.relative_to(event_dir))] = _sha256_file(f)
        return {"files": files,
                "digest": hashlib.sha256(
                    "".join(f"{k}:{files[k]}\n" for k in sorted(files)).encode()
                ).hexdigest(),
                "self_excluded": sorted(SELF_EXCLUDED_FILES)}

    # ---------------------------------------------------------------- #
    # transitions (G2/G3)                                                 #
    # ---------------------------------------------------------------- #

    def _transition(self, k: int) -> dict:
        """Event k → k+1: load verified terminal adapter, transition factors
        in an isolated staging area, persist the E5-format seed bundle for
        event k+1 (intra=0), then remove the staging area only after the
        committed seed has been re-validated (Finding A)."""
        import torch  # lazy

        r_from = self.plan["rank_schedule"][k - 1]
        r_to = self.plan["rank_schedule"][k]
        a_to = self.plan["alpha_schedule"][k]
        from_bundle = self.run_dir / "events" / f"event_{k}" / "bundles" / \
            f"step_{k * EVENT_SLICE:06d}"
        from_payload = validate_bundle_e5(
            from_bundle, self._event_ctx(k)["config_hash"],
            expected_global_step=k * EVENT_SLICE)
        from_adapter = from_bundle / "adapter"
        from_fp = from_payload["adapter_fingerprint"]

        # Finding A: staging lives under the run's transitions area, outside
        # every authoritative event checkpoint; refuse if it already exists.
        staging = self.run_dir / "transitions" / ".staging" / \
            f"transition_event_{k}_to_{k + 1}"
        if staging.exists():
            raise OrchestratorError(
                f"staging path already exists: {staging} (fail closed; "
                f"operator review required; nothing was modified)")
        (self.run_dir / "transitions" / ".staging").mkdir(parents=True,
                                                          exist_ok=True)

        # Finding B: deterministic generator derived from the run's actual
        # experiment seed and the transition boundary.
        t_seed = transition_seed(self.plan["seed"], k)
        gen = torch.Generator(device="cpu").manual_seed(t_seed)
        transition_t0 = time.perf_counter()
        resources = {"transition_start": _resource_snapshot()}

        try:
            if self.transition_fn is not None:
                res = self.transition_fn(from_adapter, staging, r_from, r_to,
                                         a_to, gen)
            else:
                res = _default_transition_fn(self.repo_root)(
                    from_adapter, staging, r_from, r_to, a_to, gen)
            self._require(res.get("fingerprint"),
                          "transition returned no fingerprint")
            transitioned = Path(res["adapter_dir"])
            self._require(transitioned.is_dir(),
                          f"transitioned adapter dir missing: {transitioned}")
            self._require_transitioned_rank(transitioned, r_to, a_to)

            seed_bundles = self.run_dir / "events" / f"event_{k + 1}" / "bundles"
            seed_bundles.mkdir(parents=True, exist_ok=True)
            next_ctx = self._event_ctx(k + 1)
            progress = {"global_step": k * EVENT_SLICE,  # event (k+1) starts here
                        "stage_index": 0, "intra_stage_index": 0,
                        "config_hash": next_ctx["config_hash"]}
            meta = {"exec_id": self.plan["exec_id"],
                    "policy": self.plan["policy"], "seed": self.plan["seed"],
                    "event_id": k + 1, "rank": r_to, "alpha": a_to,
                    "e7_transition_from_rank": r_from}
            seed = write_e5_bundle(
                seed_bundles, step=k * EVENT_SLICE, adapter_dir=transitioned,
                fingerprint=res["fingerprint"], progress=progress, meta=meta,
                state_payload=res.get("state_payload", {
                    "optimizer": {},
                    "rng": {"note": "seed bundle: never restored "
                                     "(intra=0 fresh-AdamW gate)"}}))
            # validate the committed seed (rank/alpha/step) BEFORE cleanup
            committed = validate_bundle_e5(
                seed, next_ctx["config_hash"], expected_global_step=k * EVENT_SLICE)
            self._require_transitioned_rank(seed / "adapter", r_to, a_to)
            self._require(
                committed["adapter_fingerprint"] == res["fingerprint"],
                f"committed seed fingerprint mismatch")
        except BaseException:
            # Finding A: on failure, preserve partial staging evidence for
            # operator review; never delete unrelated directories.
            raise
        # Finding A: remove the staging directory only after the seed bundle
        # has been committed and validated.
        shutil.rmtree(staging)

        record = {
            "exec_id": self.plan["exec_id"],
            "from_event": k, "to_event": k + 1,
            "from_rank": r_from, "to_rank": r_to,
            "from_alpha": 2 * r_from, "to_alpha": a_to,
            "from_checkpoint": str(from_bundle.relative_to(self.run_dir)),
            "from_fingerprint": from_fp,
            "resulting_fingerprint": res["fingerprint"],
            "rel_err": res.get("rel_err", {}),
            "transition_seed": t_seed,
            "transition_seed_derivation": ("sha256('e7-transition:' + "
                                           "experiment_seed + ':' + "
                                           "from_event)[:8] as uint63"),
            "policy_version": "v5-static-schedule",
            "timestamp": _now(),
        }
        record["seed_bundle"] = str(seed.relative_to(self.run_dir))
        record["seed_config_hash"] = next_ctx["config_hash"]
        record["transition_duration_s"] = round(
            time.perf_counter() - transition_t0, 2)
        resources["transition_end"] = _resource_snapshot()
        record["resources"] = resources
        self._seed_fingerprints[k + 1] = res["fingerprint"]
        out = self.run_dir / "transitions" / \
            f"transition_event_{k}_to_{k + 1}.json"
        _atomic_write_json(out, record)
        record["record_path"] = str(out.relative_to(self.run_dir))
        return record

    @staticmethod
    def _require_transitioned_rank(adapter_dir: Path, r_to: int,
                                   alpha_to: int) -> None:
        """Verify the resulting adapter has the expected rank and alpha."""
        import torch  # lazy
        from safetensors.torch import load_file

        cfg_path = Path(adapter_dir) / "adapter_config.json"
        cfg = json.loads(cfg_path.read_text())
        if int(cfg.get("r", -1)) != r_to or int(cfg.get("lora_alpha", -1)) != alpha_to:
            raise OrchestratorError(
                f"transitioned adapter config r/alpha "
                f"({cfg.get('r')!r}/{cfg.get('lora_alpha')!r}) != "
                f"({r_to}/{alpha_to})")
        state = load_file(str(Path(adapter_dir) / "adapter_model.safetensors"))
        for name, tensor in state.items():
            expected = (r_to, tensor.shape[1]) if "lora_A" in name \
                else (tensor.shape[0], r_to)
            if tuple(tensor.shape) != expected:
                raise OrchestratorError(
                    f"transitioned tensor {name} shape {tuple(tensor.shape)} "
                    f"!= {expected}")
            if not torch.isfinite(tensor).all():
                raise OrchestratorError(f"transitioned tensor {name} non-finite")

    # ---------------------------------------------------------------- #
    # run-level materializations + master manifest (G6/G7)                #
    # ---------------------------------------------------------------- #

    def _materialize_run_level(self) -> dict:
        run_dir = self.run_dir
        ev3 = run_dir / "events" / "event_3"
        # Finding C: never remove an existing checkpoint tree — fail closed.
        rb = run_dir / "bundles"
        if rb.exists():
            raise OrchestratorError(
                f"run-level bundles directory already exists: {rb} "
                f"(fail closed; existing checkpoint trees are never removed; "
                f"operator review required)")
        # Run-level execution summary: the event-3 E5 summary enriched with
        # the orchestrator's measured sweep EMs (E5 reports final_avg_em=None
        # when final_eval_events=None — the sweep owns evaluation). The
        # event-3 original remains immutable; this is a provenance-noted
        # run-level view.
        summary = json.loads((ev3 / "execution_summary.json").read_text())
        final_metrics = self.event_records[-1].evaluation["metrics"]
        summary["final_avg_em"] = sum(
            m["exact_match"] for m in final_metrics.values()) / len(final_metrics)
        summary["e7_provenance"] = {
            "final_avg_em_source": "events/event_3/final_eval.json "
                                   "(orchestrator sweep, measured)",
            "final_rank_schedule": self.plan["rank_schedule"]}
        _atomic_write_json(run_dir / "execution_summary.json", summary)
        shutil.copyfile(ev3 / "final_eval.json", run_dir / "final_eval.json")
        preds = run_dir / "predictions"
        preds.mkdir(parents=True, exist_ok=True)
        for k in (1, 2, 3):
            shutil.copyfile(run_dir / "events" / f"event_{k}" / "predictions.jsonl",
                            preds / f"after_event_{k}.jsonl")
        for name in ("event_log.jsonl", "audit_log.jsonl"):
            parts = []
            for k in (1, 2, 3):
                parts.append(json.dumps({
                    "event": "e7_event_boundary", "event_index": k,
                    "source": f"events/event_{k}/{name}"}))
                parts.append(
                    (run_dir / "events" / f"event_{k}" / name).read_text().rstrip("\n"))
            (run_dir / name).write_text("\n".join(parts) + "\n", encoding="utf-8")
        # run-level bundles: E5-loadable copy of the event-3 terminal bundle
        rb.mkdir(parents=True)
        src = ev3 / "bundles" / f"step_{3 * EVENT_SLICE:06d}"
        shutil.copytree(src, rb / f"step_{3 * EVENT_SLICE:06d}")
        src_latest = json.loads((ev3 / "bundles" / "LATEST.json").read_text())
        latest = dict(src_latest)
        latest["e7_provenance"] = {
            "source_event_dir": "events/event_3/bundles",
            "byte_identical_to": str(src.relative_to(run_dir)),
            "written_by": "e7-event-orchestrator"}
        (rb / "LATEST.json").write_text(
            json.dumps(latest, indent=2, sort_keys=True), encoding="utf-8")
        self._verify_run_level_copy(src, rb / f"step_{3 * EVENT_SLICE:06d}")
        latest_digest = _sha256_file(rb / "LATEST.json")
        term_digest = _canonical_hash({
            str(f.relative_to(src)): _sha256_file(f) for f in sorted(src.rglob("*"))
            if f.is_file()})
        return {"latest_digest": latest_digest,
                "terminal_bundle_digest": term_digest,
                "terminal_path": f"bundles/step_{3 * EVENT_SLICE:06d}"}

    def _verify_run_level_copy(self, src: Path, view: Path) -> None:
        """Exact run-level checkpoint identity (Finding B): the relative file
        set, directory set, sizes, and SHA-256 checksums of the authoritative
        event-3 terminal checkpoint and the materialized run-level copy must
        match exactly. Symlinks, path escapes, and unexpected filesystem
        types are rejected on BOTH sides; additional, missing, or modified
        entries fail closed. Neither checkpoint is ever modified."""
        import stat as _stat

        def walk(base: Path, side: str) -> tuple[dict, set]:
            base_resolved = base.resolve()
            files, dirs = {}, set()
            for f in sorted(base.rglob("*")):
                st = f.lstat()
                if _stat.S_ISLNK(st.st_mode):
                    raise OrchestratorError(
                        f"run-level identity: symlink rejected on the {side} "
                        f"side at {f} (including links resolving inside the "
                        f"checkpoint)")
                resolved = f.resolve()
                if resolved != base_resolved and \
                        base_resolved not in resolved.parents:
                    raise OrchestratorError(
                        f"run-level identity: path escape on the {side} side "
                        f"at {f} (resolved {resolved})")
                rel = str(f.relative_to(base))
                if f.is_dir():
                    dirs.add(rel)
                elif f.is_file():
                    files[rel] = (_sha256_file(f), f.stat().st_size)
                else:
                    raise OrchestratorError(
                        f"run-level identity: unexpected filesystem type on "
                        f"the {side} side at {f} (mode {st.st_mode:#x})")
            return files, dirs

        src_files, src_dirs = walk(Path(src), "authoritative")
        view_files, view_dirs = walk(Path(view), "run-level")

        missing = sorted(set(src_files) - set(view_files))
        additional = sorted(set(view_files) - set(src_files))
        if missing:
            raise OrchestratorError(
                f"run-level checkpoint identity: missing files "
                f"{missing} (fail closed; evidence preserved)")
        if additional:
            raise OrchestratorError(
                f"run-level checkpoint identity: additional files "
                f"{additional} (fail closed; evidence preserved)")
        missing_dirs = sorted(set(src_dirs) - set(view_dirs))
        additional_dirs = sorted(set(view_dirs) - set(src_dirs))
        if missing_dirs or additional_dirs:
            raise OrchestratorError(
                f"run-level checkpoint identity: directory structure mismatch "
                f"(missing {missing_dirs}, additional {additional_dirs})")
        for rel in sorted(src_files):
            src_digest, src_size = src_files[rel]
            view_digest, view_size = view_files[rel]
            if view_size != src_size:
                raise OrchestratorError(
                    f"run-level checkpoint identity: size mismatch at {rel} "
                    f"({view_size} != {src_size})")
            if view_digest != src_digest:
                raise OrchestratorError(
                    f"run-level checkpoint identity: SHA-256 mismatch at {rel} "
                    f"({view_digest[:12]}… != {src_digest[:12]}…)")

    def _verify_master_manifest(self, run_level: dict, telemetry: dict) -> dict:
        """Re-read the finalized manifest and verify full run integrity:
        structure, per-event inventories (self-checksum, current-file
        agreement, digest), symlink/escape scans, event directory
        references, adjacent transition fingerprints, run-level checkpoint
        byte-identity, and LATEST metadata (final hardening, Finding 1).
        Any mismatch raises — the run then becomes FAILED via
        ``_finalize_run``'s guarded finalization."""
        m = json.loads((self.run_dir / MANIFEST_NAME).read_text())
        self._require(len(m["events"]) == 3, "manifest: expected 3 event records")
        self._require([e["rank"] for e in m["events"]] ==
                      self.plan["rank_schedule"], "manifest: rank schedule mismatch")
        self._require(len(m["transitions"]) == 2, "manifest: expected 2 transitions")
        term3 = self.run_dir / "events" / "event_3" / "bundles" / \
            f"step_{3 * EVENT_SLICE:06d}"
        fp3 = json.loads((term3 / "progress.json").read_text())["adapter_fingerprint"]
        self._require(m["final_adapter_fingerprint"] == fp3,
                      "manifest: final fingerprint mismatch")
        recomputed = _canonical_hash({
            str(f.relative_to(term3)): _sha256_file(f)
            for f in sorted(term3.rglob("*")) if f.is_file()})
        self._require(
            m["final_run_level_checkpoint"]["digest"] == recomputed,
            "manifest: run-level checkpoint digest mismatch")
        self._require(
            telemetry.get("master_manifest_sha256") ==
            _sha256_file(self.run_dir / MANIFEST_NAME),
            "manifest: telemetry-pinned self-digest mismatch")

        # --- per-event verification -------------------------------------
        for ev in m["events"]:
            k = ev["event_id"]
            event_dir = self.run_dir / ev["dir"]
            self._require(event_dir.is_dir(),
                          f"manifest: event-{k} dir {ev['dir']} missing")
            # master-manifest references match actual event directories
            self._require(
                (self.run_dir / ev["output_bundle"]["path"]).is_dir(),
                f"manifest: event-{k} terminal bundle path missing")
            # inventory file exists; self-checksum matches
            inv_path = event_dir / INVENTORY_NAME
            self._require(inv_path.is_file(),
                          f"event {k}: artifact_inventory.json missing")
            inv_self = _sha256_file(inv_path)
            self._require(inv_self == ev["inventory_self_checksum"],
                          f"event {k}: inventory self-checksum mismatch "
                          f"(modified inventory)")
            inv = json.loads(inv_path.read_text())
            # current event files match the recorded inventory (and digest)
            current = {}
            import stat as _stat
            for f in sorted(event_dir.rglob("*")):
                st = f.lstat()
                if _stat.S_ISLNK(st.st_mode):
                    raise OrchestratorError(
                        f"event {k}: symlink in event directory: {f}")
                resolved = f.resolve()
                if resolved != event_dir.resolve() and \
                        event_dir.resolve() not in resolved.parents:
                    raise OrchestratorError(
                        f"event {k}: path escape in event directory: {f}")
                if f.is_dir():
                    continue  # legitimate subdirectories (bundles/, adapter/)
                if not f.is_file():
                    raise OrchestratorError(
                        f"event {k}: unexpected file type in event directory: "
                        f"{f} (mode {st.st_mode:#x})")
                rel = str(f.relative_to(event_dir))
                if rel != INVENTORY_NAME:  # self-excluded file
                    current[rel] = _sha256_file(f)
            self._require(current == inv["files"],
                          f"event {k}: current files diverge from the "
                          f"recorded inventory")
            recomputed_digest = hashlib.sha256(
                "".join(f"{r}:{inv['files'][r]}\n" for r in sorted(inv["files"])).encode()
            ).hexdigest()
            self._require(recomputed_digest == inv["digest"],
                          f"event {k}: inventory digest incorrect")
            # Finding A: linkage — the master manifest's recorded
            # EventRecord.inventory_digest must equal the canonical hash of
            # the actual verified inventory (distinct from the inventory's
            # own `digest` field and from the manifest's self-checksum).
            recorded_digest = ev.get("inventory_digest")
            self._require(
                isinstance(recorded_digest, str) and len(recorded_digest) == 64
                and all(c in "0123456789abcdef" for c in recorded_digest),
                f"event {k}: recorded inventory digest missing or malformed")
            canonical = _canonical_hash(inv)
            self._require(canonical == recorded_digest,
                          f"event {k}: inventory digest linkage mismatch "
                          f"(master manifest {recorded_digest[:12]}… != "
                          f"canonical {canonical[:12]}…); an inventory cannot "
                          f"pass on a valid self-checksum alone")

        # --- transition fingerprint adjacency ----------------------------
        for rec in m["transitions"]:
            k_from = rec["from_event"]
            term_fp = json.loads(
                (self.run_dir / f"events/event_{k_from}/bundles/"
                 f"step_{k_from * EVENT_SLICE:06d}/progress.json").read_text()
            )["adapter_fingerprint"]
            self._require(rec["from_fingerprint"] == term_fp,
                          f"transition {k_from}: source fingerprint does not "
                          f"match the adjacent terminal bundle")
            seed_fp = json.loads(
                (self.run_dir / f"events/event_{k_from + 1}/bundles/"
                 f"step_{k_from * EVENT_SLICE:06d}/progress.json").read_text()
            )["adapter_fingerprint"]
            self._require(rec["resulting_fingerprint"] == seed_fp,
                          f"transition {k_from}: resulting fingerprint does "
                          f"not match the adjacent seed bundle")

        # --- run-level checkpoint + LATEST metadata -----------------------
        view = self.run_dir / "bundles" / f"step_{3 * EVENT_SLICE:06d}"
        self._verify_run_level_copy(term3, view)
        latest = json.loads((self.run_dir / "bundles" / "LATEST.json").read_text())
        self._require(latest.get("latest") == f"step_{3 * EVENT_SLICE:06d}",
                      "run-level LATEST.json references the wrong terminal bundle")
        self._require(int(latest.get("global_step", -1)) == 3 * EVENT_SLICE,
                      "run-level LATEST.json global_step mismatch")
        self._require(latest.get("adapter_fingerprint") == fp3,
                      "run-level LATEST.json fingerprint mismatch")
        return m

    def _build_master_manifest(self, run_level: dict, telemetry: dict,
                               state: str) -> dict:
        manifest = {
            "exec_id": self.plan["exec_id"],
            "policy": self.plan["policy"],
            "seed": self.plan["seed"],
            "matrix_row_ref": "FINAL_RUN_MATRIX.csv",
            "depth": self.plan["depth"],
            "rank_schedule": self.plan["rank_schedule"],
            "alpha_schedule": self.plan["alpha_schedule"],
            "events": [r.to_dict() for r in self.event_records],
            "transitions": self.transition_records,
            "run_level_views": {
                "execution_summary": "execution_summary.json",
                "final_eval": "final_eval.json",
                "predictions": ["predictions/after_event_1.jsonl",
                                 "predictions/after_event_2.jsonl",
                                 "predictions/after_event_3.jsonl"],
                "event_log": "event_log.jsonl",
                "audit_log": "audit_log.jsonl",
                "bundles_latest": "bundles/LATEST.json",
            },
            "final_adapter_fingerprint": self.event_records[-1]
                .output_bundle["fingerprint"],
            "final_run_level_checkpoint": {
                "path": run_level["terminal_path"],
                "digest": run_level["terminal_bundle_digest"],
                "latest_digest": run_level["latest_digest"],
                "byte_identical_to": "events/event_3/bundles/step_001500"},
            "state": state,
            "self_exclusion_rule": {
                "artifact_inventory.json": "excluded from its own digest; "
                "self-checksum recorded in events[].inventory_self_checksum",
                "master_manifest.json": "excluded from all inventories; digest "
                "pinned in harness completion telemetry"},
        }
        return manifest

    def _finalize_master_manifest(self, run_level: dict, telemetry: dict) -> dict:
        """Finding D: build the payload, verify consistency in memory, write
        atomically, re-read and verify the written file, pin its digest in
        telemetry — and only then does run() report READY."""
        payload = self._build_master_manifest(run_level, telemetry,
                                              state="READY_FOR_HARNESS_COMPLETION")
        # in-memory consistency checks (manifest creation prerequisites)
        self._require(len(payload["events"]) == 3, "manifest: expected 3 events")
        self._require([e["rank"] for e in payload["events"]] ==
                      self.plan["rank_schedule"], "manifest: rank schedule mismatch")
        self._require(len(payload["transitions"]) == 2, "manifest: expected 2 transitions")
        term3 = self.run_dir / "events" / "event_3" / "bundles" / \
            f"step_{3 * EVENT_SLICE:06d}"
        fp3 = json.loads((term3 / "progress.json").read_text())["adapter_fingerprint"]
        self._require(payload["final_adapter_fingerprint"] == fp3,
                      "manifest: final fingerprint mismatch")
        recomputed = _canonical_hash({
            str(f.relative_to(term3)): _sha256_file(f)
            for f in sorted(term3.rglob("*")) if f.is_file()})
        self._require(
            payload["final_run_level_checkpoint"]["digest"] == recomputed,
            "manifest: run-level checkpoint digest mismatch")
        # atomic creation
        _atomic_write_json(self.run_dir / MANIFEST_NAME, payload)
        # re-read and verify the written file
        written = json.loads((self.run_dir / MANIFEST_NAME).read_text())
        self._require(written == payload, "manifest write verification failed")
        telemetry["master_manifest_sha256"] = _sha256_file(
            self.run_dir / MANIFEST_NAME)
        return payload

    # ---------------------------------------------------------------- #
    # main entry (G1)                                                     #
    # ---------------------------------------------------------------- #

    def run(self) -> dict:
        """Execute the three-event dynamic run. Returns harness telemetry."""
        self._require(self.plan["policy"] in SUPPORTED_DYNAMIC_POLICIES,
                      "orchestrator supports only Grow and Reduce")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "transitions").mkdir(exist_ok=True)
        self._write_state("RUNNING", started_at=_now())
        run_t0 = time.perf_counter()
        telemetry: dict = {"training": {"events": []},
                           "evaluation": {"events": []}}
        try:
            for k in (1, 2, 3):
                self._write_state("RUNNING", current_event=k)
                try:
                    record = self._run_event(k)
                except KeyboardInterrupt:
                    self._write_state("INTERRUPTED",
                                      interruption_reason=f"event {k} "
                                      f"KeyboardInterrupt", interrupted_at=_now())
                    raise
                except BaseException as e:  # noqa: BLE001
                    self._fail(f"event {k} failed: {type(e).__name__}: {e}")
                    raise OrchestratorError(
                        f"execution stopped: event {k} FAILED "
                        f"({type(e).__name__}: {e}); partial output preserved; "
                        f"no retry, no overwrite") from e
                self.event_records.append(record)
                telemetry["training"]["events"].append(
                    {"event_id": k, "status": record.training["status"],
                     "exit_code": record.training["exit_code"]})
                telemetry["evaluation"]["events"].append(
                    {"event_id": k, "status": record.evaluation["status"],
                     "exit_code": record.evaluation["exit_code"],
                     "metrics": record.evaluation["metrics"]})
                if k < 3:
                    try:
                        self.transition_records.append(self._transition(k))
                    except KeyboardInterrupt:
                        self._write_state("INTERRUPTED",
                                          interruption_reason=f"transition "
                                          f"{k}->{k+1} KeyboardInterrupt",
                                          interrupted_at=_now())
                        raise
                    except BaseException as e:  # noqa: BLE001
                        self._fail(f"transition {k}->{k+1} failed: "
                                   f"{type(e).__name__}: {e}")
                        raise OrchestratorError(
                            f"execution stopped: transition {k}->{k+1} FAILED; "
                            f"partial output preserved; no retry") from e
            self._finalize_run(telemetry)
            telemetry["total_wall_s"] = round(time.perf_counter() - run_t0, 2)
            telemetry["resources_at_exit"] = _resource_snapshot()
            return telemetry
        except OrchestratorError:
            telemetry["training"]["exit_code"] = 1
            telemetry["training"]["status"] = "failed"
            telemetry["orchestrator_state"] = self.state
            raise
        except KeyboardInterrupt:
            telemetry["training"]["exit_code"] = 130
            telemetry["training"]["status"] = "interrupted"
            telemetry["orchestrator_state"] = self.state
            raise

    def _finalize_run(self, telemetry: dict) -> None:
        """Finding D: the entire finalization sequence is error-state
        guarded — READY_FOR_HARNESS_COMPLETION is reported only after
        materialization, manifest creation, and consistency checks all
        succeed; failures mark the run FAILED (KI → INTERRUPTED), preserve
        partial output, and never create completion evidence."""
        try:
            run_level = self._materialize_run_level()
            telemetry["final_rank_schedule"] = self.plan["rank_schedule"]
            self._finalize_master_manifest(run_level, telemetry)
            self._verify_master_manifest(run_level, telemetry)
        except KeyboardInterrupt:
            self._write_state("INTERRUPTED",
                              interruption_reason="finalization "
                              "KeyboardInterrupt", interrupted_at=_now())
            telemetry["training"]["exit_code"] = 130
            telemetry["training"]["status"] = "interrupted"
            telemetry["orchestrator_state"] = self.state
            raise
        except BaseException as e:  # noqa: BLE001
            self._fail(f"finalization failed: {type(e).__name__}: {e}")
            telemetry["training"]["exit_code"] = 1
            telemetry["training"]["status"] = "failed"
            telemetry["orchestrator_state"] = self.state
            raise OrchestratorError(
                f"execution stopped: finalization FAILED "
                f"({type(e).__name__}: {e}); partial output preserved; "
                f"no retry, no overwrite") from e
        self._write_state("READY_FOR_HARNESS_COMPLETION",
                          ready_at=_now())
        telemetry["training"]["exit_code"] = 0
        telemetry["training"]["status"] = "complete"
        telemetry["evaluation"]["exit_code"] = 0
        telemetry["evaluation"]["status"] = "complete"
        telemetry["orchestrator_state"] = self.state


# ---------------------------------------------------------------------------
# default callbacks (lazy E5-backed; written for the authorized execution
# path, never invoked in design/test tasks)
# ---------------------------------------------------------------------------


def _default_config_hash_fn(repo_root: Path) -> Callable:
    """Authoritative per-event config hash: resolves the exact
    ``LMRuntimeConfig`` the E5 invocation will use (E1 pipeline, no model
    load) and applies E5's own ``config_hash`` — guaranteeing the seed
    bundle passes ``load_latest_bundle``'s gate."""
    def hash_fn(ctx: dict) -> str:
        from cycleft.config import load_config
        from cycleft.e1.pipeline import resolve_lm_config
        from cycleft.e5.resume_bundle import config_hash as e5_config_hash

        raw = load_config(Path(repo_root) / "configs" / "e1_stream.yaml")
        lm_config, _, _ = resolve_lm_config(raw, mode="full",
                                            train_examples=None,
                                            seed=ctx["seed"], device=None)
        lm_config.lora_r = ctx["rank"]
        lm_config.lora_alpha = ctx["alpha"]
        lm_config.lora_precision = "float32"
        lm_config.dtype = "float16"
        lm_config.max_new_tokens = 24
        lm_config.seed = ctx["seed"]
        return e5_config_hash(lm_config.to_dict())
    return hash_fn


def _default_train_fn(repo_root: Path) -> Callable:
    def train(ctx: dict) -> dict:
        import torch  # noqa: F401  (lazy: authorized execution path only)

        from cycleft.config import load_config
        from cycleft.e1.data import prepare_event_datasets
        from cycleft.e1.pipeline import resolve_lm_config
        from cycleft.e5.resumable_trainer import run_workload
        from cycleft.e5.resume_bundle import config_hash as e5_config_hash

        raw = load_config(Path(repo_root) / "configs" / "e1_stream.yaml")
        lm_config, _, _ = resolve_lm_config(raw, mode="full",
                                            train_examples=None,
                                            seed=ctx["seed"], device=None)
        lm_config.lora_r = ctx["rank"]
        lm_config.lora_alpha = ctx["alpha"]
        lm_config.lora_precision = "float32"
        lm_config.dtype = "float16"
        lm_config.max_new_tokens = 24
        lm_config.seed = ctx["seed"]
        cfg_hash = e5_config_hash(lm_config.to_dict())  # authoritative gate value

        events, _ = prepare_event_datasets(train_limit=None, seed=ctx["seed"])
        rows = [dict(r) for e in events for r in e.train_rows]
        event_dir = Path(ctx["event_dir"])
        summary = run_workload(
            lm_config=lm_config, stream_rows=rows, out_dir=event_dir,
            exec_id=f"{ctx['exec_id']}-event{ctx['event_id']}", attempt=1,
            policy="P1", ckpt_every=EVENT_SLICE, resume=ctx["event_id"] > 1,
            log=lambda m: print(f"[e7:{ctx['exec_id']}-e{ctx['event_id']}] {m}",
                                flush=True),
            final_eval_events=None,          # evaluation via the post-event sweep
            stage_schedule=ctx["stage_schedule"])
        if summary.get("status") != "complete":
            return {"exit_code": 1, "status": str(summary.get("status"))}
        step = ctx["event_id"] * EVENT_SLICE
        bundle = event_dir / "bundles" / f"step_{step:06d}"
        return {"exit_code": 0, "status": "complete", "terminal_bundle": str(bundle),
                "fingerprint": json.loads(
                    (bundle / "progress.json").read_text())["adapter_fingerprint"],
                "config_hash": cfg_hash}
    return train


def _default_sweep_fn(repo_root: Path) -> Callable:
    """Post-event evaluation sweep. Inference settings are IDENTICAL to the
    training callback's (Finding 3 — configuration parity): the same
    ``LMRuntimeConfig`` fields drive both ``run_workload`` training and
    ``evaluate_event``/``generate_answer`` inference (``generate_answer``
    reads ``config.max_new_tokens``; model dtype and LoRA precision come
    from the same config object — E6 worker pattern, ``scripts/e6_worker.py``
    passes one config to both)."""
    def sweep(ctx: dict, terminal_adapter_dir: Path) -> dict:
        from cycleft.config import load_config
        from cycleft.e1.data import prepare_event_datasets
        from cycleft.e1.lm import (evaluate_event,
                                   load_adapter_for_training, load_base_model,
                                   load_tokenizer)
        from cycleft.e1.pipeline import resolve_lm_config

        raw = load_config(Path(repo_root) / "configs" / "e1_stream.yaml")
        lm_config, _, _ = resolve_lm_config(raw, mode="full",
                                            train_examples=None,
                                            seed=ctx["seed"], device=None)
        # parity with the training callback (identical inference settings):
        lm_config.lora_r = ctx["rank"]
        lm_config.lora_alpha = ctx["alpha"]
        lm_config.lora_precision = "float32"
        lm_config.dtype = "float16"
        lm_config.max_new_tokens = 24
        lm_config.seed = ctx["seed"]
        events, _ = prepare_event_datasets(train_limit=None, seed=ctx["seed"])
        # Resume-equivalent construction (pilot attempt-2 regression): the
        # plain base is passed to load_adapter_for_training, which attaches
        # per the bundle's adapter_config.json and loads the trained weights.
        # Pre-attaching with attach_lora first makes PEFT look for nested
        # keys and SILENTLY skip loading (double-wrap).
        base = load_base_model(lm_config)
        model = load_adapter_for_training(base, terminal_adapter_dir,
                                          lm_config, base.device)
        tokenizer = load_tokenizer(lm_config.model_id)
        metrics, records = {}, []
        for event in events:
            evaluation = evaluate_event(model, tokenizer, event.eval_rows,
                                        lm_config)
            metrics[str(event.event_id)] = evaluation["metrics"]
            for record in evaluation["records"]:
                record = dict(record)
                record["state"] = f"after_event_{ctx['event_id']}"
                # E5's evaluate_event records carry no test-set identifier;
                # the E7 predictions schema requires it (v2.1 contract).
                record["test_event"] = event.event_id
                records.append(record)
        eval_config = {
            "model_id": lm_config.model_id,
            "dtype": lm_config.dtype,
            "lora_precision": lm_config.lora_precision,
            "rank": lm_config.lora_r,
            "alpha": lm_config.lora_alpha,
            "max_new_tokens": lm_config.max_new_tokens,
            "decoding": "greedy",
            "seed": lm_config.seed,
            "test_sets": sorted(metrics.keys(), key=lambda x: int(x)),
            "rows_per_test_set": 50,
        }
        return {"exit_code": 0, "status": "complete", "metrics": metrics,
                "records": records, "eval_config": eval_config}
    return sweep


def _default_transition_fn(repo_root: Path) -> Callable:
    """Model-based transition (execution path): real factor math (with the
    deterministic generator propagated) + E5 fingerprint over the live
    transitioned model + throwaway zero-moment AdamW payload (never
    restored: intra=0). Stages ONLY inside the provided staging directory —
    never inside an authoritative checkpoint (Finding A)."""
    def transition(from_adapter_dir: Path, staging_dir: Path, r_from: int,
                   r_to: int, alpha_to: int, generator=None) -> dict:
        import torch  # lazy

        from cycleft.config import load_config
        from cycleft.e1.lm import load_base_model, load_adapter_for_training
        from cycleft.e1.pipeline import resolve_lm_config
        from cycleft.e5.resume_bundle import (adapter_fingerprint,
                                              capture_rng_states)

        staging = Path(staging_dir)
        staging.mkdir(parents=True, exist_ok=False)  # refuse if exists
        work = staging / "adapter"
        fact = transition_adapter_factors(from_adapter_dir, work, r_to,
                                          alpha_to, generator=generator)
        raw = load_config(Path(repo_root) / "configs" / "e1_stream.yaml")
        lm_config, _, _ = resolve_lm_config(raw, mode="full",
                                            train_examples=None, seed=42,
                                            device=None)
        lm_config.lora_r = r_to
        lm_config.lora_alpha = alpha_to
        # Resume-equivalent construction (pilot attempt-2 regression): the
        # PLAIN base goes to load_adapter_for_training — from_pretrained
        # attaches per the transitioned adapter_config.json and loads the
        # transitioned weights. Pre-attaching double-wraps the model and
        # PEFT silently skips the weight load (nested-key lookup).
        base = load_base_model(lm_config)
        model = load_adapter_for_training(base, work, lm_config, base.device)
        fp = adapter_fingerprint(model)
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=1e-9)
        return {"adapter_dir": str(work), "fingerprint": fp,
                "rel_err": fact["rel_err"],
                "state_payload": {"optimizer": optimizer.state_dict(),
                                   "rng": capture_rng_states()}}
    return transition


def make_dynamic_run_fn(repo_root: Path | None = None, *,
                        train_fn: Callable | None = None,
                        sweep_fn: Callable | None = None,
                        transition_fn: Callable | None = None,
                        config_hash_fn: Callable | None = None) -> Callable:
    """Harness-compatible ``run_fn`` dispatch for the authoritative matrix.

    Dynamic rows (Grow/Reduce) → :class:`EventOrchestrator`; static rows →
    the single-invocation E5 adapter; Resource-Aware → the documented
    limitation (blocked pending the E5 signal-persistence authorization).
    Callbacks default to lazy E5-backed implementations; tests inject
    synthetic mocks.
    """
    root = Path(repo_root) if repo_root else Path.cwd()

    def run_fn(spec, run_dir: Path) -> dict:
        if spec.policy in ("Grow", "Reduce"):
            orch = EventOrchestrator(
                spec, Path(run_dir), repo_root=root,
                train_fn=train_fn, sweep_fn=sweep_fn,
                transition_fn=transition_fn,
                config_hash_fn=config_hash_fn)
            return orch.run()
        if spec.policy == "Resource-Aware":
            raise ResourceAwareUnavailable(
                f"{spec.exec_id}: {RUN_WORKLOAD_RANK_LIMITATION}")
        from cycleft.e7.e5_adapter import make_run_fn
        return make_run_fn(root)(spec, run_dir)

    return run_fn


RUN_WORKLOAD_RANK_LIMITATION = (
    "Resource-Aware requires event-window training losses that the existing "
    "E5 workflow does not persist; blocked pending the separate E5 "
    "loss-persistence authorization (no approximation, no hardcoded ranks).")


# convenience paths ----------------------------------------------------------

def seed_bundle_rel(k: int) -> str:
    return f"events/event_{k}/bundles/step_{SEED_STEPS[k]:06d}"


def terminal_bundle_rel(k: int) -> str:
    return f"events/event_{k}/bundles/step_{TERMINAL_STEPS[k]:06d}"


def seed_bundle_path(run_dir: Path, k: int) -> Path:
    return Path(run_dir) / seed_bundle_rel(k)


def terminal_bundle_path(run_dir: Path, k: int) -> Path:
    return Path(run_dir) / terminal_bundle_rel(k)
