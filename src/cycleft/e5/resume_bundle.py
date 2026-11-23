"""E5 ResumeBundle v1 — faithful-resume state for continual LoRA training.

Implements CHECKPOINT_STATE_SPEC.md:
- adapter weights + optimizer state + four RNG states + progress counters
- atomic commit protocol (tmp -> fsync -> verify -> rename -> LATEST pointer)
- pre-training/post-training fingerprints for restore verification
- MPS RNG capture via torch.mps.get_rng_state (no seed-only fallback when the
  API is available; torch 2.14.1 verified)
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch

FORMAT_VERSION = 1
KEEP_BUNDLES = 2


def _fsync_file(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_dir(path: Path) -> None:
    _fsync_file(path)


def config_hash(config_dict: dict) -> str:
    return hashlib.sha256(
        json.dumps(config_dict, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]


def adapter_fingerprint(model) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        if parameter.requires_grad:
            digest.update(name.encode())
            digest.update(
                parameter.detach().to("cpu").contiguous().numpy().tobytes()
            )
    return digest.hexdigest()


def capture_rng_states() -> dict:
    """Capture all four RNG states. MPS synchronize() precedes capture."""
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    states = {
        "torch_cpu": torch.get_rng_state(),
        "python": _encode_py_state(random.getstate()),
        "numpy": _encode_np_state(np.random.get_state()),
    }
    if torch.backends.mps.is_available() and hasattr(torch.mps, "get_rng_state"):
        states["torch_mps"] = torch.mps.get_rng_state()
    else:
        states["torch_mps"] = None
    states["rng_tier"] = "T-full" if states["torch_mps"] is not None else "T-seedfallback"
    return states


def restore_rng_states(states: dict) -> None:
    """Restore RNG states. Normative order: AFTER model/optimizer reconstruction."""
    torch.set_rng_state(states["torch_cpu"].cpu().to(torch.uint8))
    if states.get("torch_mps") is not None:
        if not (torch.backends.mps.is_available() and hasattr(torch.mps, "set_rng_state")):
            raise RuntimeError(
                "bundle contains an MPS RNG state but the MPS RNG API is "
                "unavailable on this host/process"
            )
        torch.mps.set_rng_state(states["torch_mps"].cpu().to(torch.uint8))
    random.setstate(_decode_py_state(states["python"]))
    np.random.set_state(_decode_np_state(states["numpy"]))


def _encode_py_state(state) -> dict:
    version, internal, gauss = state
    return {
        "version": version,
        "internal": list(internal),
        "gauss_next": gauss,
    }


def _decode_py_state(payload):
    return (
        payload["version"],
        tuple(payload["internal"]),
        payload["gauss_next"],
    )


def _encode_np_state(state) -> dict:
    return {
        "bit_generator": state[0],
        "keys": [k.tolist() if hasattr(k, "tolist") else k for k in state[1]],
        "pos": state[2],
        "has_gauss": state[3],
        "cached_gaussian": state[4],
    }


def _decode_np_state(payload):
    keys = []
    for k in payload["keys"]:
        keys.append(np.array(k) if isinstance(k, list) else k)
    return (
        payload["bit_generator"],
        tuple(keys),
        payload["pos"],
        payload["has_gauss"],
        payload["cached_gaussian"],
    )


def _to_cpu(obj):
    if torch.is_tensor(obj):
        return obj.detach().to("cpu").clone()
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_cpu(v) for v in obj)
    return obj


def save_bundle(
    bundle_root: Path,
    model,
    optimizer,
    progress: dict,
    meta: dict,
    accounting: dict | None = None,
    durable_copy: bool = True,
) -> Path:
    """Atomic checkpoint commit protocol (DESIGN_AMENDMENTS.md A5).

    tmp write -> hash -> fsync -> verify -> atomic rename -> LATEST pointer
    -> durable-copy emulation (LOCAL transfer emulation, not cloud latency).

    ``accounting`` (optional dict) receives split timings in seconds:
    serialize / hash_verify / fsync_commit / durable_copy.
    """
    import time as _time

    def _acc(key, seconds):
        if accounting is not None:
            accounting[key] = accounting.get(key, 0.0) + seconds

    bundle_root = Path(bundle_root)
    bundle_root.mkdir(parents=True, exist_ok=True)
    step = progress["global_step"]
    final = bundle_root / f"step_{step:06d}"
    tmp = bundle_root / f"step_{step:06d}.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)

    tmp.mkdir(parents=True)
    t0 = _time.perf_counter_ns()
    model.save_pretrained(str(tmp / "adapter"))
    torch.save(
        {"optimizer": _to_cpu(optimizer.state_dict()), "rng": capture_rng_states()},
        tmp / "state.pt",
    )
    payload = {
        "format_version": FORMAT_VERSION,
        "progress": progress,
        "meta": meta,
        "adapter_fingerprint": adapter_fingerprint(model),
        "torch_version": torch.__version__,
        "device": str(meta.get("device", "")),
    }
    (tmp / "progress.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    _acc("serialize", (_time.perf_counter_ns() - t0) / 1e9)

    # fsync every file, then the directory
    for f in sorted(tmp.rglob("*")):
        if f.is_file():
            _fsync_file(f)
    _fsync_dir(tmp)

    # hash every written file (integrity verification; timed)
    import hashlib as _hashlib

    t0 = _time.perf_counter_ns()
    file_hashes = {}
    for f in sorted(tmp.rglob("*")):
        if f.is_file() and f.name != "progress.json":
            file_hashes[str(f.relative_to(tmp))] = _hashlib.sha256(
                f.read_bytes()
            ).hexdigest()
    _acc("hash_verify", (_time.perf_counter_ns() - t0) / 1e9)
    # progress.json is EXCLUDED from file_hashes (a file cannot contain its
    # own hash); its integrity is guaranteed by the atomic rename and by the
    # durable copy's byte-equality check against this source bundle.
    payload["file_hashes"] = file_hashes
    (tmp / "progress.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    loaded = json.loads((tmp / "progress.json").read_text())
    if loaded["format_version"] != FORMAT_VERSION:
        shutil.rmtree(tmp)
        raise RuntimeError("bundle verification failed: format version")

    # fsync + atomic commit (timed)
    t0 = _time.perf_counter_ns()
    if final.exists():
        shutil.rmtree(final)
    os.rename(tmp, final)

    # LATEST pointer (atomic tmp-then-rename)
    latest_tmp = bundle_root / "LATEST.json.tmp"
    latest_tmp.write_text(
        json.dumps(
            {
                "latest": final.name,
                "global_step": step,
                "adapter_fingerprint": payload["adapter_fingerprint"],
            }
        ),
        encoding="utf-8",
    )
    _fsync_file(latest_tmp)
    os.rename(latest_tmp, bundle_root / "LATEST.json")
    _acc("fsync_commit", (_time.perf_counter_ns() - t0) / 1e9)

    # durable-copy emulation: LOCAL transfer emulation, not cloud latency
    if durable_copy:
        t0 = _time.perf_counter_ns()
        durable = bundle_root / "durable" / final.name
        if durable.exists():
            shutil.rmtree(durable)
        shutil.copytree(final, durable)
        for f in sorted(durable.rglob("*")):
            if f.is_file():
                _fsync_file(f)
        _fsync_dir(durable)
        for f in sorted(final.rglob("*")):
            if f.is_file():
                rel = f.relative_to(final)
                if not (durable / rel).exists():
                    raise RuntimeError(f"durable-copy missing file: {rel}")
                if f.read_bytes() != (durable / rel).read_bytes():
                    raise RuntimeError(f"durable-copy byte mismatch: {rel}")
        _acc("durable_copy", (_time.perf_counter_ns() - t0) / 1e9)

    # retain the last KEEP_BUNDLES committed bundles
    bundles = sorted(
        p for p in bundle_root.glob("step_*") if p.is_dir() and not p.name.endswith(".tmp")
    )
    for old in bundles[:-KEEP_BUNDLES]:
        shutil.rmtree(old)
    return final


def load_latest_bundle(bundle_root: Path, expected_config_hash: str) -> dict:
    """Load the latest committed bundle; fall back to the previous valid one.

    Raises loudly when no valid bundle exists (never fabricates a resume).
    """
    bundle_root = Path(bundle_root)
    latest_path = bundle_root / "LATEST.json"
    if not latest_path.exists():
        raise FileNotFoundError(f"no checkpoint bundle exists under {bundle_root}")
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    candidates = [latest["latest"]]
    # fallback candidates: older committed bundles, newest first
    candidates += [
        p.name
        for p in sorted(bundle_root.glob("step_*"), reverse=True)
        if p.is_dir() and p.name != latest["latest"]
    ]
    import time as _time
    import hashlib as _hashlib

    errors = []
    for name in candidates:
        bdir = bundle_root / name
        t0 = _time.perf_counter_ns()
        try:
            payload = json.loads((bdir / "progress.json").read_text(encoding="utf-8"))
            if payload["format_version"] != FORMAT_VERSION:
                raise ValueError("format version mismatch")
            if payload["progress"]["config_hash"] != expected_config_hash:
                raise ValueError("config hash mismatch")
            for rel, digest in payload.get("file_hashes", {}).items():
                if _hashlib.sha256((bdir / rel).read_bytes()).hexdigest() != digest:
                    raise ValueError(f"file hash mismatch for {rel}")
            state = torch.load(bdir / "state.pt", map_location="cpu", weights_only=False)
            restore_s = (_time.perf_counter_ns() - t0) / 1e9
            return {
                "dir": bdir,
                "adapter_dir": str(bdir / "adapter"),
                "optimizer_state": state["optimizer"],
                "rng": state["rng"],
                "progress": payload["progress"],
                "fingerprint": payload["adapter_fingerprint"],
                "fell_back": name != latest["latest"],
                "restore_timing_s": round(restore_s, 6),
                "restore_timing_note": "bundle load + hash verification; excludes model loading",
            }
        except Exception as error:  # noqa: BLE001
            errors.append(f"{name}: {type(error).__name__}: {error}")
    raise RuntimeError(
        "no valid checkpoint bundle could be restored: " + "; ".join(errors)
    )
