
"""Checkpoint persistence for CYCLEFT experiments."""

import json
from pathlib import Path

from cycleft.finetuning import MemoryQAAdapter


def save_checkpoint(
    adapter: MemoryQAAdapter,
    path: str | Path,
    metadata: dict | None = None,
) -> Path:
    """Save adapter state and metadata as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "format_version": 1,
        "adapter_type": "MemoryQAAdapter",
        "state": adapter.state_dict(),
        "metadata": metadata or {},
    }

    path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    return path


def load_checkpoint(path: str | Path) -> tuple[MemoryQAAdapter, dict]:
    """Load an adapter checkpoint and its metadata."""
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    if payload.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format.")

    if payload.get("adapter_type") != "MemoryQAAdapter":
        raise ValueError("Unsupported checkpoint adapter type.")

    adapter = MemoryQAAdapter()
    adapter.load_state_dict(payload["state"])

    return adapter, payload.get("metadata", {})
