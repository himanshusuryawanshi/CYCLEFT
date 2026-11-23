"""E4 hardware inventory collection.

Collects only factual, non-secret environment information: CPU/chip
identity, memory, accelerator availability (MPS/CUDA), framework versions,
and the PRESENCE of cloud tooling/credentials (never their contents).
"""

from __future__ import annotations

import json
import platform
import re
import shutil
import subprocess
from importlib import metadata
from pathlib import Path
from typing import Any


def _safe_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _chip_identity() -> str | None:
    try:
        output = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
        return output or None
    except (OSError, subprocess.SubprocessError):
        return None


def _memory_gb() -> float | None:
    try:
        output = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
        return round(int(output) / 1024**3, 2)
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def _cpu_cores() -> int | None:
    try:
        output = subprocess.run(
            ["sysctl", "-n", "hw.ncpu"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
        return int(output)
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def _cloud_cli_status() -> dict[str, Any]:
    """Presence and authentication status only — never credential values."""
    status: dict[str, Any] = {}
    for cli in ("aws", "gcloud", "az"):
        path = shutil.which(cli)
        entry: dict[str, Any] = {"installed": path is not None, "path": path}
        status[cli] = entry
    if status["gcloud"]["installed"]:
        entry = dict(status["gcloud"])
        try:
            result = subprocess.run(
                ["gcloud", "auth", "list", "--filter=status:ACTIVE",
                 "--format=value(account)"],
                capture_output=True, text=True, timeout=30,
            )
            accounts = [line for line in result.stdout.splitlines() if line.strip()]
            # Redact to the account domain: identity presence, not the address.
            entry["active_account_count"] = len(accounts)
            entry["active_account_domains"] = sorted(
                {account.split("@")[-1] for account in accounts if "@" in account}
            )
        except (OSError, subprocess.SubprocessError):
            entry["active_account_count"] = None
        status["gcloud"] = entry
    if status["az"]["installed"]:
        entry = dict(status["az"])
        try:
            result = subprocess.run(
                ["az", "account", "show", "--query", "name", "-o", "tsv"],
                capture_output=True, text=True, timeout=60,
            )
            authenticated = result.returncode == 0 and bool(result.stdout.strip())
            entry["authenticated"] = authenticated
            if authenticated:
                # Subscription display name is configuration, not a secret.
                entry["subscription_name"] = result.stdout.strip().splitlines()[0]
            else:
                entry["subscription_name"] = None
                stderr = " ".join(result.stderr.split())[:200]
                entry["auth_error_summary"] = re.sub(
                    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+", "<redacted-email>", stderr
                )
        except (OSError, subprocess.SubprocessError):
            entry["authenticated"] = None
        status["az"] = entry
    return status


def collect_hardware_inventory() -> dict[str, Any]:
    """Measured/verified environment facts for hardware_inventory.json."""
    import torch

    cpu_brand = _chip_identity() or platform.processor() or "unknown"
    is_arm = platform.machine() in ("arm64", "aarch64")
    device_name = None
    if torch.backends.mps.is_available():
        # torch does not expose the MPS chip name; report the platform chip.
        device_name = cpu_brand
    inventory = {
        "host": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_model": cpu_brand,
            "cpu_architecture_arm": is_arm,
            "cpu_cores": _cpu_cores(),
            "ram_gb": _memory_gb(),
        },
        "accelerators": {
            "mps_available": torch.backends.mps.is_available(),
            "mps_built": torch.backends.mps.is_built(),
            "mps_device_name": device_name,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": (
                torch.cuda.device_count() if torch.cuda.is_available() else 0
            ),
            "cuda_visible_devices": None,  # never read GPU env in detail locally
            "gpu_memory_exposed": False,
            "gpu_memory_note": (
                "MPS unified memory is not queried per-device; the OS reports "
                "unified RAM. CUDA GPUs would report capacity via torch.cuda."
            ),
        },
        "framework": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": _safe_version("transformers"),
            "peft": _safe_version("peft"),
            "accelerate": _safe_version("accelerate"),
            "datasets": _safe_version("datasets"),
        },
        "cloud_cli": _cloud_cli_status(),
        "notes": [
            "No credential values, tokens, or private keys are collected; "
            "only installation paths, account counts and display names.",
            "The machine identity reported here supersedes earlier E1-E3 "
            "reports that described the host as an 'M3' — system_profiler "
            "and sysctl identify this host as an Apple M1 MacBook Air.",
        ],
    }
    return inventory


def write_inventory(path: Path) -> dict[str, Any]:
    inventory = collect_hardware_inventory()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    return inventory
