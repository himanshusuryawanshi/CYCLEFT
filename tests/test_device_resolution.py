"""Device-resolution and GPU-metadata tests (Phase 1).

CUDA behavior is tested with mocks (no GPU on the development MacBook);
real CPU and (when available) MPS placement tests run unmocked.  Mocked
CUDA tests verify *resolution and metadata plumbing only* — actual CUDA
execution is validated exclusively on the HPC GPU (see
hpc/HPC_WIKIFACTDIFF.md) and is never claimed from these tests.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cycleft.e1.lm import (
    gpu_metadata,
    memory_status,
    release_device_memory,
    resolve_device,
)


# ---------------------------------------------------------------------------
# resolve_device
# ---------------------------------------------------------------------------


def test_cpu_requested_always_resolves():
    assert resolve_device("cpu") == torch.device("cpu")


def test_auto_prefers_cuda_then_mps_then_cpu(monkeypatch):
    import cycleft.e1.lm as lm

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert lm.resolve_device(None).type == "cuda"
    assert lm.resolve_device("auto").type == "cuda"

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert lm.resolve_device(None).type == "mps"

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert lm.resolve_device(None).type == "cpu"  # automatic last resort


def test_cuda_requested_unavailable_fails_hard(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    with pytest.raises(RuntimeError, match="CUDA was requested"):
        resolve_device("cuda")


def test_cuda_requested_available_resolves(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    device = resolve_device("cuda")
    assert device.type == "cuda"


def test_mps_requested_unavailable_fails_hard(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="MPS is not available"):
        resolve_device("mps")


def test_mps_requested_available_resolves_when_mps_present():
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available on this machine")
    assert resolve_device("mps").type == "mps"


def test_unknown_device_rejected():
    with pytest.raises(ValueError, match="Unsupported device"):
        resolve_device("tpu")


# ---------------------------------------------------------------------------
# model placement (real devices only; mocked CUDA cannot execute)
# ---------------------------------------------------------------------------


def _tiny_model():
    return torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 2))


def test_model_placement_on_cpu():
    device = resolve_device("cpu")
    model = _tiny_model().to(device)
    assert all(p.device.type == "cpu" for p in model.parameters())


@pytest.mark.skipif(not torch.backends.mps.is_available(),
                    reason="MPS not available")
def test_model_placement_on_mps():
    device = resolve_device("mps")
    model = _tiny_model().to(device)
    assert all(p.device.type == "mps" for p in model.parameters())
    # a forward pass must actually execute on the accelerator
    out = model(torch.ones(2, 4).to(device))
    assert out.device.type == "mps"
    release_device_memory(device)


@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="real CUDA GPU required (HPC validation)")
def test_model_placement_on_real_cuda():
    """REAL GPU validation — never runs on the MacBook (no CUDA there)."""
    device = resolve_device("cuda")
    model = _tiny_model().to(device)
    assert all(p.device.type == "cuda" for p in model.parameters())
    out = model(torch.ones(2, 4).to(device))
    assert out.device.type == "cuda"
    release_device_memory(device)


# ---------------------------------------------------------------------------
# GPU metadata (mocked CUDA: plumbing only)
# ---------------------------------------------------------------------------


def test_gpu_metadata_cuda_absent_is_all_none():
    monkey_cuda_absent = (False, None)
    import cycleft.e1.lm as lm
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: monkey_cuda_absent[0])
    monkeypatch.setattr(torch, "version", type("V", (), {"cuda": monkey_cuda_absent[1]}))
    info = gpu_metadata()
    assert info["cuda_available"] is False
    assert info["device_name"] is None and info["device_count"] == 0
    assert info["torch_version"] == torch.__version__
    monkeypatch.undo()


def test_gpu_metadata_with_mocked_cuda(monkeypatch):
    """MOCKED CUDA — verifies metadata plumbing, NOT real-GPU execution."""
    import cycleft.e1.lm as lm

    class FakeProps:
        name = "Mocked A100"
        major, minor = 8, 0
        total_memory = 80 * 1024**3

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i: FakeProps)
    monkeypatch.setattr(torch.cuda, "mem_get_info",
                        lambda i: (40 * 1024**3, 80 * 1024**3))
    info = gpu_metadata()
    assert info["cuda_available"] is True
    assert info["device_name"] == "Mocked A100"
    assert info["device_capability"] == "8.0"
    assert info["total_memory_bytes"] == 80 * 1024**3
    assert info["free_memory_bytes"] == 40 * 1024**3
    assert "mocked" in info["device_name"].lower()  # explicit marker: not a real run


def test_memory_status_reports_cuda_when_available(monkeypatch):
    import cycleft.e1.lm as lm

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 2 * 1024**3)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: 4 * 1024**3)
    monkeypatch.setattr(torch.cuda, "mem_get_info",
                        lambda: (60 * 1024**3, 80 * 1024**3))
    status = memory_status()
    assert status["cuda_allocated_gb"] == 2 * 1024**3 / 1e9
    assert status["cuda_reserved_gb"] == 4 * 1024**3 / 1e9


def test_release_device_memory_never_raises(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    release_device_memory(None)  # no-op without accelerators
    release_device_memory(torch.device("cpu"))
