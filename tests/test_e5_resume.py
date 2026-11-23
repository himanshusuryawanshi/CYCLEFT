"""Tests for E5 ResumeBundle: state round-trip, atomic commit, RNG restore."""

import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from cycleft.e5.resume_bundle import (
    adapter_fingerprint,
    capture_rng_states,
    config_hash,
    load_latest_bundle,
    restore_rng_states,
    save_bundle,
)


class TinyPeft:
    """Minimal PEFT-compatible model (save_pretrained + named_parameters)."""

    def __init__(self):
        self.base = torch.nn.Linear(8, 8)
        self.lora_a = torch.nn.Parameter(torch.randn(4, 8) * 0.01)
        self.lora_b = torch.nn.Parameter(torch.zeros(2, 4))

    def named_parameters(self):
        return [
            ("base.weight", self.base.weight),
            ("base.bias", self.base.bias),
            ("lora_a", self.lora_a),
            ("lora_b", self.lora_b),
        ]

    def parameters(self):
        return [p for _, p in self.named_parameters()]

    def save_pretrained(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(
            {name: p.detach().clone() for name, p in self.named_parameters()},
            path / "adapter_model.bin",
        )

    def load_fingerprint_matches(self, fingerprint):
        return adapter_fingerprint(self) == fingerprint


from pathlib import Path


@pytest.fixture
def tiny_run(tmp_path):
    model = TinyPeft()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    # consume some optimizer state so the moments are non-trivial
    batch = torch.randn(4, 8)
    loss = (model.base(batch) ** 2).mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    progress = {"global_step": 62, "stage_index": 0, "intra_stage_index": 62,
                "config_hash": config_hash({"a": 1})}
    meta = {"device": "cpu", "lm_config": {"a": 1}}
    return model, optimizer, progress, meta, tmp_path


def test_bundle_round_trip_state_exact(tiny_run):
    model, optimizer, progress, meta, tmp = tiny_run
    root = tmp / "bundles"
    save_bundle(root, model, optimizer, progress, meta)
    bundle = load_latest_bundle(root, expected_config_hash=progress["config_hash"])
    assert bundle["progress"]["global_step"] == 62
    state_before = optimizer.state_dict()
    optimizer.load_state_dict(bundle["optimizer_state"])
    state_after = optimizer.state_dict()
    for key in state_before["state"]:
        for field in ("exp_avg", "exp_avg_sq", "step"):
            assert torch.equal(state_before["state"][key][field],
                               state_after["state"][key][field]), field
    assert state_before["param_groups"] == state_after["param_groups"]


def test_bundle_round_trip_adapter_fingerprint(tiny_run):
    model, optimizer, progress, meta, tmp = tiny_run
    root = tmp / "bundles"
    fingerprint = adapter_fingerprint(model)
    save_bundle(root, model, optimizer, progress, meta)
    bundle = load_latest_bundle(root, expected_config_hash=progress["config_hash"])
    assert bundle["fingerprint"] == fingerprint


def test_rng_states_round_trip_exact():
    torch.manual_seed(7); np.random.seed(7); random.seed(7)
    captured = capture_rng_states()
    # consume randomness after capture
    _ = torch.rand(16); _ = np.random.rand(16); _ = random.random()
    restore_rng_states(captured)
    a = (torch.rand(16), np.random.rand(16), random.random())
    # reproduce the same draws from the captured state again
    restore_rng_states(captured)
    b = (torch.rand(16), np.random.rand(16), random.random())
    assert torch.equal(a[0], b[0])
    assert np.array_equal(a[1], b[1])
    assert a[2] == b[2]


import random


def test_atomic_commit_falls_back_to_previous_bundle(tiny_run):
    model, optimizer, progress, meta, tmp = tiny_run
    root = tmp / "bundles"
    progress_50 = dict(progress, global_step=50, intra_stage_index=50)
    save_bundle(root, model, optimizer, progress_50, meta)
    # corrupt the latest bundle and repoint LATEST at it
    latest = json.loads((root / "LATEST.json").read_text())
    (root / latest["latest"] / "state.pt").unlink()
    with pytest.raises(RuntimeError, match="no valid checkpoint bundle"):
        load_latest_bundle(root, expected_config_hash=progress["config_hash"])
    # a second, older valid bundle restores instead
    progress_25 = dict(progress, global_step=25, intra_stage_index=25)
    save_bundle(root, model, optimizer, progress_25, meta)  # commit AFTER corrupting
    bundle = load_latest_bundle(root, expected_config_hash=progress["config_hash"])
    assert bundle["progress"]["global_step"] == 25


def test_missing_bundle_fails_loudly(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_latest_bundle(tmp_path / "bundles", expected_config_hash="x")


def test_config_hash_mismatch_rejected(tiny_run):
    model, optimizer, progress, meta, tmp = tiny_run
    root = tmp / "bundles"
    save_bundle(root, model, optimizer, progress, meta)
    with pytest.raises(RuntimeError, match="no valid checkpoint bundle"):
        load_latest_bundle(root, expected_config_hash="different")


def test_torn_tmp_directory_is_ignored(tiny_run):
    model, optimizer, progress, meta, tmp = tiny_run
    root = tmp / "bundles"
    save_bundle(root, model, optimizer, progress, meta)
    torn = root / "step_999999.tmp"
    torn.mkdir()
    (torn / "junk").write_text("partial")
    bundle = load_latest_bundle(root, expected_config_hash=progress["config_hash"])
    assert bundle["progress"]["global_step"] == 62
    assert not (root / "step_999999").exists()
