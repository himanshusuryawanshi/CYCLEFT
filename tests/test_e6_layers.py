"""E6 tests: deterministic layer selection, regex resolution, parameter counts."""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cycleft.e6.layer_selection import (
    ALL_CONFIGS,
    E6_A_CONFIGS,
    E6_B_CONFIGS,
    NUM_LAYERS,
    depth_indices,
    early_indices,
    expected_trainable_params,
    late_indices,
    middle_indices,
    uniform_indices,
)


class TinyLlama(nn.Module):
    """16-layer dummy mirroring the module paths of the real model."""

    def __init__(self, L: int = 16):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList()
        for _ in range(L):
            layer = nn.Module()
            layer.self_attn = nn.Module()
            layer.self_attn.q_proj = nn.Linear(8, 8, bias=False)
            layer.self_attn.k_proj = nn.Linear(8, 4, bias=False)
            layer.self_attn.v_proj = nn.Linear(8, 4, bias=False)
            layer.self_attn.o_proj = nn.Linear(8, 8, bias=False)
            self.model.layers.append(layer)


def resolved_layers_and_params(indices, L: int = 16, rank: int = 8):
    from peft import get_peft_model, LoraConfig

    model = TinyLlama(L)
    pm = get_peft_model(
        model,
        LoraConfig(r=rank, lora_alpha=2 * rank,
                   target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                   layers_to_transform=indices, bias="none"),
    )
    named = [n for n, p in pm.named_parameters() if p.requires_grad]
    layers = sorted({int(n.split("layers.")[1].split(".")[0]) for n in named if "layers." in n})
    params = sum(p.numel() for p in pm.parameters() if p.requires_grad)
    return layers, len(named), params


def test_depth_indices_match_spec():
    assert depth_indices(25) == [12, 13, 14, 15]
    assert depth_indices(50) == list(range(8, 16))
    assert depth_indices(75) == list(range(4, 16))
    assert depth_indices(100) == list(range(16))


def test_location_indices_match_spec():
    assert early_indices() == [0, 1, 2, 3]
    assert middle_indices() == [6, 7, 8, 9]
    assert late_indices() == [12, 13, 14, 15]
    assert uniform_indices() == [0, 5, 10, 15]


def test_late25_equals_d25():
    assert late_indices() == depth_indices(25)


def test_nondivisible_fallback():
    assert uniform_indices(10) == [0, 3, 6, 9]
    assert uniform_indices(6) == [0, 2, 3, 5]
    assert uniform_indices(5) == [0, 1, 3, 4]


def test_regex_resolves_exact_layers():
    for name, indices in ALL_CONFIGS.items():
        layers, n_mods, _ = resolved_layers_and_params(indices)
        assert layers == sorted(indices), name
        assert n_mods == 8 * len(indices), name  # lora_A + lora_B per module


def test_trainable_param_counts_match_analytic():
    # dummy dims at rank 8: q 16r + k 12r + v 12r + o 16r = 56r per layer
    for name, indices in ALL_CONFIGS.items():
        _, _, params = resolved_layers_and_params(indices)
        assert params == len(indices) * 448, name


def test_real_model_param_formula():
    # real Llama-3.2-1B dims: 13,312 params per layer per rank, 16 layers
    for rank, expected in ((4, 851_968), (8, 1_703_936),
                           (16, 3_407_872), (32, 6_815_744)):
        assert expected == 16 * 13_312 * rank
    assert expected_trainable_params(list(range(16)), rank=8) == 1_703_936


def test_all_configs_cover_spec():
    assert set(ALL_CONFIGS) == {
        "D25", "D50", "D75", "D100", "Early25", "Middle25", "Late25", "Uniform25",
    }
    assert E6_A_CONFIGS["D100"] == list(range(NUM_LAYERS))
    assert E6_B_CONFIGS["Late25"] == E6_A_CONFIGS["D25"]
