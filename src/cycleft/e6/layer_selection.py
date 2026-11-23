"""E6 deterministic LoRA layer selection (LAYER_SELECTION_SPEC.md).

Model: Llama-3.2-1B-Instruct — 16 decoder layers, 0-indexed 0-15.
Target modules per selected layer: q_proj, k_proj, v_proj, o_proj.
"""

from __future__ import annotations

import numpy as np

NUM_LAYERS = 16
TARGET_MODULE_NAMES = ("q_proj", "k_proj", "v_proj", "o_proj")
RANK = 8
ALPHA = 16
PARAMS_PER_LAYER_PER_RANK = 13312  # q 4096 + k 2560 + v 2560 + o 4096 (rank r)


def depth_indices(pct: int, num_layers: int = NUM_LAYERS) -> list[int]:
    """E6-A: the LAST k = round(num_layers * pct / 100) layers (0-indexed)."""
    k = max(1, round(num_layers * pct / 100))
    return list(range(num_layers - k, num_layers))


def early_indices(num_layers: int = NUM_LAYERS) -> list[int]:
    k = max(1, round(num_layers * 25 / 100))
    return list(range(k))


def middle_indices(num_layers: int = NUM_LAYERS) -> list[int]:
    k = max(1, round(num_layers * 25 / 100))
    start = (num_layers - k) // 2
    return list(range(start, start + k))


def late_indices(num_layers: int = NUM_LAYERS) -> list[int]:
    return depth_indices(25, num_layers)


def uniform_indices(num_layers: int = NUM_LAYERS) -> list[int]:
    """Endpoints-inclusive even spread of exactly 4 layers (E6-B's fixed
    count): linspace(0, L-1, 4).round(), sorted unique. For num_layers < 4,
    all layers are selected."""
    if num_layers < 4:
        return list(range(num_layers))
    return sorted(
        set(np.linspace(0, num_layers - 1, 4).round().astype(int).tolist())
    )


E6_A_CONFIGS = {
    "D25": depth_indices(25),
    "D50": depth_indices(50),
    "D75": depth_indices(75),
    "D100": list(range(NUM_LAYERS)),
}

E6_B_CONFIGS = {
    "Early25": early_indices(),
    "Middle25": middle_indices(),
    "Late25": late_indices(),
    "Uniform25": uniform_indices(),
}

ALL_CONFIGS = {**E6_A_CONFIGS, **E6_B_CONFIGS}


def lora_target_config(indices: list[int]) -> dict:
    """PEFT LoraConfig arguments selecting q/k/v/o inside exactly the given
    decoder layers.

    NOTE (verified against PEFT 0.21.2): a regex STRING target_modules is NOT
    usable — PEFT converts str target_modules into a set of characters, which
    matches decoder-layer containers and crashes injection. The supported
    mechanism is name targets + `layers_to_transform` indices.
    """
    return {
        "target_modules": list(TARGET_MODULE_NAMES),
        "layers_to_transform": list(indices),
    }


def expected_trainable_params(indices: list[int], rank: int = RANK) -> int:
    return len(indices) * PARAMS_PER_LAYER_PER_RANK * rank
