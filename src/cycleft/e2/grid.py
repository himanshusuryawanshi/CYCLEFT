"""E2 experimental grid: LoRA rank x training budget configurations.

Budget semantics (verified against the E1 runtime in
``assert_budget_semantics``): batch_size=1, gradient_accumulation_steps=1,
epochs=1, so **one optimizer step per training example**. A budget of
125 examples per event therefore means exactly 125 optimizer steps per
event. Examples processed, tokens processed and optimizer steps are all
recorded separately in the resource metrics regardless.
"""

from __future__ import annotations

from dataclasses import dataclass

from cycleft.e1.lm import LMRuntimeConfig

RANKS: tuple[int, ...] = (4, 8, 16, 32)

# Budget id -> fraction of the 500-example training pool per event.
BUDGET_FRACTIONS: dict[str, float] = {
    "b125": 0.25,
    "b250": 0.50,
    "b500": 1.00,
}

TRAINING_POOL_SIZE = 500  # verified against the E1 dataset manifest


@dataclass(frozen=True)
class E2Configuration:
    """One cell of the E2 grid."""

    config_id: str
    lora_rank: int
    budget_id: str
    budget_examples: int
    budget_fraction: float

    @property
    def expected_optimizer_steps_per_event(self) -> int:
        """Optimizer steps per event under the verified budget semantics."""
        return self.budget_examples


def budget_examples_for(budget_id: str, pool_size: int = TRAINING_POOL_SIZE) -> int:
    """Resolve a budget id to a per-event training example count."""
    if budget_id not in BUDGET_FRACTIONS:
        raise ValueError(
            f"Unknown budget id '{budget_id}'. Available: {sorted(BUDGET_FRACTIONS)}"
        )
    fraction = BUDGET_FRACTIONS[budget_id]
    examples = int(round(pool_size * fraction))
    if examples <= 0:
        raise ValueError(f"Budget '{budget_id}' resolves to {examples} examples.")
    return examples


def make_configuration(lora_rank: int, budget_id: str) -> E2Configuration:
    if lora_rank not in RANKS:
        raise ValueError(f"Unknown LoRA rank {lora_rank}. Available: {RANKS}")
    return E2Configuration(
        config_id=f"rank{lora_rank}_{budget_id}",
        lora_rank=lora_rank,
        budget_id=budget_id,
        budget_examples=budget_examples_for(budget_id),
        budget_fraction=BUDGET_FRACTIONS[budget_id],
    )


def build_grid(
    ranks: tuple[int, ...] = RANKS,
    budget_ids: tuple[str, ...] = tuple(BUDGET_FRACTIONS),
) -> list[E2Configuration]:
    """Build the E2 grid in rank-major, budget-minor order."""
    return [
        make_configuration(rank, budget_id)
        for rank in ranks
        for budget_id in budget_ids
    ]


def assert_budget_semantics(config: LMRuntimeConfig) -> None:
    """Verify that optimizer steps == examples for the configured runtime.

    E2 defines training budgets in optimizer steps. With the E1 runtime
    (batch_size=1, gradient_accumulation_steps=1, epochs=1) the equality
    holds exactly; any other setting must be adapted explicitly.
    """
    problems = []
    if config.batch_size != 1:
        problems.append(f"batch_size={config.batch_size}")
    if config.gradient_accumulation_steps != 1:
        problems.append(f"gradient_accumulation_steps={config.gradient_accumulation_steps}")
    if config.epochs != 1:
        problems.append(f"epochs={config.epochs}")
    if problems:
        raise ValueError(
            "E2 budget semantics require batch_size=1, "
            f"gradient_accumulation_steps=1, epochs=1; found {', '.join(problems)}. "
            "Adapt the budget calculation explicitly before running."
        )


def configuration_row(config: E2Configuration, lm_config: LMRuntimeConfig) -> dict:
    """Serializable description of one grid cell for configurations.csv."""
    rank_scaling = lm_config.lora_alpha / config.lora_rank
    return {
        "config_id": config.config_id,
        "lora_rank": config.lora_rank,
        "lora_alpha": lm_config.lora_alpha,
        "lora_scaling": rank_scaling,
        "budget_id": config.budget_id,
        "budget_fraction": config.budget_fraction,
        "budget_examples_per_event": config.budget_examples,
        "expected_optimizer_steps_per_event": (
            config.expected_optimizer_steps_per_event
        ),
        "learning_rate": lm_config.learning_rate,
        "lora_target_modules": ",".join(lm_config.lora_target_modules),
        "seed": lm_config.seed,
        "model_id": lm_config.model_id,
        "max_length": lm_config.max_length,
        "max_new_tokens": lm_config.max_new_tokens,
    }
