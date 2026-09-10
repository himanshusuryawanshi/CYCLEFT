from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Domain:
    """A single incoming domain/time segment in continual learning."""

    name: str
    train_tokens: int
    eval_tokens: int


@dataclass(frozen=True)
class EvaluationResult:
    """Quality measurements before and after a continual-learning update."""

    new_domain_quality: float
    old_domain_quality_pre: Optional[float] = None
    old_domain_quality_post: Optional[float] = None

    @property
    def forgetting(self) -> Optional[float]:
        """Return degradation on old-domain quality caused by the update."""
        if (
            self.old_domain_quality_pre is None
            or self.old_domain_quality_post is None
        ):
            return None

        return max(
            0.0,
            self.old_domain_quality_pre - self.old_domain_quality_post,
        )

@dataclass(frozen=True)
class CycleResult:
    """Measurements produced by one continual-learning cycle."""

    cycle_id: int
    domain: str
    training_tokens: int
    training_seconds: float
    evaluation: EvaluationResult