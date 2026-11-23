"""E1: Real LLM continual adaptation experiment package."""

from cycleft.e1 import data, metrics
from cycleft.e1.data import prepare_event_datasets
from cycleft.e1.metrics import (
    average_accuracy,
    backward_transfer,
    exact_match,
    forgetting,
    normalize_answer,
    token_f1,
)

__all__ = [
    "data",
    "metrics",
    "prepare_event_datasets",
    "normalize_answer",
    "exact_match",
    "token_f1",
    "forgetting",
    "backward_transfer",
    "average_accuracy",
]
