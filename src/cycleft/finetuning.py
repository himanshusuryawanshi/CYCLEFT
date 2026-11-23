
"""Lightweight continual-learning baseline for pipeline validation."""

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field


@dataclass
class MemoryQAAdapter:
    """Memorize training question-answer pairs as a lightweight baseline."""

    answers: dict[str, str] = field(default_factory=dict)
    answer_counts: Counter = field(default_factory=Counter)

    def fit(self, dataset: Iterable[dict]) -> dict:
        """Update memory using a dataset of question/object rows."""
        seen = 0

        for row in dataset:
            question = str(row["question"]).strip().lower()
            answer = str(row["object"]).strip()

            if not question or not answer:
                continue

            self.answers[question] = answer
            self.answer_counts[answer] += 1
            seen += 1

        return {"examples_seen": seen, "memory_size": len(self.answers)}

    def predict(self, question: str) -> str:
        """Return the memorized answer, or an empty string if unknown."""
        return self.answers.get(question.strip().lower(), "")

    def state_dict(self) -> dict:
        """Return a serializable snapshot of the adapter."""
        return {
            "answers": dict(self.answers),
            "answer_counts": dict(self.answer_counts),
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore an adapter snapshot."""
        self.answers = dict(state["answers"])
        self.answer_counts = Counter(state["answer_counts"])
