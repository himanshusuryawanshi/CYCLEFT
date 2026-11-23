
"""Sequential experiment runner for CYCLEFT pipeline validation."""

import json
from pathlib import Path
from time import perf_counter

from cycleft.data.stream import build_temporal_stream, load_event_data
from cycleft.evaluation import evaluate_qa
from cycleft.finetuning import MemoryQAAdapter
from cycleft.checkpoints import save_checkpoint

def run_experiment(output_path: str | Path = "results/baseline.json") -> dict:
    """Evaluate a baseline and process temporal events in sequence."""
    stream = build_temporal_stream()
    model = MemoryQAAdapter()

    # Load all slices once so forgetting can be measured after each event.
    datasets = {}
    for event in stream:
        train_data, eval_data = load_event_data(event)
        datasets[event.event_id] = {
            "train": train_data,
            "eval": eval_data,
            "name": event.name,
        }

    def evaluate_all_slices():
        results = {}
        for event_id, data in datasets.items():
            metrics = evaluate_qa(data["eval"], model.predict)
            results[str(event_id)] = {
                "event": data["name"],
                **metrics,
            }
        return results

    started = perf_counter()
    history = [{
        "stage": "baseline",
        "evaluations": evaluate_all_slices(),
        "training_examples": 0,
        "elapsed_seconds": 0.0,
    }]

    total_training_examples = 0

    
    # total_training_examples = 0

    for event in stream:
        train_data = datasets[event.event_id]["train"]

        # Train first so train_metrics exists.
        train_metrics = model.fit(train_data)
        total_training_examples += train_metrics["examples_seen"]

        # Save the updated state after training.
        save_checkpoint(
            model,
            Path("checkpoints") / f"event_{event.event_id}.json",
            metadata={
                "event_id": event.event_id,
                "event_name": event.name,
                "training_examples": train_metrics["examples_seen"],
            },
        )

        history.append({
            "stage": event.name,
            "event_id": event.event_id,
            "evaluations": evaluate_all_slices(),
            "training_examples": total_training_examples,
            "event_training_examples": train_metrics["examples_seen"],
            "elapsed_seconds": round(perf_counter() - started, 3),
        })


    result = {
        "experiment": "CYCLEFT temporal stream pipeline baseline",
        "model": "MemoryQAAdapter",
        "dataset": "saxenan3/temporalwiki-drift-cl-easy",
        "events": len(stream),
        "history": history,
        "limitations": [
            "This is a question-memory baseline, not LLM fine-tuning.",
            "Exact-match accuracy is not semantic evaluation.",
            "No GPU/cloud cost or RL-controller measurements are included.",
        ],
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    return result


if __name__ == "__main__":
    result = run_experiment()
    print(f"Completed {result['events']} events.")
    print("Results saved to results/baseline.json")

    for stage in result["history"]:
        print(f"\n{stage['stage']}")
        for event_id, metrics in stage["evaluations"].items():
            print(
                f"  Slice {event_id}: "
                f"accuracy={metrics['accuracy']:.3f} "
                f"({metrics['correct']}/{metrics['examples']})"
            )
