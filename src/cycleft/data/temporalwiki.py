
from datasets import Dataset, load_dataset

DATASET_ID = "saxenan3/temporalwiki-drift-cl-easy"

SLICE_TAGS = (
    "pair_20251120_20251201",
    "pair_20251201_20260101",
    "pair_20260101_20260201",
)


def load_temporal_slice(
    slice_tag: str,
    split: str = "train",
) -> Dataset:
    """Load one temporal slice for CYCLEFT development."""
    if slice_tag not in SLICE_TAGS:
        raise ValueError(f"Unknown temporal slice: {slice_tag}")

    if split not in {"train", "test", "stable"}:
        raise ValueError(f"Unsupported dataset split: {split}")

    dataset = load_dataset(DATASET_ID, split=split)
    dataset = dataset.filter(
        lambda row: row["slice_tag"] == slice_tag
    )

    if len(dataset) == 0 and split != "stable":
        raise ValueError(
            f"No rows found for {slice_tag} in split {split}"
        )

    return dataset


def to_training_example(row: dict) -> dict:
    """Convert a dataset row into a simple QA training example."""
    return {
        "prompt": row["question"],
        "completion": row["object"],
        "slice_tag": row["slice_tag"],
        "snapshot_new": row["snapshot_new"],
    }


def resolve_temporal_source(source: str):
    """Resolve a source identifier such as 'train:<slice_tag>'."""
    try:
        split, slice_tag = source.split(":", 1)
    except ValueError:
        raise ValueError(
            f"Invalid source '{source}'. Expected '<split>:<slice_tag>'."
        ) from None

    if split not in {"train", "test", "stable"}:
        raise ValueError(f"Unsupported dataset split: {split}")

    return load_temporal_slice(slice_tag=slice_tag, split=split)

