
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict:
    """Load a YAML configuration file."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if config is None:
        raise ValueError(f"Configuration file is empty: {path}")

    if not isinstance(config, dict):
        raise ValueError("Configuration must contain a YAML mapping.")

    return config


def validate_stream_config(config: dict[str, Any]) -> None:
    """Validate the structure and ordering of a CYCLEFT stream config."""
    stream = config.get("stream")
    if not isinstance(stream, dict):
        raise ValueError("Missing or invalid 'stream' section.")

    dataset = stream.get("dataset")
    if not isinstance(dataset, dict):
        raise ValueError("Missing or invalid 'stream.dataset' section.")

    for field in ("name", "source"):
        if not dataset.get(field):
            raise ValueError(f"Missing required dataset field: {field}")

    events = stream.get("events")
    if not isinstance(events, list) or not events:
        raise ValueError("'stream.events' must be a non-empty list.")

    event_ids = []
    for event in events:
        if not isinstance(event, dict):
            raise ValueError("Every stream event must be a mapping.")

        for field in ("event_id", "name", "train_source", "eval_source"):
            if field not in event:
                raise ValueError(f"Missing event field: {field}")

        if not isinstance(event["event_id"], int) or isinstance(
            event["event_id"], bool
        ):
            raise ValueError("'event_id' must be an integer.")

        if not event["name"]:
            raise ValueError("Event name cannot be empty.")

        event_ids.append(event["event_id"])

    if len(event_ids) != len(set(event_ids)):
        raise ValueError("Stream event IDs must be unique.")

    if event_ids != sorted(event_ids):
        raise ValueError("Stream events must be ordered by event_id.")

    selected_slices = stream.get("selected_slices", [])
    if not isinstance(selected_slices, list):
        raise ValueError("'selected_slices' must be a list.")

    if len(selected_slices) != len(set(selected_slices)):
        raise ValueError("'selected_slices' must not contain duplicates.")

    if any(not isinstance(item, str) or not item for item in selected_slices):
        raise ValueError("Every selected slice must be a non-empty string.")
