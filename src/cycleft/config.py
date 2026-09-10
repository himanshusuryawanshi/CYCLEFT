from pathlib import Path

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

    return config