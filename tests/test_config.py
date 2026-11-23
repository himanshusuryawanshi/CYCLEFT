
import pytest

from cycleft.config import validate_stream_config


def valid_config():
    return {
        "stream": {
            "dataset": {
                "name": "example",
                "source": "example/dataset",
            },
            "selected_slices": ["slice_1"],
            "events": [
                {
                    "event_id": 0,
                    "name": "base",
                    "train_source": None,
                    "eval_source": None,
                },
                {
                    "event_id": 1,
                    "name": "update_1",
                    "train_source": "train:slice_1",
                    "eval_source": "test:slice_1",
                },
            ],
        }
    }


def test_valid_stream_config():
    validate_stream_config(valid_config())


def test_duplicate_event_ids_rejected():
    config = valid_config()
    config["stream"]["events"][1]["event_id"] = 0

    with pytest.raises(ValueError, match="unique"):
        validate_stream_config(config)


def test_unordered_events_rejected():
    config = valid_config()
    config["stream"]["events"].reverse()

    with pytest.raises(ValueError, match="ordered"):
        validate_stream_config(config)


def test_missing_dataset_source_rejected():
    config = valid_config()
    config["stream"]["dataset"]["source"] = ""

    with pytest.raises(ValueError, match="dataset field"):
        validate_stream_config(config)


def test_duplicate_slices_rejected():
    config = valid_config()
    config["stream"]["selected_slices"] = ["slice_1", "slice_1"]

    with pytest.raises(ValueError, match="duplicates"):
        validate_stream_config(config)


def test_null_base_sources_are_allowed():
    config = valid_config()
    validate_stream_config(config)
