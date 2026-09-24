from cycleft.config import load_config

def test_load_config():
    config = load_config("configs/base.yaml")

    assert config["project"]["name"] == "CYCLEFT"
    assert config["project"]["version"] == "2.0"
    assert config["experiment"]["seed"] == 42

def test_e1_config():
    config = load_config("configs/e1.yaml")

    assert config["experiment"]["name"] == "E1_real_continual_adaptation"
    assert config["experiment"]["seed"] == 42

    assert config["model"]["name"] == "LLaMA-3.1-8B"

    assert config["data"]["stream_type"] == "sequential_domain_time"

    assert config["evaluation"]["new_domain_quality"] is True
    assert config["evaluation"]["forgetting"] is True

def test_e1_tic_lm_stream_config():
    config = load_config("configs/e1_stream.yaml")

    stream = config["stream"]

    assert stream["name"] == "E1_tic_lm_temporal"
    assert stream["dataset"]["name"] == "TiC-LM"
    assert stream["type"] == "temporal"

    assert len(stream["events"]) == 5
    assert stream["events"][0]["name"] == "base"
    assert stream["events"][1]["name"] == "temporal_update_1"