from cycleft.config import load_config


def test_load_config():
    config = load_config("configs/base.yaml")

    assert config["project"]["name"] == "CYCLEFT"
    assert config["project"]["version"] == "2.0"
    assert config["experiment"]["seed"] == 42