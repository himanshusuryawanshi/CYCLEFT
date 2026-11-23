
import json

from cycleft.experiment import run_experiment


def test_experiment_writes_sequential_results(tmp_path):
    output = tmp_path / "results.json"
    result = run_experiment(output)

    assert result["events"] == 3
    assert len(result["history"]) == 4  # baseline + 3 updates
    assert result["history"][0]["stage"] == "baseline"

    for stage in result["history"]:
        assert len(stage["evaluations"]) == 3

    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["model"] == "MemoryQAAdapter"
