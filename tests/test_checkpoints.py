
from cycleft.checkpoints import load_checkpoint, save_checkpoint
from cycleft.finetuning import MemoryQAAdapter


def test_checkpoint_round_trip(tmp_path):
    adapter = MemoryQAAdapter()
    adapter.fit([
        {"question": "Who is the coach?", "object": "Alex Smith"}
    ])

    path = tmp_path / "event_1.json"
    save_checkpoint(adapter, path, {"event_id": 1})

    restored, metadata = load_checkpoint(path)

    assert restored.predict("Who is the coach?") == "Alex Smith"
    assert metadata["event_id"] == 1
