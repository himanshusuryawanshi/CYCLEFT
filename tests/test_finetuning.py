
from cycleft.finetuning import MemoryQAAdapter


def test_fit_and_predict():
    adapter = MemoryQAAdapter()
    result = adapter.fit([
        {"question": "Who is the coach?", "object": "Alex Smith"},
        {"question": "Which project?", "object": "Project Cadmus"},
    ])

    assert result["examples_seen"] == 2
    assert adapter.predict("Who is the coach?") == "Alex Smith"
    assert adapter.predict("Unknown question") == ""


def test_incremental_updates_preserve_previous_answers():
    adapter = MemoryQAAdapter()
    adapter.fit([{"question": "Q1", "object": "A1"}])
    adapter.fit([{"question": "Q2", "object": "A2"}])

    assert adapter.predict("Q1") == "A1"
    assert adapter.predict("Q2") == "A2"


def test_state_round_trip():
    adapter = MemoryQAAdapter()
    adapter.fit([{"question": "Q1", "object": "A1"}])

    restored = MemoryQAAdapter()
    restored.load_state_dict(adapter.state_dict())

    assert restored.predict("Q1") == "A1"
