"""Tests for E1 LM runtime: prompt masking, checkpoint continuation, config.

Uses a tiny random Llama model (CPU) so no real weights are downloaded;
the Llama-3.2 tokenizer is already in the local HF cache.
"""

import pytest
import torch
from peft import PeftModel, get_peft_model
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

from cycleft.e1.lm import (
    LMRuntimeConfig,
    encode_training_example,
    make_lora_config,
    resolve_device,
    set_seed,
)
from cycleft.e1.metrics import extract_answer

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)


def tiny_llama() -> LlamaForCausalLM:
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    return LlamaForCausalLM(config)


def test_resolve_device_prefers_mps_without_silent_cpu_fallback():
    if torch.backends.mps.is_available():
        assert resolve_device(None).type == "mps"
        assert resolve_device("mps").type == "mps"
    assert resolve_device("cpu").type == "cpu"
    with pytest.raises(ValueError, match="Unsupported device"):
        resolve_device("auto")
    with pytest.raises(ValueError, match="Unsupported device"):
        resolve_device("cuda:0")


def test_lm_runtime_config_roundtrip():
    config = LMRuntimeConfig(model_id="test-model", max_length=64)
    payload = config.to_dict()
    restored = LMRuntimeConfig.from_dict(payload)
    assert restored == config


def test_set_seed_is_deterministic():
    set_seed(123)
    first = torch.rand(4)
    set_seed(123)
    second = torch.rand(4)
    assert torch.equal(first, second)


def test_encode_training_example_masks_prompt(tokenizer):
    config = LMRuntimeConfig(model_id=MODEL_ID, max_length=128)
    example = encode_training_example(
        tokenizer, "Who is Conner Kent's member of?", "Project Cadmus", config
    )
    assert example is not None
    input_ids, labels = example
    assert input_ids.shape == labels.shape

    supervised = labels != -100
    assert supervised.any(), "answer tokens must be supervised"
    # The final supervised token must be the answer terminator.
    supervised_positions = supervised.nonzero().flatten()
    assert tokenizer.decode(labels[supervised_positions[-1]]) == "<|eot_id|>"
    # The prompt (chat header + question) is masked.
    decoded_supervised = tokenizer.decode(
        [token for token in labels[supervised_positions].tolist()]
    )
    assert "Project Cadmus" in decoded_supervised
    assert "Conner Kent" not in decoded_supervised


def test_encode_training_example_truncation_returns_none(tokenizer):
    config = LMRuntimeConfig(model_id=MODEL_ID, max_length=8)
    long_answer = "x" * 200
    assert (
        encode_training_example(tokenizer, "question?", long_answer, config) is None
    )


def test_lora_config_from_runtime_config():
    config = LMRuntimeConfig(
        model_id=MODEL_ID, lora_r=4, lora_alpha=8, lora_dropout=0.1,
        lora_target_modules=("q_proj", "v_proj"),
    )
    lora_config = make_lora_config(config)
    assert lora_config.r == 4
    assert lora_config.lora_alpha == 8
    assert set(lora_config.target_modules) == {"q_proj", "v_proj"}


def test_sequential_checkpoint_continuation(tmp_path):
    """Event 2 training must continue the Event 1 adapter, not a fresh one."""
    set_seed(0)
    event_1_dir = tmp_path / "event_1"
    event_2_dir = tmp_path / "event_2"

    # --- Event 1: fresh adapter, one update step, save. ---
    base_1 = tiny_llama()
    lora_config = make_lora_config(LMRuntimeConfig(model_id=MODEL_ID))
    model_1 = get_peft_model(base_1, lora_config)
    trainable = [p for p in model_1.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-2)
    loss = model_1(
        input_ids=torch.randint(0, 128, (2, 8)),
        labels=torch.randint(0, 128, (2, 8)),
    ).loss
    loss.backward()
    optimizer.step()
    model_1.save_pretrained(str(event_1_dir))

    # --- Event 2: load base again + continue from the event 1 adapter. ---
    set_seed(0)  # identical frozen base weights as event 1
    base_2 = tiny_llama()
    model_2 = PeftModel.from_pretrained(base_2, str(event_1_dir), is_trainable=True)

    saved_state = torch.load(
        event_1_dir / "adapter_model.bin", weights_only=True
    ) if (event_1_dir / "adapter_model.bin").exists() else None
    if saved_state is None:
        from safetensors.torch import load_file

        saved_state = load_file(event_1_dir / "adapter_model.safetensors")

    for name, parameter in model_2.named_parameters():
        if "lora_" in name:
            # Saved adapter files strip the adapter name (".default").
            saved_name = name.replace(".default.", ".")
            expected = _lookup_state(saved_state, saved_name)
            assert expected is not None, f"missing {saved_name} in saved adapter"
            assert torch.equal(parameter.detach(), expected), (
                "event 2 adapter must start from the event 1 adapter weights"
            )
            break

    # One more update step, save as event 2; weights must have moved.
    trainable_2 = [p for p in model_2.parameters() if p.requires_grad]
    assert trainable_2, "continued adapter must remain trainable"
    reference = trainable_2[0].detach().clone()
    optimizer_2 = torch.optim.AdamW(trainable_2, lr=1e-2)
    loss_2 = model_2(
        input_ids=torch.randint(0, 128, (2, 8)),
        labels=torch.randint(0, 128, (2, 8)),
    ).loss
    loss_2.backward()
    optimizer_2.step()
    assert not torch.equal(reference, trainable_2[0].detach()), (
        "trainable parameters must actually update during training"
    )
    model_2.save_pretrained(str(event_2_dir))
    assert (event_2_dir / "adapter_model.safetensors").exists() or (
        event_2_dir / "adapter_model.bin"
    ).exists()


def _lookup_state(state: dict, name: str):
    for key, value in state.items():
        if name.endswith(key) or key.endswith(name):
            return value
    return None


class StubTokenizer:
    """Minimal tokenizer substitute with a tiny vocabulary (CPU tests only).

    Mimics the real chat template's shape: the generation prompt adds
    header tokens, and a full conversation adds the answer plus wrapper
    tokens, so full-text tokenizations are strictly longer than prompts.
    """

    eos_token_id = 2

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False,
                            date_string=None):
        text = " ".join(message["content"] for message in messages)
        if add_generation_prompt:
            text += " <assistant>"
        elif messages and messages[-1]["role"] == "assistant":
            text += " <end> <eot>"
        return text

    def __call__(self, text, add_special_tokens=False, max_length=None,
                 truncation=False, return_tensors=None):
        words = text.split()
        ids = [(hash(word) % 100) + 5 for word in words]
        if max_length is not None and truncation:
            ids = ids[:max_length]
        if not ids:
            ids = [5]
        if return_tensors == "pt":
            return _FakeBatch({"input_ids": torch.tensor([ids])})
        return {"input_ids": ids}

    def decode(self, token, **kwargs):
        if hasattr(token, "tolist"):
            token = token.tolist()
        return " ".join(f"tok{t}" for t in token)


class _FakeBatch:
    """Mimics BatchEncoding just enough for generate_answer."""

    def __init__(self, mapping):
        self._mapping = mapping

    def to(self, device):
        return self

    def items(self):
        return self._mapping.items()

    def __getitem__(self, key):
        return self._mapping[key]


def test_evaluate_event_runs_in_eval_mode_and_restores_state():
    """Regression: generation must not run with LoRA dropout active."""
    from cycleft.e1.lm import LMRuntimeConfig, evaluate_event

    config = LMRuntimeConfig(model_id="tiny", seed=0, max_new_tokens=4)
    base = tiny_llama()
    model = get_peft_model(base, make_lora_config(config))
    model.train()

    rows = [
        {"question": "who is x?", "object": "alpha"},
        {"question": "who is y?", "object": "beta"},
    ]
    result = evaluate_event(model, StubTokenizer(), rows, config)

    assert result["metrics"]["examples"] == 2
    assert 0.0 <= result["metrics"]["exact_match"] <= 1.0
    assert 0.0 <= result["metrics"]["token_f1"] <= 1.0
    assert result["metrics"]["eval_loss"] > 0
    assert result["metrics"]["perplexity"] > 0
    assert len(result["records"]) == 2
    assert model.training is True, "previous training mode must be restored"


def test_trainable_check_flags_frozen_adapter():
    """train_event must fail loudly if the optimizer updates nothing."""
    from cycleft.e1.lm import train_event

    config = LMRuntimeConfig(model_id="tiny", learning_rate=0.0)
    base = tiny_llama()
    model = get_peft_model(base, make_lora_config(config))

    rows = [
        {"question": "q1?", "object": "a1"},
        {"question": "q2?", "object": "a2"},
    ]
    # Learning rate 0 leaves parameters untouched -> RuntimeError.
    with pytest.raises(RuntimeError, match="did not change"):
        train_event(model, StubTokenizer(), rows, config)


def test_train_event_updates_adapter_and_reports_stats():
    from cycleft.e1.lm import train_event

    config = LMRuntimeConfig(model_id="tiny", learning_rate=1e-2, seed=0)
    base = tiny_llama()
    model = get_peft_model(base, make_lora_config(config))

    rows = [
        {"question": "who is x?", "object": "alpha"},
        {"question": "who is y?", "object": "beta"},
        {"question": "who is z?", "object": "gamma"},
    ]
    stats = train_event(model, StubTokenizer(), rows, config)

    assert stats["optimizer_steps"] == 3
    assert stats["examples_skipped"] == 0
    assert stats["trainable_parameters"] > 0
    assert stats["total_parameters"] > stats["trainable_parameters"]
    assert 0 <= stats["trainable_percentage"] < 100.0
    assert len(stats["losses_per_step"]) == 3
    assert stats["average_loss"] > 0
    assert stats["answer_tokens_trained"] > 0
    assert stats["training_seconds"] >= 0


def test_train_event_fails_when_every_example_is_truncated():
    from cycleft.e1.lm import train_event

    config = LMRuntimeConfig(model_id="tiny", learning_rate=1e-2, max_length=4,
                             seed=0)
    base = tiny_llama()
    model = get_peft_model(base, make_lora_config(config))

    rows = [{"question": "a b c d e f g", "object": "answer words here"}]
    with pytest.raises(RuntimeError, match="No training examples"):
        train_event(model, StubTokenizer(), rows, config)


def test_extract_answer_shared_with_lm_module():
    assert extract_answer("The answer is CNN") == "CNN"
