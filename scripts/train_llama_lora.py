
"""First real LoRA training smoke test for CYCLEFT."""

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from cycleft.data.stream import build_temporal_stream, load_event_data


MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
OUTPUT_DIR = "checkpoints/llama_event_1"

MAX_LENGTH = 128
MAX_TRAIN_EXAMPLES = 16
LEARNING_RATE = 1e-4


def make_training_example(row, tokenizer):
    """Create prompt-masked causal LM inputs from one QA row."""

    messages = [
        {"role": "user", "content": str(row["question"])},
    ]

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    full_text = tokenizer.apply_chat_template(
        messages + [
            {"role": "assistant", "content": str(row["object"])}
        ],
        tokenize=False,
        add_generation_prompt=False,
    )

    prompt_ids = tokenizer(
        prompt_text,
        add_special_tokens=False,
    )["input_ids"]

    encoded = tokenizer(
        full_text,
        max_length=MAX_LENGTH,
        truncation=True,
        add_special_tokens=False,
    )

    input_ids = torch.tensor(encoded["input_ids"], dtype=torch.long)

    labels = input_ids.clone()
    labels[:min(len(prompt_ids), len(labels))] = -100

    if not (labels != -100).any():
        return None

    return input_ids, labels


def main():
    if not torch.backends.mps.is_available():
        raise RuntimeError("Apple MPS is not available.")

    device = torch.device("mps")

    print("Loading TemporalWiki Event 1...")
    stream = build_temporal_stream()
    train_data, _ = load_event_data(stream[0])

    rows = train_data.select(
        range(min(MAX_TRAIN_EXAMPLES, len(train_data)))
    )

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    print("Loading Llama 3.2 1B in FP16...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )

    model.config.use_cache = False

    print("Attaching LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model.to(device)

    model.print_trainable_parameters()

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=LEARNING_RATE,
    )

    model.train()
    total_loss = 0.0
    successful_steps = 0

    print(f"Training on {len(rows)} examples...")

    for index, row in enumerate(rows):
        example = make_training_example(row, tokenizer)

        if example is None:
            print(f"Skipping example {index}: no answer tokens.")
            continue

        input_ids, labels = example

        input_ids = input_ids.unsqueeze(0).to(device)
        labels = labels.unsqueeze(0).to(device)
        attention_mask = torch.ones_like(input_ids)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
        )

        loss = outputs.loss

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite loss at training step {index}: {loss.item()}"
            )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            trainable_parameters,
            max_norm=1.0,
        )

        optimizer.step()

        total_loss += loss.item()
        successful_steps += 1

        print(
            f"Step {index + 1}/{len(rows)} "
            f"loss={loss.item():.4f}"
        )

        del outputs, loss, input_ids, labels, attention_mask

    if successful_steps == 0:
        raise RuntimeError("No examples were successfully trained.")

    print("Saving LoRA adapter...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"Training steps: {successful_steps}")
    print(f"Average loss: {total_loss / successful_steps:.4f}")
    print(f"Adapter saved to: {OUTPUT_DIR}")

    del model
    torch.mps.empty_cache()


if __name__ == "__main__":
    main()
