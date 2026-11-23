
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-0.5B"

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Loading {MODEL_ID} on {device}...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,
    attn_implementation="eager",
).to(device)

inputs = tokenizer(
    "Question: What is continual learning?\nAnswer:",
    return_tensors="pt",
).to(device)

with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=30, do_sample=False)

print(tokenizer.decode(output[0], skip_special_tokens=True))
print("Model load and generation successful.")
