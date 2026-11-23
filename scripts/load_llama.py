
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

print("Loading Llama 3.2 1B...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    attn_implementation="eager",
)

device = "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device)
model.eval()

inputs = tokenizer(
    "What is continual learning?",
    return_tensors="pt",
).to(device)

with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=40)

print(tokenizer.decode(output[0], skip_special_tokens=True))
print("Llama loaded and inference successful!")
