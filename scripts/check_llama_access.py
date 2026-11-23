from transformers import AutoConfig, AutoTokenizer

model_id = "meta-llama/Llama-3.2-1B-Instruct"

try:
    AutoConfig.from_pretrained(model_id)
    AutoTokenizer.from_pretrained(model_id)
    print("SUCCESS: Llama configuration and tokenizer are accessible.")
except Exception as exc:
    print(f"ACCESS CHECK FAILED: {type(exc).__name__}: {exc}")
    raise