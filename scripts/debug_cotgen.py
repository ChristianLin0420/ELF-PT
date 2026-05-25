"""Quick interactive test: load Qwen-Math-7B and dump 2 raw generations for
the first GSM8K example to see what the model actually outputs.
"""
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-Math-7B-Instruct"

tok = AutoTokenizer.from_pretrained(MODEL_NAME)
if tok.pad_token_id is None:
    tok.pad_token_id = tok.eos_token_id
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda:0")
model.eval()

ds = load_dataset("gsm8k", "main", split="train")
ex = ds[0]
print("QUESTION:", ex["question"])
print("ANSWER:", ex["answer"])
print()

messages = [
    {"role": "system", "content": "Please reason step by step, and put your final answer after '#### '."},
    {"role": "user", "content": ex["question"]},
]
prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print("=" * 80)
print("PROMPT (raw, after chat template):")
print(prompt)
print("=" * 80)

inputs = tok(prompt, return_tensors="pt").to(model.device)
print(f"\ninput_ids shape: {inputs.input_ids.shape}")

with torch.no_grad():
    out = model.generate(
        **inputs,
        do_sample=True, temperature=0.8, top_p=0.95,
        max_new_tokens=512,
        num_return_sequences=2,
        pad_token_id=tok.pad_token_id,
    )

print(f"\nout shape: {out.shape}  (should be (2, ~512+prompt_len))")
in_len = inputs.input_ids.shape[1]
gen = out[:, in_len:]
texts = tok.batch_decode(gen, skip_special_tokens=True)

for i, t in enumerate(texts):
    print()
    print("=" * 80)
    print(f"GENERATION {i+1} (len={len(t)} chars)")
    print("=" * 80)
    print(t)
