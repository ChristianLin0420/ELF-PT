"""Offline CoT augmentation for GSM8K via Qwen2.5-Math-7B-Instruct.

For each train example, generate N=8 varied CoTs filtered by gold-answer match.
Save as a HuggingFace Dataset with cot_texts: list[str] of length N per example.

Run inside the elf-pt:cotgen docker image:

    sudo docker run --rm --gpus all \
      -v /localhome/local-chrislin/ELF-PT:/workspace \
      -v /localhome/local-chrislin/.cache/huggingface:/cache/hf \
      -e HF_HOME=/cache/hf \
      elf-pt:cotgen \
      python /workspace/scripts/generate_cot_augmentation.py

Resumable: writes a checkpoint every 200 examples. Re-running picks up from the
last checkpoint shard.
"""
from __future__ import annotations
import json
import os
import re
import random
from pathlib import Path

import torch
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

MODEL_NAME = "Qwen/Qwen2.5-Math-7B-Instruct"
N_OUTPUTS = 8
TEMP_PRIMARY = 0.8
TEMP_FALLBACK = 0.5
TOP_P = 0.95
MAX_NEW_TOKENS = 512
BATCH_PROMPTS = 4              # forward batch in prompts (each prompt → N samples)
CHECKPOINT_EVERY = 200         # rows
OUT_DIR = "/cache/hf/datasets/local-gsm8k-cot-augmented-train"
CHECKPOINT_DIR = "/cache/hf/datasets/_cotgen_ckpt"

ANS_RE = re.compile(r"####\s*(-?[\d,\.]+)")


def parse_gold(answer: str) -> str | None:
    m = ANS_RE.search(answer)
    if not m:
        return None
    return m.group(1).strip().replace(",", "").rstrip(".")


def parse_pred(text: str) -> str | None:
    m = ANS_RE.search(text)
    if not m:
        return None
    return m.group(1).strip().replace(",", "").rstrip(".")


def make_prompt(tokenizer, question: str) -> str:
    messages = [
        {"role": "system", "content": "Please reason step by step, and put your final answer after '#### '."},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate_batch(model, tokenizer, prompts: list[str], temperature: float, n: int):
    """For each prompt, return n sampled completion strings."""
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                       max_length=1024).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            top_p=TOP_P,
            max_new_tokens=MAX_NEW_TOKENS,
            num_return_sequences=n,
            pad_token_id=tokenizer.pad_token_id,
        )
    # out: (len(prompts) * n, seq_len)
    in_len = inputs.input_ids.shape[1]
    gen = out[:, in_len:]
    texts = tokenizer.batch_decode(gen, skip_special_tokens=True)
    # Reshape: per prompt, n outputs
    return [texts[i * n:(i + 1) * n] for i in range(len(prompts))]


def load_checkpoint() -> dict[int, list[str]]:
    """Returns {row_index: cot_texts_list}."""
    p = Path(CHECKPOINT_DIR)
    if not p.exists():
        return {}
    ckpt = {}
    for shard in sorted(p.glob("shard_*.jsonl")):
        with open(shard) as f:
            for line in f:
                rec = json.loads(line)
                ckpt[rec["idx"]] = rec["cot_texts"]
    return ckpt


def save_checkpoint_shard(records: list[dict], shard_idx: int):
    p = Path(CHECKPOINT_DIR)
    p.mkdir(parents=True, exist_ok=True)
    fp = p / f"shard_{shard_idx:04d}.jsonl"
    with open(fp, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def main() -> None:
    random.seed(0)
    torch.manual_seed(0)

    print(f"Loading {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda:0",
    )
    model.eval()
    print(f"  device={model.device}  dtype={next(model.parameters()).dtype}")

    raw = load_dataset("gsm8k", "main", split="train")
    print(f"GSM8K train size: {len(raw)}")

    ckpt = load_checkpoint()
    print(f"Resuming with {len(ckpt)} already-cached examples")

    new_records: list[dict] = []
    shard_idx = len(list(Path(CHECKPOINT_DIR).glob("shard_*.jsonl"))) if Path(CHECKPOINT_DIR).exists() else 0
    pending: list[tuple[int, str, str]] = []  # (idx, question, gold)

    def flush_pending():
        nonlocal shard_idx
        if not pending:
            return
        prompts = [make_prompt(tokenizer, q) for _, q, _ in pending]
        # First pass at TEMP_PRIMARY
        out_lists = generate_batch(model, tokenizer, prompts, TEMP_PRIMARY, N_OUTPUTS)
        kept_per_example: list[list[str]] = []
        retry_indices: list[int] = []
        for i, (idx, q, gold) in enumerate(pending):
            valid = [t for t in out_lists[i] if parse_pred(t) == gold]
            if len(valid) < 4:
                retry_indices.append(i)
            kept_per_example.append(valid)
        # Second pass at TEMP_FALLBACK for low-yield examples
        if retry_indices:
            retry_prompts = [prompts[i] for i in retry_indices]
            retry_outs = generate_batch(model, tokenizer, retry_prompts, TEMP_FALLBACK, N_OUTPUTS)
            for slot, ri in enumerate(retry_indices):
                _, q, gold = pending[ri]
                more = [t for t in retry_outs[slot] if parse_pred(t) == gold]
                kept_per_example[ri].extend(more)
        # Pad / dedup with random choices to exactly N_OUTPUTS
        for i, (idx, q, gold) in enumerate(pending):
            valid = kept_per_example[i]
            if not valid:
                # Last-resort: keep the gold answer verbatim once, duplicated.
                fallback = q + " #### " + gold
                valid = [fallback]
            if len(valid) < N_OUTPUTS:
                pad = random.choices(valid, k=N_OUTPUTS - len(valid))
                valid = valid + pad
            else:
                valid = valid[:N_OUTPUTS]
            new_records.append({"idx": idx, "cot_texts": valid, "question": q, "gold": gold})
        save_checkpoint_shard(new_records[-len(pending):], shard_idx)
        shard_idx += 1
        pending.clear()

    pbar = tqdm(total=len(raw), desc="generating")
    pbar.update(len(ckpt))
    for idx in range(len(raw)):
        if idx in ckpt:
            pbar.update(0)
            continue
        ex = raw[idx]
        gold = parse_gold(ex["answer"])
        if gold is None:
            # Skip malformed gold (shouldn't happen on GSM8K)
            new_records.append({"idx": idx, "cot_texts": [ex["answer"]] * N_OUTPUTS,
                                "question": ex["question"], "gold": ""})
            pbar.update(1)
            continue
        pending.append((idx, ex["question"], gold))
        if len(pending) >= BATCH_PROMPTS:
            flush_pending()
            pbar.update(BATCH_PROMPTS)
        if (idx + 1) % CHECKPOINT_EVERY == 0:
            flush_pending()
            pbar.update(0)
    flush_pending()
    pbar.close()

    # Merge ckpt + new_records by idx; assemble final dataset.
    print("Merging records ...")
    all_recs = {r["idx"]: r["cot_texts"] for r in new_records}
    all_recs.update(ckpt)
    rows: list[dict] = []
    for idx in range(len(raw)):
        ex = raw[idx]
        rows.append({
            "input": ex["question"],
            "target": ex["answer"],
            "cot_texts": all_recs.get(idx, [ex["answer"]] * N_OUTPUTS),
        })
    ds = Dataset.from_list(rows)
    ds.save_to_disk(OUT_DIR)
    print(f"saved {OUT_DIR}: n={len(ds)}")
    print(f"  example[0].cot_texts[0][:120]: {ds[0]['cot_texts'][0][:120]!r}")


if __name__ == "__main__":
    main()
