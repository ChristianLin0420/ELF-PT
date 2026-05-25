"""Tokenize GSM8K via the T5 tokenizer and save as a HuggingFace Dataset on disk.

Output paths (inside the docker container with HF_HOME=/cache/hf):
  /cache/hf/datasets/local-gsm8k-t5-train
  /cache/hf/datasets/local-gsm8k-t5-test

Each saved example has:
  condition_input_ids : list[int]  — T5 tokens of the question
  input_ids           : list[int]  — T5 tokens of the answer (reasoning + "#### N")
  input               : str        — raw question (for eval-time inspection)
  target              : str        — raw answer (for eval-time gold parsing)

The existing collator (src/utils/data_utils.py:60-100) consumes condition_input_ids
+ input_ids directly; no further changes are needed.
"""
from __future__ import annotations
import os

from datasets import load_dataset
from transformers import AutoTokenizer

OUT_DIR = "/cache/hf/datasets"
TOKENIZER_NAME = "google-t5/t5-small"


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    raw = load_dataset("gsm8k", "main")

    def encode(ex):
        return {
            "condition_input_ids": tok(ex["question"], add_special_tokens=False)["input_ids"],
            "input_ids": tok(ex["answer"], add_special_tokens=False)["input_ids"],
            "input": ex["question"],
            "target": ex["answer"],
        }

    for split, name in [("train", "local-gsm8k-t5-train"), ("test", "local-gsm8k-t5-test")]:
        ds = raw[split].map(
            encode,
            remove_columns=raw[split].column_names,
            desc=f"tokenize {split}",
        )
        out = os.path.join(OUT_DIR, name)
        ds.save_to_disk(out)
        print(f"saved {name}: n={len(ds)} -> {out}")
        if len(ds) > 0:
            ex0 = ds[0]
            print(f"  example[0] input_len={len(ex0['condition_input_ids'])} "
                  f"target_len={len(ex0['input_ids'])}")
            print(f"  question: {ex0['input'][:120]!r}")
            print(f"  answer  : {ex0['target'][:120]!r}")


if __name__ == "__main__":
    main()
