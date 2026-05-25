"""Phase 2: pre-compute frozen-VAE encodings of the 9 reasoning candidates per
GSM8K example and save them as a new HuggingFace dataset column.

For each row of the augmented dataset:
    candidates = [strip_answer(target)] + [strip_answer(c) for c in cot_texts]   # length 9
    For each candidate:
        T5-tokenize -> chunk(S) -> per chunk: T5 hidden -> VAE encoder -> (mem_size, D) mu
        Stack S chunks -> (S, mem_size, D), pad to (S_max, mem_size, D)
    Stack 9 candidates -> (9, S_max, mem_size, D) per row
    Also store n_segments per candidate.

Additionally tokenises the final-answer text ("#### N") and stores both
condition_input_ids (the question) and input_ids (just the final-answer line)
so the existing collate_fn / encode_text path can produce the answer slot's x0
without further changes.

Run inside `elf-pt:smoke`:

    sudo docker run --rm --gpus all \
      -v /localhome/local-chrislin/ELF-PT:/workspace \
      -v /localhome/local-chrislin/.cache/huggingface:/cache/hf \
      -e HF_HOME=/cache/hf \
      -e PYTHONPATH=/workspace/src \
      elf-pt:smoke \
      python /workspace/scripts/encode_cot_vae.py
"""
from __future__ import annotations
import argparse
import math
import os
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from datasets import load_from_disk, Dataset
from transformers import AutoTokenizer
from tqdm import tqdm

from modules.t5_encoder import get_encoder
from modules.cot_vae import CotVAE
from utils.cot_preprocessing import strip_answer, extract_final, chunk_token_ids
from utils.checkpoint_utils import load_encoder_checkpoint

IN_DIR = "/cache/hf/datasets/local-gsm8k-cot-augmented-train"
OUT_DIR = "/cache/hf/datasets/local-gsm8k-cot-vae-train"
VAE_PATH = "/cache/hf/vae/cot_vae_m3_d512.pkl"
T5_ENC_CKPT = "embedded-language-flows/t5_small_encoder_jax/t5_small_encoder_jax.pkl"
TOKENIZER_NAME = "t5-small"

MEM_SIZE = 3
COMPRESSION_RATE = 8
MAX_SEGMENTS = 8
CHUNK_PAD_LEN = 32
LATENT_DIM = 512
LATENT_MEAN = 0.0
LATENT_STD = 0.2
BATCH_CHUNKS = 128


def encode_one_candidate(
    text: str, tokenizer, encode_t5_fn, encode_vae_fn,
) -> tuple[np.ndarray, int]:
    """Encode one reasoning candidate.

    Returns (encoding, n_segments):
        encoding : (MAX_SEGMENTS, MEM_SIZE, LATENT_DIM) float32, zero-padded.
        n_segments: actual S in [0, MAX_SEGMENTS]
    """
    enc = np.zeros((MAX_SEGMENTS, MEM_SIZE, LATENT_DIM), dtype=np.float32)
    text = text.strip()
    if not text:
        return enc, 0
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    segs = chunk_token_ids(ids, MEM_SIZE, COMPRESSION_RATE, MAX_SEGMENTS)
    if not segs:
        return enc, 0
    # Pad each segment to CHUNK_PAD_LEN
    seg_ids = np.zeros((len(segs), CHUNK_PAD_LEN), dtype=np.int32)
    seg_mask = np.zeros((len(segs), CHUNK_PAD_LEN), dtype=np.int32)
    for i, seg in enumerate(segs):
        seg = seg[:CHUNK_PAD_LEN]
        seg_ids[i, :len(seg)] = seg
        seg_mask[i, :len(seg)] = 1
    hidden = encode_t5_fn(jnp.asarray(seg_ids), jnp.asarray(seg_mask))     # (S, L, D)
    mu = encode_vae_fn(hidden, jnp.asarray(seg_mask))                       # (S, mem_size, D)
    enc[:len(segs)] = np.asarray(mu)
    return enc, len(segs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", default=IN_DIR)
    parser.add_argument("--out_dir", default=OUT_DIR)
    parser.add_argument("--vae_path", default=VAE_PATH)
    args = parser.parse_args()

    print(f"loading dataset: {args.in_dir}")
    ds = load_from_disk(args.in_dir)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    # Frozen T5 encoder
    encoder_config, encoder_model, _ = get_encoder(TOKENIZER_NAME, jnp.float32)
    encoder_params = load_encoder_checkpoint(T5_ENC_CKPT)

    # Frozen VAE
    print(f"loading VAE: {args.vae_path}")
    with open(args.vae_path, "rb") as f:
        vae_params = pickle.load(f)
    vae = CotVAE(hidden_size=LATENT_DIM, memory_tokens=MEM_SIZE, num_enc_layers=2, num_heads=8)

    @jax.jit
    def encode_t5(ids, mask):
        out = encoder_model.apply(
            {"params": encoder_params}, input_ids=ids, attention_mask=mask, deterministic=True,
        )
        return (out - LATENT_MEAN) / LATENT_STD

    @jax.jit
    def encode_vae(hidden, mask):
        # Use mu (no reparameterisation) for deterministic encoding.
        out = vae.apply({"params": vae_params}, hidden, mask, rng=None, deterministic=True)
        return out[1]   # (recon, mu, log_var, z) -> mu

    print(f"encoding {len(ds)} rows × 9 candidates ...")
    cot_vae_encodings: list[np.ndarray] = []
    cot_n_segments: list[np.ndarray] = []
    answer_texts: list[str] = []

    for row in tqdm(ds, desc="encode"):
        candidates = [strip_answer(row["target"])] + [strip_answer(c) for c in row["cot_texts"]]
        candidates = candidates[:9]                                          # safety
        while len(candidates) < 9:
            candidates.append(candidates[-1])
        per_row_enc = np.zeros((9, MAX_SEGMENTS, MEM_SIZE, LATENT_DIM), dtype=np.float32)
        per_row_S = np.zeros(9, dtype=np.int32)
        for k, cand in enumerate(candidates):
            enc, S = encode_one_candidate(cand, tokenizer, encode_t5, encode_vae)
            per_row_enc[k] = enc
            per_row_S[k] = S
        cot_vae_encodings.append(per_row_enc.reshape(-1))                    # flatten for Arrow
        cot_n_segments.append(per_row_S)
        answer_texts.append(extract_final(row["target"]))

    # Tokenise question + final answer for the new dataset's input_ids / condition columns
    print("tokenising questions and final-answer texts ...")
    questions_ids = [tokenizer(r["input"], add_special_tokens=False)["input_ids"] for r in tqdm(ds, desc="tok-Q")]
    final_ids = [tokenizer(t, add_special_tokens=False)["input_ids"] for t in tqdm(answer_texts, desc="tok-A")]

    # Build the output dataset
    out_rows = []
    for i, row in enumerate(ds):
        out_rows.append({
            "input": row["input"],
            "target": answer_texts[i],
            "condition_input_ids": questions_ids[i],
            "input_ids": final_ids[i],
            "cot_vae_encodings": cot_vae_encodings[i].tolist(),
            "cot_n_segments": cot_n_segments[i].tolist(),
        })

    print(f"saving to {args.out_dir} ...")
    out_ds = Dataset.from_list(out_rows)
    out_ds.save_to_disk(args.out_dir)

    # Print storage summary
    enc_bytes = 9 * MAX_SEGMENTS * MEM_SIZE * LATENT_DIM * 4
    total_mb = len(ds) * enc_bytes / (1024 * 1024)
    print(f"  per-row encoding size: {enc_bytes/1024:.1f} KB  total: ~{total_mb:.1f} MB")
    print(f"  example[0]: S per candidate = {cot_n_segments[0].tolist()}")
    print(f"  example[0]: input='{out_rows[0]['input'][:80]}'  target='{out_rows[0]['target']}'")


if __name__ == "__main__":
    main()
