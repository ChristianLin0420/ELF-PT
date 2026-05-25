"""Pre-train the LaDiR-style CoT β-VAE on stripped GSM8K reasoning chains.

Input dataset:  /cache/hf/datasets/local-gsm8k-cot-augmented-train
                (output of scripts/generate_cot_augmentation.py)

For each row:
    1 gold reasoning (from `target`, stripped of `#### N`)
  + 8 LLM reasoning   (from `cot_texts`, each stripped of `#### N` / `\\boxed{N}`)
  = 9 reasoning chains per example
  → ~7,473 × 9 = ~67K reasoning chains total

Pipeline per chain:
    text -> T5 tokens -> S fixed-length chunks (LaDiR rule) ->
    per chunk: frozen T5-small encoder -> CotVAE -> (3 memory tokens, mu, log_var)
    loss: MSE(recon, target) + β·KL(mu, log_var)
    target: per-memory-token mean-pool of the chunk's T5 hidden states
            (split chunk into mem_size equal sub-spans)

Run inside `elf-pt:smoke`:

    sudo docker run --rm --gpus all \
      -v /localhome/local-chrislin/ELF-PT:/workspace \
      -v /localhome/local-chrislin/.cache/huggingface:/cache/hf \
      -e HF_HOME=/cache/hf \
      -e PYTHONPATH=/workspace/src \
      elf-pt:smoke \
      python /workspace/scripts/train_cot_vae.py
"""
from __future__ import annotations
import argparse
import os
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax.linen as nn
from datasets import load_from_disk
from flax.training import train_state
from transformers import AutoTokenizer
from tqdm import tqdm

from modules.t5_encoder import get_encoder
from modules.cot_vae import CotVAE
from utils.cot_preprocessing import strip_answer, chunk_token_ids
from utils.checkpoint_utils import load_encoder_checkpoint

DATA_DIR = "/cache/hf/datasets/local-gsm8k-cot-augmented-train"
T5_ENC_CKPT = "embedded-language-flows/t5_small_encoder_jax/t5_small_encoder_jax.pkl"
TOKENIZER_NAME = "t5-small"
OUT_PATH = "/cache/hf/vae/cot_vae_m3_d512.pkl"

MEM_SIZE = 3
COMPRESSION_RATE = 8
MAX_SEGMENTS = 8
CHUNK_PAD_LEN = 32          # pad each chunk to this length for batched T5 + VAE forward
LATENT_DIM = 512            # T5-small d_model

EPOCHS = 20
BATCH_CHUNKS = 128
LR = 3e-4
BETA_MAX = 1.0
BETA_WARMUP_STEPS = 2000
LATENT_MEAN = 0.0
LATENT_STD = 0.2


def collect_chunks(ds, tokenizer):
    """Materialise (chunk_token_ids list, chunk_attention_mask list) over all
    9 reasoning candidates × N rows. Returns a list of (tokens, mask) tuples,
    each of length CHUNK_PAD_LEN.
    """
    chunks: list[tuple[np.ndarray, np.ndarray]] = []
    print(f"Tokenising and chunking {len(ds)} rows × 9 reasoning chains ...")
    for row in tqdm(ds, desc="prep"):
        candidates = [strip_answer(row["target"])] + [strip_answer(c) for c in row["cot_texts"]]
        for chain in candidates:
            if not chain:
                continue
            ids = tokenizer(chain, add_special_tokens=False)["input_ids"]
            segs = chunk_token_ids(ids, MEM_SIZE, COMPRESSION_RATE, MAX_SEGMENTS)
            for seg in segs:
                seg = seg[:CHUNK_PAD_LEN]
                ids_padded = np.zeros(CHUNK_PAD_LEN, dtype=np.int32)
                mask = np.zeros(CHUNK_PAD_LEN, dtype=np.int32)
                ids_padded[:len(seg)] = seg
                mask[:len(seg)] = 1
                chunks.append((ids_padded, mask))
    print(f"  total chunks: {len(chunks)}")
    return chunks


def encode_chunks_to_hidden(encoder_apply_fn, encoder_params, ids, mask):
    """Run frozen T5-small encoder on a batch of chunks. Returns the
    (B, CHUNK_PAD_LEN, 512) hidden state, normalised by (latent_mean, latent_std).
    """
    out = encoder_apply_fn(
        {"params": encoder_params}, input_ids=ids, attention_mask=mask, deterministic=True,
    )
    return (out - LATENT_MEAN) / LATENT_STD


def per_memtoken_target(hidden, mask, mem_size):
    """Build the recon target: split each chunk's hidden into mem_size equal sub-spans
    along the token axis, mean-pool each. Returns (B, mem_size, D).
    """
    B, L, D = hidden.shape
    seg_len = L // mem_size
    # Sub-spans 0..mem_size-1 each of length seg_len; last sub-span absorbs remainder.
    targets = []
    for i in range(mem_size):
        s = i * seg_len
        e = (i + 1) * seg_len if i < mem_size - 1 else L
        sub_h = hidden[:, s:e, :]
        sub_m = mask[:, s:e].astype(hidden.dtype)[..., None]   # (B, seg_len, 1)
        pooled = (sub_h * sub_m).sum(axis=1) / jnp.maximum(sub_m.sum(axis=1), 1.0)
        targets.append(pooled)
    return jnp.stack(targets, axis=1)   # (B, mem_size, D)


def beta_schedule(step):
    """Linear warm-up of β from 0.01 to BETA_MAX over BETA_WARMUP_STEPS."""
    return jnp.clip(0.01 + (BETA_MAX - 0.01) * (step / BETA_WARMUP_STEPS), 0.01, BETA_MAX)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch", type=int, default=BATCH_CHUNKS)
    parser.add_argument("--out", type=str, default=OUT_PATH)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print(f"loading dataset: {DATA_DIR}")
    ds = load_from_disk(DATA_DIR)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    chunks = collect_chunks(ds, tokenizer)
    rng = np.random.default_rng(0)
    chunk_ids = np.stack([c[0] for c in chunks])
    chunk_masks = np.stack([c[1] for c in chunks])
    N = len(chunks)
    print(f"  chunks tensor: {chunk_ids.shape}  masks: {chunk_masks.shape}")

    # Frozen T5 encoder
    print(f"loading T5 encoder: {TOKENIZER_NAME}")
    encoder_config, encoder_model, _ = get_encoder(TOKENIZER_NAME, jnp.float32)
    encoder_params = load_encoder_checkpoint(T5_ENC_CKPT)
    encoder_apply_fn = encoder_model.apply

    # VAE
    print(f"building CotVAE (mem_size={MEM_SIZE}, D={LATENT_DIM}) ...")
    model = CotVAE(hidden_size=LATENT_DIM, memory_tokens=MEM_SIZE, num_enc_layers=2, num_heads=8)
    dummy_x = jnp.ones((1, CHUNK_PAD_LEN, LATENT_DIM))
    dummy_mask = jnp.ones((1, CHUNK_PAD_LEN), dtype=jnp.int32)
    init_rng, sample_rng = jax.random.split(jax.random.PRNGKey(0))
    init_vars = model.init(init_rng, dummy_x, dummy_mask, rng=sample_rng, deterministic=False)
    params = init_vars["params"]
    n_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print(f"  VAE params: {n_params:,}")

    tx = optax.adamw(LR, weight_decay=0.0)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    @jax.jit
    def encode_t5(p_enc, ids, mask):
        return encode_chunks_to_hidden(encoder_apply_fn, p_enc, ids, mask)

    @jax.jit
    def train_step(state, hidden, mask, rng_key, step):
        def loss_fn(params):
            recon, mu, log_var, _ = model.apply(
                {"params": params}, hidden, mask, rng=rng_key, deterministic=False,
            )
            target = per_memtoken_target(hidden, mask, MEM_SIZE)
            recon_loss = jnp.mean((recon - target) ** 2)
            kl = -0.5 * jnp.mean(1 + log_var - mu ** 2 - jnp.exp(log_var))
            beta = beta_schedule(step)
            return recon_loss + beta * kl, (recon_loss, kl, beta)
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss, aux

    steps_per_epoch = max(1, N // args.batch)
    total_steps = args.epochs * steps_per_epoch
    print(f"training: epochs={args.epochs} batch={args.batch} "
          f"steps/epoch={steps_per_epoch} total={total_steps}")

    step = 0
    rng = jax.random.PRNGKey(1)
    for epoch in range(args.epochs):
        perm = np.random.permutation(N)
        running_loss, running_recon, running_kl = 0.0, 0.0, 0.0
        pbar = tqdm(range(steps_per_epoch), desc=f"epoch {epoch+1}/{args.epochs}")
        for i in pbar:
            idx = perm[i * args.batch:(i + 1) * args.batch]
            batch_ids = jnp.asarray(chunk_ids[idx])
            batch_mask = jnp.asarray(chunk_masks[idx])
            hidden = encode_t5(encoder_params, batch_ids, batch_mask)
            rng, step_rng = jax.random.split(rng)
            state, loss, (recon_loss, kl, beta) = train_step(state, hidden, batch_mask, step_rng, step)
            running_loss += float(loss)
            running_recon += float(recon_loss)
            running_kl += float(kl)
            step += 1
            if (i + 1) % 50 == 0:
                pbar.set_postfix(
                    loss=f"{running_loss/(i+1):.4f}",
                    recon=f"{running_recon/(i+1):.4f}",
                    kl=f"{running_kl/(i+1):.4f}",
                    beta=f"{float(beta):.3f}",
                )

    print(f"\nsaving VAE params to {args.out}")
    with open(args.out, "wb") as f:
        pickle.dump(jax.device_get(state.params), f)

    print("done.")


if __name__ == "__main__":
    main()
