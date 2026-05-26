# ELF-PT-R with LaDiR-Style CoT-VAE — Method

> Parallel-Thought diffusion language model trained on GSM8K with K_reasoning
> latent thought slots whose targets come from a frozen VAE that encodes
> diverse Chain-of-Thought (CoT) solutions to each problem.

---

## Table of Contents

1. [Motivation](#1-motivation)
2. [End-to-end pipeline overview](#2-end-to-end-pipeline-overview)
3. [Data preparation](#3-data-preparation)
4. [VAE pre-training](#4-vae-pre-training)
5. [Per-step batch construction](#5-per-step-batch-construction)
6. [Model architecture](#6-model-architecture)
7. [Attention masks](#7-attention-masks)
8. [Loss computation](#8-loss-computation)
9. [What each loss term enforces](#9-what-each-loss-term-enforces)
10. [Inference (post-training)](#10-inference-post-training)
11. [File map](#11-file-map)

---

## 1. Motivation

GSM8K math reasoning has a well-defined evaluation objective — the **final
numeric answer** after `#### N`. The reasoning chain is *instrumental*, not
scored. Yet a model that can explicitly carry multiple distinct reasoning
trajectories in parallel and condition the final answer on them should be
better at exploring solution paths.

We follow **LaDiR** (Liu et al., 2025, arXiv:2510.04573):

- Encode each CoT into a fixed-size set of latent memory tokens via a β-VAE.
- Train a diffusion model whose K_reasoning slots are each denoised toward a
  different VAE-encoded CoT, while a separate answer slot is denoised toward
  the actual final answer text.
- Use inference-time RBF repulsion to make the K reasoning trajectories
  populate distinct modes of the learned latent space.

The departure from LaDiR is minimal: we reuse the frozen T5-small encoder
already used by ELF instead of training a separate VAE encoder LM, and we
amplify diversity by generating **N=8 LLM CoTs per GSM8K problem** so each
training step can sample K_reasoning *distinct* CoTs from a real
distribution of solutions (rather than reusing the single gold chain).

---

## 2. End-to-end pipeline overview

```
┌─────────────────┐
│  GSM8K train    │  7,473 (question, gold_answer) pairs
│  (HuggingFace)  │
└────────┬────────┘
         │
         ▼   Phase 0:  LLM CoT augmentation         (Qwen2.5-Math-7B, T=0.8)
         │           N=8 varied CoTs per problem,
         │           filter by gold-answer match
         ▼
┌──────────────────────────────────┐
│ local-gsm8k-cot-augmented-train  │  rows = {input, target, cot_texts: 8}
└────────┬─────────────────────────┘
         │
         ▼   Phase 1:  Train CoT β-VAE
         │           (~67K reasoning chunks, mem_size=3, 20 epochs)
         ▼
┌──────────────────────────────────┐
│ cot_vae_m3_d512.pkl              │  frozen, 11.3M params
└────────┬─────────────────────────┘
         │
         ▼   Phase 2:  Pre-compute VAE encodings
         │           (9 candidates × S_max=8 segments × 3 mem × 512 dim per row)
         ▼
┌──────────────────────────────────┐
│ local-gsm8k-cot-vae-train        │  6.6 GB Arrow, ready for diffusion
└────────┬─────────────────────────┘
         │
         ▼   Phase 3:  Diffusion training
         │           ELF-PT-B, K_total=3, 100 epochs, MSE+CE+inference-diversity
         ▼
┌──────────────────────────────────┐
│ checkpoint                       │
└──────────────────────────────────┘
```

---

## 3. Data preparation

### 3.1 Augmentation (`scripts/generate_cot_augmentation.py`)

For each GSM8K example, generate N=8 CoTs with `Qwen/Qwen2.5-Math-7B-Instruct`
in bfloat16 at `temperature=0.8, top_p=0.95, max_new_tokens=512`.

Filter pass: keep only generations whose `parse_pred(text)` (matches any of
`#### N`, `\boxed{N}`, `the answer is N`) equals the gold answer. If fewer
than 4 valid → retry at `temperature=0.5`. Pad to exactly N=8 with
duplicates if needed.

**Stored row**:
```json
{
  "input":  "<question>",
  "target": "<gold answer: reasoning + '#### N'>",
  "cot_texts": ["<LLM CoT 1>", ..., "<LLM CoT 8>"]
}
```

### 3.2 Candidate construction (`src/utils/cot_preprocessing.py`)

At training and encoding time we derive **9 reasoning candidates** per row by
stripping the final-answer suffix from each available CoT:

```python
candidates = [strip_answer(row["target"])]              # 1 gold reasoning
           + [strip_answer(c) for c in row["cot_texts"]] # 8 LLM reasoning
```

Where `strip_answer` removes a trailing `#### N` and / or `\boxed{N}`.

Answer-slot target = `extract_final(row["target"])` = `"#### N"`.

### 3.3 Chunking and encoding (`scripts/encode_cot_vae.py`)

For each candidate (length `T` T5 tokens), apply LaDiR's fixed-length
chunking with `mem_size=3, mean_compression_rate=8`:

```
S = ceil(T / (mem_size × mean_compression_rate)) = ceil(T / 24)
segment_length = ceil(T / S)
```

`S` is capped at `S_max=8` for fixed storage shape. For each segment:
1. Pad to `CHUNK_PAD_LEN=32` tokens with attention mask.
2. Forward through **frozen** T5-small encoder → `(32, 512)` hidden states.
3. Forward through **frozen** CoT-VAE encoder → mean memory tokens `μ` of
   shape `(mem_size=3, 512)`.

Stack S segments → `(S, 3, 512)`, zero-pad to `(8, 3, 512)`.
Stack 9 candidates → `(9, 8, 3, 512)` per row.

**Stored row** (Arrow):
```json
{
  "condition_input_ids": [<question tokens>],
  "input_ids":           [<"#### N" tokens>],
  "cot_vae_encodings":   [9*8*3*512 floats, flattened],
  "cot_n_segments":      [S_0, S_1, ..., S_8]
}
```

| Quantity | Value |
|---|---|
| Rows | 7,473 |
| Bytes per row | 9 × 8 × 3 × 512 × 4 = 442 KB |
| Total Arrow size | ~6.6 GB |

---

## 4. VAE pre-training

### 4.1 Module (`src/modules/cot_vae.py`)

```
CotEncoder(L_cot, 512) ─ 2 × {RMSNorm → Self-Attn → SwiGLU FFN}
                         ─ mean-pool over valid tokens → (B, 512)
                         ─ Linear → (B, 2 × M × 512)
                         ─ reshape → μ (B, M, 512), log_σ² (B, M, 512)
                         ─ reparameterise: z = μ + exp(log_σ²/2) · ε

CotDecoder(z) ─ learnable position queries (M, 512)
              ─ cross-attention(query=positions, key=z, value=z)
              ─ SwiGLU FFN
              ─ Linear → (B, M, 512)   reconstructed memory tokens
```

`M = mem_size = 3`. 11.3M total params.

### 4.2 Training loss

Per chunk:
- Input: T5 hidden states of one (≤32-token) segment.
- Target: mean-pool the chunk's T5 hidden into M=3 equal sub-spans, one
  512-d vector per sub-span. Shape `(3, 512)`.
- Reconstruction: VAE decoder output `(3, 512)`.
- Loss:

$$
\mathcal{L}_{\text{VAE}} = \underbrace{\frac{1}{M \cdot D}\|\hat{x} - x_{\text{target}}\|_2^2}_{\text{recon (MSE)}}
                          + \beta \cdot \underbrace{-\frac{1}{2}\sum (1 + \log \sigma^2 - \mu^2 - \sigma^2)}_{\text{KL}(\mathcal{N}(\mu,\sigma)\,\|\,\mathcal{N}(0,I))}
$$

β linearly warms 0.01 → 1.0 over 2,000 steps.

20 epochs at batch=128 over 501K chunks (= 9 candidates × ~7 segments × 7,473
rows). Total ~78K steps. ~12 min wall-clock on A100.

---

## 5. Per-step batch construction

`src/utils/data_utils.py:get_dataloader → collate_fn`.

For each row in a batch of size B:

```python
# Pick K_r distinct CoTs from 9 (random per step, different each epoch)
idx = np.random.choice(9, size=K_r, replace=False)            # e.g. [3, 7]
for k, ci in enumerate(idx):
    S = cot_n_segments[ci]                                    # 1..8
    latent = encoded[ci, :S, :, :].reshape(S * 3, 512)        # actual positions
    reasoning_targets[b, k, :S*3, :] = latent                 # zero-pad to L
    reasoning_loss_mask[b, k, :S*3] = True                    # mask is True
                                                              # only at valid pos
```

Resulting batch fields:

| Field | Shape | Source |
|---|---|---|
| `input_ids` | (B, 512) | `[condition_input_ids \| input_ids]`, padded |
| `condition_input_ids` | (B, ?) | T5 tokens of the question |
| `attention_mask` | (B, 512) | non-padding positions |
| `cond_seq_mask` | (B, 512) | 1 for question (condition) positions |
| `encoder_attention_mask` | (B, 512, 512) | asymmetric cond mask for T5 |
| **`reasoning_targets`** | **(B, K_r, 512, 512)** | sampled VAE latents per slot |
| **`reasoning_loss_mask`** | **(B, K_r, 512)** | which positions to score |

---

## 6. Model architecture

### 6.1 Sequence layout

```
K_total = K_reasoning + 1 = 3  slots, each L = 512 tokens.
Total flat sequence: K_total × L = 1,536 tokens

┌────────────── slot R1 ──────────────┬────── slot R2 ──────┬──── slot A ────────┐
│  VAE latents (S_i · 3 positions)    │  VAE latents (S_j·3) │  T5(question +     │
│  zero-padded to 512                 │  zero-padded to 512  │       "#### N")    │
│                                     │                      │  zero-padded       │
└─────────────────────────────────────┴──────────────────────┴────────────────────┘
   ε_R1,  shared t per example          ε_R2,  shared t      ε_A,  shared t
   (independent noise per slot,                              (its target is the
    one t per example shared across slots)                    final answer text)
```

### 6.2 Forward pass

```
INPUT  (B, 1536, 512)
   │
   │  text projection (per-position):
   │     Dense 512 → 128  (bottleneck)
   │     Dense 128 → 768  (hidden)
   ▼
   (B, 1536, 768)
   │
   │  prepend 12 learnable prefix tokens (4 time + 4 SC-CFG + 4 mode):
   │     time_emb(t) → 4 tokens of (768)
   │     sc_cfg_emb(γ) → 4 tokens of (768)
   │     mode_tokens → 4 tokens of (768)
   ▼
   (B, 1548, 768)
   │
   │  RoPE positional encoding
   │     ft_seq_len = K_total · L = 1,536
   │     pt_seq_len = L = 512  (NTK-style position interpolation across slots)
   ▼
   ┌──────────────────────────────────────────────────────────────┐
   │     12 Transformer Blocks, alternating intra / inter          │
   │                                                              │
   │  Layer 0:  IntraGroupBlock   ─ uses intra_mask                │
   │  Layer 1:  InterGroupBlock   ─ uses inter_mask   ◆ zero-init  │
   │  Layer 2:  IntraGroupBlock                                    │
   │  Layer 3:  InterGroupBlock                       ◆ zero-init  │
   │  ...                                                          │
   │  Layer 11: InterGroupBlock                       ◆ zero-init  │
   │                                                              │
   │  Each block:                                                  │
   │    x  ← x + Attn(RMSNorm(x), mask)                            │
   │    x  ← x + SwiGLU(RMSNorm(x))                                │
   │                                                              │
   │  ◆ Inter blocks zero-init the attention out-proj kernel and   │
   │    SwiGLU's w3 so the block is exact identity at init.        │
   └──────────────────────────────────────────────────────────────┘
   │
   │  strip prefix → (B, 1536, 768)
   ▼
   ┌────────────────────────────────────────────┐
   │ FinalLayer:  RMSNorm → Dense 768 → 512     │  (velocity / x_pred head)
   └────────────────────────────────────────────┘
   ▼
OUTPUT  v_pred  (B, 1536, 512)
   │
   │  reshape: (B, 3, 512, 512) = (B, K_total, L, D)
   ▼
   per-slot velocity predictions used in the loss
```

### 6.3 Block-type counts

ELF-PT-B has 12 layers in the pattern `intra,inter` repeated 6×:
**6 IntraGroupBlocks + 6 InterGroupBlocks = 105M total parameters**.

---

## 7. Attention masks

Both intra and inter masks are shape `(B, K_total · L, K_total · L) = (B, 1536, 1536)`, ANDed with ELF's existing `encoder_attention_mask` (which encodes per-batch label-drop and CFG semantics).

### 7.1 Intra-group mask — within-slot attention only

```
              keys:    R1 cols       R2 cols       A cols
                       (0..511)     (512..1023)  (1024..1535)
  queries
  ──────────────────────────────────────────────────────────────────
  R1 rows         ┌──────────┬─────────────┬─────────────┐
  (0..511)        │  ✓ valid │   ✗  ZERO   │   ✗  ZERO   │
                  │   pairs  │             │             │
                  ├──────────┼─────────────┼─────────────┤
  R2 rows         │ ✗ ZERO   │  ✓ valid    │   ✗  ZERO   │
  (512..1023)     │          │   pairs     │             │
                  ├──────────┼─────────────┼─────────────┤
  A rows          │ ✗ ZERO   │   ✗ ZERO    │  ✓ valid    │
  (1024..1535)    │          │             │   pairs     │
                  └──────────┴─────────────┴─────────────┘

  Cond/source token columns (the question positions inside each slot)
  are visible to ALL slots — so source tokens act as shared context.
```

### 7.2 Inter-group mask — CAUSAL (answer reads reasoning; reasoning ↛ answer)

```
              keys:    R1 cols       R2 cols       A cols
  queries
  ──────────────────────────────────────────────────────────────────
  R1 rows         ┌──────────┬─────────────┬─────────────┐
                  │  ✓ valid │  ✓ valid    │  ✗  ZERO    │  ← reasoning
                  │   pairs  │   pairs     │  (causal)   │     does not
                  ├──────────┼─────────────┼─────────────┤     attend
  R2 rows         │ ✓ valid  │  ✓ valid    │  ✗  ZERO    │     answer
                  │  pairs   │   pairs     │  (causal)   │
                  ├──────────┼─────────────┼─────────────┤
  A rows          │ ✓ valid  │  ✓ valid    │  ✓ valid    │  ← answer
                  │  pairs   │   pairs     │   pairs     │     reads all
                  └──────────┴─────────────┴─────────────┘
```

This causal block-mask is what gives our architecture its "reasoning →
answer" information flow. Reasoning slots can develop independent thoughts;
the answer slot conditionally synthesises them.

---

## 8. Loss computation

### 8.1 Branch selection

Each training step picks one branch via `jax.lax.cond` on a Bernoulli draw:
- **Denoiser branch (80% of steps)** — MSE on flow-matching velocity.
- **Decoder branch (20% of steps)** — cross-entropy on answer tokens.

### 8.2 Denoiser branch

Per slot `k ∈ {R1, R2, A}` independently sample noise `ε_k ∼ N(0,I)` (and one
shared time `t ∼ logit-normal` per example, broadcast to all slots):

$$
z_t^{(k)} = t \cdot x_0^{(k)} + (1 - t) \cdot \varepsilon^{(k)} \cdot \texttt{noise\_scale}
$$

Per-slot velocity target:

$$
v_{\text{target}}^{(k)} = \frac{x_0^{(k)} - z_t^{(k)}}{\max(1 - t, t_\text{eps})}
$$

Forward through the model: `v_pred` shape `(B, K_total, L, D)`.

**Per-slot masked MSE**:

```python
per_dim_loss   = (v_pred - v_target_per) ** 2          # (B, K_total, L, D)
per_token_loss = jnp.mean(per_dim_loss, axis=-1)       # (B, K_total, L)
masked         = per_token_loss * loss_mask_per         # (B, K_total, L)
l2_loss        = masked.sum() / max(loss_mask_per.sum(), 1.0)
```

Where `loss_mask_per` is built by concatenating the reasoning slots' VAE-position mask with the answer slot's text mask:

```python
loss_mask_per = jnp.concatenate([
    reasoning_loss_mask,                # (B, K_r, L)  ─ True at first S*3 positions
    answer_text_mask[:, None, :],        # (B, 1,   L)  ─ True at "#### N" positions
], axis=1)                                # (B, K_total, L)
```

Effective supervised positions per slot (typical GSM8K numbers):
- **Slot R1**: `S_i × 3` — usually 6–24 positions out of 512
- **Slot R2**: `S_j × 3` — usually 6–24 positions out of 512
- **Slot A**: 3–5 positions out of 512 (just the `"#### N"` tokens)

Even though only ~30 positions are scored per example, the model has to
attend across all 1,536 sequence positions to compute the prediction → the
masked positions get high-quality gradient.

### 8.3 Decoder branch

Re-run the model with `decoder_step_active=True` and `return_pre_unembed=True`:
in R-mode this returns the answer slot's pre-unembed hidden states
`(B, L, 768)`.

Apply the shared decoder MLP (parameters `proj_kernel`, `proj_bias`,
`unembed_kernel`, `unembed_bias`):

```python
logits  = GELU(x_answer @ proj_kernel + proj_bias) @ unembed_kernel + unembed_bias
                                                              # (B, L, V=32100)
log_p   = jax.nn.log_softmax(logits.astype(jnp.float32), -1)
ce_per_token = -jnp.take_along_axis(log_p, decoder_targets[..., None], -1).squeeze(-1)
ce_loss = (ce_per_token * loss_mask).sum() / max(loss_mask.sum(), 1.0)
```

Where `loss_mask` is the answer slot's text loss mask (non-cond, non-pad
positions of `"#### N"`).

### 8.4 Total loss

Per step (the branches are stochastic):

$$
\mathcal{L}_{\text{step}} =
\begin{cases}
\mathcal{L}_{\text{L2,per-slot}} + \lambda_{\text{div}} \cdot \mathcal{L}_{\text{div}} & \text{w.p.}~0.8 \\
\mathcal{L}_{\text{CE}} & \text{w.p.}~0.2
\end{cases}
$$

`λ_div = 0.0` in our default config — diversity is **inference-only**. The
term is computed and logged for diagnostics but contributes zero gradient.

---

## 9. What each loss term enforces

| Term | What it pushes the model to learn |
|---|---|
| **L2 on R1** | The model's velocity prediction at R1's first `S_i × 3` positions must denoise toward the VAE encoding of randomly sampled CoT i. Different sample each step. R1 learns "I am one reasoning latent block." |
| **L2 on R2** | Same as R1 but for a different sampled CoT j. Architectural separation (causal mask + per-slot noise) + sampling diversity push R1 and R2 to develop distinct representations. |
| **L2 on A** | At the answer slot's ~3–5 supervised positions, the velocity must denoise toward T5(`"#### N"`). Small position count but very high signal — this is the actual GSM8K eval objective. |
| **CE on A** | When the decoder branch fires, the decoded answer slot must explicitly predict `input_ids` of `"#### N"`. Trains the shared unembed kernel to behave like a tiny text decoder. |
| **Diversity (off)** | Not used at training time. Reported as a diagnostic to verify R1 and R2 hidden states are differentiating. Inference-time RBF repulsion (in `apply_diversity_repulsion`) provides the actual diversity pressure. |

---

## 10. Inference (post-training)

For each test question:

1. T5-encode question to build the answer-slot's condition prefix.
2. Initialize K_total slots of noise `z_1 ∼ N(0, I)`.
3. For each denoising step `t: 1 → 0`:
   - Forward through the model → velocity `v(z_t, t)`.
   - Euler step: `z_{t - Δt} = z_t - Δt · v`.
   - (Optional) Apply RBF repulsion across reasoning slots only:
     ```python
     z_r ← z_r + γ(t) · ∑_{j ≠ i} (1 - exp(-||z_i - z_j||² / σ²)) · (z_i - z_j)
     ```
     with `γ(t) = γ_max · (t/T)²` (LaDiR annealing).
4. At `t = 0`: decode the **answer slot only** via the shared unembed head:
   ```
   logits = GELU(x_answer @ proj_kernel + proj_bias) @ unembed_kernel + bias
   text   = tokenizer.decode(argmax(logits, axis=-1))
   ```
5. Parse `#### N` from the decoded text → predicted answer.
6. Compare to gold → accuracy.

The diversity-ablation comparison happens at **eval time** by toggling
`config.diversity_repulsion_inference`. No re-training needed.

---

## 11. File map

| File | Role |
|---|---|
| `scripts/generate_cot_augmentation.py` | Phase 0: Qwen2.5-Math-7B CoT generation + filter |
| `scripts/train_cot_vae.py` | Phase 1: pre-train CoT β-VAE |
| `scripts/encode_cot_vae.py` | Phase 2: pre-compute frozen-VAE encodings |
| `src/modules/cot_vae.py` | `CotEncoder`, `CotDecoder`, `CotVAE` Flax modules |
| `src/utils/cot_preprocessing.py` | `strip_answer`, `extract_final`, `chunk_token_ids` |
| `src/utils/data_utils.py` | Collator: K-of-9 CoT sampling, per-slot mask |
| `src/thought_train_step_r.py` | R-mode train step with per-slot heterogeneous targets |
| `src/modules/parallel_thought.py` | `IntraGroupBlock`, `InterGroupBlock`, `ELF_PT` |
| `src/modules/layers.py` | `Attention`, `SwiGLUFFN`, `RMSNorm`, `RoPE` |
| `src/utils/thought_mask_utils.py` | `build_thought_masks_with_answer` (causal inter) |
| `src/utils/thought_sampling_utils.py` | LaDiR-style RBF repulsion for inference |
| `src/configs/training_configs/train_gsm8k_cot_vae.yml` | Main training config |
| `Dockerfile` / `Dockerfile.cotgen` | Two images: JAX-CUDA for training, PyTorch-GPU for LLM gen |

---

## Hyperparameter quick reference (default config)

| Parameter | Value |
|---|---|
| Model | ELF-PT-B (105M params) |
| Hidden size | 768 |
| Transformer layers | 12 (6 intra + 6 inter, alternating) |
| Heads | 12 |
| `K_reasoning` | 2 |
| `K_total` (slots) | 3 |
| Slot length L | 512 |
| Total sequence length | 1,536 + 12 prefix = 1,548 |
| VAE `mem_size` | 3 |
| VAE `mean_compression_rate` | 8 |
| VAE `S_max` | 8 segments |
| Num CoT candidates per row | 9 (1 gold + 8 LLM) |
| Diversity loss weight (training) | 0.0 (inference-only) |
| Diversity RBF γ_max (inference) | 0.1 |
| Diversity RBF σ | 1.0 |
| Causal inter-mask | enabled (reasoning ↛ answer) |
| Inter-block zero-init | enabled |
| Batch size | 8 |
| Learning rate | 3.0 × 10⁻⁴ (Muon, 500-step warmup) |
| Epochs | 100 |
| Steps per epoch | 934 |
| Total steps | 93,400 |
| Denoiser/decoder branch split | 80% / 20% |
| Self-cond (R-mode) | off |
| Optimizer | Muon |
| EMA decay | 0.9999 |
