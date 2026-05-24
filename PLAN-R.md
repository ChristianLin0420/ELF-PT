# ELF-PT-R Implementation Plan (Reasoning + Answer with Causal Flow)

> Follow-up plan to PLAN.md. Implements asymmetric "K reasoning + 1 answer" latent thought structure with causal inter-block attention and a diversity loss to prevent reasoning collapse.

**Goal:** Extend ELF (and reuse most of ELF-PT) with K latent *reasoning* thoughts plus 1 dedicated *answer* thought. Reasoning thoughts are pure latents (no MSE target); they're shaped only by gradient from the answer + a diversity loss that forces them to specialize.

**Architecture summary:**

```
   Reasoning thoughts                Answer thought
   ┌────────┬────────┬────────┐      ┌────────┐
   │  R1    │  R2    │  R3    │      │   A    │
   │ ε_r1   │ ε_r2   │ ε_r3   │      │  ε_a   │
   │ t_r1   │ t_r2   │ t_r3   │      │  t_a   │
   └────────┴────────┴────────┘      └────────┘
        │       │       │                ▲
        └───────┴───────┴────────────────┘
            answer attends to reasoning
            (causal: reasoning ↛ answer)

   Loss: L2(answer only) + CE(answer only) + λ_div · L_div(reasoning)
   Inference: decode only the answer slot.
```

**Design decisions locked in (per user 2026-05-24):**
1. **No MSE on reasoning thoughts.** They're pure latents; only gradient comes through the answer thought + diversity loss.
2. **Causal inter mask.** Answer attends to all reasoning thoughts; reasoning thoughts attend among themselves and to cond, but NOT to the answer. One-way information flow → CoT analog.
3. **Decoder branch unchanged from ELF.** 80% L2 (random t), 20% CE (separate decoder_z at t≈1). L2 only on answer; CE only on answer.
4. **Diversity loss on reasoning thoughts** to prevent collapse (without it, with no MSE target, reasoning thoughts converge to whatever single representation maximizes the answer's likelihood).

**Reuse from PLAN.md:** Most infrastructure carries over — intra/inter blocks (Task 3), ELF_PT model (Task 4 with edits), K-thought train step (Task 6 with edits), K-thought sampler (Task 7 with edits). Aggregator (Task 5) is **dropped** — no aggregation needed since answer thought is decoded directly.

**Tech stack:** Same as PLAN.md. Docker `elf-pt:smoke`, 1× A100 80GB.

---

## Phase 0 — Prerequisites (already done via PLAN.md)

- Dockerfile, Plan, smoke test, config fields, mask builder, blocks, model, aggregators, train step, sampler, eval metrics, K=2 WMT14 baseline run.

---

## Phase 1 — Foundation extensions

### Task R1: Config fields for reasoning mode

**Files:** Modify `src/configs/config.py`.

- [ ] **Step R1.1** Add to `Config` near the existing parallel-thought fields:

```python
# Parallel-thought reasoning extension (ELF-PT-R)
num_reasoning_thoughts: int = 0          # K reasoning thoughts; 0 disables R mode and falls back to ELF-PT
use_causal_inter_mask: bool = False      # if True, answer attends reasoning but reasoning ↛ answer
diversity_loss_weight: float = 0.01      # λ_div for pairwise cosine penalty on reasoning slots
diversity_loss_t_gating: bool = True     # if True, weight diversity loss by mean(t)*(1-mean(t))
```

The combined "total slots" K_total at runtime is `num_reasoning_thoughts + 1` when `num_reasoning_thoughts > 0`. The existing `num_thoughts` field still routes to symmetric ELF-PT and is mutually exclusive with `num_reasoning_thoughts > 0`.

- [ ] **Step R1.2** Verify import:
  ```bash
  sudo docker run --rm -v /localhome/local-chrislin/ELF-PT:/workspace -e PYTHONPATH=/workspace/src elf-pt:smoke \
    python -c "from configs.config import Config; c = Config(); assert c.num_reasoning_thoughts == 0; print('OK')"
  ```
- [ ] **Step R1.3** Commit: `feat(config): add ELF-PT-R reasoning + answer fields`

### Task R2: Causal mask builder

**Files:** Modify `src/utils/thought_mask_utils.py`, add tests to `tests/test_thought_mask_utils.py`.

The current `build_thought_masks(is_cond, is_valid, K)` produces symmetric intra/inter masks. Add a new function:

```python
def build_thought_masks_with_answer(is_cond, is_valid, K_reasoning, xp=jnp):
    """Build intra/inter masks for K reasoning + 1 answer layout.

    Sequence is laid out as: [reasoning_1 | reasoning_2 | ... | reasoning_K | answer]
    Each block is L tokens long → total sequence length is (K+1)*L.

    Intra mask: each thought attends only to itself + ALL cond keys across slots.
    Inter mask (CAUSAL):
      - Answer attends to all reasoning thoughts AND itself
      - Reasoning thoughts attend among themselves and to cond
      - Reasoning thoughts do NOT attend to the answer

    Returns: intra_mask (B, (K+1)*L, (K+1)*L), inter_mask same shape.
    """
```

- [ ] **Step R2.1** Write failing tests for the causal property:

```python
def test_causal_inter_mask_answer_attends_reasoning():
    is_cond = jnp.zeros((1, 4), dtype=jnp.bool_)
    is_valid = jnp.ones((1, 4), dtype=jnp.bool_)
    K_r = 2
    intra, inter = build_thought_masks_with_answer(is_cond, is_valid, K_reasoning=K_r)
    L = 4
    # Sequence layout: [R1 | R2 | Answer], each of length 4 → total 12.
    # Answer slot indices: 8..11. Reasoning slot indices: 0..7.
    # Answer queries (rows 8..11) attending reasoning keys (cols 0..7): allowed
    assert bool(inter[0, 8, 0])
    assert bool(inter[0, 11, 7])
    # Reasoning queries (rows 0..7) attending answer keys (cols 8..11): FORBIDDEN
    assert not bool(inter[0, 0, 8])
    assert not bool(inter[0, 7, 11])
    # Reasoning ↔ reasoning is allowed
    assert bool(inter[0, 0, 4])  # R1 sees R2
    assert bool(inter[0, 4, 0])  # R2 sees R1


def test_intra_mask_within_each_slot_unchanged():
    is_cond = jnp.zeros((1, 4), dtype=jnp.bool_)
    is_valid = jnp.ones((1, 4), dtype=jnp.bool_)
    intra, _ = build_thought_masks_with_answer(is_cond, is_valid, K_reasoning=2)
    # Within slot 0 (rows 0..3, cols 0..3): allowed
    assert bool(intra[0, 0, 1])
    # Across slots (row 0 in R1 to col 4 in R2): forbidden (intra-isolation)
    assert not bool(intra[0, 0, 4])
```

- [ ] **Step R2.2** Implement:

```python
def build_thought_masks_with_answer(is_cond, is_valid, K_reasoning, xp=jnp):
    """K reasoning + 1 answer; causal: reasoning ↛ answer."""
    B, L = is_cond.shape
    K_total = K_reasoning + 1
    # Reuse the symmetric builder for the intra part
    intra_sym, inter_sym = build_thought_masks(is_cond, is_valid, K=K_total, xp=xp)
    # Causal modification of inter: zero out (reasoning_query, answer_key) entries
    # Answer slot is the LAST slot, spanning rows/cols [K_reasoning*L : (K_reasoning+1)*L].
    ans_start = K_reasoning * L
    ans_end = (K_reasoning + 1) * L
    # Build a mask that's True everywhere except the (reasoning rows, answer cols) block
    rows = xp.arange(K_total * L)
    cols = xp.arange(K_total * L)
    is_reasoning_query = (rows < ans_start)[None, :, None]   # (1, K*L+L, 1)
    is_answer_key = ((cols >= ans_start) & (cols < ans_end))[None, None, :]
    forbidden = is_reasoning_query & is_answer_key
    inter_causal = inter_sym & (~forbidden)
    return intra_sym, inter_causal
```

- [ ] **Step R2.3** Run tests; expect all pass.
- [ ] **Step R2.4** Commit: `feat(masks): causal mask for K reasoning + 1 answer`

## Phase 2 — Architecture extension

### Task R3: ELF_PT_R model — drop aggregator, expose answer slot

**Files:** Modify `src/modules/parallel_thought.py`.

Don't create a new model class; **extend `ELF_PT`** with a new mode that activates when `num_reasoning_thoughts > 0`:

- [ ] **Step R3.1** Add a new Flax field on `ELF_PT`:

```python
num_reasoning_thoughts: int = 0
```

When `num_reasoning_thoughts > 0`:
- `num_thoughts` (internal sequence-replication count) should equal `num_reasoning_thoughts + 1`
- `aggregation` is ignored (no aggregator runs)
- `return_pre_unembed=True` returns ONLY the answer slot: `(B, L, hidden_size)` instead of `(B, K*L, hidden_size)` aggregated.

The cleanest place to slice is at the end of `__call__` (lines around 158-185 of the current `parallel_thought.py`):

```python
# x has shape (B, K_total * L, hidden_size) after block stack + prefix strip
if self.num_reasoning_thoughts > 0:
    K_total = self.num_reasoning_thoughts + 1
    L = x.shape[1] // K_total
    if return_pre_unembed:
        # Slice the answer slot (last L positions)
        return x[:, -L:, :], None
    # Denoiser branch: also return only the answer slot's velocity
    # (FinalLayer is applied to the whole sequence then sliced)
    output = FinalLayer(self.hidden_size, patch_size, self.text_encoder_dim, name='final_layer')(x)
    output = output[:, -L:, :]   # (B, L, text_encoder_dim) — only the answer slot
    return output, None
else:
    # Existing K-symmetric path (aggregator runs internally if return_pre_unembed)
    ...  # unchanged
```

- [ ] **Step R3.2** Add a test `tests/test_parallel_thought_model.py`:

```python
def test_elf_pt_r_returns_answer_slot_only():
    cls = ELF_PT_models['ELF-PT-B']
    K_r = 2
    K_total = K_r + 1
    L = 32
    model = cls(text_encoder_dim=512, max_length=L, num_thoughts=K_total,
                num_reasoning_thoughts=K_r, block_pattern="intra,inter",
                vocab_size=32100)
    B = 1
    x = jnp.ones((B, K_total * L, 512))
    t = jnp.ones((B,))
    mask = jnp.ones((B, K_total * L, K_total * L), dtype=jnp.bool_)
    rng = jax.random.PRNGKey(0)
    params = model.init(rng, x, t, intra_mask=mask, inter_mask=mask)
    out, _ = model.apply(params, x, t, intra_mask=mask, inter_mask=mask)
    # In R-mode, output is (B, L, text_encoder_dim) — only the answer slot
    assert out.shape == (B, L, 512)
```

- [ ] **Step R3.3** Run tests, commit: `feat(arch): ELF_PT-R answer-slot output`

## Phase 3 — Diversity loss

### Task R4: Diversity loss helper

**Files:** Modify `src/utils/diversity_metrics.py` (add training-time loss, not just eval metric), tests in `tests/test_diversity_metrics.py`.

The existing `pairwise_thought_diversity(x)` is a (positive) diversity *metric* — we want to *maximize* diversity. The corresponding *loss* is its negative, but to keep loss scales bounded we use squared cosine similarity averaged over pairs, which is directly minimized.

- [ ] **Step R4.1** Add to `src/utils/diversity_metrics.py`:

```python
def reasoning_diversity_loss(hidden, K_reasoning, t_gating=None):
    """Pairwise squared cosine similarity loss on K reasoning slots.

    Args:
        hidden: (B, K_total, L, D) where K_total = K_reasoning + 1.
                The last slot (index K_reasoning) is the answer and is excluded.
        K_reasoning: number of reasoning slots to apply diversity to.
        t_gating: optional (B,) time tensor — loss is multiplied by mean(t)*(1-mean(t)).
                  Use the reasoning thoughts' time average. If None, no gating.

    Returns:
        Scalar loss to be minimized. Bounded in [0, 1].
    """
    eps = 1e-8
    reasoning = hidden[:, :K_reasoning]                       # (B, K_r, L, D)
    norm = reasoning / (jnp.linalg.norm(reasoning, axis=-1, keepdims=True) + eps)
    # Pairwise cosine sim across the K_r dimension
    sim = jnp.einsum('bkld,bjld->bkjl', norm, norm)            # (B, K_r, K_r, L)
    # Mask out diagonal
    iu = jnp.triu_indices(K_reasoning, k=1)
    pair_sim = sim[:, iu[0], iu[1], :]                         # (B, P, L), P=K_r(K_r-1)/2
    loss = (pair_sim ** 2).mean()                              # scalar in [0, 1]
    if t_gating is not None:
        gate = (t_gating * (1.0 - t_gating)).mean() * 4.0      # normalize peak at 0.5 → 1.0
        loss = loss * gate
    return loss
```

- [ ] **Step R4.2** Tests:

```python
def test_reasoning_diversity_loss_zero_when_orthogonal():
    # Two thoughts that are unit vectors in orthogonal directions
    B, K_r, L, D = 1, 2, 1, 4
    h_r1 = jnp.array([[1, 0, 0, 0]], dtype=jnp.float32)[None, :, :, None].transpose(0,3,1,2)
    # Easier: build manually as (B, K_total=3, L, D), reasoning slots are first 2
    r1 = jnp.zeros((1, 1, 1, 4)).at[..., 0].set(1.0)
    r2 = jnp.zeros((1, 1, 1, 4)).at[..., 1].set(1.0)
    answer = jnp.zeros((1, 1, 1, 4))   # ignored
    h = jnp.concatenate([r1, r2, answer], axis=1)
    loss = reasoning_diversity_loss(h, K_reasoning=2)
    assert float(loss) < 1e-6


def test_reasoning_diversity_loss_max_when_identical():
    # Two identical reasoning thoughts → cosine sim = 1 → loss = 1
    r1 = jnp.ones((1, 1, 1, 4))
    r2 = jnp.ones((1, 1, 1, 4))
    answer = jnp.zeros((1, 1, 1, 4))
    h = jnp.concatenate([r1, r2, answer], axis=1)
    loss = reasoning_diversity_loss(h, K_reasoning=2)
    assert abs(float(loss) - 1.0) < 1e-5


def test_reasoning_diversity_loss_t_gating_peak_at_half():
    r1 = jnp.ones((2, 1, 1, 4))
    r2 = jnp.ones((2, 1, 1, 4))   # identical → high loss
    answer = jnp.zeros((2, 1, 1, 4))
    h = jnp.concatenate([r1, r2, answer], axis=1)
    # t=0.5 → gate ≈ 1.0; t=0.0 → gate=0
    loss_mid = reasoning_diversity_loss(h, K_reasoning=2, t_gating=jnp.array([0.5, 0.5]))
    loss_edge = reasoning_diversity_loss(h, K_reasoning=2, t_gating=jnp.array([0.0, 0.0]))
    assert float(loss_mid) > float(loss_edge)
```

- [ ] **Step R4.3** Run tests; commit: `feat(eval): reasoning diversity training loss`

## Phase 4 — Training

### Task R5: K-reasoning train step

**Files:** Create `src/thought_train_step_r.py` (forked from `src/thought_train_step.py`), modify `src/train.py` routing.

The forked step changes:

1. `K = num_reasoning_thoughts + 1` instead of `K = num_thoughts`.
2. Build masks via `build_thought_masks_with_answer` (causal) instead of `build_thought_masks`.
3. **Denoiser branch:** sample per-slot noise/time as before. Compute v_target_per as before. **But the loss only uses the ANSWER slot** — slice `v_pred_per[:, -1]` and `v_target_per[:, -1]`, compute MSE on these only.
4. **Decoder branch:** decoder_z is K_total replicated. After the model forward with `return_pre_unembed=True`, the model now returns only the answer slot's hidden states `(B, L, hidden_size)` — apply the shared unembed kernel directly (no aggregator).
5. **Diversity loss:** in the denoiser branch only, after the model forward, capture the post-block hidden states for the reasoning slots and compute `reasoning_diversity_loss(...)`. Add `λ_div * L_div` to the total loss.

The hidden states needed for diversity: pre-FinalLayer, after the prefix is stripped, shape `(B, K_total * L, hidden_size)`. The current `ELF_PT.__call__` does not expose this in the denoiser path — only in `return_pre_unembed=True` (which is the decoder path). **Add a second flag** to expose intermediates via `self.sow('intermediates', 'hidden_pre_final', x)` in the denoiser path too, then the train step extracts it via `apply(..., capture_intermediates=True, mutable=['intermediates'])`.

- [ ] **Step R5.1** Modify `src/modules/parallel_thought.py:ELF_PT.__call__` to sow `hidden_pre_final` always (cheap; just a sow call, no compute change).
- [ ] **Step R5.2** Fork `thought_train_step.py` → `thought_train_step_r.py`. Apply the 5 mutations above.
- [ ] **Step R5.3** Modify `src/train.py` routing: when `config.num_reasoning_thoughts > 0`, import `thought_train_step_r.train_step`; else fall back to existing logic.

Routing order in `train.py`:

```python
if config.num_reasoning_thoughts > 0:
    from thought_train_step_r import train_step
elif config.num_thoughts > 1:
    from thought_train_step import train_step
else:
    from train_step import train_step
```

- [ ] **Step R5.4** Add a smoke test `tests/test_thought_train_step_r.py`:

```python
def test_thought_train_step_r_importable():
    from thought_train_step_r import train_step
    assert callable(train_step)


def test_train_step_r_routing():
    """num_reasoning_thoughts > 0 must route to thought_train_step_r."""
    import importlib
    sym_mod = importlib.import_module('thought_train_step')
    r_mod = importlib.import_module('thought_train_step_r')
    assert sym_mod.train_step is not r_mod.train_step
```

Plus a real K_r=2 forward-pass test that runs one model.apply through the new train logic and asserts: (a) finite loss, (b) diversity loss term is non-zero, (c) all gradients exist.

- [ ] **Step R5.5** Run tests; commit: `feat(train): K-reasoning + 1-answer train step with diversity loss`

## Phase 5 — Sampling

### Task R6: K-reasoning sampler

**Files:** Modify `src/utils/thought_sampling_utils.py` to handle the R variant.

Two functions need updates:
- `init_thought_state(rng, B, K, L, D)` — currently uses K-way split; for R-mode, K = K_total (reasoning + answer). No change needed; just call with K=K_total.
- `thought_final_decode(state, z_kl, intra_mask, inter_mask, ...)` — currently the model aggregates K thoughts internally and returns `(B, L, H)` aggregated. For R-mode, the model returns `(B, L, H)` already (just the answer slot). Same downstream code works.

The only architecturally novel piece for R-mode sampling is **building the causal inter mask at inference**. Update `_build_thought_masks_batch` in `src/utils/generation_utils.py`:

```python
def _build_thought_masks_batch(cond_seq_mask, attention_mask, config):
    is_cond = cond_seq_mask.astype(jnp.bool_)
    is_valid = attention_mask.astype(jnp.bool_)
    if config.num_reasoning_thoughts > 0:
        from utils.thought_mask_utils import build_thought_masks_with_answer
        intra, inter = build_thought_masks_with_answer(
            is_cond, is_valid, K_reasoning=config.num_reasoning_thoughts,
        )
    else:
        from utils.thought_mask_utils import build_thought_masks
        K_total = config.num_thoughts
        intra, inter = build_thought_masks(is_cond, is_valid, K=K_total)
    return intra, inter
```

- [ ] **Step R6.1** Apply the mask builder dispatch.
- [ ] **Step R6.2** Add a smoke test (extend `tests/test_thought_sampling.py`): generate from K_r=2+1 with random params, assert output token shape and finite.
- [ ] **Step R6.3** Commit: `feat(sampling): K-reasoning + 1-answer sampler`

## Phase 6 — Feasibility test

### Task R7: K_r=2 WMT14 feasibility run

**Files:** Create `src/configs/training_configs/train_de-en_ELF-PT-R-B.yml`.

- [ ] **Step R7.1** Config:

```yaml
data_path: "embedded-language-flows/wmt14_de-en_train_t5"
eval_data_path: "embedded-language-flows/wmt14_de-en_validation_t5"

encoder_model_name: t5-small
encoder_checkpoint: "embedded-language-flows/t5_small_encoder_jax/t5_small_encoder_jax.pkl"
latent_mean: 0.0
latent_std: 0.2

model: ELF-PT-B
num_thoughts: 3                       # K_total = K_reasoning + 1 = 2 + 1
num_reasoning_thoughts: 2
use_causal_inter_mask: true
diversity_loss_weight: 0.01
diversity_loss_t_gating: true
thought_block_pattern: "intra,inter"
thought_aggregation: mean              # ignored in R-mode but kept for fallback compatibility
inter_block_zero_init: true

bottleneck_dim: 128
num_time_tokens: 4
num_self_cond_cfg_tokens: 4
num_model_mode_tokens: 4
max_length: 128

denoiser_p_mean: -1.5
denoiser_p_std: 0.8
denoiser_noise_scale: 2.0
t_eps: 0.05
time_schedule: "logit_normal"

decoder_prob: 0.2
decoder_noise_scale: 5.0
decoder_p_mean: 0.8
decoder_p_std: 0.8
self_cond_prob: 0.5

# K_total=3 triples the sequence length over K=1; expect ~50 GB at batch 64.
# If memory permits push to batch 96-128 after first 200 steps.
epochs: 1
global_batch_size: 64
grad_accum_steps: 1
blr: 0.001
warmup_steps: 500
optimizer: muon

ema_decay1: 0.9999

sampling_configs_path: "configs/sampling_configs/cond_sampling_configs.yml"
num_samples: 8

log_freq: 50
save_freq: 0.1
eval_freq: 9999

output_dir: outputs/pt_r_k2_wmt14
online_eval: false
use_wandb: true
wandb_project: elf-pt
wandb_entity: crlc112358
wandb_run_name: pt_r_k2_wmt14_bs64
```

- [ ] **Step R7.2** Launch in a fresh tmux session (after the current PLAN.md K=2 run completes or is killed):

```bash
tmux new-session -d -s elf_pt_r "sudo docker run --rm --gpus all --name elf_pt_r_train \
  -v /localhome/local-chrislin/ELF-PT:/workspace \
  -v /localhome/local-chrislin/.cache/huggingface:/cache/hf \
  -e HF_HOME=/cache/hf \
  -e WANDB_API_KEY=<set_at_launch> \
  elf-pt:smoke bash -c 'cd /workspace/src && timeout 28800 python train.py --config configs/training_configs/train_de-en_ELF-PT-R-B.yml' 2>&1 | tee /localhome/local-chrislin/ELF-PT/outputs/pt_r_k2_wmt14.log"
```

- [ ] **Step R7.3** Success criteria — same as PLAN.md Task 9, plus:
  - **`L_div` (diversity loss) is finite and decreases** over the first 1000 steps (or stays bounded; not catastrophic increase).
  - **Per-thought reasoning representations differ** at step 500 vs step 0 — confirm via the `pairwise_thought_diversity` metric (should be >0; ideally >0.1).
  - If `L_div` stays at the same value of ~1.0 throughout training, the reasoning thoughts have collapsed → λ_div is too low. Increase to 0.1 and re-run.

- [ ] **Step R7.4** Commit config + run notes: `chore(smoke): K_r=2 WMT14 ELF-PT-R feasibility config`

## Phase 7 — Comparison ablations (post-feasibility)

After R7 passes, the comparison story has three points:

| Variant | Setup | Hypothesis |
|---|---|---|
| ELF (baseline) | K=1, no thoughts | Lower bound from PLAN.md compute-matched K=1 |
| ELF-PT (symmetric) | K=2 thoughts, mean aggregator | Test parallel-thought without role asymmetry |
| ELF-PT-R | K_r=2 reasoning + 1 answer, causal mask, diversity loss | Test asymmetric reasoning|

Ablations within ELF-PT-R:
- `K_reasoning ∈ {1, 2, 4}` — does more reasoning help, and where does it saturate?
- `diversity_loss_weight ∈ {0, 0.01, 0.1, 1.0}` — does diversity loss prevent collapse and improve BLEU? At weight=0, reasoning thoughts should collapse.
- `use_causal_inter_mask ∈ {True, False}` — does the causal flow help vs symmetric full attention?
- `diversity_loss_t_gating ∈ {True, False}` — is t-gating necessary?

Each ablation = one short (1-epoch) run, logged to `EXPERIMENTS.md` with config path, headline BLEU, diversity metric at convergence, training loss curve.

---

## Open question

The current ELF-PT (symmetric K=2) run finishes in ~6.5h at batch 96. ELF-PT-R adds 50% more sequence length (K_total=3 vs K_total=2) so I've conservatively set batch=64. Once the run starts and we see actual memory use, push batch higher.

## What this plan deliberately omits

- The eval.py integration for cond K-thought generation (still deferred from PLAN.md Task 7).
- The K=8 ablation at L=1024 (infeasible on A100 80GB).
- The compute-matched K=1 baseline retrain (will be added once ELF-PT-R is running and we want a clean three-way comparison).
