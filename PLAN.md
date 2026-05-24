# ELF-PT Implementation Plan

> **For agentic workers:** Use superpowers:subagent-driven-development. Each task below is implemented by a fresh subagent, then spec-reviewed and code-reviewed before moving on.

**Goal:** Extend ELF with K parallel latent "thought" streams that interleave intra-group attention (each thought attends only to itself + shared cond tokens) and inter-group attention (full cross-thought), trained end-to-end with no extra VAE.

**Architecture:** Each transformer block is either `IntraGroupBlock` (block-diagonal attention over K thought groups, with cond tokens visible to all) or `InterGroupBlock` (full attention across all K groups). Layers alternate `[intra, inter, intra, inter, ...]`. Inter blocks zero-init their output projection so they start as identity (no Stage-1 warmup needed). K thoughts share weights; the only thing that varies is the noise sample, time, and intra-block isolation.

**Tech Stack:** JAX 0.4.38, Flax 0.10.2, Optax 0.2.5, `elf-pt:smoke` Docker image (built at `/localhome/local-chrislin/ELF-PT/Dockerfile`), 1× A100 80GB.

**Design choices locked in:**

1. Sequence layout: K replicas of ELF's existing (B, L, D) latent. No separate Q region — cond tokens live in-sequence as ELF already encodes them. After replication: shape is (B, K·L, D) plus prefix tokens. The mask uses `(B, K·L, K·L)`.
2. Mask composition: the new K-group mask is **ANDed** with ELF's existing `encoder_attention_mask` — does not replace it. Label-drop / CFG behavior is preserved.
3. Attention layer: no modification needed. `scaled_dot_product_attention` already accepts `(B, L, S)` masks (`src/modules/layers.py:130-148`).
4. CE/decoder branch: each thought gets its own `decoder_z`; aggregate K *pre-unembed* embeddings before applying the shared unembed matrix. Trains the aggregator end-to-end.
5. Self-conditioning: per-thought x_pred is fed back to its own thought; no cross-thought sharing in self-cond. Preserves ELF's existing self-cond semantics.
6. Stability: zero-init the out-proj of inter blocks → they start as identity. Train jointly from step 0. No Stage-1 warmup.
7. Diversity loss: **dropped from training** (single shared x0 target makes per-step repulsion pathological). Diversity is purely inference-time, optional, LaDiR-style.
8. K=8 dropped from OWT ablations (infeasible at L=1024 on A100); kept for WMT14 (L=128).

**Standard run command (every subagent uses this):**

```bash
sudo docker run --rm --gpus all \
  -v /localhome/local-chrislin/ELF-PT:/workspace \
  -v /localhome/local-chrislin/.cache/huggingface:/cache/hf \
  -e HF_HOME=/cache/hf -e WANDB_MODE=disabled \
  elf-pt:smoke <command>
```

Working directory inside the container is `/workspace/src`. The repo is at `/localhome/local-chrislin/ELF-PT` and mounts read-write at `/workspace`.

For tests: `sudo docker run --rm --gpus all -v /localhome/local-chrislin/ELF-PT:/workspace -e PYTHONPATH=/workspace/src elf-pt:smoke pytest /workspace/tests/<file>.py -v`.

---

## Phase 0 — Environment (DONE)

- `Dockerfile` and `.dockerignore` at repo root.
- `elf-pt:smoke` image built.
- Smoke test passed: ELF-B HF checkpoint loads, SDE 32-step + 64-step sampling works on A100, generated text is fluent. Output at `outputs/smoke/`.

## Phase 1 — Foundation

### Task 1: Config fields for ELF-PT

**Files:** Modify `src/configs/config.py`.

- [ ] **Step 1.1** Add to `Config` class (anywhere among the existing fields, but grouped together):

```python
# Parallel-thought (ELF-PT)
num_thoughts: int = 1                          # K. 1 = vanilla ELF.
thought_block_pattern: str = "intra,inter"     # repeating unit; len(depth) must be divisible by len(pattern.split(','))
thought_aggregation: str = "mean"              # "mean" | "learned"
inter_block_zero_init: bool = True             # init out_proj/w3 to zero for stability
diversity_repulsion_inference: bool = False
diversity_repulsion_gamma_max: float = 0.5
diversity_repulsion_sigma: float = 1.0
```

- [ ] **Step 1.2** Verify import:
  ```bash
  sudo docker run --rm -v /localhome/local-chrislin/ELF-PT:/workspace -e PYTHONPATH=/workspace/src elf-pt:smoke \
    python -c "from configs.config import Config; c = Config(); assert c.num_thoughts == 1; print('OK')"
  ```
  Expected: `OK`.
- [ ] **Step 1.3** Commit: `feat(config): add ELF-PT thought fields`

### Task 2: Mask builder + tests

**Files:** Create `src/utils/thought_mask_utils.py`, create `tests/test_thought_mask_utils.py`.

- [ ] **Step 2.1** Write failing test at `tests/test_thought_mask_utils.py`:

```python
import jax.numpy as jnp
from utils.thought_mask_utils import build_thought_masks


def test_intra_blocks_cross_group_attention():
    is_cond = jnp.zeros((2, 4), dtype=jnp.bool_)
    is_valid = jnp.ones((2, 4), dtype=jnp.bool_)
    intra, inter = build_thought_masks(is_cond, is_valid, K=3)
    assert intra.shape == (2, 12, 12)
    assert inter.shape == (2, 12, 12)
    assert int(intra[0, 0, 4]) == 0
    assert int(intra[0, 0, 1]) == 1
    assert int(inter[0, 0, 4]) == 1


def test_cond_tokens_visible_to_all_groups_in_intra_mask():
    is_cond = jnp.array([[1, 1, 0, 0]], dtype=jnp.bool_)
    is_valid = jnp.ones((1, 4), dtype=jnp.bool_)
    intra, _ = build_thought_masks(is_cond, is_valid, K=2)
    assert int(intra[0, 0, 4]) == 1  # cond G0 -> cond G1
    assert int(intra[0, 2, 0]) == 1  # non-cond G0 -> cond G0
    assert int(intra[0, 2, 4]) == 1  # non-cond G0 -> cond G1 (cond is shared)
    assert int(intra[0, 2, 6]) == 0  # non-cond G0 -> non-cond G1 (different groups)


def test_padding_zeroed():
    is_cond = jnp.zeros((1, 4), dtype=jnp.bool_)
    is_valid = jnp.array([[1, 1, 0, 0]], dtype=jnp.bool_)
    _, inter = build_thought_masks(is_cond, is_valid, K=2)
    assert int(inter[0, 0, 2]) == 0
    assert int(inter[0, 0, 6]) == 0
```

- [ ] **Step 2.2** Run `sudo docker run --rm -v /localhome/local-chrislin/ELF-PT:/workspace -e PYTHONPATH=/workspace/src elf-pt:smoke pytest /workspace/tests/test_thought_mask_utils.py -v` — expect ImportError.
- [ ] **Step 2.3** Implement `src/utils/thought_mask_utils.py`:

```python
"""Mask builders for ELF-PT parallel-thought attention."""
import jax.numpy as jnp


def build_thought_masks(is_cond, is_valid, K, xp=jnp):
    """Build intra-group and inter-group attention masks.

    Inputs:
      is_cond:  (B, L) bool. True for condition tokens.
      is_valid: (B, L) bool. True for non-padding tokens.
      K:        number of thought groups.

    Returns:
      intra_mask: (B, K*L, K*L) int. 1 where attention is allowed.
        - Cond keys visible to all groups.
        - Non-cond keys only visible within their own group.
        - Padded queries/keys never attend.
      inter_mask: (B, K*L, K*L) int. 1 for all (valid_query, valid_key) pairs.
    """
    B, L = is_cond.shape
    is_cond_k = xp.tile(is_cond, (1, K))      # (B, K*L)
    is_valid_k = xp.tile(is_valid, (1, K))    # (B, K*L)

    group_id = xp.repeat(xp.arange(K), L)[None, :]
    group_id = xp.broadcast_to(group_id, (B, K * L))

    valid_pair = is_valid_k[:, :, None] & is_valid_k[:, None, :]
    inter_mask = valid_pair.astype(xp.int32)

    same_group = group_id[:, :, None] == group_id[:, None, :]
    key_is_cond = is_cond_k[:, None, :]
    allowed = same_group | key_is_cond
    intra_mask = (valid_pair & allowed).astype(xp.int32)
    return intra_mask, inter_mask
```

- [ ] **Step 2.4** Run the pytest command from Step 2.2 — expect all 3 tests pass.
- [ ] **Step 2.5** Commit: `feat(masks): K-group intra/inter attention masks`

## Phase 2 — Architecture

### Task 3: Intra/Inter blocks + zero-init plumbing

**Files:** Create `src/modules/parallel_thought.py`, modify `src/modules/layers.py` (thread `out_kernel_init`), create `tests/test_parallel_thought_blocks.py`.

- [ ] **Step 3.1** Write failing test at `tests/test_parallel_thought_blocks.py`:

```python
import jax, jax.numpy as jnp
from modules.parallel_thought import IntraGroupBlock, InterGroupBlock


def test_inter_block_zero_init_is_identity():
    block = InterGroupBlock(hidden_size=64, num_heads=4, zero_init_out=True)
    rng = jax.random.PRNGKey(0)
    x = jax.random.normal(rng, (2, 16, 64))
    mask = jnp.ones((2, 16, 16), dtype=jnp.int32)
    params = block.init(rng, x, attention_mask=mask)
    y = block.apply(params, x, attention_mask=mask)
    assert jnp.allclose(y, x, atol=1e-5)


def test_intra_block_shape_preserved():
    block = IntraGroupBlock(hidden_size=64, num_heads=4)
    rng = jax.random.PRNGKey(0)
    x = jax.random.normal(rng, (2, 16, 64))
    mask = jnp.ones((2, 16, 16), dtype=jnp.int32)
    params = block.init(rng, x, attention_mask=mask)
    y = block.apply(params, x, attention_mask=mask)
    assert y.shape == x.shape
```

- [ ] **Step 3.2** Run pytest — expect ImportError.
- [ ] **Step 3.3** Modify `src/modules/layers.py`:
  - Add an `out_kernel_init` class field to `Attention` (default `DEFAULT_KERNEL_INIT`), and use it in the `proj` Dense at the end of `__call__`.
  - Add an `out_kernel_init` class field to `SwiGLUFFN` (default `DEFAULT_KERNEL_INIT`), and use it in the `w3` Dense.
- [ ] **Step 3.4** Implement `src/modules/parallel_thought.py`:

```python
import flax.linen as nn
from modules.layers import Attention, RMSNorm, SwiGLUFFN


class IntraGroupBlock(nn.Module):
    hidden_size: int
    num_heads: int
    mlp_ratio: float = 4.0
    attn_drop: float = 0.0
    proj_drop: float = 0.0

    @nn.compact
    def __call__(self, x, rope_fn=None, attention_mask=None, deterministic=True):
        mlp_hidden = int(self.hidden_size * self.mlp_ratio)
        x = x + Attention(
            self.hidden_size, self.num_heads, qkv_bias=True, qk_norm=True,
            attn_drop=self.attn_drop, proj_drop=self.proj_drop, name='attn',
        )(RMSNorm(self.hidden_size, eps=1e-6, name='norm1')(x), rope_fn,
          attention_mask=attention_mask, deterministic=deterministic)
        x = x + SwiGLUFFN(self.hidden_size, mlp_hidden, drop=self.proj_drop, name='mlp')(
            RMSNorm(self.hidden_size, eps=1e-6, name='norm2')(x), deterministic=deterministic,
        )
        return x


class InterGroupBlock(nn.Module):
    hidden_size: int
    num_heads: int
    mlp_ratio: float = 4.0
    attn_drop: float = 0.0
    proj_drop: float = 0.0
    zero_init_out: bool = True

    @nn.compact
    def __call__(self, x, rope_fn=None, attention_mask=None, deterministic=True):
        mlp_hidden = int(self.hidden_size * self.mlp_ratio)
        out_init = nn.initializers.zeros if self.zero_init_out else nn.initializers.xavier_uniform()
        x = x + Attention(
            self.hidden_size, self.num_heads, qkv_bias=True, qk_norm=True,
            attn_drop=self.attn_drop, proj_drop=self.proj_drop, name='attn',
            out_kernel_init=out_init,
        )(RMSNorm(self.hidden_size, eps=1e-6, name='norm1')(x), rope_fn,
          attention_mask=attention_mask, deterministic=deterministic)
        x = x + SwiGLUFFN(
            self.hidden_size, mlp_hidden, drop=self.proj_drop, name='mlp',
            out_kernel_init=out_init,
        )(RMSNorm(self.hidden_size, eps=1e-6, name='norm2')(x),
          deterministic=deterministic)
        return x
```

- [ ] **Step 3.5** Run pytest — expect both pass.
- [ ] **Step 3.6** Commit: `feat(arch): IntraGroupBlock + InterGroupBlock with zero-init`

### Task 4: ELF_PT model + factory

**Files:** Modify `src/modules/parallel_thought.py` (add `ELF_PT`, factories), create `tests/test_parallel_thought_model.py`.

- [ ] **Step 4.1** Write failing test:

```python
import jax, jax.numpy as jnp
from modules.parallel_thought import ELF_PT_models


def test_elf_pt_b_forward_pass_shape():
    cls = ELF_PT_models['ELF-PT-B']
    model = cls(text_encoder_dim=512, max_length=128, num_thoughts=2,
                block_pattern="intra,inter", vocab_size=32100)
    B, L_per, K = 1, 128, 2
    x = jnp.ones((B, K * L_per, 512))
    t = jnp.ones((B,))
    intra = jnp.ones((B, K * L_per, K * L_per), dtype=jnp.int32)
    inter = jnp.ones((B, K * L_per, K * L_per), dtype=jnp.int32)
    rng = jax.random.PRNGKey(0)
    params = model.init(rng, x, t, intra_mask=intra, inter_mask=inter)
    out, _ = model.apply(params, x, t, intra_mask=intra, inter_mask=inter)
    assert out.shape == (B, K * L_per, 512)
```

- [ ] **Step 4.2** Run pytest — ImportError.
- [ ] **Step 4.3** Implement `ELF_PT`. Approach: subclass `ELF` from `src/modules/model.py` and override `__call__` to (a) accept `intra_mask` and `inter_mask` separately, (b) at each block layer, pick `IntraGroupBlock` or `InterGroupBlock` per `block_pattern`, and (c) pass the corresponding mask (extended with prefix-token mask, same as ELF does for `attention_mask`).

Key constraint: the prefix tokens (time/cfg/mode, ~12 tokens) must be visible to ALL positions in both masks. Extend the mask by prepending a `(prefix_len, K*L + prefix_len)` block of ones to each row, and a matching column block.

```python
from modules.model import ELF
# Add at end of parallel_thought.py:

class ELF_PT(ELF):
    num_thoughts: int = 1
    block_pattern: str = "intra,inter"

    def _select_block_cls(self, i):
        pat = self.block_pattern.split(',')
        return IntraGroupBlock if pat[i % len(pat)].strip() == 'intra' else InterGroupBlock

    # __call__ override:
    # - Reuse ELF's projection / prefix-token construction.
    # - In the block loop, select block class via _select_block_cls(i).
    # - Pass intra_mask or inter_mask (with prefix extension) as `attention_mask`
    #   depending on block kind.
    # - The rest (final_layer, unembed) is unchanged.


def ELF_PT_B(**kw): return ELF_PT(depth=12, hidden_size=768,  num_heads=12, **kw)
def ELF_PT_M(**kw): return ELF_PT(depth=24, hidden_size=1056, num_heads=16, **kw)
def ELF_PT_L(**kw): return ELF_PT(depth=32, hidden_size=1280, num_heads=16, **kw)

ELF_PT_models = {'ELF-PT-B': ELF_PT_B, 'ELF-PT-M': ELF_PT_M, 'ELF-PT-L': ELF_PT_L}
```

The implementer must read `src/modules/model.py` lines 75-157 and replicate the prefix/projection logic, varying only the block loop.

- [ ] **Step 4.4** Run pytest — expect pass.
- [ ] **Step 4.5** Commit: `feat(arch): ELF_PT model with alternating block types`

## Phase 3 — Aggregation

### Task 5: Aggregators

**Files:** Create `src/modules/thought_aggregation.py`, create `tests/test_thought_aggregation.py`.

- [ ] **Step 5.1** Write tests:

```python
import jax, jax.numpy as jnp
from modules.thought_aggregation import (
    MeanPoolAggregator, LearnedWeightAggregator, get_aggregator,
)


def test_mean_pool_averages():
    agg = MeanPoolAggregator()
    # x: (B=2, K=4, L=8, D=16); thought k filled with value k
    x = jnp.stack([jnp.full((2, 8, 16), float(k)) for k in range(4)], axis=1)
    params = agg.init(jax.random.PRNGKey(0), x)
    out = agg.apply(params, x)
    assert out.shape == (2, 8, 16)
    assert jnp.allclose(out, jnp.ones_like(out) * 1.5)


def test_learned_weight_shape():
    agg = LearnedWeightAggregator(hidden_dim=32)
    x = jax.random.normal(jax.random.PRNGKey(0), (2, 4, 8, 16))
    params = agg.init(jax.random.PRNGKey(1), x)
    out = agg.apply(params, x)
    assert out.shape == (2, 8, 16)


def test_factory_returns_mean_by_default():
    class _C: thought_aggregation = "mean"
    assert isinstance(get_aggregator(_C()), MeanPoolAggregator)
```

- [ ] **Step 5.2** Run pytest — ImportError.
- [ ] **Step 5.3** Implement:

```python
import jax.numpy as jnp
import flax.linen as nn


class MeanPoolAggregator(nn.Module):
    @nn.compact
    def __call__(self, x):
        # x: (B, K, L, D)
        return x.mean(axis=1)


class LearnedWeightAggregator(nn.Module):
    hidden_dim: int = 64

    @nn.compact
    def __call__(self, x):
        B, K, L, D = x.shape
        summary = jnp.concatenate([x.mean(1), x.std(1)], axis=-1)   # (B, L, 2D)
        h = nn.gelu(nn.Dense(self.hidden_dim)(summary))
        logits = nn.Dense(K)(h)                                     # (B, L, K)
        w = nn.softmax(logits, axis=-1)[..., None]                  # (B, L, K, 1)
        x_blkd = x.transpose(0, 2, 1, 3)                            # (B, L, K, D)
        return (w * x_blkd).sum(axis=2)


def get_aggregator(config):
    kind = getattr(config, 'thought_aggregation', 'mean')
    if kind == 'mean':    return MeanPoolAggregator()
    if kind == 'learned': return LearnedWeightAggregator()
    raise ValueError(f"unknown aggregator: {kind}")
```

- [ ] **Step 5.4** Run pytest — expect pass.
- [ ] **Step 5.5** Commit: `feat(agg): mean + learned thought aggregators`

## Phase 4 — Training

### Task 6: K-thought training step

**Files:** Create `src/thought_train_step.py` (forked from `src/train_step.py`), modify `src/train.py` (router), create `tests/test_thought_train_step.py` (smoke test).

The implementer must (a) read `src/train_step.py` in full, (b) understand the four mutation points listed below, (c) preserve ALL of ELF's behavior at K=1, (d) extend cleanly for K>1.

**Four mutation points in train_step.py:**
- L65-71: time + noise sampling — now produces K independent samples
- L81: `denoiser_z = add_noise(...)` — now K versions, stacked as (B, K, L, D)
- L93-99: `decoder_z` — same, K versions
- L120-136 (`get_z_input`) and L143-162 (`get_sc_cond_and_uncond`): self-cond — per-thought
- L189-198 (`_decoder_branch`): aggregate K *pre-unembed* hidden states before applying the shared unembed

**Important:** `ELF_PT.__call__` needs to expose the pre-unembed hidden state separately from the final output, so the aggregator can run on `(B, K, L, D)` and then the shared unembed kernel is applied to the aggregated `(B, L, D)`. The implementer may need to extend `ELF_PT.__call__` with a flag like `return_pre_unembed=True` for the decoder branch.

- [ ] **Step 6.1** Write smoke test `tests/test_thought_train_step.py`:

```python
import jax, jax.numpy as jnp
from configs.config import Config
# Smoke test: one train step on CPU-shaped batch, verify loss finite + grads nonzero.
# Use num_thoughts=2, depth=2, hidden_size=64, max_length=8, batch=2.
# Implementation deferred; placeholder asserts that thought_train_step.train_step is importable.

def test_thought_train_step_importable():
    from thought_train_step import train_step
    assert callable(train_step)
```

(A fuller numeric smoke test is impractical without standing up the full train state; the empirical validation is Task 9.)

- [ ] **Step 6.2** Run pytest — ImportError.
- [ ] **Step 6.3** Implement `src/thought_train_step.py`. Start by copying `src/train_step.py` then apply mutations. Key code patterns the implementer must use:

```python
K = config.num_thoughts
B, L, D = x0.shape

# K independent noises + times
noise_keys = jax.random.split(noise_rng, K)
t_keys = jax.random.split(t_rng, K)
noise_k = jnp.stack([jax.random.normal(noise_keys[k], x0.shape) for k in range(K)], axis=1)
t_k = jnp.stack([sample_timesteps(t_keys[k], B, P_mean=config.denoiser_p_mean,
                                  P_std=config.denoiser_p_std,
                                  time_schedule=config.time_schedule)
                 for k in range(K)], axis=1)  # (B, K)

# x0 broadcast and noise interpolation, per-thought
x0_k = jnp.broadcast_to(x0[:, None], (B, K, L, D))
t_exp = t_k[:, :, None, None]
denoiser_z_k = t_exp * x0_k + (1 - t_exp) * noise_k  # (B, K, L, D)
v_target_k = (x0_k - denoiser_z_k) / jnp.maximum(1 - t_exp, t_eps)
denoiser_z = denoiser_z_k.reshape(B, K * L, D)

# Masks
from utils.thought_mask_utils import build_thought_masks
intra_mask, inter_mask = build_thought_masks(
    batch["cond_seq_mask"].astype(jnp.bool_),
    batch["attention_mask"].astype(jnp.bool_),
    K=K,
)
# Block-tile the existing encoder_attention_mask across K groups and AND it in:
# encoder_attention_mask has shape (B, L, L). After tiling to (B, K*L, K*L), do:
enc_mask_k = jnp.tile(encoder_attention_mask, (1, K, K))
intra_mask = intra_mask * enc_mask_k
inter_mask = inter_mask * enc_mask_k

# Forward (denoiser branch)
net_out_flat, _ = state.apply_fn(
    {"params": params}, denoiser_z, t_k.mean(axis=1),
    intra_mask=intra_mask, inter_mask=inter_mask,
    deterministic=False, rngs={"dropout": model_dropout_rng},
    self_cond_cfg_scale=self_cond_cfg_scale,
    decoder_step_active=jnp.array(False),
)
net_out_k = net_out_flat.reshape(B, K, L, D)

# Per-thought velocity loss; AVERAGE over K (not sum) so LR transfers from K=1
per_thought_mse = ((net_out_k - v_target_k) ** 2).mean(axis=-1)  # (B, K, L)
loss_mask_k = jnp.broadcast_to(loss_mask[:, None], (B, K, L))
l2_loss = reduce_token_loss(per_thought_mse.mean(axis=1), loss_mask)
```

For the decoder branch, the model must return the pre-unembed hidden state so we can aggregate K → 1:

```python
# Decoder branch
pre_unembed_flat, _ = state.apply_fn(
    {"params": params}, decoder_z_flat, jnp.ones(B),
    intra_mask=intra_mask, inter_mask=inter_mask,
    deterministic=False, rngs={"dropout": model_dropout_rng},
    self_cond_cfg_scale=self_cond_cfg_scale,
    decoder_step_active=jnp.array(True),
    return_pre_unembed=True,
)
pre_unembed_k = pre_unembed_flat.reshape(B, K, L, -1)
from modules.thought_aggregation import get_aggregator
aggregator = get_aggregator(config)
x_agg = aggregator.apply({'params': params['aggregator']}, pre_unembed_k)  # (B, L, D)
# Shared unembed (reuse model's unembed kernel parameters via apply_fn path or extract)
logits = x_agg @ params['unembed_kernel'] + params['unembed_bias']
# CE loss same as ELF, against decoder_targets[..., None]
```

The implementer should carefully verify how to thread aggregator params and the unembed kernel through Flax `apply_fn`. One clean option: make the aggregator a submodule of `ELF_PT` and have the model do the aggregation internally when `num_thoughts > 1` and `decoder_step_active = True`.

- [ ] **Step 6.4** Modify `src/train.py` to route based on `config.num_thoughts`:

```python
if config.num_thoughts > 1:
    from thought_train_step import train_step
else:
    from train_step import train_step
```

- [ ] **Step 6.5** Run smoke test — expect pass (just import succeeds).
- [ ] **Step 6.6** Commit: `feat(train): K-thought train step with mask composition + aggregated CE`

## Phase 5 — Sampling

### Task 7: K-thought sampler

**Files:** Create `src/utils/thought_sampling_utils.py`, modify `src/generation.py` to dispatch when `config.num_thoughts > 1`.

- [ ] **Step 7.1** Read `src/utils/sampling_utils.py` and `src/generation.py:test_generation_uncond` end-to-end.
- [ ] **Step 7.2** Implement `src/utils/thought_sampling_utils.py`:
  - `init_thought_state(rng, B, K, L, D)`: K independent z_0 ~ N(0,I).
  - `thought_ode_step(state, z_kl, t, t_next, intra_mask, inter_mask, ...)`: one ODE step. Reshape (B, K, L, D) ↔ (B, K·L, D), single model call, update each of K z's by its own velocity.
  - `thought_sde_step(...)`: same as ODE but with per-thought noise re-injection (split RNG K ways).
  - `apply_diversity_repulsion(z_k, gamma, sigma)`: optional. For each pair (i,j) and each token position, add `gamma * (1 - exp(-||z_i - z_j||² / σ²)) * (z_i - z_j)` to z_i. Pure JAX, no model call.
  - `thought_final_decode(state, z_kl, intra_mask, inter_mask)`: forward pass with `decoder_step_active=True, return_pre_unembed=True`, aggregate, apply shared unembed, argmax.
- [ ] **Step 7.3** Smoke test `tests/test_thought_sampling.py`: 4 ODE steps with freshly-initialized random model params at K=2, L=16, tiny model. Assert no NaN, output finite, final token ids in `[0, vocab_size)`.
- [ ] **Step 7.4** Modify `src/generation.py` — when `config.num_thoughts > 1`, call into `thought_sampling_utils` instead of `sampling_utils` for both ODE and SDE paths.
- [ ] **Step 7.5** Run smoke test — expect pass.
- [ ] **Step 7.6** Commit: `feat(sampling): K-thought ODE/SDE sampler with optional repulsion`

## Phase 6 — Evaluation

### Task 8: Diversity metrics + Pass@K

**Files:** Create `src/utils/diversity_metrics.py`, modify `src/eval.py` to log new metrics when `config.num_thoughts > 1`, create `tests/test_diversity_metrics.py`.

- [ ] **Step 8.1** Write tests:

```python
import jax.numpy as jnp
import jax
from utils.diversity_metrics import pairwise_thought_diversity, oracle_pass_at_k


def test_identical_thoughts_have_zero_diversity():
    x = jnp.ones((2, 4, 8, 16))  # B, K, L, D — all identical
    d = pairwise_thought_diversity(x)
    assert float(d) < 1e-6


def test_random_thoughts_have_positive_diversity():
    rng = jax.random.PRNGKey(0)
    x = jax.random.normal(rng, (1, 4, 8, 16))
    d = pairwise_thought_diversity(x)
    assert 0 < float(d) < 2.0


def test_oracle_pass_at_k_picks_best():
    preds_per_example = [["xyz", "the quick brown fox", "abc"]]   # K=3 candidates
    refs = ["the quick brown fox jumps"]
    def score(p, r):
        return float(p in r)
    s = oracle_pass_at_k(preds_per_example, refs, score, K=3)
    assert s == 1.0
```

- [ ] **Step 8.2** Run — ImportError.
- [ ] **Step 8.3** Implement `src/utils/diversity_metrics.py`:
  - `pairwise_thought_diversity(x)`: x is (B, K, L, D). Compute mean cosine *distance* `1 - cos(x_i, x_j)` averaged over all `K*(K-1)/2` pairs and over (B, L).
  - `oracle_pass_at_k(preds, refs, scorer, K)`: `preds[i]` is a list of K candidate strings for example i; `refs[i]` is the reference; `scorer(p, r)` returns a float. Return mean over examples of `max_k scorer(preds[i][k], refs[i])`.
- [ ] **Step 8.4** Extend `src/eval.py`: when `config.num_thoughts > 1`, additionally (a) capture the (B, K, L, D) pre-unembed embeddings at t=1 and log `pairwise_thought_diversity`, (b) decode each thought independently (skip aggregation) to produce K candidate outputs per example, then compute Pass@K against the eval references (only for cond tasks where references exist).
- [ ] **Step 8.5** Run pytest — expect pass.
- [ ] **Step 8.6** Commit: `feat(eval): diversity + Pass@K metrics`

## Phase 7 — Feasibility test

### Task 9: K=2 WMT14 smoke training run

**Files:** Create `src/configs/training_configs/train_de-en_ELF-PT-B_smoke.yml`.

- [ ] **Step 9.1** Create config (small enough to fit and finish in ~20 min):

```yaml
data_path: "embedded-language-flows/wmt14_de-en_train_t5"
eval_data_path: "embedded-language-flows/wmt14_de-en_validation_t5"

encoder_model_name: t5-small
encoder_checkpoint: "embedded-language-flows/t5_small_encoder_jax/t5_small_encoder_jax.pkl"
latent_mean: 0.0
latent_std: 0.2

model: ELF-PT-B
num_thoughts: 2
thought_block_pattern: "intra,inter"
thought_aggregation: mean
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

epochs: 1
global_batch_size: 16
blr: 0.001
warmup_steps: 50
optimizer: muon

ema_decay1: 0.9999

sampling_configs_path: "configs/sampling_configs/cond_sampling_configs.yml"
num_samples: 8

log_freq: 10
save_freq: 9999
eval_freq: 9999

output_dir: outputs/smoke_pt_k2
online_eval: false
use_wandb: false
```

- [ ] **Step 9.2** Run:

```bash
sudo docker run --rm --gpus all \
  -v /localhome/local-chrislin/ELF-PT:/workspace \
  -v /localhome/local-chrislin/.cache/huggingface:/cache/hf \
  -e HF_HOME=/cache/hf -e WANDB_MODE=disabled \
  elf-pt:smoke timeout 1500 \
  python train.py --config configs/training_configs/train_de-en_ELF-PT-B_smoke.yml \
  2>&1 | tee /workspace/outputs/smoke_pt_k2.log
```

- [ ] **Step 9.3** Verify success criteria from log:
  - `l2_loss` decreases monotonically over the first 200 steps. No spike at any point (confirms zero-init worked).
  - `ce_loss` finite throughout.
  - No OOM.
  - Wall-clock per step ≤ 4× the ELF-B baseline at same batch size (`global_batch_size=16`, `max_length=128`).
- [ ] **Step 9.4** If any criterion fails, STOP and report; do not proceed to Phase 8. Common failure modes:
  - Loss spike at step ~10 → zero-init not threaded correctly into `Attention.proj` and `SwiGLUFFN.w3`.
  - OOM at step 1 → mask shape wrong; check K=2, max_length=128 → (B, 256, 256) mask is small, but enc_mask_k tiling may have a bug.
  - Per-thought MSEs identical → noise RNGs aren't actually different per thought.
- [ ] **Step 9.5** Commit log and config: `chore(smoke): K=2 WMT14 feasibility run passed`

## Phase 8 — Ablations (post-feasibility)

After Task 9 passes, run these experiments and log outcomes to `EXPERIMENTS.md`. Each is launched with the docker pattern above, swapping configs.

- Run A — Compute-matched K=1 baseline retrained from scratch in this environment.
- Run B — K=2 full WMT14.
- Run C — K=4 full WMT14.
- Run D — K=2 on OWT at L=1024 (reduce batch as needed).
- Run E — K=4 on OWT.
- Run F — Block-pattern ablation on WMT14 with K=4: "intra" only, "inter" only, "intra,inter", "intra,intra,inter".
- Run G — Inference-time repulsion sweep γ ∈ {0.1, 0.3, 0.5} on the K=4 WMT14 checkpoint.
- Run H — K=8 on WMT14 (skip OWT due to memory).

For each: config path, command, git SHA, wall-clock per step, final loss, headline metric, one-sentence interpretation.

---

## Open decisions (defaults locked in unless overridden)

| Decision | Default |
|---|---|
| Self-cond x_pred per thought or shared? | Per-thought |
| Per-thought time `t_k` independent or shared? | Independent |
| Aggregate logits or pre-unembed embeddings for CE? | Pre-unembed |
| K=1 ablation: tied or distinct intra/inter weights? | Distinct (K=1 ELF-PT is its own number, not exactly ELF-B) |
