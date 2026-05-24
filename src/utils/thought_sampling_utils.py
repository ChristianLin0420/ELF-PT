"""K-thought sampler for ELF-PT (ODE + SDE + optional diversity repulsion).

Noise formula: matches ELF's existing SDE churn in sampling_utils.py exactly.
  alpha = clip(1 - gamma * h, 0, 1),  h = t_next - t
  t_back = alpha * t
  z_back = alpha * z + (1 - alpha) * noise   (per-thought, independently sampled)

The caller is responsible for building the intra/inter masks via
build_thought_masks (utils/thought_mask_utils.py) before calling these
functions.
"""
import jax
import jax.numpy as jnp

from utils.sampling_utils import net_out_to_v_x   # reuse existing helper


# ============================================
# Initialisation
# ============================================

def init_thought_state(rng, B, K, L, D, dtype=jnp.float32):
    """Return (B, K*L, D) of K independent standard-normal samples.

    The RNG is split K ways so every thought gets a different noise draw.
    Using jnp.tile would produce K *identical* thoughts — we explicitly avoid
    that here.
    """
    rngs = jax.random.split(rng, K)
    z_k = jnp.stack([
        jax.random.normal(rngs[k], (B, L, D), dtype=dtype)
        for k in range(K)
    ], axis=1)                              # (B, K, L, D)
    return z_k.reshape(B, K * L, D)


# ============================================
# ODE step
# ============================================

def thought_ode_step(state, z_kl, t, t_next, intra_mask, inter_mask,
                     t_eps=5e-2, **mk):
    """One Euler step in latent space using K-thought model.

    Args:
        state:      Flax TrainState (has .apply_fn and .params).
        z_kl:       (B, K*L, D) current latent.
        t:          scalar or (B,) current time.
        t_next:     scalar or (B,) next time.
        intra_mask: (B, K*L, K*L) intra-group attention mask.
        inter_mask: (B, K*L, K*L) inter-group attention mask.
        t_eps:      minimum denominator for velocity computation.
        **mk:       extra kwargs forwarded to model (e.g. decoder_step_active).

    Returns:
        z_kl_new: (B, K*L, D) updated latent after one Euler step.
    """
    B = z_kl.shape[0]
    t_batch = jnp.full((B,), t) if jnp.ndim(t) == 0 else t

    net_out, _ = state.apply_fn(
        {"params": state.params}, z_kl, t_batch,
        intra_mask=intra_mask, inter_mask=inter_mask,
        deterministic=True, **mk,
    )
    v, _ = net_out_to_v_x(net_out, z_kl, t_batch, t_eps)
    dt = jnp.array(t_next - t)
    return z_kl + v * dt.reshape(-1, 1, 1) if dt.ndim > 0 else z_kl + v * dt


# ============================================
# SDE step
# ============================================

def thought_sde_step(rng, state, z_kl, t, t_next, intra_mask, inter_mask,
                     gamma, K, noise_scale=1.0, t_eps=5e-2, **mk):
    """SDE-style churn step with per-thought independent noise re-injection.

    Matches ELF's existing SDE formula (sampling_utils._sde_step):
      alpha  = clip(1 - gamma * h, 0, 1)         where h = t_next - t
      t_back = alpha * t
      z_back = alpha * z + (1 - alpha) * noise   (per thought, independently)

    Then runs thought_ode_step from (z_back, t_back) to t_next.

    Args:
        rng:        JAX PRNG key for noise generation.
        state:      Flax TrainState.
        z_kl:       (B, K*L, D) current latent.
        t:          scalar current time.
        t_next:     scalar next time.
        intra_mask: (B, K*L, K*L) intra-group mask.
        inter_mask: (B, K*L, K*L) inter-group mask.
        gamma:      SDE churn magnitude (0 = pure ODE).
        K:          number of thoughts.
        noise_scale: denoiser noise scale (config.denoiser_noise_scale).
        t_eps:      minimum denominator for velocity computation.
        **mk:       extra kwargs forwarded to model.

    Returns:
        z_kl_new: (B, K*L, D) updated latent.
    """
    if gamma <= 0:
        return thought_ode_step(state, z_kl, t, t_next, intra_mask, inter_mask,
                                t_eps, **mk)

    h = t_next - t
    alpha = jnp.clip(1.0 - gamma * h, 0.0, 1.0)
    t_back = float(alpha * t)

    B, S, D = z_kl.shape
    L = S // K
    z_per = z_kl.reshape(B, K, L, D)

    # Per-thought independent noise (split RNG K ways)
    noise_rngs = jax.random.split(rng, K)
    noise_per = jnp.stack([
        jax.random.normal(noise_rngs[k], (B, L, D), dtype=z_kl.dtype)
        for k in range(K)
    ], axis=1)                              # (B, K, L, D)
    noise_per = noise_per * noise_scale

    z_back_per = alpha * z_per + (1.0 - alpha) * noise_per
    z_back = z_back_per.reshape(B, K * L, D)

    return thought_ode_step(state, z_back, t_back, t_next, intra_mask, inter_mask,
                            t_eps, **mk)


# ============================================
# Diversity repulsion (LaDiR-style, inference only)
# ============================================

def apply_diversity_repulsion(z_kl, K, gamma, sigma):
    """Optional LaDiR-style repulsion between K thoughts.

    For each pair (i, j) with i != j, adds a repulsion force proportional to
    (1 - exp(-||z_i - z_j||^2 / sigma^2)) * (z_i - z_j) scaled by gamma.

    No-op when gamma == 0 or K == 1.

    Args:
        z_kl:  (B, K*L, D) latent.
        K:     number of thoughts.
        gamma: repulsion step size.
        sigma: RBF kernel bandwidth.

    Returns:
        z_kl updated: (B, K*L, D).
    """
    if gamma <= 0 or K == 1:
        return z_kl

    B, S, D = z_kl.shape
    L = S // K
    z_per = z_kl.reshape(B, K, L, D)

    # Pairwise squared distances ||z_i - z_j||^2 per (B, L) position
    diff = z_per[:, :, None] - z_per[:, None, :]    # (B, K, K, L, D)
    sq = jnp.sum(diff ** 2, axis=-1)                 # (B, K, K, L)
    weight = (1 - jnp.exp(-sq / (sigma ** 2)))[..., None]  # (B, K, K, L, 1)

    # Mask out diagonal (i == j contributes zero force)
    eye_mask = jnp.eye(K, dtype=jnp.bool_)[None, :, :, None, None]  # (1, K, K, 1, 1)
    weight = jnp.where(eye_mask, 0.0, weight)

    # Force on thought i = sum_{j != i} weight_ij * (z_i - z_j)
    force = (weight * diff).sum(axis=2)             # (B, K, L, D)
    z_per = z_per + gamma * force
    return z_per.reshape(B, K * L, D)


# ============================================
# Final decode
# ============================================

def thought_final_decode(state, z_kl, intra_mask, inter_mask, t_final=None, **mk):
    """Final decode at t=1: aggregate K thoughts and return token ids.

    Calls model with return_pre_unembed=True so the model aggregates K thoughts
    internally (MeanPoolAggregator or LearnedWeightAggregator), obtaining
    (B, L, hidden_size).  Then applies the shared factored unembed head from
    state.params to produce logits (B, L, V) and returns argmax token ids (B, L).

    Args:
        state:      Flax TrainState.
        z_kl:       (B, K*L, D) final latent.
        intra_mask: (B, K*L, K*L) intra-group attention mask.
        inter_mask: (B, K*L, K*L) inter-group attention mask.
        t_final:    (B,) final time values.  Defaults to jnp.ones((B,)) when
                    None, which is correct for schedules that end at t=1.
                    Pass an explicit value when the schedule ends at t != 1.
        **mk:       extra kwargs forwarded to model.

    Returns:
        token_ids: (B, L) int32 argmax token predictions.
    """
    B = z_kl.shape[0]
    if t_final is None:
        t_final = jnp.ones((B,), dtype=z_kl.dtype)

    x_agg, _ = state.apply_fn(
        {"params": state.params}, z_kl, t_final,
        intra_mask=intra_mask, inter_mask=inter_mask,
        deterministic=True,
        decoder_step_active=jnp.array(True),
        return_pre_unembed=True,
        **mk,
    )
    # x_agg: (B, L, hidden_size)
    # Apply shared factored unembed: hidden -> text_encoder_dim -> vocab
    params = state.params
    logits = (
        jax.nn.gelu(x_agg @ params['proj_kernel'] + params['proj_bias'])
        @ params['unembed_kernel'] + params['unembed_bias']
    )
    return jnp.argmax(logits, axis=-1)   # (B, L)
