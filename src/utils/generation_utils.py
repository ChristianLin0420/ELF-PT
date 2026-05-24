from functools import partial

import jax
import jax.numpy as jnp
from flax import jax_utils
from jax import Array

from configs.config import Config, SamplingConfig
from utils.logging_utils import log_for_0
from utils.sampling_utils import (
    restore_cond, _ode_step, _sde_step, get_sampling_steps,
)
from modules.t5_encoder import get_encoder
from utils.thought_mask_utils import build_thought_masks

PRNGKey = jax.random.PRNGKey


# ============================================
# Generation utilities
# ============================================


def mask_after_eos(predicted_ids, eos_token_id, pad_token_id):
    """Mask everything at/after first EOS token per sequence."""
    eos_mask = predicted_ids == eos_token_id
    keep_mask = jnp.cumsum(eos_mask, axis=1) == 0
    return jnp.where(keep_mask, predicted_ids, pad_token_id)


def shift_left(x, shift_per_sample, pad_value=0, axis=1):
    """Shift each sample left along the sequence axis; pad emptied positions."""
    if x.ndim < 2:
        raise ValueError("x must have at least batch and sequence dimensions")
    axis = axis if axis >= 0 else x.ndim + axis
    if axis == 0:
        raise ValueError("axis=0 is the batch axis and cannot be shifted")
    shift_per_sample = shift_per_sample.astype(jnp.int32)
    if axis != 1:
        x = jnp.moveaxis(x, axis, 1)
    seq_len = x.shape[1]
    base_idx = jnp.arange(seq_len)[None, :]
    gather_idx = shift_per_sample[:, None] + base_idx
    valid = gather_idx < seq_len
    gather_idx = jnp.clip(gather_idx, 0, seq_len - 1)
    if x.ndim == 2:
        shifted = jnp.take_along_axis(x, gather_idx, axis=1)
        shifted = jnp.where(valid, shifted, pad_value)
    else:
        expand_axes = tuple(range(2, x.ndim))
        shifted = jnp.take_along_axis(x, jnp.expand_dims(gather_idx, expand_axes), axis=1)
        shifted = jnp.where(jnp.expand_dims(valid, expand_axes), shifted, pad_value)
    if axis != 1:
        shifted = jnp.moveaxis(shifted, 1, axis)
    return shifted


# ============================================
# Multi-device helpers (pmap)
# ============================================

def _sample_step_for_scan(
    model_apply_fn, model_params, config, sampling_config: SamplingConfig,
    cfg_scale, self_cond_cfg_scale, cond_seq, cond_seq_mask, rng=None,
):
    """Create a scan-compatible step function.

    For method == "sde", `rng` must be provided and the scan carry must include a step index
    (z, x_pred, step_idx); fold_in is done per step. Other methods use a (z, x_pred) carry.
    """
    method = sampling_config.sampling_method
    base_kwargs = dict(
        model_apply_fn=model_apply_fn, model_params=model_params,
        config=config,
        cfg_scale=cfg_scale, self_cond_cfg_scale=self_cond_cfg_scale,
        cond_seq=cond_seq, cond_seq_mask=cond_seq_mask,
    )

    if method == "sde":
        assert rng is not None, "SDE method requires rng to be passed to _sample_step_for_scan"
        sde_gamma = getattr(sampling_config, "sde_gamma", 0.0)

        def step_fn(carry, t_pair):
            z, x_pred, step_idx = carry
            t, t_next = t_pair
            step_rng = jax.random.fold_in(rng, step_idx)
            z_new, x_pred_new = _sde_step(
                z=z, t=t, t_next=t_next, x_pred_prev=x_pred,
                gamma=sde_gamma, rng=step_rng, **base_kwargs,
            )
            return (z_new, x_pred_new, step_idx + 1), None
        return step_fn

    if method == "ode":
        base_step_fn = _ode_step
    else:
        raise ValueError(f"Invalid sampling method: {method}")

    def step_fn(carry, t_pair):
        z, x_pred = carry
        t, t_next = t_pair
        z_new, x_pred_new = base_step_fn(
            z=z, t=t, t_next=t_next, x_pred_prev=x_pred, **base_kwargs,
        )
        return (z_new, x_pred_new), None
    return step_fn


def _generate_samples_single_batch(
    model_params, model_apply_fn, rng: PRNGKey, z: Array, t_steps: Array,
    cond_seq: Array, cond_seq_mask: Array, config: Config, sampling_config: SamplingConfig,
    cfg_scale: float, self_cond_cfg_scale: float,
) -> Array:
    """Generate samples for a single batch (pmap-compatible, uses lax.scan)."""
    method = sampling_config.sampling_method
    batch_size, max_length, d_model = z.shape
    if cond_seq is None:
        cond_seq = jnp.zeros((batch_size, max_length, d_model))
        cond_seq_mask = jnp.zeros((batch_size, max_length))
    step_kwargs = dict(
        model_apply_fn=model_apply_fn, model_params=model_params,
        config=config,
        cfg_scale=cfg_scale, self_cond_cfg_scale=self_cond_cfg_scale,
        cond_seq=cond_seq, cond_seq_mask=cond_seq_mask,
    )

    z = restore_cond(z, cond_seq, cond_seq_mask)
    x_pred = restore_cond(jnp.zeros_like(z), cond_seq, cond_seq_mask)

    t_pairs = jnp.stack([t_steps[:-2], t_steps[1:-1]], axis=1)
    if method == "sde":
        step_fn = _sample_step_for_scan(sampling_config=sampling_config, rng=rng, **step_kwargs)
        (z, x_pred, _), _ = jax.lax.scan(step_fn, (z, x_pred, jnp.int32(0)), t_pairs)
    else:
        step_fn = _sample_step_for_scan(sampling_config=sampling_config, **step_kwargs)
        (z, x_pred), _ = jax.lax.scan(step_fn, (z, x_pred), t_pairs)

    # Last step always with ode
    z, x_pred = _ode_step(
        z=z, t=t_steps[-2], t_next=t_steps[-1], x_pred_prev=x_pred, **step_kwargs,
    )
    return z


def _dlm_decode_batch(z, model_params, model_apply_fn, t_final_val, config, self_cond_cfg_scale):
    """Decode z→tokens with the DLM decoder head."""
    batch_size = z.shape[0]
    t_final = jnp.full((batch_size,), t_final_val, dtype=z.dtype)
    self_cond_cfg_scale_batch = (
        jnp.full((batch_size,), self_cond_cfg_scale, dtype=z.dtype)
        if config.num_self_cond_cfg_tokens > 0 else None
    )
    z_input = jnp.concatenate([z, jnp.zeros_like(z)], axis=-1) if config.self_cond_prob > 0 else z
    _, decoder_logits = model_apply_fn(
        {"params": model_params}, z_input, t_final,
        deterministic=True,
        self_cond_cfg_scale=self_cond_cfg_scale_batch,
        decoder_step_active=jnp.array(True),
    )
    return jnp.argmax(decoder_logits, axis=-1)


# ============================================
# Shared generation scaffolding
# ============================================
def _make_pmap_pair(model_apply_fn, config, sampling_config, cfg_scale, self_cond_cfg_scale):
    """Build pmapped (generate, decode) pair for a (cfg, sccfg) combo."""
    p_generate = jax.pmap(
        partial(
            _generate_samples_single_batch,
            model_apply_fn=model_apply_fn, config=config, sampling_config=sampling_config,
            cfg_scale=cfg_scale, self_cond_cfg_scale=self_cond_cfg_scale,
        ),
        axis_name="batch",
    )
    p_decode_ids = jax.pmap(
        partial(
            _dlm_decode_batch, model_apply_fn=model_apply_fn, config=config,
            self_cond_cfg_scale=self_cond_cfg_scale,
        )
    )
    return p_generate, p_decode_ids


def _build_run_name(sampling_method, num_sampling_steps, cfg_scale, self_cond_cfg_scale,
                    time_schedule, sde_gamma, suffix):
    ts_str = f"-ts_{time_schedule}"
    sccfg_str = f"-sccfg{self_cond_cfg_scale}" if self_cond_cfg_scale != 1.0 else ""
    sde_str = f"-gamma{sde_gamma}" if sampling_method == "sde" else ""
    return f"{sampling_method}-steps{num_sampling_steps}-cfg{cfg_scale}{sccfg_str}{ts_str}{sde_str}-{suffix}"


def _shard_timesteps(t_rng, num_local_devices, num_sampling_steps, time_schedule, config):
    t_device_rngs = jax.random.split(t_rng, num_local_devices)
    return jnp.stack([
        get_sampling_steps(
            t_device_rngs[i], n_steps=num_sampling_steps,
            time_schedule=time_schedule, P_mean=config.denoiser_p_mean, P_std=config.denoiser_p_std,
        )
        for i in range(num_local_devices)
    ])


def _shard_noise(device_rngs, num_local_devices, per_device, max_length, d_model, noise_scale):
    return jnp.stack([
        jax.random.normal(device_rngs[i], (per_device, max_length, d_model)) * noise_scale
        for i in range(num_local_devices)
    ])


def _setup_generation(state, config, batch_size, header):
    """Shared setup: log header, unreplicate state, build replicated model_params, compute batch sizes."""
    log_for_0("\n" + "=" * 70)
    log_for_0(header)
    log_for_0("=" * 70)

    num_local_devices = jax.local_device_count()
    log_for_0(f"Using {num_local_devices} local devices for generation")

    state_unreplicated = jax_utils.unreplicate(state)
    model_apply_fn = state_unreplicated.apply_fn

    encoder_config, _, _ = get_encoder(config.encoder_model_name, None)
    d_model = encoder_config.d_model

    model_params_replicated = jax_utils.replicate(state_unreplicated.ema_params1)

    per_device_batch = max(1, batch_size // num_local_devices)
    effective_batch_size = per_device_batch * num_local_devices
    log_for_0(f"Per-device batch size: {per_device_batch}, effective batch size: {effective_batch_size}")

    return state_unreplicated, model_apply_fn, model_params_replicated, d_model, num_local_devices, effective_batch_size


# ============================================
# K-thought generation helpers
# ============================================

def _shard_thought_noise(device_rngs, num_local_devices, per_device, K, max_length,
                         d_model, noise_scale):
    """Return sharded (num_devices, per_device, K*max_length, d_model) noise."""
    return jnp.stack([
        jax.random.normal(device_rngs[i], (per_device, K * max_length, d_model)) * noise_scale
        for i in range(num_local_devices)
    ])


def _build_thought_masks_batch(B, L, K, cond_seq_mask=None):
    """Build (intra_mask, inter_mask) for a batch of size B.

    For unconditional generation (cond_seq_mask is None), all tokens are treated
    as non-cond valid tokens.
    """
    if cond_seq_mask is not None:
        is_cond = (cond_seq_mask > 0).astype(jnp.bool_)
    else:
        is_cond = jnp.zeros((B, L), dtype=jnp.bool_)
    is_valid = jnp.ones((B, L), dtype=jnp.bool_)
    return build_thought_masks(is_cond, is_valid, K)


def _thought_generate_single_batch(
    model_params, model_apply_fn, rng: PRNGKey, z: Array, t_steps: Array,
    cond_seq: Array, cond_seq_mask: Array, config: Config, sampling_config: SamplingConfig,
) -> Array:
    """K-thought sampling loop for a single batch (pmap-compatible).

    Args:
        z: (B, K*L, D) initial noise.
        t_steps: (n_steps+1,) time schedule.
        cond_seq: (B, L, D) or None.
        cond_seq_mask: (B, L) float in {0, 1} or None.

    Returns:
        z_final: (B, K*L, D).
    """
    from utils.thought_sampling_utils import (
        thought_ode_step, thought_sde_step, apply_diversity_repulsion,
    )

    K = config.num_thoughts
    method = sampling_config.sampling_method
    sde_gamma = getattr(sampling_config, "sde_gamma", 0.0)
    div_gamma = (
        config.diversity_repulsion_gamma_max
        if config.diversity_repulsion_inference else 0.0
    )
    div_sigma = config.diversity_repulsion_sigma
    t_eps = config.t_eps
    noise_scale = config.denoiser_noise_scale

    B = z.shape[0]
    L = z.shape[1] // K

    # Build attention masks once — constant across steps
    intra_mask, inter_mask = _build_thought_masks_batch(
        B, L, K, cond_seq_mask=cond_seq_mask,
    )

    # Replicate cond tokens K times and restore them into z
    if cond_seq is not None:
        cond_seq_kl = jnp.tile(cond_seq, (1, K, 1))       # (B, K*L, D)
        cond_mask_kl = jnp.tile(cond_seq_mask, (1, K))    # (B, K*L)
        z = restore_cond(z, cond_seq_kl, cond_mask_kl[..., None])

    # Build a fake TrainState-like carrier so thought_ode/sde_step can call apply_fn
    class _FakeState:
        def __init__(self, apply_fn, params):
            self.apply_fn = apply_fn
            self.params = params

    state = _FakeState(model_apply_fn, model_params)

    t_pairs = jnp.stack([t_steps[:-2], t_steps[1:-1]], axis=1)  # (n_steps-1, 2)
    n_inner = t_pairs.shape[0]

    def ode_step_fn(carry, t_pair):
        z_c = carry
        t, t_next = t_pair
        z_new = thought_ode_step(state, z_c, t, t_next, intra_mask, inter_mask, t_eps)
        if div_gamma > 0:
            z_new = apply_diversity_repulsion(z_new, K, div_gamma, div_sigma)
        return z_new, None

    def sde_step_fn(carry, t_pair):
        z_c, step_idx = carry
        t, t_next = t_pair
        step_rng = jax.random.fold_in(rng, step_idx)
        z_new = thought_sde_step(
            step_rng, state, z_c, t, t_next, intra_mask, inter_mask,
            gamma=sde_gamma, K=K, noise_scale=noise_scale, t_eps=t_eps,
        )
        if div_gamma > 0:
            z_new = apply_diversity_repulsion(z_new, K, div_gamma, div_sigma)
        return (z_new, step_idx + 1), None

    if method == "sde":
        (z, _), _ = jax.lax.scan(sde_step_fn, (z, jnp.int32(0)), t_pairs)
    else:
        z, _ = jax.lax.scan(ode_step_fn, z, t_pairs)

    # Final ODE step (always ODE, as in the K=1 path)
    z = thought_ode_step(state, z, t_steps[-2], t_steps[-1], intra_mask, inter_mask, t_eps)
    return z


def _thought_decode_batch(z, model_params, model_apply_fn, t_final_val, config):
    """Decode K-thought latent z → token ids (B, L).

    Uses return_pre_unembed=True path: model aggregates K thoughts internally,
    then we apply the factored unembed head.
    """
    from utils.thought_sampling_utils import thought_final_decode

    K = config.num_thoughts
    B = z.shape[0]
    L = z.shape[1] // K

    # Build masks (unconditional: no cond tokens)
    intra_mask, inter_mask = _build_thought_masks_batch(B, L, K, cond_seq_mask=None)

    class _FakeState:
        def __init__(self, apply_fn, params):
            self.apply_fn = apply_fn
            self.params = params

    state = _FakeState(model_apply_fn, model_params)
    return thought_final_decode(state, z, intra_mask, inter_mask)


def _make_thought_pmap_pair(model_apply_fn, config, sampling_config):
    """Build pmapped (generate, decode) pair for K-thought sampling."""
    p_generate = jax.pmap(
        partial(
            _thought_generate_single_batch,
            model_apply_fn=model_apply_fn,
            config=config,
            sampling_config=sampling_config,
        ),
        axis_name="batch",
    )
    p_decode_ids = jax.pmap(
        partial(
            _thought_decode_batch,
            model_apply_fn=model_apply_fn,
            config=config,
        )
    )
    return p_generate, p_decode_ids


