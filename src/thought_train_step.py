"""Per-device pmap'd training step for ELF-PT with K parallel thought streams.

Forked from train_step.py.  The K=1 path produces the same loss values as
the original train_step when using the same RNG and noise, because at K=1
the per-thought tensors all reduce to the single-thought tensors.

Key differences vs. train_step.py:
  - Time and noise are sampled K-independently.
  - denoiser_z / decoder_z are (B, K, L, D) flattened to (B, K*L, D).
  - intra_mask / inter_mask are built by build_thought_masks and ANDed with
    the existing encoder_attention_mask (tiled K times).
  - Model forward uses ELF_PT's intra_mask / inter_mask keyword args instead
    of a single attention_mask.
  - Decoder CE branch: model returns pre-unembed hidden states, which are
    aggregated across K thoughts, then the shared unembed kernel is applied.
  - Denoiser L2 branch: per-thought velocity MSE, averaged (not summed) over K.
"""

from typing import Dict, Tuple

import jax
import jax.numpy as jnp

from utils.train_utils import TrainState
from utils.encoder_utils import encode_text
from utils.sampling_utils import (
    sample_cfg_scale, add_noise, sample_timesteps,
    net_out_to_v_x, restore_cond,
)
from utils.thought_mask_utils import build_thought_masks
from modules.thought_aggregation import get_aggregator


Array = jnp.ndarray


def train_step(
    state: TrainState,
    encoder_params: Dict,
    encoder_apply_fn,
    batch: Dict[str, Array],
    config,
) -> Tuple[TrainState, Dict[str, float]]:
    """Perform a single K-thought training step."""
    t_eps = config.t_eps
    self_cond_prob = config.self_cond_prob
    latent_mean, latent_std = config.latent_mean, config.latent_std

    decoder_prob = config.decoder_prob
    decoder_noise_scale = config.decoder_noise_scale

    K = config.num_thoughts

    new_dropout_rng, current_step_rng = jax.random.split(state.dropout_rng, 2)
    current_step_rng = jax.random.fold_in(current_step_rng, jax.lax.axis_index(axis_name="batch"))
    (
        t_rng, noise_rng, self_cond_mask_rng, self_cond_cfg_rng,
        model_dropout_rng, decoder_step_rng, decoder_rng,
        decoder_lambda_rng, decoder_noise_rng,
    ) = jax.random.split(current_step_rng, 9)

    # encoder_attention_mask: cond sees cond, x sees all
    encoder_attention_mask = batch["encoder_attention_mask"]

    # Label drop before encoding (same as original)
    if config.label_drop_prob > 0:
        drop = batch["label_drop_mask"][:, None, None]  # (B, 1, 1)
        cond_mask = batch["cond_seq_mask"]  # (B, S)
        block_mask = (1 - cond_mask)[:, :, None] * cond_mask[:, None, :]
        encoder_attention_mask = encoder_attention_mask * (1 - drop * block_mask)

    x0 = encode_text(
        input_ids=batch["input_ids"],
        attention_mask=encoder_attention_mask,
        encoder_apply_fn=encoder_apply_fn,
        encoder_params=encoder_params,
        latent_mean=latent_mean,
        latent_std=latent_std,
    )

    batch_size, seq_length = x0.shape[0], x0.shape[1]

    # ── Per-thought RNG split ──────────────────────────────────────────────
    thought_t_rngs = jax.random.split(t_rng, K)
    thought_noise_rngs = jax.random.split(noise_rng, K)

    # Sample K independent times, shape (B, K)
    t_per = jnp.stack([
        sample_timesteps(
            thought_t_rngs[k], batch_size,
            P_mean=config.denoiser_p_mean, P_std=config.denoiser_p_std,
            time_schedule=config.time_schedule,
        ) for k in range(K)
    ], axis=1)  # (B, K)

    # Sample K independent noises, shape (B, K, L, D)
    noise_per = jnp.stack([
        jax.random.normal(thought_noise_rngs[k], x0.shape, dtype=x0.dtype)
        for k in range(K)
    ], axis=1)  # (B, K, L, D)

    # Broadcast x0 across K
    x0_per = jnp.broadcast_to(x0[:, None], (batch_size, K, seq_length, x0.shape[-1]))

    cond_seq_mask = batch["cond_seq_mask"][:, :, None]  # (B, L, 1)
    attention_mask = batch["attention_mask"]
    if config.pad_token == "pad":
        loss_mask = attention_mask
    else:
        loss_mask = jnp.ones_like(attention_mask)
    loss_mask = loss_mask * (1 - batch["cond_seq_mask"])

    # ── Per-thought denoiser_z ─────────────────────────────────────────────
    # Mirror add_noise logic: z = t * x0 + (1-t) * noise * scale
    t_exp = t_per[:, :, None, None]   # (B, K, 1, 1)
    denoiser_z_per = (
        t_exp * x0_per + (1 - t_exp) * noise_per * config.denoiser_noise_scale
    )
    # Re-apply cond_seq_mask: cond positions keep x0 (no noise)
    cond_mask_exp = cond_seq_mask[:, None]  # (B, 1, L, 1)
    denoiser_z_per = jnp.where(cond_mask_exp > 0, x0_per, denoiser_z_per)

    # Label drop on denoiser_z and x0 (same as original)
    drop = batch["label_drop_mask"][:, None]
    if config.label_drop_prob > 0:
        drop_exp = drop[:, :, None, None]   # (B, 1, 1, 1)
        denoiser_z_per = jnp.where(
            drop_exp & (cond_mask_exp > 0),
            jnp.zeros_like(denoiser_z_per),
            denoiser_z_per,
        )
        x0 = jnp.where(drop[:, :, None] & (cond_seq_mask > 0), jnp.zeros_like(x0), x0)
        x0_per = jnp.broadcast_to(x0[:, None], (batch_size, K, seq_length, x0.shape[-1]))

    # Flatten K dim into S = K*L
    denoiser_z = denoiser_z_per.reshape(batch_size, K * seq_length, -1)

    # ── Per-thought velocity target ────────────────────────────────────────
    v_target_per = (x0_per - denoiser_z_per) / jnp.maximum(1 - t_exp, t_eps)  # (B, K, L, D)

    # ── Decoder branch inputs ──────────────────────────────────────────────
    decoder_targets = batch["input_ids"]  # (B, S)
    decoder_step_active = jax.random.bernoulli(decoder_step_rng, decoder_prob)

    # K independent decoder lambda + noise (mirroring the single-thought construction)
    thought_decoder_lambda_rngs = jax.random.split(decoder_lambda_rng, K)
    thought_decoder_noise_rngs = jax.random.split(decoder_noise_rng, K)

    decoder_lambda_t_per = jnp.stack([
        jax.nn.sigmoid(
            jax.random.normal(thought_decoder_lambda_rngs[k], (batch_size * seq_length,))
            * config.decoder_p_std + config.decoder_p_mean
        ).reshape(batch_size, seq_length, 1)
        for k in range(K)
    ], axis=1)  # (B, K, L, 1)

    decoder_noise_per = jnp.stack([
        jax.random.normal(thought_decoder_noise_rngs[k], x0.shape, dtype=x0.dtype)
        * decoder_noise_scale
        for k in range(K)
    ], axis=1)  # (B, K, L, D)

    decoder_z_per = (
        decoder_lambda_t_per * x0_per + (1 - decoder_lambda_t_per) * decoder_noise_per
    )  # (B, K, L, D)
    decoder_z = decoder_z_per.reshape(batch_size, K * seq_length, -1)

    # ── Self-conditioning setup ────────────────────────────────────────────
    if self_cond_prob > 0:
        use_self_cond_mask = (
            (jax.random.uniform(self_cond_mask_rng, (batch_size,)) < self_cond_prob)
            .reshape(-1, 1, 1).astype(x0.dtype)
        )
    else:
        use_self_cond_mask = None

    if config.num_self_cond_cfg_tokens > 0:
        self_cond_cfg_scale = sample_cfg_scale(
            self_cond_cfg_rng, batch_size,
            cfg_min=config.self_cond_cfg_min, cfg_max=config.self_cond_cfg_max,
        )
    else:
        self_cond_cfg_scale = None

    # ── Build K-group attention masks ─────────────────────────────────────
    intra_mask_kk, inter_mask_kk = build_thought_masks(
        batch["cond_seq_mask"].astype(jnp.bool_),
        batch["attention_mask"].astype(jnp.bool_),
        K=K,
    )
    # Tile encoder_attention_mask to (B, K*L, K*L) and AND it in.
    # encoder_attention_mask shape: (B, L, L).
    enc_mask_kk = jnp.tile(encoder_attention_mask, (1, K, K))
    intra_mask_kk = intra_mask_kk & (enc_mask_kk > 0)
    inter_mask_kk = inter_mask_kk & (enc_mask_kk > 0)

    # ── Helpers ────────────────────────────────────────────────────────────

    def _net_out_to_v_x_per(net_out_per, z_per, t_per_bk):
        """Apply net_out_to_v_x over K thoughts using (B*K, L, D) reshape."""
        BK = batch_size * K
        net_flat = net_out_per.reshape(BK, seq_length, -1)
        z_flat = z_per.reshape(BK, seq_length, -1)
        t_flat = t_per_bk.reshape(BK)
        v_flat, x_flat = net_out_to_v_x(net_flat, z_flat, t_flat, t_eps)
        v_per = v_flat.reshape(batch_size, K, seq_length, -1)
        x_per = x_flat.reshape(batch_size, K, seq_length, -1)
        return v_per, x_per

    def get_z_input(params, z_flat, t_input, self_cond_cfg_input, x_tokens_per):
        """Self-conditioning on the flattened (B, K*L, D) representation.

        x_tokens_per: (B, K, L, D) — broadcast x0 for all K thoughts.
        Flattened to (B, K*L, D) for the stop-gradient forward pass.
        """
        if self_cond_prob == 0:
            return z_flat
        # x_tokens flattened to (B, K*L, D)
        x_tokens_flat = x_tokens_per.reshape(batch_size, K * seq_length, -1)
        # cond_seq_mask tiled to (B, K*L, 1)
        cond_mask_kl = jnp.tile(cond_seq_mask, (1, K, 1))  # (B, K*L, 1)
        z_uncond = restore_cond(jnp.zeros_like(z_flat), x_tokens_flat, cond_mask_kl)
        z_with_zeros = jnp.concatenate([z_flat, z_uncond], axis=-1)
        # Use mean time across K for the model's time signal
        t_mean = t_input.mean(axis=1) if t_input.ndim == 2 else t_input
        net_out_init = state.apply_fn(
            {"params": params}, z_with_zeros, t_mean,
            intra_mask=intra_mask_kk, inter_mask=inter_mask_kk,
            deterministic=True,
            self_cond_cfg_scale=self_cond_cfg_input,
        )
        net_out_init = jax.lax.stop_gradient(net_out_init)
        # net_out_init is a tuple (output, decoder_logits); output shape (B, K*L, D_enc)
        net_out_flat = net_out_init[0] if isinstance(net_out_init, tuple) else net_out_init
        # Expand net_out_flat to (B, K, L, D) for _net_out_to_v_x_per
        net_out_per_sc = net_out_flat.reshape(batch_size, K, seq_length, -1)
        _, x_pred_per = _net_out_to_v_x_per(net_out_per_sc, denoiser_z_per, t_per)
        # Restore cond tokens per-thought then flatten
        x_pred_per = jnp.where(cond_mask_exp > 0, x0_per, x_pred_per)
        x_pred_flat = x_pred_per.reshape(batch_size, K * seq_length, -1)
        x_pred_cond = x_pred_flat * jnp.tile(use_self_cond_mask, (1, K, 1)).astype(z_flat.dtype)
        x_pred_cond = jnp.where(
            jnp.tile(cond_seq_mask, (1, K, 1)) > 0, x_tokens_flat, x_pred_cond
        )
        return jnp.concatenate([z_flat, x_pred_cond], axis=-1)

    def reduce_token_loss(per_token_loss, mask):
        mask = mask.astype(per_token_loss.dtype)
        safe_loss = jnp.where(mask > 0, per_token_loss, jnp.zeros_like(per_token_loss))
        return (safe_loss * mask).sum() / jnp.maximum(mask.sum(), 1.0)

    def get_sc_cond_and_uncond(params, z_flat, t_input, cond_mask_kl, x_tokens_flat):
        kwargs = {
            "intra_mask": intra_mask_kk,
            "inter_mask": inter_mask_kk,
            "self_cond_cfg_scale": self_cond_cfg_scale,
            "deterministic": True,
        }
        t_mean = t_input.mean(axis=1) if t_input.ndim == 2 else t_input
        if config.self_cond_prob == 0:
            net_out_uncond = state.apply_fn({"params": params}, z_flat, t_mean, **kwargs)
            net_flat = net_out_uncond[0] if isinstance(net_out_uncond, tuple) else net_out_uncond
            net_per = net_flat.reshape(batch_size, K, seq_length, -1)
            v_uncond_per, _ = _net_out_to_v_x_per(net_per, denoiser_z_per, t_input)
            v_uncond_flat = v_uncond_per.reshape(batch_size, K * seq_length, -1)
            return v_uncond_flat, v_uncond_flat

        z_uncond = restore_cond(jnp.zeros_like(z_flat), x_tokens_flat, cond_mask_kl)
        z_input_uncond = jnp.concatenate([z_flat, z_uncond], axis=-1)
        net_out_uncond = state.apply_fn({"params": params}, z_input_uncond, t_mean, **kwargs)
        net_flat = net_out_uncond[0] if isinstance(net_out_uncond, tuple) else net_out_uncond
        net_per = net_flat.reshape(batch_size, K, seq_length, -1)
        v_uncond_per, x_uncond_per = _net_out_to_v_x_per(net_per, denoiser_z_per, t_input)
        x_uncond_per = jnp.where(cond_mask_exp > 0, x0_per, x_uncond_per)

        x_uncond_flat = x_uncond_per.reshape(batch_size, K * seq_length, -1)
        z_input_cond = jnp.concatenate([z_flat, x_uncond_flat], axis=-1)
        net_out_cond = state.apply_fn({"params": params}, z_input_cond, t_mean, **kwargs)
        net_flat_cond = net_out_cond[0] if isinstance(net_out_cond, tuple) else net_out_cond
        net_per_cond = net_flat_cond.reshape(batch_size, K, seq_length, -1)
        v_cond_per, _ = _net_out_to_v_x_per(net_per_cond, denoiser_z_per, t_input)

        v_cond_flat = v_cond_per.reshape(batch_size, K * seq_length, -1)
        v_uncond_flat = v_uncond_per.reshape(batch_size, K * seq_length, -1)
        return v_cond_flat, v_uncond_flat

    def get_sc_guided_v(params, z_flat, t_input, base_v_target_per, x_tokens_flat):
        """v target with self-conditioning guidance (K-flat representation)."""
        cond_mask_kl = jnp.tile(cond_seq_mask, (1, K, 1))
        v_cond_flat, v_uncond_flat = get_sc_cond_and_uncond(
            params, z_flat, t_input, cond_mask_kl, x_tokens_flat
        )
        sc_w = self_cond_cfg_scale.reshape(batch_size, 1, 1)
        sc_guidance_flat = (1 - 1 / sc_w) * (v_cond_flat - v_uncond_flat)
        use_sc_tiled = jnp.tile(use_self_cond_mask, (1, K, 1))
        sc_guidance_flat = jnp.where(use_sc_tiled, sc_guidance_flat, jnp.zeros_like(sc_guidance_flat))
        # Reshape guidance to per-thought and add to per-thought target
        sc_guidance_per = sc_guidance_flat.reshape(batch_size, K, seq_length, -1)
        return jax.lax.stop_gradient(base_v_target_per + sc_guidance_per)

    def get_v_target(params, z_flat, t_input, base_v_target_per, x_tokens_flat):
        if config.num_self_cond_cfg_tokens > 0 and config.self_cond_prob > 0:
            return get_sc_guided_v(
                params, z_flat, t_input,
                base_v_target_per=base_v_target_per,
                x_tokens_flat=x_tokens_flat,
            )
        return base_v_target_per

    # ── Loss function ──────────────────────────────────────────────────────

    def loss_fn(params):

        def _decoder_branch(_):
            # Decoder mode: encoder-noised latents (decoder_z_per → decoder_z) at t=1,
            # CE loss on tokens. Aggregate K pre-unembed hidden states before unembed.
            decoder_t = jnp.ones((batch_size,))
            decoder_input = (
                jnp.concatenate([decoder_z, jnp.zeros_like(decoder_z)], axis=-1)
                if config.self_cond_prob > 0 else decoder_z
            )
            x_pre_kl, _ = state.apply_fn(
                {"params": params}, decoder_input, decoder_t,
                intra_mask=intra_mask_kk, inter_mask=inter_mask_kk,
                deterministic=False,
                rngs={"dropout": model_dropout_rng},
                self_cond_cfg_scale=self_cond_cfg_scale,
                decoder_step_active=jnp.array(True),
                return_pre_unembed=True,
            )
            # x_pre_kl: (B, K*L, hidden_size)
            x_pre_per = x_pre_kl.reshape(batch_size, K, seq_length, -1)
            aggregator = get_aggregator(config)
            if config.thought_aggregation == 'mean':
                x_agg = aggregator.apply({}, x_pre_per)
            else:
                # Learned aggregator needs params; access via 'aggregator' subkey if present
                agg_params = params.get('aggregator', {})
                x_agg = aggregator.apply({'params': agg_params}, x_pre_per)
            # Apply shared decoder unembed kernel (same params as the model's unembed)
            proj_kernel = params['proj_kernel']
            proj_bias = params['proj_bias']
            unembed_kernel = params['unembed_kernel']
            unembed_bias = params['unembed_bias']
            decoder_logits = (
                jax.nn.gelu(x_agg @ proj_kernel + proj_bias) @ unembed_kernel + unembed_bias
            )
            log_probs = jax.nn.log_softmax(decoder_logits.astype(jnp.float32), axis=-1)
            ce = -jnp.take_along_axis(log_probs, decoder_targets[..., None], axis=-1).squeeze(-1)
            ce_loss = (ce * loss_mask).sum() / jnp.maximum(loss_mask.sum(), 1.0)
            return ce_loss, ce_loss, jnp.zeros(())

        def _denoiser_branch(_):
            # Denoiser mode: x0-noised latent (denoiser_z) at random t_per, L2 loss.
            denoiser_t = t_per  # (B, K)
            t_mean = denoiser_t.mean(axis=1)  # (B,)
            x_tokens_flat = x0_per.reshape(batch_size, K * seq_length, -1)
            denoiser_input = get_z_input(
                params, denoiser_z, denoiser_t,
                self_cond_cfg_input=self_cond_cfg_scale,
                x_tokens_per=x0_per,
            )
            net_out_flat, _ = state.apply_fn(
                {"params": params}, denoiser_input, t_mean,
                intra_mask=intra_mask_kk, inter_mask=inter_mask_kk,
                deterministic=False,
                rngs={"dropout": model_dropout_rng},
                self_cond_cfg_scale=self_cond_cfg_scale,
                decoder_step_active=jnp.array(False),
            )
            # net_out_flat: (B, K*L, D_enc)
            net_out_per = net_out_flat.reshape(batch_size, K, seq_length, -1)
            v_pred_per, _ = _net_out_to_v_x_per(net_out_per, denoiser_z_per, denoiser_t)

            v_final_target_per = get_v_target(
                params, denoiser_z, denoiser_t,
                base_v_target_per=v_target_per,
                x_tokens_flat=x_tokens_flat,
            )

            # Per-thought MSE averaged over K and D
            per_dim_loss = (v_pred_per - v_final_target_per) ** 2
            per_token_loss = jnp.mean(per_dim_loss, axis=-1).mean(axis=1)  # (B, L)
            l2_loss = reduce_token_loss(per_token_loss, loss_mask)
            return l2_loss, jnp.zeros(()), l2_loss

        loss, ce_loss, l2_loss = jax.lax.cond(
            decoder_step_active, _decoder_branch, _denoiser_branch, None,
        )
        return loss, (l2_loss, ce_loss)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (l2_loss_val, ce_loss_val)), grads = grad_fn(state.params)

    grads = jax.lax.pmean(grads, axis_name="batch")
    loss = jax.lax.pmean(loss, axis_name="batch")
    l2_loss_val = jax.lax.pmean(l2_loss_val, axis_name="batch")
    ce_loss_val = jax.lax.pmean(ce_loss_val, axis_name="batch")

    new_state = state.apply_gradients(grads=grads, dropout_rng=new_dropout_rng)

    def ema_update(ema_params, params, decay):
        return jax.tree_util.tree_map(lambda e, p: e * decay + p * (1 - decay), ema_params, params)

    is_optimizer_step = (new_state.step % config.grad_accum_steps) == 0
    new_ema_params1 = jax.lax.cond(
        is_optimizer_step,
        lambda: ema_update(state.ema_params1, new_state.params, config.ema_decay1),
        lambda: state.ema_params1,
    )
    new_state = new_state.replace(ema_params1=new_ema_params1, dropout_rng=new_dropout_rng)

    decoder_prob_arr = jnp.asarray(decoder_prob, dtype=jnp.float32)
    denoiser_prob_arr = jnp.asarray(1.0 - decoder_prob, dtype=jnp.float32)
    active_ce_loss_val = jnp.where(
        decoder_prob_arr > 0.0, ce_loss_val / decoder_prob_arr, jnp.zeros_like(ce_loss_val),
    )
    active_l2_loss_val = jnp.where(
        denoiser_prob_arr > 0.0, l2_loss_val / denoiser_prob_arr, jnp.zeros_like(l2_loss_val),
    )
    metrics = {
        "loss": loss,
        "l2_loss": active_l2_loss_val,
        "ce_loss": active_ce_loss_val,
    }
    return new_state, metrics
