"""Per-device pmap'd training step for ELF-PT-R: K reasoning + 1 answer slots.

Forked from thought_train_step.py (the symmetric K-thought version).

Key differences vs. thought_train_step.py:
  - K_total = K_reasoning + 1 (K_r reasoning slots + 1 answer slot).
  - intra/inter masks built via build_thought_masks_with_answer (causal inter: answer
    attends reasoning, reasoning does NOT attend answer).
  - Denoiser L2 loss is computed ONLY on the answer slot (model returns sliced answer
    in R-mode); velocity target is likewise restricted to the answer slot.
  - Diversity loss on the K_r reasoning slots' pre-FinalLayer hidden states, captured
    via self.sow('intermediates', 'hidden_pre_final_full', ...) in the model.
  - Decoder CE branch: model returns (B, L, hidden_size) directly (answer slot already
    sliced); no aggregator is needed.
  - train.py routes to this step when config.num_reasoning_thoughts > 0.
"""

from typing import Dict, Tuple

import jax
import jax.numpy as jnp

from utils.train_utils import TrainState
from utils.encoder_utils import encode_text
from utils.sampling_utils import (
    sample_cfg_scale, sample_timesteps,
    net_out_to_v_x, restore_cond,
)
from utils.thought_mask_utils import build_thought_masks_with_answer
from utils.diversity_metrics import reasoning_diversity_loss


Array = jnp.ndarray


def train_step(
    state: TrainState,
    encoder_params: Dict,
    encoder_apply_fn,
    batch: Dict[str, Array],
    config,
) -> Tuple[TrainState, Dict[str, float]]:
    """Perform a single K-reasoning + 1-answer training step."""
    t_eps = config.t_eps
    self_cond_prob = config.self_cond_prob
    latent_mean, latent_std = config.latent_mean, config.latent_std

    decoder_prob = config.decoder_prob
    decoder_noise_scale = config.decoder_noise_scale

    K_r = config.num_reasoning_thoughts        # number of reasoning slots
    K_total = K_r + 1                          # total slots (reasoning + answer)

    new_dropout_rng, current_step_rng = jax.random.split(state.dropout_rng, 2)
    current_step_rng = jax.random.fold_in(current_step_rng, jax.lax.axis_index(axis_name="batch"))
    (
        t_rng, noise_rng, self_cond_mask_rng, self_cond_cfg_rng,
        model_dropout_rng, decoder_step_rng,
        decoder_lambda_rng, decoder_noise_rng,
    ) = jax.random.split(current_step_rng, 8)

    # encoder_attention_mask: cond sees cond, x sees all
    encoder_attention_mask = batch["encoder_attention_mask"]

    # Label drop before encoding
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

    # ── Per-slot noise + single t per example (LaDiR-aligned) ─────────────
    thought_noise_rngs = jax.random.split(noise_rng, K_total)

    # ONE t per example, broadcast to all K_total slots. Matches LaDiR: each
    # data sample under diffusion_batch_mul gets a single t. Removes the
    # t_mean mismatch where the model's time embedding disagreed with per-slot
    # noise levels.
    t_single = sample_timesteps(
        t_rng, batch_size,
        P_mean=config.denoiser_p_mean, P_std=config.denoiser_p_std,
        time_schedule=config.time_schedule,
    )                                                          # (B,)
    t_per = jnp.broadcast_to(t_single[:, None], (batch_size, K_total))  # (B, K_total)

    # Sample K_total independent noises, shape (B, K_total, L, D)
    noise_per = jnp.stack([
        jax.random.normal(thought_noise_rngs[k], x0.shape, dtype=x0.dtype)
        for k in range(K_total)
    ], axis=1)  # (B, K_total, L, D)

    # ── Per-slot x0_per construction ──────────────────────────────────────
    # CoT-VAE recipe: reasoning slots get pre-computed VAE encodings of
    # sampled CoTs; the answer slot gets T5(final-answer) as its target.
    # Legacy mode (no reasoning_targets in batch): broadcast x0 to all slots.
    if "reasoning_targets" in batch:
        reasoning_targets = batch["reasoning_targets"]               # (B, K_r, L, D)
        answer_target = x0[:, None, :, :]                            # (B, 1, L, D)
        x0_per = jnp.concatenate([reasoning_targets, answer_target], axis=1)  # (B, K_total, L, D)
    else:
        x0_per = jnp.broadcast_to(x0[:, None], (batch_size, K_total, seq_length, x0.shape[-1]))

    cond_seq_mask = batch["cond_seq_mask"][:, :, None]  # (B, L, 1)
    attention_mask = batch["attention_mask"]
    if config.pad_token == "pad":
        loss_mask = attention_mask
    else:
        loss_mask = jnp.ones_like(attention_mask)
    loss_mask = loss_mask * (1 - batch["cond_seq_mask"])

    # ── Per-slot loss mask ─────────────────────────────────────────────────
    # Reasoning slots: only the first S*M positions are supervised (mask in batch).
    # Answer slot: existing loss_mask (non-cond, non-pad positions).
    if "reasoning_loss_mask" in batch:
        reasoning_loss_mask = batch["reasoning_loss_mask"].astype(jnp.float32)   # (B, K_r, L)
        loss_mask_per = jnp.concatenate([
            reasoning_loss_mask,
            loss_mask[:, None, :].astype(jnp.float32),                            # answer slot
        ], axis=1)                                                                # (B, K_total, L)
    else:
        loss_mask_per = jnp.broadcast_to(loss_mask[:, None, :], (batch_size, K_total, seq_length))

    # ── Per-slot denoiser_z ────────────────────────────────────────────────
    t_exp = t_per[:, :, None, None]   # (B, K_total, 1, 1)
    denoiser_z_per = (
        t_exp * x0_per + (1 - t_exp) * noise_per * config.denoiser_noise_scale
    )
    # Re-apply cond_seq_mask: only the ANSWER slot has meaningful cond tokens
    # (its target is the T5-encoded [question | "#### N"] sequence).
    # Reasoning slots' content is pure VAE latent — no cond positions to fix.
    cond_mask_exp = cond_seq_mask[:, None]  # (B, 1, L, 1)
    if "reasoning_targets" in batch:
        # Build a (B, K_total, L, 1) mask: cond positions ON only for the answer slot.
        slot_is_answer = jnp.arange(K_total) == (K_total - 1)         # (K_total,)
        slot_is_answer = slot_is_answer[None, :, None, None]           # (1, K_total, 1, 1)
        cond_per_slot = jnp.where(slot_is_answer, cond_mask_exp, jnp.zeros_like(cond_mask_exp))
        denoiser_z_per = jnp.where(cond_per_slot > 0, x0_per, denoiser_z_per)
    else:
        denoiser_z_per = jnp.where(cond_mask_exp > 0, x0_per, denoiser_z_per)

    # Label drop on denoiser_z and x0 (answer slot only in CoT-VAE mode)
    drop = batch["label_drop_mask"][:, None]
    if config.label_drop_prob > 0:
        drop_exp = drop[:, :, None, None]   # (B, 1, 1, 1)
        if "reasoning_targets" in batch:
            # Only zero cond positions of the answer slot
            zero_mask = drop_exp & (cond_per_slot > 0)
            denoiser_z_per = jnp.where(zero_mask, jnp.zeros_like(denoiser_z_per), denoiser_z_per)
            # Update the answer slot's x0 in x0_per (reasoning targets stay as-is)
            x0 = jnp.where(drop[:, :, None] & (cond_seq_mask > 0), jnp.zeros_like(x0), x0)
            answer_x0 = x0[:, None, :, :]                              # (B, 1, L, D)
            x0_per = jnp.concatenate([x0_per[:, :K_r, :, :], answer_x0], axis=1)
        else:
            denoiser_z_per = jnp.where(
                drop_exp & (cond_mask_exp > 0),
                jnp.zeros_like(denoiser_z_per),
                denoiser_z_per,
            )
            x0 = jnp.where(drop[:, :, None] & (cond_seq_mask > 0), jnp.zeros_like(x0), x0)
            x0_per = jnp.broadcast_to(x0[:, None], (batch_size, K_total, seq_length, x0.shape[-1]))

    # Flatten K_total dim into S = K_total * L
    denoiser_z = denoiser_z_per.reshape(batch_size, K_total * seq_length, -1)

    # ── Per-slot velocity target ───────────────────────────────────────────
    v_target_per = (x0_per - denoiser_z_per) / jnp.maximum(1 - t_exp, t_eps)  # (B, K_total, L, D)

    # ── Decoder branch inputs ──────────────────────────────────────────────
    decoder_targets = batch["input_ids"]  # (B, S)
    decoder_step_active = jax.random.bernoulli(decoder_step_rng, decoder_prob)

    # K_total independent decoder lambda + noise
    thought_decoder_lambda_rngs = jax.random.split(decoder_lambda_rng, K_total)
    thought_decoder_noise_rngs = jax.random.split(decoder_noise_rng, K_total)

    decoder_lambda_t_per = jnp.stack([
        jax.nn.sigmoid(
            jax.random.normal(thought_decoder_lambda_rngs[k], (batch_size * seq_length,))
            * config.decoder_p_std + config.decoder_p_mean
        ).reshape(batch_size, seq_length, 1)
        for k in range(K_total)
    ], axis=1)  # (B, K_total, L, 1)

    decoder_noise_per = jnp.stack([
        jax.random.normal(thought_decoder_noise_rngs[k], x0.shape, dtype=x0.dtype)
        * decoder_noise_scale
        for k in range(K_total)
    ], axis=1)  # (B, K_total, L, D)

    decoder_z_per = (
        decoder_lambda_t_per * x0_per + (1 - decoder_lambda_t_per) * decoder_noise_per
    )  # (B, K_total, L, D)
    decoder_z = decoder_z_per.reshape(batch_size, K_total * seq_length, -1)

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

    # ── Build causal K-group attention masks ─────────────────────────────
    # Use build_thought_masks_with_answer: answer sees reasoning, reasoning ↛ answer.
    intra_mask_kk, inter_mask_kk = build_thought_masks_with_answer(
        batch["cond_seq_mask"].astype(jnp.bool_),
        batch["attention_mask"].astype(jnp.bool_),
        K_reasoning=K_r,
    )
    # Tile encoder_attention_mask to (B, K_total*L, K_total*L) and AND it in.
    enc_mask_kk = jnp.tile(encoder_attention_mask, (1, K_total, K_total))
    intra_mask_kk = intra_mask_kk & (enc_mask_kk > 0)
    inter_mask_kk = inter_mask_kk & (enc_mask_kk > 0)

    # ── Helpers ────────────────────────────────────────────────────────────

    def reduce_token_loss(per_token_loss, mask):
        mask = mask.astype(per_token_loss.dtype)
        safe_loss = jnp.where(mask > 0, per_token_loss, jnp.zeros_like(per_token_loss))
        return (safe_loss * mask).sum() / jnp.maximum(mask.sum(), 1.0)

    def get_z_input(params, z_flat, t_input, self_cond_cfg_input, x_tokens_per):
        """Self-conditioning on the flattened (B, K_total*L, D) representation."""
        if self_cond_prob == 0:
            return z_flat
        x_tokens_flat = x_tokens_per.reshape(batch_size, K_total * seq_length, -1)
        cond_mask_kl = jnp.tile(cond_seq_mask, (1, K_total, 1))  # (B, K_total*L, 1)
        z_uncond = restore_cond(jnp.zeros_like(z_flat), x_tokens_flat, cond_mask_kl)
        z_with_zeros = jnp.concatenate([z_flat, z_uncond], axis=-1)
        t_mean = t_input.mean(axis=1) if t_input.ndim == 2 else t_input
        # In R-mode, model.apply returns (answer_slot, None). We only need it for
        # the self-cond forward; the result is discarded (stop_gradient below).
        net_out_init_tuple = state.apply_fn(
            {"params": params}, z_with_zeros, t_mean,
            intra_mask=intra_mask_kk, inter_mask=inter_mask_kk,
            deterministic=True,
            self_cond_cfg_scale=self_cond_cfg_input,
        )
        net_out_init_tuple = jax.lax.stop_gradient(net_out_init_tuple)
        # net_out_init_tuple[0] shape: (B, L, D_enc) — answer slot only in R-mode.
        # For self-cond we need the full (B, K_total*L, D) representation, but the
        # R-mode model no longer returns that. We use zeros for reasoning slots and
        # the answer-slot prediction for the answer slot only.
        net_out_answer = net_out_init_tuple[0]  # (B, L, D_enc)
        # Reconstruct a (B, K_total, L, D_enc) tensor: zeros for reasoning, prediction for answer.
        reasoning_placeholder = jnp.zeros(
            (batch_size, K_r, seq_length, net_out_answer.shape[-1]),
            dtype=net_out_answer.dtype,
        )
        net_out_4d = jnp.concatenate(
            [reasoning_placeholder, net_out_answer[:, None, :, :]], axis=1
        )  # (B, K_total, L, D_enc)
        net_out_per_sc = net_out_4d
        # Compute v/x only for the reconstructed (answer-only) prediction.
        # Since reasoning slots are zero, their x_pred is unreliable; we use x0 for them.
        BK = batch_size * K_total
        net_flat = net_out_per_sc.reshape(BK, seq_length, -1)
        z_flat_bk = denoiser_z_per.reshape(BK, seq_length, -1)
        t_flat = t_input.reshape(BK) if t_input.ndim == 2 else jnp.tile(t_input, K_total)
        v_flat, x_flat = net_out_to_v_x(net_flat, z_flat_bk, t_flat, t_eps)
        x_per = x_flat.reshape(batch_size, K_total, seq_length, -1)
        x_per = jnp.where(cond_mask_exp > 0, x0_per, x_per)
        x_pred_flat = x_per.reshape(batch_size, K_total * seq_length, -1)
        x_tokens_flat = x_tokens_per.reshape(batch_size, K_total * seq_length, -1)
        x_pred_cond = x_pred_flat * use_self_cond_mask.astype(z_flat.dtype)
        x_pred_cond = jnp.where(
            jnp.tile(cond_seq_mask, (1, K_total, 1)) > 0, x_tokens_flat, x_pred_cond
        )
        return jnp.concatenate([z_flat, x_pred_cond], axis=-1)

    # ── Loss function ──────────────────────────────────────────────────────

    def loss_fn(params):

        def _decoder_branch(_):
            # Decoder mode: CE loss on answer slot.
            # In R-mode, model with return_pre_unembed=True returns (B, L, hidden_size)
            # already sliced to the answer slot — no aggregator needed.
            decoder_t = jnp.ones((batch_size,))
            decoder_input = (
                jnp.concatenate([decoder_z, jnp.zeros_like(decoder_z)], axis=-1)
                if config.self_cond_prob > 0 else decoder_z
            )
            x_pre_answer, _ = state.apply_fn(
                {"params": params}, decoder_input, decoder_t,
                intra_mask=intra_mask_kk, inter_mask=inter_mask_kk,
                deterministic=False,
                rngs={"dropout": model_dropout_rng},
                self_cond_cfg_scale=self_cond_cfg_scale,
                decoder_step_active=jnp.array(True),
                return_pre_unembed=True,
            )
            # x_pre_answer: (B, L, hidden_size). No aggregator — answer slot directly.
            proj_kernel = params['proj_kernel']
            proj_bias   = params['proj_bias']
            unembed_kernel = params['unembed_kernel']
            unembed_bias   = params['unembed_bias']
            decoder_logits = (
                jax.nn.gelu(x_pre_answer @ proj_kernel + proj_bias)
                @ unembed_kernel + unembed_bias
            )
            log_probs = jax.nn.log_softmax(decoder_logits.astype(jnp.float32), axis=-1)
            ce = -jnp.take_along_axis(log_probs, decoder_targets[..., None], axis=-1).squeeze(-1)
            ce_loss = (ce * loss_mask).sum() / jnp.maximum(loss_mask.sum(), 1.0)
            # div_loss is zero in the decoder branch (no hidden_pre_final sow here)
            return ce_loss, ce_loss, jnp.zeros(()), jnp.zeros(())

        def _denoiser_branch(_):
            # LaDiR-style: MSE over ALL K_total slots toward per-slot v_target.
            # Each slot has independent noise (per-slot ε_k) and shared t (single
            # t per example, broadcast). Diversity is INFERENCE-ONLY; training-time
            # diversity is preserved for ablation but governed by diversity_loss_weight.
            denoiser_t = t_per[:, 0]   # (B,) — all slots share the same t in this design
            denoiser_input = get_z_input(
                params, denoiser_z, denoiser_t,
                self_cond_cfg_input=self_cond_cfg_scale,
                x_tokens_per=x0_per,
            )
            # In R-mode the model now returns the FULL sequence (B, K_total*L, D_enc).
            (net_out_full, _decoder_logits_unused), sow_state = state.apply_fn(
                {"params": params}, denoiser_input, denoiser_t,
                intra_mask=intra_mask_kk, inter_mask=inter_mask_kk,
                deterministic=False,
                rngs={"dropout": model_dropout_rng},
                self_cond_cfg_scale=self_cond_cfg_scale,
                decoder_step_active=jnp.array(False),
                mutable=['intermediates'],
            )
            # Reshape (B, K_total*L, D) -> (B, K_total, L, D); per-slot masked MSE.
            B_, S_, D_ = net_out_full.shape
            net_out_per = net_out_full.reshape(B_, K_total, seq_length, D_)
            per_dim_loss = (net_out_per - v_target_per) ** 2                # (B, K_total, L, D)
            per_token_loss = jnp.mean(per_dim_loss, axis=-1)                # (B, K_total, L)
            # Per-slot masking (reasoning slots: VAE-latent positions only;
            # answer slot: non-cond text positions only).
            masked = per_token_loss * loss_mask_per
            l2_loss = masked.sum() / jnp.maximum(loss_mask_per.sum(), 1.0)

            # ── Optional diversity loss (kept for ablation; weight 0 by default) ──
            h_full = sow_state['intermediates']['hidden_pre_final_full'][0]
            B_h, S_h, H_h = h_full.shape
            h_full_4d = h_full.reshape(B_h, K_total, seq_length, H_h)
            t_reasoning = t_per[:, :K_r].mean(axis=1)
            t_gate = t_reasoning if config.diversity_loss_t_gating else None
            div_loss = reasoning_diversity_loss(h_full_4d, K_reasoning=K_r, t_gating=t_gate)

            total_l2 = l2_loss + config.diversity_loss_weight * div_loss
            return total_l2, jnp.zeros(()), l2_loss, div_loss

        loss, ce_loss, l2_loss, div_loss = jax.lax.cond(
            decoder_step_active, _decoder_branch, _denoiser_branch, None,
        )
        return loss, (l2_loss, ce_loss, div_loss)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (l2_loss_val, ce_loss_val, div_loss_val)), grads = grad_fn(state.params)

    grads = jax.lax.pmean(grads, axis_name="batch")
    loss = jax.lax.pmean(loss, axis_name="batch")
    l2_loss_val = jax.lax.pmean(l2_loss_val, axis_name="batch")
    ce_loss_val = jax.lax.pmean(ce_loss_val, axis_name="batch")
    div_loss_val = jax.lax.pmean(div_loss_val, axis_name="batch")

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
        "diversity_loss": div_loss_val,
    }
    return new_state, metrics
