"""Parallel-thought attention blocks for ELF-PT."""
import jax
import jax.numpy as jnp
import flax.linen as nn

from modules.layers import (
    Attention, BottleneckTextProj, FinalLayer, RMSNorm, SwiGLUFFN,
    TextRotaryEmbeddingFast, TimestepEmbedder,
    DEFAULT_KERNEL_INIT, DEFAULT_BIAS_INIT, NORMAL_INIT_002,
)
from modules.model import ELF, ELFBlock


class IntraGroupBlock(ELFBlock):
    """Standard ELF block; caller supplies an intra-group attention mask.

    Functionally identical to ELFBlock — the class name signals the calling
    convention (intra-group mask expected) rather than a behavioral difference.
    Kept as a distinct class so future per-group modifications (e.g., group-aware
    positional encoding) can override here without touching ELFBlock callers.
    """
    pass


class InterGroupBlock(nn.Module):
    """ELF block with zero-init output projections so it starts as identity.
    Caller supplies an inter-group attention mask."""
    hidden_size: int
    num_heads: int
    mlp_ratio: float = 4.0
    attn_drop: float = 0.0
    proj_drop: float = 0.0
    zero_init_out: bool = True

    @nn.compact
    def __call__(self, x, rope_fn=None, attention_mask=None, deterministic=True):
        mlp_hidden = int(self.hidden_size * self.mlp_ratio)
        out_init = nn.initializers.zeros if self.zero_init_out else DEFAULT_KERNEL_INIT
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


class ELF_PT(ELF):
    """ELF with alternating intra-group and inter-group attention blocks.

    The caller is responsible for replicating the input K times along the
    sequence dim and supplying separate (B, K*L, K*L) masks for the two
    block types.
    """
    num_thoughts: int = 1
    block_pattern: str = "intra,inter"
    aggregation: str = "mean"
    num_reasoning_thoughts: int = 0

    def _select_block_cls(self, i):
        pat = [s.strip() for s in self.block_pattern.split(',')]
        kind = pat[i % len(pat)]
        if kind == 'intra':
            return IntraGroupBlock
        elif kind == 'inter':
            return InterGroupBlock
        raise ValueError(f"unknown block kind: {kind}")

    @nn.compact
    def __call__(
        self, x, t, intra_mask=None, inter_mask=None,
        deterministic=True, self_cond_cfg_scale=None, decoder_step_active=None,
        return_pre_unembed: bool = False,
    ):
        """x: (B, K*L, C). t: (B,).
        intra_mask/inter_mask: (B, K*L, K*L), 1=attend 0=masked.

        Note: callers must encode padding/conditioning constraints directly into intra_mask
        and inter_mask. ELF's single-mask attention_mask parameter is not supported here.

        When return_pre_unembed=True, returns (x_agg, None) where x_agg has
        shape (B, L, hidden_size) after aggregating across K thoughts.
        FinalLayer is always constructed to keep its params registered; its
        output is discarded in the return_pre_unembed path.
        """
        patch_size = 1
        head_dim = self.hidden_size // self.num_heads
        B = x.shape[0]

        # Self-conditioning: input is [z, x_pred] when 2x encoder dim
        if x.shape[-1] == 2 * self.text_encoder_dim:
            x = nn.Dense(
                self.text_encoder_dim, use_bias=True,
                kernel_init=DEFAULT_KERNEL_INIT, bias_init=DEFAULT_BIAS_INIT,
                name='self_cond_proj',
            )(x)

        # Text projection (with bottleneck)
        x = BottleneckTextProj(
            self.text_encoder_dim, self.hidden_size, self.bottleneck_dim,
            name='text_proj',
        )(x)

        # Prepend learnable model-mode tokens
        model_mode_offset = 0
        if self.num_model_mode_tokens > 0:
            mode_tokens = jnp.tile(
                self.param('mode_tokens', NORMAL_INIT_002,
                           (1, self.num_model_mode_tokens, self.hidden_size)),
                (B, 1, 1),
            )
            active_gate = jnp.array(False) if decoder_step_active is None else decoder_step_active
            mode_tokens = mode_tokens * active_gate.astype(mode_tokens.dtype)
            x = jnp.concatenate([mode_tokens, x], axis=1)
            model_mode_offset = self.num_model_mode_tokens

        prefix_len = 0
        context_prefix_tokens = self.build_context(t, self_cond_cfg_scale)
        if context_prefix_tokens:
            prefix_tokens = jnp.concatenate(context_prefix_tokens, axis=1)
            prefix_len = prefix_tokens.shape[1]
            x = jnp.concatenate([prefix_tokens, x], axis=1)

        prefix_total = prefix_len + model_mode_offset

        def _extend_mask(m):
            """Extend a (B, S, S) mask to (B, S+prefix_total, S+prefix_total).
            Prefix tokens attend to everything; original tokens attend to prefix."""
            if m is None:
                return None
            B_, S, _ = m.shape
            # Left cols: original queries see prefix keys (ones = attend)
            left = jnp.ones((B_, S, prefix_total), dtype=m.dtype)
            # Bottom rows augmented: (B, S, S+prefix_total)
            extended = jnp.concatenate([left, m], axis=2)
            # Top rows: prefix queries see everything (ones = attend)
            top = jnp.ones((B_, prefix_total, S + prefix_total), dtype=m.dtype)
            # Full: (B, S+prefix_total, S+prefix_total)
            extended = jnp.concatenate([top, extended], axis=1)
            return extended

        intra_mask_x = _extend_mask(intra_mask)
        inter_mask_x = _extend_mask(inter_mask)

        feat_rope = TextRotaryEmbeddingFast(
            dim=head_dim, pt_seq_len=self.max_length,
            ft_seq_len=self.num_thoughts * self.max_length,
            num_empty_token=prefix_total, name='feat_rope',
        )

        q1, q3 = self.depth // 4, self.depth // 4 * 3
        for i in range(self.depth):
            block_cls = self._select_block_cls(i)
            in_drop_range = q3 > i >= q1
            block = block_cls(
                self.hidden_size, self.num_heads, mlp_ratio=self.mlp_ratio,
                attn_drop=self.attn_drop if in_drop_range else 0.0,
                proj_drop=self.proj_drop if in_drop_range else 0.0,
                name=f'blocks_{i}',
            )
            mask_for_this = intra_mask_x if block_cls is IntraGroupBlock else inter_mask_x
            x = block(x, rope_fn=feat_rope, attention_mask=mask_for_this,
                      deterministic=deterministic)

        x = x[:, prefix_total:]

        # Factored decoder unembedding: hidden -> text_encoder_dim -> vocab
        # Register all unembed params unconditionally so they are always initialized.
        bn = self.text_encoder_dim
        proj_kernel = self.param('proj_kernel', DEFAULT_KERNEL_INIT, (self.hidden_size, bn))
        proj_bias = self.param('proj_bias', DEFAULT_BIAS_INIT, (bn,))
        unembed_kernel = self.param('unembed_kernel', DEFAULT_KERNEL_INIT, (bn, self.vocab_size))
        unembed_bias = self.param('unembed_bias', DEFAULT_BIAS_INIT, (self.vocab_size,))

        # Always run FinalLayer so its params are registered during init, even in
        # pre-unembed mode.  Store result but only return it when not pre-unembed.
        output = FinalLayer(self.hidden_size, patch_size, self.text_encoder_dim, name='final_layer')(x)

        # R-mode: num_reasoning_thoughts > 0. LaDiR-style: denoiser branch returns the
        # FULL (B, K_total*L, D) velocity prediction so the caller can MSE all slots
        # against their per-slot v_target. Decoder branch still slices the answer slot
        # for the unembed call (only the answer is decoded to text).
        if self.num_reasoning_thoughts > 0:
            K_total = self.num_reasoning_thoughts + 1
            L_slot = x.shape[1] // K_total
            # Sow full pre-final hidden states (for optional training-time diversity).
            self.sow('intermediates', 'hidden_pre_final_full', x)
            if return_pre_unembed:
                # Decoder branch: only the answer slot is decoded; slice it for unembed.
                return x[:, -L_slot:, :], None
            # Denoiser branch: return full sequence; the train step will reshape to
            # (B, K_total, L, D) and compute per-slot MSE against v_target_per.
            return output, None

        # Pre-unembed return path: aggregate K thoughts internally then return (B, L, hidden_size).
        if return_pre_unembed:
            if self.num_thoughts > 1:
                from modules.thought_aggregation import MeanPoolAggregator, LearnedWeightAggregator
                B_, S, H = x.shape
                L = S // self.num_thoughts
                x_per = x.reshape(B_, self.num_thoughts, L, H)
                if self.aggregation == 'mean':
                    agg = MeanPoolAggregator(name='aggregator')
                elif self.aggregation == 'learned':
                    agg = LearnedWeightAggregator(name='aggregator')
                else:
                    raise ValueError(f"unknown aggregation: {self.aggregation!r}")
                x = agg(x_per)  # (B, L, H)
            return x, None

        decoder_logits = None
        if decoder_step_active is not None:
            decoder_logits = jax.lax.cond(
                decoder_step_active,
                lambda xi: jax.nn.gelu(xi @ proj_kernel + proj_bias) @ unembed_kernel + unembed_bias,
                lambda xi: jnp.zeros((*xi.shape[:2], self.vocab_size), dtype=xi.dtype),
                x,
            )

        return output, decoder_logits


# Factory functions
def ELF_PT_B(**kw): return ELF_PT(depth=12, hidden_size=768,  num_heads=12, **kw)
def ELF_PT_M(**kw): return ELF_PT(depth=24, hidden_size=1056, num_heads=16, **kw)
def ELF_PT_L(**kw): return ELF_PT(depth=32, hidden_size=1280, num_heads=16, **kw)

ELF_PT_models = {
    'ELF-PT-B': ELF_PT_B,
    'ELF-PT-M': ELF_PT_M,
    'ELF-PT-L': ELF_PT_L,
}
