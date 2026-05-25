"""Small β-VAE for compressing CoT text into M memory tokens × D-dim latents.

Used by the ELF-PT-R CoT-augmented recipe (see PLAN.md / plan file). The VAE
is trained on T5-encoded CoT hidden states and then used (frozen) to encode
N=8 augmented CoTs per GSM8K example into per-slot targets for the diffusion
training step.

Reuses Attention / SwiGLUFFN / RMSNorm from src/modules/layers.py.
"""
import jax.numpy as jnp
import flax.linen as nn

from modules.layers import (
    Attention, RMSNorm, SwiGLUFFN,
    DEFAULT_KERNEL_INIT, DEFAULT_BIAS_INIT, NORMAL_INIT_002,
)


class _EncoderBlock(nn.Module):
    """Pre-norm self-attention block (no rope, no cond mask — variable length)."""
    hidden_size: int
    num_heads: int
    mlp_ratio: float = 4.0

    @nn.compact
    def __call__(self, x, attention_mask=None, deterministic=True):
        mlp_hidden = int(self.hidden_size * self.mlp_ratio)
        x = x + Attention(
            self.hidden_size, self.num_heads, qkv_bias=True, qk_norm=True, name='attn',
        )(RMSNorm(self.hidden_size, eps=1e-6, name='norm1')(x), rope_fn=None,
          attention_mask=attention_mask, deterministic=deterministic)
        x = x + SwiGLUFFN(self.hidden_size, mlp_hidden, name='mlp')(
            RMSNorm(self.hidden_size, eps=1e-6, name='norm2')(x), deterministic=deterministic,
        )
        return x


class _CrossAttnDecoderBlock(nn.Module):
    """Cross-attention block: queries attend keys/values from z."""
    hidden_size: int
    num_heads: int
    mlp_ratio: float = 4.0

    @nn.compact
    def __call__(self, q, kv, deterministic=True):
        mlp_hidden = int(self.hidden_size * self.mlp_ratio)
        # Manual cross-attention: project q from q, k+v from kv.
        head_dim = self.hidden_size // self.num_heads
        q_n = RMSNorm(self.hidden_size, eps=1e-6, name='norm_q')(q)
        kv_n = RMSNorm(self.hidden_size, eps=1e-6, name='norm_kv')(kv)
        q_proj = nn.Dense(self.hidden_size, use_bias=True, name='q_proj')(q_n)
        k_proj = nn.Dense(self.hidden_size, use_bias=True, name='k_proj')(kv_n)
        v_proj = nn.Dense(self.hidden_size, use_bias=True, name='v_proj')(kv_n)
        B, Q, _ = q_proj.shape
        _, K, _ = k_proj.shape
        q_proj = q_proj.reshape(B, Q, self.num_heads, head_dim).transpose(0, 2, 1, 3)
        k_proj = k_proj.reshape(B, K, self.num_heads, head_dim).transpose(0, 2, 1, 3)
        v_proj = v_proj.reshape(B, K, self.num_heads, head_dim).transpose(0, 2, 1, 3)
        scale = 1.0 / jnp.sqrt(head_dim)
        attn = jnp.einsum('bhqd,bhkd->bhqk', q_proj, k_proj) * scale
        attn = nn.softmax(attn.astype(jnp.float32), axis=-1).astype(q_proj.dtype)
        out = jnp.einsum('bhqk,bhkd->bhqd', attn, v_proj)
        out = out.transpose(0, 2, 1, 3).reshape(B, Q, self.hidden_size)
        out = nn.Dense(self.hidden_size, use_bias=True, name='out_proj')(out)
        x = q + out
        x = x + SwiGLUFFN(self.hidden_size, mlp_hidden, name='mlp')(
            RMSNorm(self.hidden_size, eps=1e-6, name='norm2')(x), deterministic=deterministic,
        )
        return x


class CotEncoder(nn.Module):
    """T5-encoded CoT hidden -> M memory tokens (mu, log_var)."""
    hidden_size: int = 512
    num_layers: int = 2
    num_heads: int = 8
    memory_tokens: int = 16

    @nn.compact
    def __call__(self, x, attention_mask=None, deterministic=True):
        # x: (B, L, hidden_size). attention_mask: (B, L) 1=valid.
        for i in range(self.num_layers):
            x = _EncoderBlock(self.hidden_size, self.num_heads, name=f'enc_{i}')(
                x, attention_mask=attention_mask, deterministic=deterministic,
            )
        # Mean-pool over valid tokens
        if attention_mask is not None:
            mask = attention_mask.astype(x.dtype)[..., None]   # (B, L, 1)
            pooled = (x * mask).sum(axis=1) / jnp.maximum(mask.sum(axis=1), 1.0)
        else:
            pooled = x.mean(axis=1)
        # Linear to (M * H * 2): mu and log_var
        H = self.hidden_size
        M = self.memory_tokens
        head = nn.Dense(M * H * 2, use_bias=True, name='to_mu_logvar')(pooled)
        head = head.reshape(-1, M, H * 2)
        mu, log_var = jnp.split(head, 2, axis=-1)   # each (B, M, H)
        return mu, log_var


class CotDecoder(nn.Module):
    """M memory tokens (z) -> M reconstructed memory tokens."""
    hidden_size: int = 512
    num_heads: int = 8
    memory_tokens: int = 16

    @nn.compact
    def __call__(self, z, deterministic=True):
        # z: (B, M, hidden_size). Use learnable position queries.
        B = z.shape[0]
        M = self.memory_tokens
        H = self.hidden_size
        pos_queries = self.param('pos_queries', NORMAL_INIT_002, (1, M, H))
        q = jnp.tile(pos_queries, (B, 1, 1))
        x = _CrossAttnDecoderBlock(self.hidden_size, self.num_heads, name='cross_dec')(
            q, z, deterministic=deterministic,
        )
        # Final projection
        x = RMSNorm(self.hidden_size, eps=1e-6, name='norm_out')(x)
        out = nn.Dense(self.hidden_size, use_bias=True, name='out_proj',
                       kernel_init=DEFAULT_KERNEL_INIT)(x)
        return out


class CotVAE(nn.Module):
    """β-VAE for CoT compression.

    Train target: mean-pooled T5 hidden over M equal chunks of the CoT
    (so the decoder reconstructs the structure-aware mean per chunk).
    """
    hidden_size: int = 512
    memory_tokens: int = 16
    num_enc_layers: int = 2
    num_heads: int = 8

    @nn.compact
    def __call__(self, x, attention_mask=None, rng=None, deterministic=True):
        # x: (B, L, hidden_size) — T5-encoded CoT hidden states.
        # attention_mask: (B, L) 1=valid.
        # rng: jax PRNGKey for reparameterisation. If None, returns mu (deterministic).
        encoder = CotEncoder(self.hidden_size, self.num_enc_layers, self.num_heads,
                             self.memory_tokens, name='encoder')
        decoder = CotDecoder(self.hidden_size, self.num_heads,
                             self.memory_tokens, name='decoder')
        mu, log_var = encoder(x, attention_mask, deterministic=deterministic)
        if rng is not None and not deterministic:
            std = jnp.exp(0.5 * log_var)
            eps = jax.random.normal(rng, mu.shape, dtype=mu.dtype)
            z = mu + std * eps
        else:
            z = mu
        recon = decoder(z, deterministic=deterministic)
        return recon, mu, log_var, z


def encode_only(params, x, attention_mask, hidden_size=512, memory_tokens=16,
                num_enc_layers=2, num_heads=8):
    """Run the encoder of a frozen CotVAE and return the mean latent (mu).
    Used by encode_cot_vae.py.
    """
    import jax
    model = CotVAE(hidden_size=hidden_size, memory_tokens=memory_tokens,
                   num_enc_layers=num_enc_layers, num_heads=num_heads)
    # Apply with deterministic=True and no rng -> returns mu via recon path,
    # but we want just mu. Call encoder submodule directly:
    @jax.jit
    def _enc(p, xi, mi):
        out = model.apply({'params': p}, xi, mi, rng=None, deterministic=True)
        # out = (recon, mu, log_var, z=mu when rng None)
        return out[1]
    return _enc(params, x, attention_mask)


# Avoid jax import at module level for the @jax.jit inside encode_only
import jax  # noqa: E402
