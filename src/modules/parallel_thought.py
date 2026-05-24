"""Parallel-thought attention blocks for ELF-PT."""
import flax.linen as nn

from modules.layers import Attention, DEFAULT_KERNEL_INIT, RMSNorm, SwiGLUFFN
from modules.model import ELFBlock


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
