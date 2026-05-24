"""Aggregators that reduce K parallel thought embeddings to one (B, L, D) tensor."""
import jax.numpy as jnp
import flax.linen as nn


class MeanPoolAggregator(nn.Module):
    """Average across the K dimension. No learnable parameters."""

    @nn.compact
    def __call__(self, x):
        """x: (B, K, L, D) -> (B, L, D)."""
        return x.mean(axis=1)


class LearnedWeightAggregator(nn.Module):
    """Per-(B, L) softmax-weighted sum across K. Weights are predicted from
    a small MLP over the K-thought summary statistics (mean and std).

    Captures the softmax weights as 'thought_weights' intermediate (shape (B, L, K, 1))
    so tests can verify the sum-to-one property.
    """
    hidden_dim: int = 64

    @nn.compact
    def __call__(self, x):
        """x: (B, K, L, D) -> (B, L, D)."""
        B, K, L, D = x.shape
        summary = jnp.concatenate([x.mean(axis=1), x.std(axis=1)], axis=-1)  # (B, L, 2D)
        h = nn.gelu(nn.Dense(self.hidden_dim, name='proj1')(summary))
        logits = nn.Dense(K, name='proj2')(h)                                 # (B, L, K)
        w = nn.softmax(logits, axis=-1)[..., None]                            # (B, L, K, 1)
        self.sow('intermediates', 'thought_weights', w)
        x_blkd = x.transpose(0, 2, 1, 3)                                      # (B, L, K, D)
        return (w * x_blkd).sum(axis=2)


def get_aggregator(config):
    """Build an aggregator module instance from config.thought_aggregation."""
    kind = getattr(config, 'thought_aggregation', 'mean')
    if kind == 'mean':
        return MeanPoolAggregator()
    if kind == 'learned':
        return LearnedWeightAggregator()
    raise ValueError(f"unknown thought_aggregation: {kind!r}; expected 'mean' or 'learned'")
