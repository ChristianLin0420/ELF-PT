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


def test_learned_weight_weights_sum_to_one_per_position():
    """Internal property check: the K softmax weights at each (B, L) should sum to 1."""
    agg = LearnedWeightAggregator(hidden_dim=32)
    x = jax.random.normal(jax.random.PRNGKey(0), (1, 4, 8, 16))
    params = agg.init(jax.random.PRNGKey(1), x)
    # Use capture_intermediates to access internal softmax weights:
    out, mutated = agg.apply(params, x, capture_intermediates=True)
    # Walk the captured tree to find the weights tensor; pytest will fail if not present.
    # Implementation hint: name the softmax output 'thought_weights' via self.sow('intermediates', 'thought_weights', w)
    weights = mutated['intermediates']['thought_weights'][0]   # (B, L, K, 1)
    sums = weights.sum(axis=2)
    assert jnp.allclose(sums, jnp.ones_like(sums), atol=1e-5)


def test_factory_returns_mean_by_default():
    class _C: thought_aggregation = "mean"
    assert isinstance(get_aggregator(_C()), MeanPoolAggregator)


def test_factory_returns_learned_when_configured():
    class _C: thought_aggregation = "learned"
    assert isinstance(get_aggregator(_C()), LearnedWeightAggregator)


def test_factory_rejects_unknown():
    class _C: thought_aggregation = "median"
    import pytest
    with pytest.raises(ValueError, match="median"):
        get_aggregator(_C())
