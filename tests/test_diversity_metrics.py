import jax, jax.numpy as jnp
from utils.diversity_metrics import pairwise_thought_diversity, oracle_pass_at_k


def test_identical_thoughts_have_zero_diversity():
    x = jnp.ones((2, 4, 8, 16))  # B=2, K=4, L=8, D=16 — all identical
    d = pairwise_thought_diversity(x)
    assert float(d) < 1e-6


def test_random_thoughts_have_positive_diversity():
    rng = jax.random.PRNGKey(0)
    x = jax.random.normal(rng, (1, 4, 8, 16))
    d = pairwise_thought_diversity(x)
    assert 0 < float(d) < 2.0   # cosine distance is in [0, 2]


def test_opposite_thoughts_have_max_diversity():
    """Two thoughts that are exact opposites should have cosine distance = 2."""
    x_pos = jnp.ones((1, 1, 4, 8))
    x_neg = -jnp.ones((1, 1, 4, 8))
    x = jnp.concatenate([x_pos, x_neg], axis=1)  # (1, 2, 4, 8)
    d = pairwise_thought_diversity(x)
    assert abs(float(d) - 2.0) < 1e-5


def test_oracle_pass_at_k_picks_best():
    preds_per_example = [["xyz", "the quick brown fox", "abc"]]   # K=3 candidates
    refs = ["the quick brown fox jumps"]
    def scorer(p, r):
        return float(p in r)
    s = oracle_pass_at_k(preds_per_example, refs, scorer, K=3)
    assert s == 1.0


def test_oracle_pass_at_k_averages_over_examples():
    """Average of best-of-K over 2 examples."""
    preds = [
        ["bad", "good"],     # K=2: best score 1.0
        ["meh", "ok"],       # K=2: best score 0.0
    ]
    refs = ["good", "great"]
    def scorer(p, r):
        return float(p == r)
    s = oracle_pass_at_k(preds, refs, scorer, K=2)
    assert s == 0.5


def test_oracle_pass_at_k_validates_input():
    """Must raise if preds row length != K."""
    preds = [["a", "b"]]
    refs = ["a"]
    import pytest
    with pytest.raises(ValueError):
        oracle_pass_at_k(preds, refs, lambda p, r: 1.0, K=3)
