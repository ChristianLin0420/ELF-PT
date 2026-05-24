"""Smoke tests for K-thought sampler utilities."""
import jax
import jax.numpy as jnp

from utils.thought_sampling_utils import (
    init_thought_state,
    apply_diversity_repulsion,
)


def test_init_thought_state_shape():
    z = init_thought_state(jax.random.PRNGKey(0), B=2, K=3, L=8, D=16)
    assert z.shape == (2, 3 * 8, 16)


def test_init_thought_state_thoughts_are_independent():
    """The K slices along the K dim should be statistically independent (not identical)."""
    z = init_thought_state(jax.random.PRNGKey(0), B=1, K=4, L=64, D=32)
    z_per = z.reshape(1, 4, 64, 32)
    # Compare pairs; they should differ substantially
    assert not jnp.allclose(z_per[0, 0], z_per[0, 1], atol=1e-3)
    assert not jnp.allclose(z_per[0, 0], z_per[0, 2], atol=1e-3)


def test_diversity_repulsion_noop_at_gamma_zero():
    z = init_thought_state(jax.random.PRNGKey(0), B=2, K=4, L=8, D=16)
    z2 = apply_diversity_repulsion(z, K=4, gamma=0.0, sigma=1.0)
    assert jnp.array_equal(z, z2)


def test_diversity_repulsion_noop_at_k1():
    z = init_thought_state(jax.random.PRNGKey(0), B=2, K=1, L=8, D=16)
    z2 = apply_diversity_repulsion(z, K=1, gamma=0.5, sigma=1.0)
    assert jnp.array_equal(z, z2)


def test_diversity_repulsion_pushes_thoughts_apart():
    """With gamma > 0, K thoughts should move further apart (mean pairwise dist increases)."""
    z = init_thought_state(jax.random.PRNGKey(0), B=2, K=4, L=8, D=16) * 0.1   # close together

    def mean_pair_dist(z, K=4):
        B, S, D = z.shape
        z_per = z.reshape(B, K, S // K, D)
        diff = z_per[:, :, None] - z_per[:, None, :]   # (B, K, K, L, D)
        return float(jnp.sqrt((diff ** 2).sum(axis=-1)).mean())

    d_before = mean_pair_dist(z)
    z2 = apply_diversity_repulsion(z, K=4, gamma=0.5, sigma=1.0)
    d_after = mean_pair_dist(z2)
    assert d_after > d_before, f"Expected d_after ({d_after:.4f}) > d_before ({d_before:.4f})"
