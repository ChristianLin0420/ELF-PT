import jax, jax.numpy as jnp
from modules.parallel_thought import IntraGroupBlock, InterGroupBlock


def test_inter_block_zero_init_is_identity():
    block = InterGroupBlock(hidden_size=64, num_heads=4, zero_init_out=True)
    rng = jax.random.PRNGKey(0)
    x = jax.random.normal(rng, (2, 16, 64))
    mask = jnp.ones((2, 16, 16), dtype=jnp.int32)
    params = block.init(rng, x, attention_mask=mask)
    y = block.apply(params, x, attention_mask=mask)
    assert jnp.array_equal(y, x), "inter block must start as identity (bit-exact)"


def test_intra_block_shape_preserved():
    block = IntraGroupBlock(hidden_size=64, num_heads=4)
    rng = jax.random.PRNGKey(0)
    x = jax.random.normal(rng, (2, 16, 64))
    mask = jnp.ones((2, 16, 16), dtype=jnp.int32)
    params = block.init(rng, x, attention_mask=mask)
    y = block.apply(params, x, attention_mask=mask)
    assert y.shape == x.shape


def test_inter_block_default_init_is_not_identity():
    """Sanity check: without zero-init, the inter block must transform the input."""
    block = InterGroupBlock(hidden_size=64, num_heads=4, zero_init_out=False)
    rng = jax.random.PRNGKey(0)
    x = jax.random.normal(rng, (2, 16, 64))
    mask = jnp.ones((2, 16, 16), dtype=jnp.bool_)
    params = block.init(rng, x, attention_mask=mask)
    y = block.apply(params, x, attention_mask=mask)
    assert not jnp.allclose(y, x, atol=1e-3), "without zero-init, block should not be identity"
