import jax, jax.numpy as jnp
from modules.parallel_thought import ELF_PT_models


def test_elf_pt_b_forward_pass_shape():
    cls = ELF_PT_models['ELF-PT-B']
    model = cls(text_encoder_dim=512, max_length=128, num_thoughts=2,
                block_pattern="intra,inter", vocab_size=32100)
    B, L_per, K = 1, 128, 2
    x = jnp.ones((B, K * L_per, 512))
    t = jnp.ones((B,))
    intra = jnp.ones((B, K * L_per, K * L_per), dtype=jnp.bool_)
    inter = jnp.ones((B, K * L_per, K * L_per), dtype=jnp.bool_)
    rng = jax.random.PRNGKey(0)
    params = model.init(rng, x, t, intra_mask=intra, inter_mask=inter)
    out, _ = model.apply(params, x, t, intra_mask=intra, inter_mask=inter)
    assert out.shape == (B, K * L_per, 512)


def test_elf_pt_k1_forward_pass():
    """K=1 must still produce sensible output."""
    cls = ELF_PT_models['ELF-PT-B']
    model = cls(text_encoder_dim=512, max_length=128, num_thoughts=1,
                block_pattern="intra,inter", vocab_size=32100)
    x = jnp.ones((1, 128, 512))
    t = jnp.ones((1,))
    intra = jnp.ones((1, 128, 128), dtype=jnp.bool_)
    inter = jnp.ones((1, 128, 128), dtype=jnp.bool_)
    rng = jax.random.PRNGKey(0)
    params = model.init(rng, x, t, intra_mask=intra, inter_mask=inter)
    out, _ = model.apply(params, x, t, intra_mask=intra, inter_mask=inter)
    assert out.shape == (1, 128, 512)
