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


def test_intra_mask_actually_affects_output():
    """A non-trivial intra_mask must produce different output than all-ones intra_mask.

    Note: FinalLayer uses zero-init weights, so the final output is always zero at
    initialisation. We compare the last transformer block's hidden states (captured
    via capture_intermediates) to verify the mask is actually plumbed through the
    attention layers.
    """
    cls = ELF_PT_models['ELF-PT-B']
    model = cls(text_encoder_dim=512, max_length=64, num_thoughts=2,
                block_pattern="intra,inter", vocab_size=32100)
    B, L_per, K = 1, 64, 2
    S = K * L_per
    x = jax.random.normal(jax.random.PRNGKey(0), (B, S, 512))
    t = jnp.ones((B,))

    ones_mask = jnp.ones((B, S, S), dtype=jnp.bool_)

    # block-diagonal: each thought attends only to itself
    blk = jnp.zeros((B, S, S), dtype=jnp.bool_)
    blk = blk.at[:, :L_per, :L_per].set(True)
    blk = blk.at[:, L_per:, L_per:].set(True)

    rng = jax.random.PRNGKey(1)
    params = model.init(rng, x, t, intra_mask=ones_mask, inter_mask=ones_mask)

    # Capture intermediate block outputs: FinalLayer has zero-init weights so the
    # model output is zero at initialisation regardless of the mask.
    _, state_ones = model.apply(params, x, t, intra_mask=ones_mask, inter_mask=ones_mask,
                                capture_intermediates=True)
    _, state_blk = model.apply(params, x, t, intra_mask=blk, inter_mask=ones_mask,
                               capture_intermediates=True)

    # Last transformer block hidden states must differ — confirms intra_mask is
    # plumbed through the attention layers.
    depth = 12  # ELF-PT-B depth
    last_block_key = f'blocks_{depth - 1}'
    hidden_ones = state_ones['intermediates'][last_block_key]['__call__'][0]
    hidden_blk = state_blk['intermediates'][last_block_key]['__call__'][0]

    assert not jnp.allclose(hidden_ones, hidden_blk, atol=1e-4), \
        "intra_mask appears to be ignored — last block hidden states identical with very different masks"


def test_elf_pt_k4_forward_pass():
    cls = ELF_PT_models['ELF-PT-B']
    model = cls(text_encoder_dim=512, max_length=32, num_thoughts=4,
                block_pattern="intra,inter", vocab_size=32100)
    B, L_per, K = 1, 32, 4
    S = K * L_per
    x = jnp.ones((B, S, 512))
    t = jnp.ones((B,))
    mask = jnp.ones((B, S, S), dtype=jnp.bool_)
    rng = jax.random.PRNGKey(0)
    params = model.init(rng, x, t, intra_mask=mask, inter_mask=mask)
    out, _ = model.apply(params, x, t, intra_mask=mask, inter_mask=mask)
    assert out.shape == (B, S, 512)


def test_elf_pt_return_pre_unembed():
    cls = ELF_PT_models['ELF-PT-B']
    model = cls(text_encoder_dim=512, max_length=32, num_thoughts=2,
                block_pattern="intra,inter", vocab_size=32100)
    B, L_per, K = 1, 32, 2
    S = K * L_per
    x = jnp.ones((B, S, 512))
    t = jnp.ones((B,))
    mask = jnp.ones((B, S, S), dtype=jnp.bool_)
    rng = jax.random.PRNGKey(0)
    params = model.init(rng, x, t, intra_mask=mask, inter_mask=mask)
    out, second = model.apply(params, x, t, intra_mask=mask, inter_mask=mask, return_pre_unembed=True)
    assert out.shape == (B, S, 768)   # hidden_size of ELF-PT-B is 768
    assert second is None
