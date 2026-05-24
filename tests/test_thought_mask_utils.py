import jax.numpy as jnp
from utils.thought_mask_utils import build_thought_masks


def test_intra_blocks_cross_group_attention():
    is_cond = jnp.zeros((2, 4), dtype=jnp.bool_)
    is_valid = jnp.ones((2, 4), dtype=jnp.bool_)
    intra, inter = build_thought_masks(is_cond, is_valid, K=3)
    assert intra.shape == (2, 12, 12)
    assert inter.shape == (2, 12, 12)
    assert int(intra[0, 0, 4]) == 0       # G0 -> G1 forbidden
    assert int(intra[0, 0, 1]) == 1       # same group OK
    assert int(inter[0, 0, 4]) == 1       # inter allows cross-group


def test_cond_tokens_visible_to_all_groups_in_intra_mask():
    is_cond = jnp.array([[1, 1, 0, 0]], dtype=jnp.bool_)   # first 2 positions are cond
    is_valid = jnp.ones((1, 4), dtype=jnp.bool_)
    intra, _ = build_thought_masks(is_cond, is_valid, K=2)
    assert int(intra[0, 0, 4]) == 1       # cond G0 -> cond G1
    assert int(intra[0, 2, 0]) == 1       # non-cond G0 -> cond G0
    assert int(intra[0, 2, 4]) == 1       # non-cond G0 -> cond G1 (cond is shared)
    assert int(intra[0, 2, 6]) == 0       # non-cond G0 -> non-cond G1: forbidden


def test_padding_zeroed():
    is_cond = jnp.zeros((1, 4), dtype=jnp.bool_)
    is_valid = jnp.array([[1, 1, 0, 0]], dtype=jnp.bool_)  # last 2 padded
    intra, inter = build_thought_masks(is_cond, is_valid, K=2)
    assert int(inter[0, 0, 2]) == 0       # padded key never attended to
    assert int(inter[0, 0, 6]) == 0       # padded key in group 1
    assert int(intra[0, 0, 2]) == 0       # intra: padded key in group 0 zeroed
    assert int(intra[0, 0, 6]) == 0       # intra: padded key in group 1 zeroed


def test_k1_intra_equals_inter():
    is_cond = jnp.array([[1, 1, 0, 0]], dtype=jnp.bool_)
    is_valid = jnp.array([[1, 1, 1, 0]], dtype=jnp.bool_)
    intra, inter = build_thought_masks(is_cond, is_valid, K=1)
    assert jnp.all(intra == inter)
    assert intra.shape == (1, 4, 4)
