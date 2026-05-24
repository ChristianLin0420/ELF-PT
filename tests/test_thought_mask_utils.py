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


from utils.thought_mask_utils import build_thought_masks_with_answer


def test_causal_inter_mask_answer_attends_reasoning():
    # 1 batch, L=4 per slot, K_r=2 reasoning + 1 answer → 12 total positions
    is_cond = jnp.zeros((1, 4), dtype=jnp.bool_)
    is_valid = jnp.ones((1, 4), dtype=jnp.bool_)
    intra, inter = build_thought_masks_with_answer(is_cond, is_valid, K_reasoning=2)
    L = 4
    assert intra.shape == (1, 12, 12)
    assert inter.shape == (1, 12, 12)
    # Sequence layout: [R1 (rows 0..3) | R2 (rows 4..7) | Answer (rows 8..11)].
    # Answer queries (rows 8..11) attending reasoning keys (cols 0..7): allowed
    assert bool(inter[0, 8, 0])
    assert bool(inter[0, 11, 7])
    # Answer queries (rows 8..11) attending answer keys (cols 8..11): allowed
    assert bool(inter[0, 8, 8])
    assert bool(inter[0, 11, 11])
    # Reasoning queries (rows 0..7) attending answer keys (cols 8..11): FORBIDDEN (causal block)
    assert not bool(inter[0, 0, 8])
    assert not bool(inter[0, 7, 11])
    # Reasoning ↔ reasoning is allowed
    assert bool(inter[0, 0, 4])  # R1 sees R2
    assert bool(inter[0, 4, 0])  # R2 sees R1
    assert bool(inter[0, 0, 1])  # R1 sees itself


def test_causal_intra_mask_within_each_slot_unchanged():
    is_cond = jnp.zeros((1, 4), dtype=jnp.bool_)
    is_valid = jnp.ones((1, 4), dtype=jnp.bool_)
    intra, _ = build_thought_masks_with_answer(is_cond, is_valid, K_reasoning=2)
    # Within slot 0 (rows 0..3, cols 0..3): allowed
    assert bool(intra[0, 0, 1])
    # Across slots (row 0 in R1 to col 4 in R2): forbidden (intra isolation)
    assert not bool(intra[0, 0, 4])
    # Across slots: R1 to Answer (row 0 to col 8): forbidden
    assert not bool(intra[0, 0, 8])


def test_causal_padding_zeroed():
    """Padded positions should be unreachable in both intra and inter (even for the answer)."""
    is_cond = jnp.zeros((1, 4), dtype=jnp.bool_)
    is_valid = jnp.array([[1, 1, 0, 0]], dtype=jnp.bool_)  # last 2 positions of each slot are padded
    intra, inter = build_thought_masks_with_answer(is_cond, is_valid, K_reasoning=2)
    # Padded answer position (col 10): even answer-self attention should not reach it
    assert not bool(inter[0, 8, 10])
    # Padded reasoning position (col 2 in R1): not attendable
    assert not bool(inter[0, 8, 2])
    assert not bool(intra[0, 0, 2])
