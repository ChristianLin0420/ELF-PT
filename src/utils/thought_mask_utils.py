"""Mask builders for ELF-PT parallel-thought attention."""
import jax.numpy as jnp


def build_thought_masks(is_cond, is_valid, K, xp=jnp):
    """Build intra-group and inter-group attention masks.

    Inputs:
      is_cond:  (B, L) bool. True for condition tokens.
      is_valid: (B, L) bool. True for non-padding tokens.
      K:        number of thought groups.
      xp:       array namespace (default jnp). Pass numpy for CPU precomputation in data loaders.

    Returns:
      intra_mask: (B, K*L, K*L) bool. True where attention is allowed.
        - Cond keys visible to all groups.
        - Non-cond keys only visible within their own group.
        - Padded queries/keys never attend.
      inter_mask: (B, K*L, K*L) bool. True for all (valid_query, valid_key) pairs.
    """
    B, L = is_cond.shape
    is_cond_k = xp.tile(is_cond, (1, K))      # (B, K*L)
    is_valid_k = xp.tile(is_valid, (1, K))    # (B, K*L)

    group_id = xp.repeat(xp.arange(K), L)[None, :]
    group_id = xp.broadcast_to(group_id, (B, K * L))

    valid_pair = is_valid_k[:, :, None] & is_valid_k[:, None, :]
    inter_mask = valid_pair.astype(xp.bool_)

    same_group = group_id[:, :, None] == group_id[:, None, :]
    key_is_cond = is_cond_k[:, None, :]
    allowed = same_group | key_is_cond
    intra_mask = (valid_pair & allowed).astype(xp.bool_)
    return intra_mask, inter_mask


def build_thought_masks_with_answer(is_cond, is_valid, K_reasoning, xp=jnp):
    """Build intra/inter masks for K reasoning + 1 answer layout (causal inter).

    Inputs:
      is_cond:     (B, L) bool. True for condition tokens within each slot.
      is_valid:    (B, L) bool. True for non-padding tokens.
      K_reasoning: number of reasoning slots; answer is the (K_reasoning+1)-th slot.

    Sequence layout (length (K_reasoning+1)*L):
      [reasoning_1 | reasoning_2 | ... | reasoning_K | answer]

    Returns:
      intra_mask: (B, (K_r+1)*L, (K_r+1)*L) bool. Standard intra: each slot sees
        itself + cond keys across all slots; non-cond across slots is forbidden.
      inter_mask: (B, (K_r+1)*L, (K_r+1)*L) bool. CAUSAL:
        - Answer queries attend all reasoning + answer keys.
        - Reasoning queries attend reasoning + cond keys, but NOT answer keys.
    """
    B, L = is_cond.shape
    K_total = K_reasoning + 1
    intra_sym, inter_sym = build_thought_masks(is_cond, is_valid, K=K_total, xp=xp)
    # Causal mask: zero out (reasoning_query, answer_key) block.
    ans_start = K_reasoning * L
    ans_end = (K_reasoning + 1) * L
    rows = xp.arange(K_total * L)
    cols = xp.arange(K_total * L)
    is_reasoning_query = (rows < ans_start)[None, :, None]
    is_answer_key = ((cols >= ans_start) & (cols < ans_end))[None, None, :]
    forbidden = is_reasoning_query & is_answer_key
    inter_causal = inter_sym & (~forbidden)
    return intra_sym, inter_causal
