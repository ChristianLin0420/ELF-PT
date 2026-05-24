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
