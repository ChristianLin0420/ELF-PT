"""Diversity and Pass@K metrics for K-thought ELF-PT evaluation."""
from typing import Callable, List
import jax.numpy as jnp


def pairwise_thought_diversity(x):
    """Mean pairwise cosine distance across K thoughts.

    Args:
        x: (B, K, L, D). The K thought embeddings per (B, L).

    Returns:
        Scalar: mean over all K*(K-1)/2 pairs and over (B, L) of
        1 - cos(x_i, x_j). Range: [0, 2]; 0 = identical, 1 = orthogonal,
        2 = exact opposites.
    """
    B, K, L, D = x.shape
    if K < 2:
        return jnp.float32(0.0)
    # Normalize along D for cosine similarity
    eps = 1e-8
    norm = jnp.sqrt((x ** 2).sum(axis=-1, keepdims=True) + eps)
    x_n = x / norm                                              # (B, K, L, D)
    # Cosine similarity matrix per (B, L): (B, K, K, L)
    cos = jnp.einsum('bkld,bjld->bkjl', x_n, x_n)
    # Take strictly upper-triangle pairs (i < j) to avoid double-counting and diagonal
    iu = jnp.triu_indices(K, k=1)
    pair_cos = cos[:, iu[0], iu[1], :]                          # (B, K*(K-1)/2, L)
    pair_dist = 1.0 - pair_cos
    return pair_dist.mean()


def reasoning_diversity_loss(hidden, K_reasoning, t_gating=None):
    """Pairwise squared cosine similarity loss on K reasoning slots.

    Minimized when reasoning thoughts are orthogonal. Bounded in [0, 1].

    Args:
        hidden:      (B, K_total, L, D) where K_total = K_reasoning + 1.
                     The last slot (index K_reasoning) is the answer and is excluded.
        K_reasoning: number of reasoning slots; loss is computed over these.
        t_gating:    optional (B,) time tensor. If provided, the loss is multiplied
                     by mean(4 * t * (1 - t)), which peaks at t=0.5 and vanishes at
                     t=0 and t=1. Use the reasoning thoughts' mean time.

    Returns:
        Scalar loss in [0, 1] (or [0, 1] * gate when t_gating is provided).
    """
    if K_reasoning < 2:
        return jnp.float32(0.0)
    eps = 1e-8
    reasoning = hidden[:, :K_reasoning]                              # (B, K_r, L, D)
    norm = reasoning / (jnp.linalg.norm(reasoning, axis=-1, keepdims=True) + eps)
    # Pairwise cosine similarity across the K_r axis
    sim = jnp.einsum('bkld,bjld->bkjl', norm, norm)                   # (B, K_r, K_r, L)
    # Upper-triangle pairs (i < j)
    iu = jnp.triu_indices(K_reasoning, k=1)
    pair_sim = sim[:, iu[0], iu[1], :]                                # (B, P, L)
    loss = (pair_sim ** 2).mean()
    if t_gating is not None:
        t_mean = t_gating.mean()
        gate = 4.0 * t_mean * (1.0 - t_mean)                          # peak 1.0 at t=0.5, 0 at endpoints
        loss = loss * gate
    return loss


def oracle_pass_at_k(preds, refs, scorer: Callable, K: int):
    """Oracle Pass@K: for each example, take the best score among K candidates.

    Args:
        preds: list of length N; each element is a list of K candidate strings.
        refs:  list of length N; each element is the reference (string or list of strings).
        scorer: callable scorer(p, r) -> float. Higher is better.
        K:     expected number of candidates per example.

    Returns:
        Mean over examples of max-over-K scorer(preds[i][k], refs[i]).
    """
    if len(preds) != len(refs):
        raise ValueError(f"preds (n={len(preds)}) and refs (n={len(refs)}) length mismatch")
    if not preds:
        return 0.0
    for i, row in enumerate(preds):
        if len(row) != K:
            raise ValueError(f"preds[{i}] has {len(row)} candidates; expected K={K}")
    best_scores = [max(scorer(p, r) for p in row) for row, r in zip(preds, refs)]
    return sum(best_scores) / len(best_scores)
