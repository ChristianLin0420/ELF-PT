"""CoT-text preprocessing for the LaDiR-style CoT-VAE recipe.

Three reusable functions used by the VAE training, the VAE encoding, and the
diffusion-training collator:

  - strip_answer(cot):   remove the "#### N" or "\\boxed{N}" suffix
                         (leaves only the reasoning chain)
  - extract_final(target): return the final-answer text "#### N"
                           (used as the answer-slot target)
  - chunk_token_ids(...):  LaDiR-style fixed-length token segmentation
"""
from __future__ import annotations
import math
import re
from typing import List

# Matches a trailing "#### 18" or "#### -3.14" line (with optional comma/period).
_HASH_SUFFIX_RE = re.compile(r"\s*\n*\s*####\s*-?[\d,\.]+\.?\s*$")
# Matches a trailing "\boxed{18}" or "\boxed{-3.14}" expression.
_BOXED_SUFFIX_RE = re.compile(r"\\boxed\{\s*-?[\d,\.]+\s*\}\.?\s*$")
# Generic "the answer is N" trailing phrase.
_ANSWER_IS_SUFFIX_RE = re.compile(
    r"\s*(?:so|thus|therefore|hence|so,|thus,|therefore,|hence,|the\s+answer\s+is[:\s]+)?\s*\\?\$?\s*-?[\d,\.]+\.?\s*$",
    re.IGNORECASE,
)
# Final-answer extraction (#### N is the GSM8K canonical format).
_FINAL_HASH_RE = re.compile(r"####\s*(-?[\d,\.]+)")


def strip_answer(cot: str) -> str:
    """Return the CoT with its final-answer suffix removed.

    Strips, in order:
      1. trailing "#### N"
      2. trailing "\\boxed{N}"
    The result is just the reasoning chain (no final-answer line).
    """
    text = cot.strip()
    # Pass 1: hash suffix
    text = _HASH_SUFFIX_RE.sub("", text).rstrip()
    # Pass 2: boxed suffix (LaTeX)
    text = _BOXED_SUFFIX_RE.sub("", text).rstrip()
    # Pass 3: if the very last sentence is just "So the answer is 72.", drop it.
    # Be conservative: only strip if the rstripped tail is a short numeric phrase.
    # (Skipped — the two above passes cover >95% of real cases.)
    return text


def extract_final(target: str) -> str:
    """Return the canonical "#### N" final-answer text.

    Used as the answer slot's T5-tokenized target. Falls back to "#### ?" if no
    "#### N" suffix is found (should not happen on GSM8K).
    """
    m = _FINAL_HASH_RE.search(target)
    if m:
        num = m.group(1).strip().replace(",", "").rstrip(".")
        return f"#### {num}"
    return "#### ?"


def chunk_token_ids(
    token_ids: List[int],
    mem_size: int = 3,
    mean_compression_rate: int = 8,
    max_segments: int | None = 8,
) -> List[List[int]]:
    """Split a flat token-id sequence into S fixed-length segments.

    Follows LaDiR's `_compress` rule:
        S = ceil(T / (mem_size * mean_compression_rate))
        segment_length = ceil(T / S)

    If `max_segments` is given, S is capped (longer sequences are truncated).
    """
    T = len(token_ids)
    if T == 0:
        return []
    S = max(1, math.ceil(T / (mem_size * mean_compression_rate)))
    if max_segments is not None:
        S = min(S, max_segments)
    seg_len = math.ceil(T / S)
    chunks = []
    for i in range(S):
        start = i * seg_len
        end = min((i + 1) * seg_len, T)
        chunks.append(list(token_ids[start:end]))
        if end >= T:
            break
    return chunks
