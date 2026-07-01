"""Copy problem dataset for long-range dependency experiments."""

from __future__ import annotations

import numpy as np


def copy_problem_shapes(seq_len: int, delay: int) -> tuple[int, int]:
    """Return (L, total_len) where total_len = 2*L + delay."""
    if (seq_len - delay) % 2 != 0 or seq_len <= delay:
        raise ValueError("seq_len must be 2*L + delay with L >= 1")
    L = (seq_len - delay) // 2
    return L, seq_len


def generate_copy_batch(
    batch_size: int,
    seq_len: int = 30,
    delay: int = 10,
    vocab_size: int = 8,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate copy problem sequences.

    Layout: [L symbols][D blanks][L recall outputs]
    - Input: symbols in first L steps, blank (vocab_size) for rest
    - Target: -1 (ignore) for first L+D steps, then symbols to recall

    Returns:
        x: (batch, seq_len) int token ids
        y: (batch, seq_len) int targets, -1 = ignore
        mask: (batch, seq_len) bool, True where loss applies
    """
    rng = np.random.default_rng(seed)
    L, total = copy_problem_shapes(seq_len, delay)
    blank = vocab_size

    x = np.full((batch_size, total), blank, dtype=np.int64)
    y = np.full((batch_size, total), -1, dtype=np.int64)

    symbols = rng.integers(0, vocab_size, size=(batch_size, L))
    x[:, :L] = symbols
    y[:, L + delay :] = symbols
    mask = y >= 0
    return x, y, mask


def one_hot_batch(x: np.ndarray, vocab_size: int) -> np.ndarray:
    """Convert token ids to one-hot (batch, seq, vocab+1)."""
    n_classes = vocab_size + 1
    batch, seq_len = x.shape
    out = np.zeros((batch, seq_len, n_classes))
    out[np.arange(batch)[:, None], np.arange(seq_len), x] = 1.0
    return out
