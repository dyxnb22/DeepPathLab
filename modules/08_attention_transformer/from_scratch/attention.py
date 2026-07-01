"""Scaled dot-product attention from scratch."""

from __future__ import annotations

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def attention(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Scaled dot-product attention.

    query: (..., seq_q, d_k)
    key:   (..., seq_k, d_k)
    value: (..., seq_k, d_v)
    Returns: output (..., seq_q, d_v), weights (..., seq_q, seq_k)
    """
    d_k = query.shape[-1]
    scores = query @ np.swapaxes(key, -2, -1) / np.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask, scores, -1e9)
    weights = softmax(scores, axis=-1)
    output = weights @ value
    return output, weights


def self_attention(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Self-attention on input sequence x: (seq_len, d_model)."""
    q = x @ W_q
    k = x @ W_k
    v = x @ W_v
    return attention(q, k, v)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    seq_len, d_model = 6, 8
    x = rng.normal(size=(seq_len, d_model))
    W_q = rng.normal(scale=0.1, size=(d_model, d_model))
    W_k = rng.normal(scale=0.1, size=(d_model, d_model))
    W_v = rng.normal(scale=0.1, size=(d_model, d_model))

    out, weights = self_attention(x, W_q, W_k, W_v)
    print("Output shape:", out.shape)
    print("Attention weights shape:", weights.shape)
    print("Weights row sums (should be 1):", weights.sum(axis=-1))
