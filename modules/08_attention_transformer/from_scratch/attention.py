"""Scaled dot-product attention from scratch (NumPy).

Core formula:
    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

Run: python modules/08_attention_transformer/from_scratch/attention.py
"""

from __future__ import annotations

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
  """Numerically stable softmax along axis."""
  shifted = x - np.max(x, axis=axis, keepdims=True)
  exp_x = np.exp(shifted)
  return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def attention(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Scaled dot-product attention — 缩放点积注意力.

    Args:
        query: (..., seq_q, d_k)
        key:   (..., seq_k, d_k)
        value: (..., seq_k, d_v)
        mask:  optional bool, True = keep, False = mask out (set score to -1e9)

    Returns:
        output: (..., seq_q, d_v) — 加权 value 之和
        weights: (..., seq_q, seq_k) — 注意力权重，每行和为 1
    """
    d_k = query.shape[-1]
    # scores[i,j] = q_i · k_j / sqrt(d_k)
    scores = query @ np.swapaxes(key, -2, -1) / np.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask, scores, -1e9)  # causal / padding mask
    weights = softmax(scores, axis=-1)
    output = weights @ value
    return output, weights


def self_attention(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Self-attention: Q, K, V all projected from the same input x.

    自注意力：同一序列 x 经不同权重矩阵投影为 Q/K/V，
    使每个位置能 attend 到所有位置（含自身）。
    """
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
