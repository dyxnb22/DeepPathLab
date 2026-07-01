"""Minimal transformer encoder block from scratch (NumPy, single-head).

Architecture (Pre-LN style simplified as post-residual LN here):
    x -> SelfAttention -> Add&Norm -> FFN -> Add&Norm -> out

Run: python modules/08_attention_transformer/from_scratch/transformer_block.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from attention import attention


def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Layer normalization over the last dimension (per-token)."""
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return gamma * (x - mean) / np.sqrt(var + eps) + beta


def transformer_encoder_block(
    x: np.ndarray,
    params: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """One encoder block: self-attention + FFN with residual connections.

    Args:
        x: (seq_len, d_model)
        params: W_q/k/v/o, FFN weights, LayerNorm gamma/beta

    Returns:
        output: (seq_len, d_model)
        weights: (seq_len, seq_len) attention weights from first sub-layer
    """
    d_model = x.shape[-1]

    # --- Sub-layer 1: Self-Attention + residual + LayerNorm ---
    q = x @ params["W_q"]
    k = x @ params["W_k"]
    v = x @ params["W_v"]
    attn_out, weights = attention(q, k, v)
    attn_out = attn_out @ params["W_o"]  # output projection
    x = layer_norm(x + attn_out, params["ln1_g"], params["ln1_b"])  # skip from Module 05

    # --- Sub-layer 2: FFN + residual + LayerNorm ---
    h = np.maximum(0, x @ params["W_ff1"] + params["b_ff1"])  # ReLU FFN
    ff = h @ params["W_ff2"] + params["b_ff2"]
    x = layer_norm(x + ff, params["ln2_g"], params["ln2_b"])

    return x, weights


def init_transformer_block(d_model: int, d_ff: int, seed: int = 0) -> dict[str, np.ndarray]:
    """Initialize single-head encoder block parameters."""
    rng = np.random.default_rng(seed)
    s = 0.1
    return {
        "W_q": rng.normal(scale=s, size=(d_model, d_model)),
        "W_k": rng.normal(scale=s, size=(d_model, d_model)),
        "W_v": rng.normal(scale=s, size=(d_model, d_model)),
        "W_o": rng.normal(scale=s, size=(d_model, d_model)),
        "W_ff1": rng.normal(scale=s, size=(d_model, d_ff)),
        "b_ff1": np.zeros(d_ff),
        "W_ff2": rng.normal(scale=s, size=(d_ff, d_model)),
        "b_ff2": np.zeros(d_model),
        "ln1_g": np.ones(d_model),
        "ln1_b": np.zeros(d_model),
        "ln2_g": np.ones(d_model),
        "ln2_b": np.zeros(d_model),
    }


if __name__ == "__main__":
    d_model, d_ff, seq_len = 16, 32, 8
    rng = np.random.default_rng(42)
    x = rng.normal(size=(seq_len, d_model))
    params = init_transformer_block(d_model, d_ff)
    out, weights = transformer_encoder_block(x, params)
    print(f"Transformer block: in {x.shape} -> out {out.shape}")
    print(f"Attention weights: {weights.shape}")
