"""Residual connection from scratch (numpy) to illustrate the core idea."""

from __future__ import annotations

import numpy as np


def conv2d_single_channel(x: np.ndarray, kernel: np.ndarray, padding: int = 0) -> np.ndarray:
    """2D convolution with optional zero padding."""
    if padding > 0:
        x = np.pad(x, padding, mode="constant")
    kh, kw = kernel.shape
    h, w = x.shape
    out = np.zeros((h - kh + 1, w - kw + 1))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = np.sum(x[i : i + kh, j : j + kw] * kernel)
    return out


def conv2d_same(x: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Same-size output for odd-sized kernels."""
    pad = kernel.shape[0] // 2
    return conv2d_single_channel(x, kernel, padding=pad)


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def residual_block_forward(x: np.ndarray, kernel1: np.ndarray, kernel2: np.ndarray) -> np.ndarray:
    """Compute y = ReLU(conv2(ReLU(conv1(x))) + x) with same spatial size."""
    h1 = relu(conv2d_same(x, kernel1))
    h2 = conv2d_same(h1, kernel2)
    return relu(h2 + x)


def plain_block_forward(x: np.ndarray, kernel1: np.ndarray, kernel2: np.ndarray) -> np.ndarray:
    """Same conv stack without skip connection."""
    h1 = relu(conv2d_same(x, kernel1))
    h2 = conv2d_same(h1, kernel2)
    return relu(h2)


def gradient_highway_demo() -> None:
    """Show that skip path preserves unit gradient even when F'(x) is small."""
    # Symbolic/numeric: y = f(x) + x, dy/dx = f'(x) + 1
    x_val = 1.0
    f_prime_small = 0.01  # saturated conv path
    plain_grad = f_prime_small
    residual_grad = f_prime_small + 1.0
    print("Gradient highway demo (symbolic):")
    print(f"  Plain block dL/dx ~ {plain_grad:.4f}  (only through F')")
    print(f"  Residual dL/dx ~ {residual_grad:.4f}  (F' + 1 from skip)")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x = rng.normal(size=(8, 8))
    k1 = rng.normal(size=(3, 3)) * 0.1
    k2 = rng.normal(size=(3, 3)) * 0.1

    y_res = residual_block_forward(x, k1, k2)
    y_plain = plain_block_forward(x, k1, k2)

    print("Input shape:", x.shape)
    print("Residual output shape:", y_res.shape)
    print("Plain output shape:", y_plain.shape)
    print("Outputs differ (skip changes signal):", not np.allclose(y_res, y_plain))
    gradient_highway_demo()
