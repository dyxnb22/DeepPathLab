"""Residual block from scratch (NumPy) — 残差块与梯度高速公路演示.

Demonstrates the core ResNet idea without PyTorch:
  y = ReLU(F(x) + x)   vs   y = ReLU(F(x))

Run: python modules/05_modern_cnn/from_scratch/residual_block.py
"""

from __future__ import annotations

import numpy as np


def conv2d_single_channel(x: np.ndarray, kernel: np.ndarray, padding: int = 0) -> np.ndarray:
    """2D convolution with optional zero padding (single channel, no stride).

    单通道 2D 卷积，用于演示残差块内部的两层卷积堆叠。
    """
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
    """Same-size output for odd-sized kernels (padding = kernel_size // 2).

    保持空间尺寸不变，使 skip 路径的 x 与 F(x) 可直接相加。
    """
    pad = kernel.shape[0] // 2
    return conv2d_single_channel(x, kernel, padding=pad)


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def residual_block_forward(x: np.ndarray, kernel1: np.ndarray, kernel2: np.ndarray) -> np.ndarray:
    """Residual block: y = ReLU(conv2(ReLU(conv1(x))) + x).

    残差块核心：学习 F(x) = conv2(relu(conv1(x)))，输出 y = relu(F(x) + x)。
    skip 路径要求 conv 输出与 x 同 shape（此处通过 same padding 保证）。
    """
    h1 = relu(conv2d_same(x, kernel1))
    h2 = conv2d_same(h1, kernel2)
    return relu(h2 + x)  # 关键：+ x 即 skip connection


def plain_block_forward(x: np.ndarray, kernel1: np.ndarray, kernel2: np.ndarray) -> np.ndarray:
    """Plain block without skip — 对照组，无恒等映射捷径。

    相同卷积堆叠但不加 x，用于对比残差路径对信号与梯度的影响。
    """
    h1 = relu(conv2d_same(x, kernel1))
    h2 = conv2d_same(h1, kernel2)
    return relu(h2)


def gradient_highway_demo() -> None:
    """Show skip path preserves unit gradient: dy/dx = dF/dx + 1.

    符号演示：y = F(x) + x 时，即使 dF/dx 很小（饱和区），
    梯度仍可通过 +1 项回传——这就是 ResNet 能训练极深网络的原因。
    """
    f_prime_small = 0.01  # 模拟卷积路径梯度接近零（饱和）
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
