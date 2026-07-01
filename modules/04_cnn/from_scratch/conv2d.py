"""Naive 2D 卷积与最大池化（Module 04 from_scratch）

教学用四重循环实现 conv2d / conv2d_multi / max_pool2d，便于对照笔记中的输出尺寸公式。
sanity_check_against_torch() 与 F.conv2d 比对，确认前向实现正确；训练用 reproduce/ 中 PyTorch。
"""

from __future__ import annotations

import numpy as np


def conv2d(
    x: np.ndarray,
    kernel: np.ndarray,
    stride: int = 1,
    padding: int = 0,
) -> np.ndarray:
    """Convolve a single-channel 2D input with one kernel.

    x: (H, W)
    kernel: (kH, kW)
    Returns: (outH, outW)
    """
    if padding > 0:
        x = np.pad(x, padding, mode="constant")
    h, w = x.shape
    kh, kw = kernel.shape
    out_h = (h - kh) // stride + 1
    out_w = (w - kw) // stride + 1
    out = np.zeros((out_h, out_w))
    # 在滑动窗口位置做 patch 与 kernel 逐元素乘再求和
    for i in range(out_h):
        for j in range(out_w):
            patch = x[i * stride : i * stride + kh, j * stride : j * stride + kw]
            out[i, j] = np.sum(patch * kernel)
    return out


def conv2d_multi(
    x: np.ndarray,
    kernels: np.ndarray,
    stride: int = 1,
    padding: int = 0,
) -> np.ndarray:
    """Multi-channel input, multiple output channels.

    x: (C_in, H, W)
    kernels: (C_out, C_in, kH, kW)
  Returns: (C_out, outH, outW)
    """
    outputs = []
    for oc in range(kernels.shape[0]):
        channel_sum = np.zeros(conv2d(x[0], kernels[oc, 0], stride, padding).shape)
        # 每个输出通道：对所有输入通道卷积后累加
        for ic in range(x.shape[0]):
            channel_sum += conv2d(x[ic], kernels[oc, ic], stride, padding)
        outputs.append(channel_sum)
    return np.stack(outputs)


def max_pool2d(x: np.ndarray, pool_size: int = 2, stride: int | None = None) -> np.ndarray:
    """Max pooling on (C, H, W) or (H, W)."""
    if x.ndim == 2:
        x = x[np.newaxis, ...]
        squeeze = True
    else:
        squeeze = False
    if stride is None:
        stride = pool_size
    c, h, w = x.shape
    out_h = (h - pool_size) // stride + 1
    out_w = (w - pool_size) // stride + 1
    out = np.zeros((c, out_h, out_w))
    for ch in range(c):
        for i in range(out_h):
            for j in range(out_w):
                patch = x[ch, i * stride : i * stride + pool_size, j * stride : j * stride + pool_size]
                out[ch, i, j] = np.max(patch)
    if squeeze:
        return out[0]
    return out


def output_size(
    input_size: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
) -> int:
    """Compute spatial output dimension."""
    return (input_size + 2 * padding - kernel_size) // stride + 1


def sanity_check_against_torch() -> bool:
    """Verify conv2d matches PyTorch on random small input."""
    import torch
    import torch.nn.functional as F

    rng = np.random.default_rng(0)
    x = rng.normal(size=(1, 3, 8, 8)).astype(np.float32)
    w = rng.normal(size=(4, 3, 3, 3)).astype(np.float32)

    scratch = conv2d_multi(x[0], w, stride=1, padding=1)
    x_t = torch.tensor(x)
    w_t = torch.tensor(w)
    torch_out = F.conv2d(x_t, w_t, padding=1).numpy()[0]

    max_err = np.max(np.abs(scratch - torch_out))
    print(f"conv2d max error vs PyTorch: {max_err:.2e}")
    return max_err < 1e-5


if __name__ == "__main__":
    print("Output size (28, 3, stride=1, pad=1):", output_size(28, 3, stride=1, padding=1))
    ok = sanity_check_against_torch()
    print("Sanity check:", "PASSED" if ok else "FAILED")
