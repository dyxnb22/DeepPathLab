"""从零实现基础图像增强（numpy，教学用）。

支持单通道图像 shape (C, H, W)。流水线：水平翻转 → 随机裁剪 → 归一化。
训练时每次调用 augment() 产生不同随机变换；推理应使用原图或固定预处理。

几何直觉与陷阱参见 notes.md。
"""

from __future__ import annotations

import numpy as np


def random_horizontal_flip(image: np.ndarray, rng: np.random.Generator, p: float = 0.5) -> np.ndarray:
    """以概率 p 沿宽度轴翻转图像（左右镜像）。

    Args:
        image: shape (C, H, W)
        rng: numpy 随机数生成器，保证可复现
        p: 翻转概率
    """
    if rng.random() < p:
        return image[:, ::-1]
    return image


def random_crop(image: np.ndarray, crop_h: int, crop_w: int, rng: np.random.Generator) -> np.ndarray:
    """从图像中随机选取 crop_h × crop_w 区域。

    要求 image 高宽不小于 crop 尺寸；左上角坐标均匀采样。
    """
    h, w = image.shape[-2], image.shape[-1]
    top = rng.integers(0, h - crop_h + 1)
    left = rng.integers(0, w - crop_w + 1)
    return image[..., top : top + crop_h, left : left + crop_w]


def normalize(image: np.ndarray, mean: float = 0.5, std: float = 0.5) -> np.ndarray:
    """逐像素标准化：(x - mean) / std，与 ToTensor + Normalize 语义一致。"""
    return (image - mean) / std


def augment(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """简单增强流水线：翻转 → 裁剪（若尺寸足够）→ 归一化。

    对 28×28 等小图，裁剪仅缩小 2 像素边长，模拟轻微平移不变性。
    """
    out = image.copy()
    out = random_horizontal_flip(out, rng)
    # 仅当图像足够大时才裁剪，避免退化到过小尺寸
    if out.shape[-1] > 4 and out.shape[-2] > 4:
        out = random_crop(out, out.shape[-2] - 2, out.shape[-1] - 2, rng)
    return normalize(out)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    img = np.random.rand(1, 28, 28).astype(np.float32)
    aug = augment(img, rng)
    print("Original shape:", img.shape, "Augmented shape:", aug.shape)
