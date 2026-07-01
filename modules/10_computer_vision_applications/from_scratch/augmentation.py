"""Basic image augmentation from scratch (numpy)."""

from __future__ import annotations

import numpy as np


def random_horizontal_flip(image: np.ndarray, rng: np.random.Generator, p: float = 0.5) -> np.ndarray:
    if rng.random() < p:
        return image[:, ::-1]
    return image


def random_crop(image: np.ndarray, crop_h: int, crop_w: int, rng: np.random.Generator) -> np.ndarray:
    h, w = image.shape[-2], image.shape[-1]
    top = rng.integers(0, h - crop_h + 1)
    left = rng.integers(0, w - crop_w + 1)
    return image[..., top : top + crop_h, left : left + crop_w]


def normalize(image: np.ndarray, mean: float = 0.5, std: float = 0.5) -> np.ndarray:
    return (image - mean) / std


def augment(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Simple augmentation pipeline for 1xHxW images."""
    out = image.copy()
    out = random_horizontal_flip(out, rng)
    if out.shape[-1] > 4 and out.shape[-2] > 4:
        out = random_crop(out, out.shape[-2] - 2, out.shape[-1] - 2, rng)
    return normalize(out)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    img = np.random.rand(1, 28, 28).astype(np.float32)
    aug = augment(img, rng)
    print("Original shape:", img.shape, "Augmented shape:", aug.shape)
