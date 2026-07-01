#!/usr/bin/env python3
"""Visualize first-layer convolution kernels and feature maps."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import datasets, transforms

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "modules/04_cnn/reproduce"))

from lib.plotting import module_output_dir
from lenet_fashion_mnist import LeNet


def main() -> None:
    weights_path = module_output_dir("04_cnn") / "lenet_weights.pt"
    model = LeNet()
    if weights_path.exists():
        model.load_state_dict(torch.load(weights_path, weights_only=True))
    else:
        print("Warning: no trained weights found, using random init for demo.")

    model.eval()
    conv1 = model.features[0]
    kernels = conv1.weight.detach().numpy()  # (6, 1, 5, 5)

    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(kernels[i, 0], cmap="gray")
        ax.set_title(f"Filter {i}")
        ax.axis("off")
    fig.suptitle("First Conv Layer Kernels")
    out_kernels = module_output_dir("04_cnn") / "conv_kernels.png"
    fig.tight_layout()
    fig.savefig(out_kernels, dpi=120)
    plt.close(fig)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    ds = datasets.FashionMNIST(str(REPO_ROOT / "data"), train=False, download=True, transform=transform)
    img, label = ds[0]
    with torch.no_grad():
        feat = model.features[0:2](img.unsqueeze(0))
    maps = feat[0].numpy()

    fig2, axes2 = plt.subplots(2, 3, figsize=(9, 6))
    for i, ax in enumerate(axes2.flat):
        ax.imshow(maps[i], cmap="viridis")
        ax.set_title(f"Feature map {i}")
        ax.axis("off")
    fig2.suptitle(f"Feature Maps (label={label})")
    out_maps = module_output_dir("04_cnn") / "feature_maps.png"
    fig2.tight_layout()
    fig2.savefig(out_maps, dpi=120)
    plt.close(fig2)

    print(f"Saved kernels to {out_kernels}")
    print(f"Saved feature maps to {out_maps}")


if __name__ == "__main__":
    main()
