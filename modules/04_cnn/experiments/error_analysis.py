#!/usr/bin/env python3
"""Error analysis: per-class accuracy and misclassified examples."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "modules/04_cnn/reproduce"))

from lib.plotting import module_output_dir
from lenet_fashion_mnist import LeNet

CLASS_NAMES = [
    "T-shirt", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def main() -> None:
    weights_path = module_output_dir("04_cnn") / "lenet_weights.pt"
    model = LeNet()
    if not weights_path.exists():
        print("Error: train lenet first (reproduce/lenet_fashion_mnist.py)")
        sys.exit(1)
    model.load_state_dict(torch.load(weights_path, weights_only=True))
    model.eval()

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    test_ds = datasets.FashionMNIST(str(REPO_ROOT / "data"), train=False, download=True, transform=transform)
    loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    class_correct = np.zeros(10)
    class_total = np.zeros(10)
    confusion = np.zeros((10, 10), dtype=int)
    misclassified: list[tuple] = []

    with torch.no_grad():
        for X, y in loader:
            logits = model(X)
            preds = logits.argmax(1)
            for i in range(len(y)):
                true_label = y[i].item()
                pred_label = preds[i].item()
                class_total[true_label] += 1
                confusion[true_label, pred_label] += 1
                if pred_label == true_label:
                    class_correct[true_label] += 1
                elif len(misclassified) < 12:
                    misclassified.append((X[i], true_label, pred_label))

    print("Per-class accuracy:")
    for c in range(10):
        acc = class_correct[c] / max(class_total[c], 1)
        print(f"  {CLASS_NAMES[c]:12s}: {acc:.3f} ({int(class_correct[c])}/{int(class_total[c])})")

    # Most confused pairs
    confusion_offdiag = confusion.copy()
    np.fill_diagonal(confusion_offdiag, 0)
    flat_idx = np.argmax(confusion_offdiag)
    true_c, pred_c = divmod(flat_idx, 10)
    print(f"\nMost confused pair: {CLASS_NAMES[true_c]} -> {CLASS_NAMES[pred_c]} ({confusion[true_c, pred_c]} errors)")

    fig, axes = plt.subplots(3, 4, figsize=(10, 8))
    for ax, (img, true_l, pred_l) in zip(axes.flat, misclassified):
        ax.imshow(img.squeeze(), cmap="gray")
        ax.set_title(f"T:{CLASS_NAMES[true_l]}\nP:{CLASS_NAMES[pred_l]}", fontsize=8)
        ax.axis("off")
    fig.suptitle("Misclassified Examples")
    out_path = module_output_dir("04_cnn") / "error_analysis.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"\nSaved misclassified examples to {out_path}")


if __name__ == "__main__":
    main()
