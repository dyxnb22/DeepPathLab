#!/usr/bin/env python3
"""Visualize softmax regression decision boundaries."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "modules/02_linear_models/from_scratch"))

from lib.plotting import module_output_dir
from lib.synthetic_data import multiclass_blobs
from softmax_regression import softmax, train_softmax


def plot_decision_boundary(
    X: np.ndarray,
    y: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
    n_classes: int,
    output_path: Path,
) -> None:
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = softmax(grid @ W + b)
    preds = np.argmax(probs, axis=1).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.contourf(xx, yy, preds, alpha=0.3, levels=n_classes)
    for c in range(n_classes):
        mask = y == c
        ax.scatter(X[mask, 0], X[mask, 1], label=f"class {c}", s=20)
    ax.set_title("Softmax Regression Decision Boundary")
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def main() -> None:
    X, y = multiclass_blobs(n_classes=3)
    W, b, losses, accs = train_softmax(X, y, n_classes=3, lr=0.5, n_epochs=500)
    out_path = module_output_dir("02_linear_models") / "decision_boundary.png"
    plot_decision_boundary(X, y, W, b, n_classes=3, output_path=out_path)
    print(f"Final accuracy: {accs[-1]:.4f}")
    print(f"Saved decision boundary to {out_path}")


if __name__ == "__main__":
    main()
