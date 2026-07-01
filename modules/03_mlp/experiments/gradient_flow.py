#!/usr/bin/env python3
"""Measure gradient norms at each layer during training."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "modules/03_mlp/from_scratch"))

from lib.plotting import module_output_dir, save_metric_curves
from lib.synthetic_data import spiral_data
from mlp_numpy import MLP, cross_entropy_grad, softmax


def train_with_grad_tracking(
    X: np.ndarray,
    y: np.ndarray,
    activation: str,
    n_epochs: int = 300,
) -> tuple[list[float], list[float]]:
    model = MLP(2, 32, 3, activation=activation, seed=42)
    w1_norms, w2_norms = [], []
    lr = 0.5

    for _ in range(n_epochs):
        logits = model.forward(X)
        probs = softmax(logits)
        grad = cross_entropy_grad(probs, y)
        model.backward(grad)
        w1_norms.append(float(np.linalg.norm(model.dW1)))
        w2_norms.append(float(np.linalg.norm(model.dW2)))
        model.W2 -= lr * model.dW2
        model.b2 -= lr * model.db2
        model.W1 -= lr * model.dW1
        model.b1 -= lr * model.db1

    return w1_norms, w2_norms


def main() -> None:
    X, y = spiral_data(n_per_class=100, n_classes=3)
    for act in ["relu", "sigmoid"]:
        w1, w2 = train_with_grad_tracking(X, y, activation=act)
        print(f"\n{act}:")
        print(f"  W1 grad norm (first 5 epochs): {[f'{v:.4f}' for v in w1[:5]]}")
        print(f"  W2 grad norm (first 5 epochs): {[f'{v:.4f}' for v in w2[:5]]}")
        print(f"  W1 grad norm (final): {w1[-1]:.6f}")
        print(f"  W2 grad norm (final): {w2[-1]:.6f}")

        out = module_output_dir("03_mlp") / f"gradient_flow_{act}.png"
        save_metric_curves(
            {"W1_grad_norm": w1, "W2_grad_norm": w2},
            out,
            title=f"Gradient Flow ({act})",
            xlabel="Epoch",
        )
        print(f"  Saved to {out}")


if __name__ == "__main__":
    main()
