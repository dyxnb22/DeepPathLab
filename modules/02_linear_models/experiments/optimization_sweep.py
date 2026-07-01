#!/usr/bin/env python3
"""Learning rate and batch size sweep for linear regression."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "modules/02_linear_models/from_scratch"))

from lib.plotting import module_output_dir, save_metric_curves
from lib.synthetic_data import linear_regression_data
from linear_regression import mse_loss, predict


def train_gd_lr(X: np.ndarray, y: np.ndarray, lr: float, n_epochs: int = 200) -> list[float]:
    rng = np.random.default_rng(0)
    w = rng.normal(scale=0.01, size=(X.shape[1], 1))
    b = 0.0
    n_samples = X.shape[0]
    losses = []
    for _ in range(n_epochs):
        y_pred = predict(X, w, b)
        losses.append(mse_loss(y_pred, y))
        grad_w = (2.0 / n_samples) * (X.T @ (y_pred - y))
        grad_b = (2.0 / n_samples) * float(np.sum(y_pred - y))
        w -= lr * grad_w
        b -= lr * grad_b
    return losses


def main() -> None:
    X, y, _ = linear_regression_data()
    learning_rates = [0.001, 0.01, 0.1, 0.5, 1.0]
    curves: dict[str, list[float]] = {}

    print("Learning rate sweep:")
    for lr in learning_rates:
        losses = train_gd_lr(X, y, lr=lr)
        curves[f"lr={lr}"] = losses
        final = losses[-1]
        status = "converged" if final < 0.5 else ("diverged" if final > 100 else "slow")
        print(f"  lr={lr:5.3f}: final_mse={final:10.4f} ({status})")

    out_dir = module_output_dir("02_linear_models")
    path = save_metric_curves(
        curves,
        out_dir / "lr_sweep.png",
        title="Linear Regression: Learning Rate Sweep",
        xlabel="Epoch",
    )
    print(f"\nSaved plot to {path}")


if __name__ == "__main__":
    main()
