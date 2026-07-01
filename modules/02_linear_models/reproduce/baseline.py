#!/usr/bin/env python3
"""PyTorch baseline for linear regression and softmax classification."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "modules/02_linear_models/from_scratch"))

from lib.synthetic_data import linear_regression_data, multiclass_blobs
from linear_regression import closed_form_solution, mse_loss, predict, train_gd
from softmax_regression import accuracy, cross_entropy, softmax, train_softmax


def pytorch_linear_regression(X: np.ndarray, y: np.ndarray, n_epochs: int = 300) -> float:
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    model = nn.Linear(X.shape[1], 1, bias=True)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.MSELoss()

    for _ in range(n_epochs):
        optimizer.zero_grad()
        pred = model(X_t)
        loss = loss_fn(pred, y_t)
        loss.backward()
        optimizer.step()
    return float(loss.item())


def pytorch_softmax(X: np.ndarray, y: np.ndarray, n_classes: int, n_epochs: int = 300) -> tuple[float, float]:
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    model = nn.Linear(X.shape[1], n_classes)
    optimizer = optim.SGD(model.parameters(), lr=0.5)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(n_epochs):
        optimizer.zero_grad()
        logits = model(X_t)
        loss = loss_fn(logits, y_t)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits = model(X_t)
        probs = torch.softmax(logits, dim=1).numpy()
    return cross_entropy(probs, y, n_classes), accuracy(probs, y)


def main() -> None:
    print("=== Linear Regression ===")
    X, y, _ = linear_regression_data()
    w_cf, b_cf = closed_form_solution(X, y)
    cf_mse = mse_loss(predict(X, w_cf, b_cf), y)
    _, _, gd_losses = train_gd(X, y, lr=0.1, n_epochs=300)
    pt_mse = pytorch_linear_regression(X, y)
    print(f"  Closed-form MSE: {cf_mse:.6f}")
    print(f"  Scratch GD MSE:  {gd_losses[-1]:.6f}")
    print(f"  PyTorch MSE:     {pt_mse:.6f}")

    print("\n=== Softmax Classification ===")
    X, y = multiclass_blobs(n_classes=3)
    _, _, scratch_losses, scratch_accs = train_softmax(X, y, n_classes=3)
    pt_loss, pt_acc = pytorch_softmax(X, y, n_classes=3)
    print(f"  Scratch loss/acc: {scratch_losses[-1]:.4f} / {scratch_accs[-1]:.4f}")
    print(f"  PyTorch loss/acc: {pt_loss:.4f} / {pt_acc:.4f}")


if __name__ == "__main__":
    main()
