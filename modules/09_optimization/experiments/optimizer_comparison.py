#!/usr/bin/env python3
"""Compare SGD, Momentum, and Adam on spiral classification."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from lib.plotting import module_output_dir, save_metric_curves
from lib.synthetic_data import spiral_data


class SpiralMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_with_optimizer(opt_name: str, lr: float, n_epochs: int = 100) -> list[float]:
    X, y = spiral_data(n_per_class=100, n_classes=3)
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    model = SpiralMLP()
    loss_fn = nn.CrossEntropyLoss()

    if opt_name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif opt_name == "momentum":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif opt_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        raise ValueError(opt_name)

    accs = []
    for _ in range(n_epochs):
        optimizer.zero_grad()
        logits = model(X_t)
        loss = loss_fn(logits, y_t)
        loss.backward()
        optimizer.step()
        accs.append(float((logits.argmax(1) == y_t).float().mean()))
    return accs


def main() -> None:
    configs = [("sgd", "SGD", 0.5), ("momentum", "Momentum", 0.1), ("adam", "Adam", 0.01)]
    curves: dict[str, list[float]] = {}
    print("Optimizer comparison on spiral data:\n")
    for key, label, lr in configs:
        accs = train_with_optimizer(key, lr=lr)
        curves[label] = accs
        print(f"  {label:10s} (lr={lr}): final_acc={accs[-1]:.4f}")

    out = module_output_dir("09_optimization") / "optimizer_comparison.png"
    save_metric_curves(curves, out, title="Optimizer Comparison (Spiral MLP)", xlabel="Epoch")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
