#!/usr/bin/env python3
"""PyTorch MLP baseline for comparison with scratch implementation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "modules/03_mlp/from_scratch"))

from lib.synthetic_data import spiral_data
from mlp_numpy import accuracy as scratch_accuracy, train as scratch_train


class TorchMLP(nn.Module):
    def __init__(self, n_input: int, n_hidden: int, n_output: int, activation: str = "relu"):
        super().__init__()
        self.fc1 = nn.Linear(n_input, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_output)
        acts = {"relu": nn.ReLU(), "sigmoid": nn.Sigmoid(), "tanh": nn.Tanh()}
        self.act = acts[activation]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


def train_pytorch(
    X: np.ndarray,
    y: np.ndarray,
    n_hidden: int = 32,
    activation: str = "relu",
    lr: float = 0.5,
    n_epochs: int = 500,
    seed: int = 42,
) -> tuple[list[float], list[float]]:
    torch.manual_seed(seed)
    n_output = len(np.unique(y))
    model = TorchMLP(X.shape[1], n_hidden, n_output, activation)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)

    losses, accs = [], []
    for _ in range(n_epochs):
        optimizer.zero_grad()
        logits = model(X_t)
        loss = loss_fn(logits, y_t)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
        accs.append(float((logits.argmax(1) == y_t).float().mean()))
    return losses, accs


def main() -> None:
    X, y = spiral_data(n_per_class=100, n_classes=3)
    _, scratch_losses, scratch_accs = scratch_train(X, y, n_hidden=32, n_epochs=500)
    pt_losses, pt_accs = train_pytorch(X, y, n_hidden=32, n_epochs=500)
    print(f"Scratch: loss={scratch_losses[-1]:.4f}, acc={scratch_accs[-1]:.4f}")
    print(f"PyTorch: loss={pt_losses[-1]:.4f}, acc={pt_accs[-1]:.4f}")


if __name__ == "__main__":
    main()
