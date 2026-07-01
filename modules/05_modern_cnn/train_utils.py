#!/usr/bin/env python3
"""Training utilities for Module 05 CNN benchmarks."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from lib.plotting import module_output_dir, save_metric_curves


def get_fashion_mnist_loaders(batch_size: int = 128) -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_ds = datasets.FashionMNIST(str(REPO_ROOT / "data"), train=True, download=True, transform=transform)
    test_ds = datasets.FashionMNIST(str(REPO_ROOT / "data"), train=False, download=True, transform=transform)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False),
    )


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        total_loss += loss_fn(logits, y).item() * X.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += X.size(0)
    return total_loss / total, correct / total


def train_model(
    model: nn.Module,
    name: str,
    n_epochs: int = 8,
    lr: float = 0.001,
    device: str | None = None,
) -> dict[str, list[float]]:
    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device_t)
    train_loader, test_loader = get_fashion_mnist_loaders()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    history: dict[str, list[float]] = {
        "train_loss": [], "train_acc": [], "test_loss": [], "test_acc": [],
    }

    for epoch in range(n_epochs):
        model.train()
        epoch_loss, correct, total = 0.0, 0, 0
        for X, y in train_loader:
            X, y = X.to(device_t), y.to(device_t)
            optimizer.zero_grad()
            logits = model(X)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * X.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += X.size(0)

        train_loss = epoch_loss / total
        train_acc = correct / total
        test_loss, test_acc = evaluate(model, test_loader, device_t)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        print(
            f"[{name}] Epoch {epoch + 1}/{n_epochs}: "
            f"train_acc={train_acc:.4f}, test_acc={test_acc:.4f}"
        )

    out_dir = module_output_dir("05_modern_cnn")
    save_metric_curves(
        {"train_acc": history["train_acc"], "test_acc": history["test_acc"]},
        out_dir / f"{name}_accuracy.png",
        title=f"{name} Accuracy",
    )
    return history
