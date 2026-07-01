#!/usr/bin/env python3
"""LeNet-style classifier on Fashion-MNIST."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from lib.plotting import module_output_dir, save_metric_curves


class LeNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def get_dataloaders(batch_size: int = 128) -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_ds = datasets.FashionMNIST(str(REPO_ROOT / "data"), train=True, download=True, transform=transform)
    test_ds = datasets.FashionMNIST(str(REPO_ROOT / "data"), train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    loss_fn = nn.CrossEntropyLoss()
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            total_loss += loss_fn(logits, y).item() * X.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += X.size(0)
    return total_loss / total, correct / total


def train_lenet(
    n_epochs: int = 5,
    lr: float = 0.01,
    device: str | None = None,
) -> tuple[list[float], list[float], list[float], list[float]]:
    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    train_loader, test_loader = get_dataloaders()
    model = LeNet().to(device_t)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    train_losses, train_accs, test_losses, test_accs = [], [], [], []

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
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        print(
            f"Epoch {epoch + 1}/{n_epochs}: "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"test_acc={test_acc:.4f}"
        )

    out_dir = module_output_dir("04_cnn")
    save_metric_curves({"train_acc": train_accs, "test_acc": test_accs}, out_dir / "lenet_accuracy.png")
    save_metric_curves({"train_loss": train_losses, "test_loss": test_losses}, out_dir / "lenet_loss.png")
    torch.save(model.state_dict(), out_dir / "lenet_weights.pt")
    return train_losses, train_accs, test_losses, test_accs


def main() -> None:
    _, _, _, test_accs = train_lenet(n_epochs=5)
    print(f"\nFinal test accuracy: {test_accs[-1]:.4f}")


if __name__ == "__main__":
    main()
