#!/usr/bin/env python3
"""Fine-tuning vs training from scratch on Fashion-MNIST."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "modules/04_cnn/reproduce"))

from lenet_fashion_mnist import LeNet, evaluate
from lib.plotting import module_output_dir, save_metric_curves


def get_loaders(subset_size: int = 5000, batch_size: int = 128) -> tuple[DataLoader, DataLoader]:
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_full = datasets.FashionMNIST(str(REPO_ROOT / "data"), train=True, download=True, transform=tfm)
    test_ds = datasets.FashionMNIST(str(REPO_ROOT / "data"), train=False, download=True, transform=tfm)
    indices = list(range(subset_size))
    train_ds = Subset(train_full, indices)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False),
    )


def train_model(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader, n_epochs: int, lr: float, freeze_features: bool = False) -> list[float]:
    device = torch.device("cpu")
    model = model.to(device)
    if freeze_features:
        for p in model.features.parameters():
            p.requires_grad = False
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    accs = []

    for epoch in range(n_epochs):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(X), y)
            loss.backward()
            optimizer.step()
        _, acc = evaluate(model, test_loader, device)
        accs.append(acc)
        if (epoch + 1) % 3 == 0:
            print(f"  epoch {epoch + 1}: test_acc={acc:.4f}")
    return accs


def main() -> None:
    train_loader, test_loader = get_loaders(subset_size=5000)
    weights_path = module_output_dir("04_cnn") / "lenet_weights.pt"

    print("=== Train from scratch (5k subset, 6 epochs) ===")
    scratch = LeNet()
    scratch_accs = train_model(scratch, train_loader, test_loader, n_epochs=6, lr=0.01)

    print("\n=== Fine-tune pretrained features (freeze conv) ===")
    finetune = LeNet()
    if weights_path.exists():
        finetune.load_state_dict(torch.load(weights_path, weights_only=True))
    else:
        print("  (no pretrained weights, using random init as fallback)")
    ft_accs = train_model(finetune, train_loader, test_loader, n_epochs=6, lr=0.001, freeze_features=True)

    out = module_output_dir("10_computer_vision_applications") / "transfer_learning.png"
    save_metric_curves(
        {"from_scratch": scratch_accs, "fine_tune_features": ft_accs},
        out, title="Transfer Learning vs Scratch", xlabel="Epoch",
    )
    print(f"\nFinal: scratch={scratch_accs[-1]:.4f}, fine_tune={ft_accs[-1]:.4f}")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
