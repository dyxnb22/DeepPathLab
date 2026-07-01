#!/usr/bin/env python3
"""Data augmentation ablation on Fashion-MNIST subset."""

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


def train_with_transform(tfm: transforms.Compose, label: str, n_epochs: int = 5) -> list[float]:
    train_full = datasets.FashionMNIST(str(REPO_ROOT / "data"), train=True, download=True, transform=tfm)
    test_ds = datasets.FashionMNIST(str(REPO_ROOT / "data"), train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),
    ]))
    train_ds = Subset(train_full, list(range(3000)))
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=128)

    model = LeNet()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    device = torch.device("cpu")
    accs = []

    for epoch in range(n_epochs):
        model.train()
        for X, y in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(X), y)
            loss.backward()
            optimizer.step()
        _, acc = evaluate(model, test_loader, device)
        accs.append(acc)
    print(f"  {label}: final_acc={accs[-1]:.4f}")
    return accs


def main() -> None:
    base = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    aug = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(28, padding=2),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    curves = {
        "no_augmentation": train_with_transform(base, "no_aug"),
        "with_augmentation": train_with_transform(aug, "with_aug"),
    }
    out = module_output_dir("10_computer_vision_applications") / "augmentation_ablation.png"
    save_metric_curves(curves, out, title="Data Augmentation Ablation")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
