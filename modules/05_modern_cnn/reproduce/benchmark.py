#!/usr/bin/env python3
"""Reproduce plain deep CNN, small ResNet, and VGG-mini on Fashion-MNIST."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "modules/05_modern_cnn"))

from models import PlainDeepCNN, SmallResNet, VGGMini, count_parameters
from train_utils import train_model


def main() -> None:
    configs = [
        ("plain_deep", PlainDeepCNN()),
        ("small_resnet", SmallResNet()),
        ("vgg_mini", VGGMini()),
    ]
    for name, model in configs:
        params = count_parameters(model)
        print(f"\n=== Training {name} ({params:,} params) ===")
        history = train_model(model, name=name, n_epochs=8)
        print(f"Final test acc: {history['test_acc'][-1]:.4f}")


if __name__ == "__main__":
    main()
