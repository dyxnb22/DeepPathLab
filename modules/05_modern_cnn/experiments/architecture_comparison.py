#!/usr/bin/env python3
"""Architecture comparison table: params, depth proxy, final accuracy."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "modules/05_modern_cnn"))

from models import PlainDeepCNN, SmallResNet, VGGMini, count_parameters
from train_utils import train_model


def count_conv_layers(model) -> int:
    import torch.nn as nn
    return sum(1 for m in model.modules() if isinstance(m, nn.Conv2d))


def main() -> None:
    models = {
        "VGGMini": VGGMini(),
        "PlainDeepCNN": PlainDeepCNN(),
        "SmallResNet": SmallResNet(),
    }

    print(f"{'Model':<16} {'Params':>10} {'Conv2d':>8} {'TestAcc':>10}")
    print("-" * 48)

    for name, model in models.items():
        hist = train_model(model, name=name.lower(), n_epochs=6)
        params = count_parameters(model)
        convs = count_conv_layers(model)
        acc = hist["test_acc"][-1]
        print(f"{name:<16} {params:>10,} {convs:>8} {acc:>10.4f}")


if __name__ == "__main__":
    main()
