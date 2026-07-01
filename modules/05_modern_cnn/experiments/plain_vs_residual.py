#!/usr/bin/env python3
"""Compare plain deep CNN vs residual network training dynamics."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "modules/05_modern_cnn"))

from lib.plotting import module_output_dir, save_metric_curves
from models import PlainDeepCNN, SmallResNet, count_parameters
from train_utils import train_model


def main() -> None:
    print("Hypothesis: at similar depth, ResNet trains more stably than plain CNN.\n")
    plain = PlainDeepCNN()
    resnet = SmallResNet()
    print(f"PlainDeepCNN params: {count_parameters(plain):,}")
    print(f"SmallResNet params:  {count_parameters(resnet):,}\n")

    plain_hist = train_model(plain, name="plain_vs_residual_plain", n_epochs=5)
    resnet_hist = train_model(resnet, name="plain_vs_residual_resnet", n_epochs=5)

    out = module_output_dir("05_modern_cnn") / "plain_vs_residual.png"
    save_metric_curves(
        {
            "plain_train": plain_hist["train_acc"],
            "plain_test": plain_hist["test_acc"],
            "resnet_train": resnet_hist["train_acc"],
            "resnet_test": resnet_hist["test_acc"],
        },
        out,
        title="Plain Deep CNN vs Small ResNet",
        xlabel="Epoch",
    )

    print(f"\nFinal test accuracy:")
    print(f"  Plain:  {plain_hist['test_acc'][-1]:.4f}")
    print(f"  ResNet: {resnet_hist['test_acc'][-1]:.4f}")
    print(f"Saved comparison plot to {out}")


if __name__ == "__main__":
    main()
