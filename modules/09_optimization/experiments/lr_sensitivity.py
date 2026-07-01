#!/usr/bin/env python3
"""Learning rate sensitivity for SGD vs Adam."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "modules/09_optimization/experiments"))

from lib.plotting import module_output_dir, save_metric_curves
from optimizer_comparison import train_with_optimizer


def main() -> None:
    lrs = [0.001, 0.01, 0.05, 0.1, 0.5]
    sgd_finals, adam_finals = [], []

    print("LR sensitivity (final accuracy after 80 epochs):\n")
    for lr in lrs:
        sgd_acc = train_with_optimizer("sgd", lr=lr, n_epochs=80)[-1]
        adam_acc = train_with_optimizer("adam", lr=lr, n_epochs=80)[-1]
        sgd_finals.append(sgd_acc)
        adam_finals.append(adam_acc)
        print(f"  lr={lr:5.3f}: SGD={sgd_acc:.3f}, Adam={adam_acc:.3f}")

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([str(l) for l in lrs], sgd_finals, marker="o", label="SGD")
    ax.plot([str(l) for l in lrs], adam_finals, marker="s", label="Adam")
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Final Accuracy")
    ax.set_title("LR Sensitivity: SGD vs Adam")
    ax.legend()
    ax.grid(True, alpha=0.3)
    out = module_output_dir("09_optimization") / "lr_sensitivity.png"
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
