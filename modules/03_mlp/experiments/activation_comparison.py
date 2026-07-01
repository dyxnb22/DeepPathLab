#!/usr/bin/env python3
"""Compare ReLU, sigmoid, and tanh activations on spiral data."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "modules/03_mlp/from_scratch"))

from lib.plotting import module_output_dir, save_metric_curves
from lib.synthetic_data import spiral_data
from mlp_numpy import train


def main() -> None:
    X, y = spiral_data(n_per_class=100, n_classes=3)
    activations = ["relu", "sigmoid", "tanh"]
    acc_curves: dict[str, list[float]] = {}

    print("Hypothesis: sigmoid/tanh saturate in deeper nets, ReLU converges faster.\n")
    for act in activations:
        _, losses, accs = train(X, y, n_hidden=32, activation=act, lr=0.5, n_epochs=500, seed=42)
        acc_curves[act] = accs
        print(f"  {act:8s}: final_loss={losses[-1]:.4f}, final_acc={accs[-1]:.4f}")

    out = module_output_dir("03_mlp") / "activation_comparison.png"
    save_metric_curves(acc_curves, out, title="Activation Comparison (Accuracy)", xlabel="Epoch")
    print(f"\nSaved plot to {out}")


if __name__ == "__main__":
    main()
