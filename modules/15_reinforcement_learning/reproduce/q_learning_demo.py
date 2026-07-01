#!/usr/bin/env python3
"""Q-learning demo with learning curve."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "modules/15_reinforcement_learning/from_scratch"))

from lib.plotting import module_output_dir, save_metric_curves
from q_learning import q_learning


def main() -> None:
    Q, returns = q_learning(n_episodes=600, alpha=0.5, gamma=0.99, epsilon=0.2)
    # smooth returns
    window = 20
    smooth = [np.mean(returns[max(0, i - window) : i + 1]) for i in range(len(returns))]
    out = module_output_dir("15_reinforcement_learning") / "qlearning_returns.png"
    save_metric_curves({"raw": returns, "smoothed": smooth}, out, title="Q-learning Returns", xlabel="Episode",)
    print(f"Final smoothed return: {smooth[-1]:.3f}")
    print(f"Q[0] = {Q[0]}")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
