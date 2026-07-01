#!/usr/bin/env python3
"""Compare popularity baseline vs matrix factorization."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "modules/14_recommender_systems"))
sys.path.insert(0, str(REPO_ROOT / "modules/14_recommender_systems/from_scratch"))

from lib.plotting import module_output_dir, save_loss_curve
from matrix_factorization import train_matrix_factorization
from rating_data import generate_synthetic_ratings


def popularity_test_mse(ratings: np.ndarray, train_mask: np.ndarray, test_mask: np.ndarray) -> float:
    mean_rating = ratings[train_mask].mean()
    preds = np.full_like(ratings, mean_rating)
    return float(np.mean((preds[test_mask] - ratings[test_mask]) ** 2))


def train_test_split_mask(mask: np.ndarray, frac: float = 0.2, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    obs = np.argwhere(mask)
    rng.shuffle(obs)
    n_test = int(len(obs) * frac)
    test_mask = np.zeros_like(mask, dtype=bool)
    train_mask = mask.copy()
    for u, i in obs[:n_test]:
        test_mask[u, i] = True
        train_mask[u, i] = False
    return train_mask, test_mask


def eval_mse(P: np.ndarray, Q: np.ndarray, ratings: np.ndarray, eval_mask: np.ndarray) -> float:
    preds = P @ Q.T
    return float(np.mean((preds[eval_mask] - ratings[eval_mask]) ** 2))


def main() -> None:
    ratings, mask = generate_synthetic_ratings()
    train_mask, test_mask = train_test_split_mask(mask)

    pop_mse = popularity_test_mse(ratings, train_mask, test_mask)
    print(f"Popularity baseline test MSE: {pop_mse:.4f}")

    P, Q, losses = train_matrix_factorization(ratings, train_mask, n_factors=10, n_epochs=400, lr=0.01, reg=0.05)
    mf_mse = eval_mse(P, Q, ratings, test_mask)
    print(f"Matrix Factorization test MSE: {mf_mse:.4f}")

    out = module_output_dir("14_recommender_systems") / "mf_training_loss.png"
    save_loss_curve(losses, out, title="MF Training MSE", ylabel="MSE")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
