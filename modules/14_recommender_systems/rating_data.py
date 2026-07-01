"""Synthetic rating data for recommender experiments."""

from __future__ import annotations

import numpy as np


def generate_synthetic_ratings(
    n_users: int = 50,
    n_items: int = 40,
    n_factors: int = 6,
    density: float = 0.25,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    P = rng.normal(size=(n_users, n_factors))
    Q = rng.normal(size=(n_items, n_factors))
    full = P @ Q.T
    mask = rng.random((n_users, n_items)) < density
    ratings = np.zeros_like(full)
    ratings[mask] = np.clip(full[mask] + rng.normal(scale=0.05, size=mask.sum()), 1, 5)
    return ratings, mask
