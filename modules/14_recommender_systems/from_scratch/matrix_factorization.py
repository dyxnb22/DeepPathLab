"""Matrix factorization from scratch with SGD.

Learns low-rank user (P) and item (Q) embeddings by minimizing
squared error on observed ratings plus L2 regularization.
"""

from __future__ import annotations

import numpy as np


def train_matrix_factorization(
    ratings: np.ndarray,
    mask: np.ndarray,
    n_factors: int = 8,
    lr: float = 0.01,
    reg: float = 0.01,
    n_epochs: int = 200,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Alternating SGD matrix factorization.

    ratings: (n_users, n_items) with 0 for missing
    mask: boolean, True where rating observed
    """
    rng = np.random.default_rng(seed)
    n_users, n_items = ratings.shape
    P = rng.normal(scale=0.1, size=(n_users, n_factors))
    Q = rng.normal(scale=0.1, size=(n_items, n_factors))
    losses = []

    for _ in range(n_epochs):
        for u in range(n_users):
            for i in range(n_items):
                if not mask[u, i]:
                    continue
                pred = P[u] @ Q[i]
                err = ratings[u, i] - pred
                # Save P[u] before update — Q[i] gradient uses the old user vector
                P_u = P[u].copy()
                P[u] += lr * (err * Q[i] - reg * P[u])
                Q[i] += lr * (err * P_u - reg * Q[i])
        preds = P @ Q.T
        mse = float(np.mean((preds[mask] - ratings[mask]) ** 2))
        losses.append(mse)
    return P, Q, losses


def predict(P: np.ndarray, Q: np.ndarray, user: int, item: int) -> float:
    """Dot product of learned user and item latent vectors."""
    return float(P[user] @ Q[item])


def top_k_recommendations(P: np.ndarray, Q: np.ndarray, user: int, k: int = 3, seen: set[int] | None = None) -> list[int]:
    """Rank items by predicted score, excluding already-seen items."""
    scores = P[user] @ Q.T
    seen = seen or set()
    ranked = np.argsort(-scores)
    return [int(i) for i in ranked if i not in seen][:k]


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n_users, n_items, n_factors = 20, 30, 6
    true_P = rng.normal(size=(n_users, n_factors))
    true_Q = rng.normal(size=(n_items, n_factors))
    full = true_P @ true_Q.T
    mask = rng.random((n_users, n_items)) < 0.3
    ratings = np.zeros_like(full)
    ratings[mask] = full[mask] + rng.normal(scale=0.1, size=mask.sum())

    P, Q, losses = train_matrix_factorization(ratings, mask, n_factors=n_factors, n_epochs=100)
    print(f"Final MSE: {losses[-1]:.4f}")
    recs = top_k_recommendations(P, Q, user=0, k=3)
    print(f"Top-3 for user 0: {recs}")
