"""线性回归从零实现（Module 02 from_scratch）

提供两种求解路径：
1. closed_form_solution — 增广设计矩阵 + lstsq 求 MSE 闭式解
2. train_gd — 全批量梯度下降，手写 MSE 对 w、b 的梯度

运行本文件可在合成数据上对比两种方法的 MSE 与权重差异。
"""

from __future__ import annotations

import numpy as np


def closed_form_solution(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """Solve min ||Xw + b - y||^2 via normal equations."""
    n_samples = X.shape[0]
    # 把偏置 b 并入最后一列 1，一次 lstsq 解 [w; b]
    X_aug = np.hstack([X, np.ones((n_samples, 1))])
    theta = np.linalg.lstsq(X_aug, y, rcond=None)[0]
    w = theta[:-1]
    b = float(theta[-1, 0]) if theta[-1].ndim else float(theta[-1])
    return w, b


def predict(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    return X @ w.reshape(-1, 1) + b


def mse_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.mean((y_pred - y_true) ** 2))


def train_gd(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.1,
    n_epochs: int = 200,
    seed: int = 0,
) -> tuple[np.ndarray, float, list[float]]:
    """Train linear regression with full-batch gradient descent."""
    rng = np.random.default_rng(seed)
    n_features = X.shape[1]
    w = rng.normal(scale=0.01, size=(n_features, 1))
    b = 0.0
    n_samples = X.shape[0]
    losses: list[float] = []

    for _ in range(n_epochs):
        y_pred = predict(X, w, b)
        loss = mse_loss(y_pred, y)
        losses.append(loss)

        # MSE 梯度：∇_w = (2/n) X^T (ŷ - y)
        grad_w = (2.0 / n_samples) * (X.T @ (y_pred - y))
        grad_b = (2.0 / n_samples) * float(np.sum(y_pred - y))

        w -= lr * grad_w
        b -= lr * grad_b

    return w, b, losses


if __name__ == "__main__":
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo))
    from lib.synthetic_data import linear_regression_data

    X, y, true_w = linear_regression_data()
    w_cf, b_cf = closed_form_solution(X, y)
    w_gd, b_gd, losses = train_gd(X, y, lr=0.1, n_epochs=300)

    print("Closed-form MSE:", mse_loss(predict(X, w_cf, b_cf), y))
    print("GD final MSE:", losses[-1])
    print("Weight diff (CF vs GD):", np.linalg.norm(w_cf - w_gd))
