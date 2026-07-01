"""两层 MLP 手写矩阵反向传播（Module 03 from_scratch）

结构：input → Linear → 激活 → Linear → logits；损失为 softmax + 交叉熵。
forward 缓存 X/z1/a1；backward 从 grad_logits 反传，隐藏层乘 activation'(z1)。

对比 mlp_autograd.py：同一网络可用标量 Value 展开，验证本文件梯度公式。
"""

from __future__ import annotations

from typing import Callable

import numpy as np


ACTIVATIONS: dict[str, tuple[Callable, Callable]] = {}


def _register_activation(name: str, fn: Callable, deriv: Callable) -> None:
    ACTIVATIONS[name] = (fn, deriv)


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def relu_deriv(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(float)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def sigmoid_deriv(x: np.ndarray) -> np.ndarray:
    s = sigmoid(x)
    return s * (1 - s)


def tanh_fn(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def tanh_deriv(x: np.ndarray) -> np.ndarray:
    t = np.tanh(x)
    return 1 - t * t


_register_activation("relu", relu, relu_deriv)
_register_activation("sigmoid", sigmoid, sigmoid_deriv)
_register_activation("tanh", tanh_fn, tanh_deriv)


class MLP:
    """2-layer MLP: input -> hidden -> output."""

    def __init__(
        self,
        n_input: int,
        n_hidden: int,
        n_output: int,
        activation: str = "relu",
        seed: int = 42,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(scale=0.5, size=(n_input, n_hidden))
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.normal(scale=0.5, size=(n_hidden, n_output))
        self.b2 = np.zeros(n_output)
        self.activation_name = activation
        self.act_fn, self.act_deriv = ACTIVATIONS[activation]
        self.cache: dict[str, np.ndarray] = {}

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.cache["X"] = X
        z1 = X @ self.W1 + self.b1
        self.cache["z1"] = z1
        a1 = self.act_fn(z1)
        self.cache["a1"] = a1
        z2 = a1 @ self.W2 + self.b2
        self.cache["z2"] = z2
        return z2

    def backward(self, grad_logits: np.ndarray) -> None:
        a1 = self.cache["a1"]
        z1 = self.cache["z1"]
        X = self.cache["X"]

        # 输出层：dL/dW2 = a1^T @ dL/dz2
        self.dW2 = a1.T @ grad_logits
        self.db2 = np.sum(grad_logits, axis=0)
        grad_a1 = grad_logits @ self.W2.T
        # 隐藏层：链式法则 × 激活局部导数（逐元素）
        grad_z1 = grad_a1 * self.act_deriv(z1)
        self.dW1 = X.T @ grad_z1
        self.db1 = np.sum(grad_z1, axis=0)

    def get_grads(self) -> dict[str, np.ndarray]:
        return {"W1": self.dW1, "b1": self.db1, "W2": self.dW2, "b2": self.db2}


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_z = np.exp(shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def cross_entropy_grad(probs: np.ndarray, y: np.ndarray) -> np.ndarray:
    n_samples = y.shape[0]
    n_classes = probs.shape[1]
    one_hot = np.zeros((n_samples, n_classes))
    one_hot[np.arange(n_samples), y] = 1.0
    return (probs - one_hot) / n_samples


def accuracy(logits: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.argmax(logits, axis=1) == y))


def train(
    X: np.ndarray,
    y: np.ndarray,
    n_hidden: int = 16,
    activation: str = "relu",
    lr: float = 0.5,
    n_epochs: int = 500,
    seed: int = 42,
) -> tuple[MLP, list[float], list[float]]:
    n_input = X.shape[1]
    n_output = len(np.unique(y))
    model = MLP(n_input, n_hidden, n_output, activation=activation, seed=seed)
    losses, accs = [], []

    for _ in range(n_epochs):
        logits = model.forward(X)
        probs = softmax(logits)
        one_hot = np.zeros_like(probs)
        one_hot[np.arange(len(y)), y] = 1.0
        loss = float(-np.mean(np.sum(one_hot * np.log(probs + 1e-12), axis=1)))
        losses.append(loss)
        accs.append(accuracy(logits, y))

        grad = cross_entropy_grad(probs, y)
        model.backward(grad)
        model.W2 -= lr * model.dW2
        model.b2 -= lr * model.db2
        model.W1 -= lr * model.dW1
        model.b1 -= lr * model.db1

    return model, losses, accs


if __name__ == "__main__":
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo))
    from lib.synthetic_data import spiral_data

    X, y = spiral_data(n_per_class=100, n_classes=3)
    model, losses, accs = train(X, y, n_hidden=32, activation="relu", n_epochs=500)
    print(f"Final loss: {losses[-1]:.4f}, accuracy: {accs[-1]:.4f}")
