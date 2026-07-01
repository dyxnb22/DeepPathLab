"""从零实现 SGD、Momentum、Adam 优化器（numpy，教学用）。

本模块不依赖 autograd：调用方负责计算 grads，优化器只执行参数更新。
接口与 PyTorch optimizer 类似：step(params, grads) + zero_state()。

公式参见 notes.md；单步 sanity check 见 __main__。
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Optimizer(Protocol):
    """优化器协议：对参数字典就地更新。"""

    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> None: ...
    def zero_state(self) -> None: ...


class SGD:
    """随机梯度下降：θ ← θ - η·g。

    最简更新规则；对学习率敏感，无历史状态。
    """

    def __init__(self, lr: float = 0.01) -> None:
        self.lr = lr

    def zero_state(self) -> None:
        pass  # SGD 无内部状态

    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> None:
        for k in params:
            params[k] -= self.lr * grads[k]


class SGDMomentum:
    """SGD + 动量：v ← μv + g，θ ← θ - ηv。

    velocity 按参数名分别维护，与 PyTorch SGD(momentum=μ) 一致。
    """

    def __init__(self, lr: float = 0.01, momentum: float = 0.9) -> None:
        self.lr = lr
        self.momentum = momentum
        self.velocity: dict[str, np.ndarray] = {}

    def zero_state(self) -> None:
        self.velocity = {}

    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> None:
        for k in params:
            if k not in self.velocity:
                self.velocity[k] = np.zeros_like(params[k])
            # 指数移动平均梯度方向
            self.velocity[k] = self.momentum * self.velocity[k] + grads[k]
            params[k] -= self.lr * self.velocity[k]


class Adam:
    """Adam：自适应学习率 + 动量，含偏差修正。

    m ← β₁m + (1-β₁)g
    v ← β₂v + (1-β₂)g²
    m̂ = m/(1-β₁ᵗ),  v̂ = v/(1-β₂ᵗ)
    θ ← θ - η·m̂/(√v̂ + ε)
    """

    def __init__(self, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> None:
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0  # 全局步数，用于偏差修正
        self.m: dict[str, np.ndarray] = {}
        self.v: dict[str, np.ndarray] = {}

    def zero_state(self) -> None:
        self.t = 0
        self.m = {}
        self.v = {}

    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> None:
        self.t += 1
        for k in params:
            if k not in self.m:
                self.m[k] = np.zeros_like(params[k])
                self.v[k] = np.zeros_like(params[k])
            # 一阶矩（动量）与二阶矩（梯度平方 EMA）
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * grads[k]
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * (grads[k] ** 2)
            # 偏差修正：训练初期 m,v 偏向 0，需除以 (1-β^t)
            m_hat = self.m[k] / (1 - self.beta1 ** self.t)
            v_hat = self.v[k] / (1 - self.beta2 ** self.t)
            params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


if __name__ == "__main__":
    # 固定梯度下对比三种优化器的单步更新量
    params = {"w": np.array([1.0, 2.0])}
    grads = {"w": np.array([0.5, -0.3])}
    for name, opt in [("SGD", SGD(0.1)), ("Momentum", SGDMomentum(0.1)), ("Adam", Adam(0.1))]:
        p = {"w": params["w"].copy()}
        opt.step(p, grads)
        print(f"{name}: w -> {p['w']}")
