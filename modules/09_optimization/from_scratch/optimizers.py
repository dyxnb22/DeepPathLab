"""Optimizer update rules from scratch (numpy, educational)."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Optimizer(Protocol):
    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> None: ...
    def zero_state(self) -> None: ...


class SGD:
    def __init__(self, lr: float = 0.01) -> None:
        self.lr = lr

    def zero_state(self) -> None:
        pass

    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> None:
        for k in params:
            params[k] -= self.lr * grads[k]


class SGDMomentum:
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
            self.velocity[k] = self.momentum * self.velocity[k] + grads[k]
            params[k] -= self.lr * self.velocity[k]


class Adam:
    def __init__(self, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> None:
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
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
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * grads[k]
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * (grads[k] ** 2)
            m_hat = self.m[k] / (1 - self.beta1 ** self.t)
            v_hat = self.v[k] / (1 - self.beta2 ** self.t)
            params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


if __name__ == "__main__":
    params = {"w": np.array([1.0, 2.0])}
    grads = {"w": np.array([0.5, -0.3])}
    for name, opt in [("SGD", SGD(0.1)), ("Momentum", SGDMomentum(0.1)), ("Adam", Adam(0.1))]:
        p = {"w": params["w"].copy()}
        opt.step(p, grads)
        print(f"{name}: w -> {p['w']}")
