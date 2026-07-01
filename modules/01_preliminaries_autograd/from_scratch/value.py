"""标量自动微分引擎（Module 01 from_scratch）

本文件实现一个最小可用的标量 autograd：
- 前向：运算符重载构建计算图（节点存 data、子节点、局部反向闭包）
- 反向：对输出节点做拓扑排序，逆序调用各节点的 _backward，沿链式法则传梯度

学习重点：观察 __mul__ / relu 等里 _backward 如何捕获前向操作数，以及 backward() 为何需要逆拓扑序。
完整训练请用 PyTorch；本引擎用于理解框架 autograd 的底层机制。
"""

from __future__ import annotations

from typing import Callable


class Value:
    """A scalar node in a computational graph with automatic differentiation."""

    def __init__(self, data: float, _children: tuple[Value, ...] = (), _op: str = "") -> None:
        self.data = float(data)
        self.grad = 0.0
        self._backward: Callable[[], None] = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self) -> str:
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    def __add__(self, other: Value | float) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward() -> None:
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __radd__(self, other: float) -> Value:
        return self + other

    def __mul__(self, other: Value | float) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward() -> None:
            # 乘法链式法则：∂(a*b)/∂a = b，闭包捕获的是前向时的 other.data
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other: float) -> Value:
        return self * other

    def __neg__(self) -> Value:
        return self * -1

    def __sub__(self, other: Value | float) -> Value:
        return self + (-other if isinstance(other, Value) else -other)

    def __rsub__(self, other: float) -> Value:
        return Value(other) - self

    def __truediv__(self, other: Value | float) -> Value:
        return self * (other ** -1 if isinstance(other, Value) else other ** -1)

    def __rtruediv__(self, other: float) -> Value:
        return Value(other) / self

    def __pow__(self, other: float) -> Value:
        out = Value(self.data ** other, (self,), f"**{other}")

        def _backward() -> None:
            self.grad += other * (self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out

    def relu(self) -> Value:
        out = Value(max(0.0, self.data), (self,), "relu")

        def _backward() -> None:
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad

        out._backward = _backward
        return out

    def tanh(self) -> Value:
        t = __import__("math").tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward() -> None:
            self.grad += (1.0 - t * t) * out.grad

        out._backward = _backward
        return out

    def zero_grad(self) -> None:
        self.grad = 0.0

    def backward(self) -> None:
        """Reverse-mode autodiff via topological sort."""
        topo: list[Value] = []
        visited: set[Value] = set()

        def build_topo(v: Value) -> None:
            # 后序遍历：子节点先入栈，保证 topo 中父节点在子节点之后
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1.0  # 对标量 loss，∂L/∂L = 1
        # 逆拓扑序：从 loss 往叶子传梯度，下游就绪后再调 _backward
        for node in reversed(topo):
            node._backward()

    def graph_label(self) -> str:
        """Short label for graph visualization."""
        op = f" ({self._op})" if self._op else ""
        return f"{self.data:.3f}{op}"
