#!/usr/bin/env python3
"""Visualize computational graph structure and backward order."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "modules/01_preliminaries_autograd/from_scratch"))

from value import Value


def build_topo(root: Value) -> list[Value]:
    topo: list[Value] = []
    visited: set[Value] = set()

    def visit(v: Value) -> None:
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                visit(child)
            topo.append(v)

    visit(root)
    return topo


def print_graph(root: Value) -> None:
    topo = build_topo(root)
    print("Forward topological order (inputs -> output):")
    for i, node in enumerate(topo):
        children = ", ".join(c.graph_label() for c in node._prev) or "leaf"
        print(f"  [{i}] {node.graph_label()}  <-  {children}")

    print("\nBackward order (output -> inputs):")
    for i, node in enumerate(reversed(topo)):
        print(f"  [{i}] {node.graph_label()}")


def main() -> None:
    a = Value(2.0)
    b = Value(-3.0)
    c = Value(1.0)
    d = a * b
    e = d + c
    f = e.relu()
    g = f * b  # b shared: used in d and g

    print("Expression: g = relu(a*b + c) * b")
    print_graph(g)

    g.backward()
    print("\nGradients after backward:")
    for name, node in [("a", a), ("b", b), ("c", c)]:
        print(f"  d{name}/dg = {node.grad:.6f}")


if __name__ == "__main__":
    main()
