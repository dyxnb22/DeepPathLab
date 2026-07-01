#!/usr/bin/env python3
"""Numerical gradient checks for the scalar autograd engine."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "modules/01_preliminaries_autograd/from_scratch"))

from lib.gradient_check import check_scalar_gradients, print_gradient_report
from value import Value


def loss_simple(vars_dict: dict[str, float]) -> float:
    """f = (a * b + c).relu()"""
    a, b, c = vars_dict["a"], vars_dict["b"], vars_dict["c"]
    return max(0.0, a * b + c)


def loss_shared(vars_dict: dict[str, float]) -> float:
    """f = a * b + b * c  (b used twice, tests gradient accumulation)"""
    a, b, c = vars_dict["a"], vars_dict["b"], vars_dict["c"]
    return a * b + b * c


def loss_power_relu(vars_dict: dict[str, float]) -> float:
    """f = (x**2).relu()"""
    x = vars_dict["x"]
    return max(0.0, x ** 2)


def run_autograd(loss_name: str, variables: dict[str, float]) -> dict[str, float]:
    nodes = {k: Value(v) for k, v in variables.items()}
    if loss_name == "simple":
        a, b, c = nodes["a"], nodes["b"], nodes["c"]
        loss = (a * b + c).relu()
    elif loss_name == "shared":
        a, b, c = nodes["a"], nodes["b"], nodes["c"]
        loss = a * b + b * c
    elif loss_name == "power_relu":
        x = nodes["x"]
        loss = (x ** 2).relu()
    else:
        raise ValueError(f"Unknown loss: {loss_name}")

    for n in nodes.values():
        n.zero_grad()
    loss.backward()
    return {k: nodes[k].grad for k in variables}


def main() -> None:
    print("=== Gradient Check: simple relu expression ===")
    vars_simple = {"a": 2.0, "b": -3.0, "c": 1.0}
    analytic = run_autograd("simple", vars_simple)
    results = check_scalar_gradients(vars_simple, loss_simple, analytic)
    passed = print_gradient_report(results)

    print("\n=== Gradient Check: shared variable (b used twice) ===")
    vars_shared = {"a": 1.5, "b": 2.0, "c": -1.0}
    analytic = run_autograd("shared", vars_shared)
    results = check_scalar_gradients(vars_shared, loss_shared, analytic)
    passed &= print_gradient_report(results)

    print("\n=== Gradient Check: power + relu ===")
    vars_power = {"x": 0.5}
    analytic = run_autograd("power_relu", vars_power)
    results = check_scalar_gradients(vars_power, loss_power_relu, analytic)
    passed &= print_gradient_report(results)

    if passed:
        print("\nAll gradient checks PASSED.")
    else:
        print("\nSome gradient checks FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
