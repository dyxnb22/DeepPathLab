"""Numerical gradient checking utilities."""

from __future__ import annotations

from typing import Callable


def numerical_gradient(
    fn: Callable[[float], float],
    x: float,
    eps: float = 1e-5,
) -> float:
    """Finite-difference gradient for a scalar function."""
    return (fn(x + eps) - fn(x - eps)) / (2.0 * eps)


def relative_error(analytic: float, numeric: float, eps: float = 1e-12) -> float:
    """Relative error between analytic and numeric gradients."""
    return abs(analytic - numeric) / max(abs(analytic), abs(numeric), eps)


def check_scalar_gradients(
    variables: dict[str, float],
    loss_fn: Callable[[dict[str, float]], float],
    analytic_grads: dict[str, float],
    eps: float = 1e-5,
    tol: float = 1e-5,
) -> dict[str, dict[str, float]]:
    """Compare analytic gradients against finite differences.

    Returns a dict mapping variable names to analytic, numeric, and error values.
    """
    results: dict[str, dict[str, float]] = {}
    for name in variables:
        def perturbed(val: float, var=name) -> float:
            perturbed_vars = dict(variables)
            perturbed_vars[var] = val
            return loss_fn(perturbed_vars)

        numeric = numerical_gradient(perturbed, variables[name], eps=eps)
        analytic = analytic_grads[name]
        err = relative_error(analytic, numeric)
        results[name] = {
            "analytic": analytic,
            "numeric": numeric,
            "relative_error": err,
            "passed": err < tol,
        }
    return results


def print_gradient_report(results: dict[str, dict[str, float]]) -> bool:
    """Print a gradient check report. Returns True if all checks passed."""
    all_passed = True
    for name, info in results.items():
        status = "PASS" if info["passed"] else "FAIL"
        print(
            f"  {name}: analytic={info['analytic']:.6f}, "
            f"numeric={info['numeric']:.6f}, "
            f"rel_err={info['relative_error']:.2e} [{status}]"
        )
        if not info["passed"]:
            all_passed = False
    return all_passed
