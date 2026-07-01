#!/usr/bin/env python3
"""Compare our scalar autograd against PyTorch on the same expressions."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "modules/01_preliminaries_autograd/from_scratch"))

from value import Value


def check_expression(name: str, build_scratch, build_torch) -> bool:
    scratch_loss, scratch_vars = build_scratch()
    scratch_loss.backward()
    scratch_grads = {k: v.grad for k, v in scratch_vars.items()}

    torch_loss, torch_vars = build_torch()
    torch_loss.backward()
    torch_grads = {k: float(v.grad) for k, v in torch_vars.items()}

    print(f"\n=== {name} ===")
    all_ok = True
    for key in scratch_grads:
        s, t = scratch_grads[key], torch_grads[key]
        err = abs(s - t) / max(abs(s), abs(t), 1e-12)
        ok = err < 1e-5
        status = "PASS" if ok else "FAIL"
        print(f"  {key}: scratch={s:.6f}, torch={t:.6f}, rel_err={err:.2e} [{status}]")
        all_ok &= ok
    return all_ok


def main() -> None:
    passed = True

    def simple_scratch():
        a, b, c = Value(2.0), Value(-3.0), Value(1.0)
        loss = (a * b + c).relu()
        return loss, {"a": a, "b": b, "c": c}

    def simple_torch():
        a = torch.tensor(2.0, requires_grad=True)
        b = torch.tensor(-3.0, requires_grad=True)
        c = torch.tensor(1.0, requires_grad=True)
        loss = torch.relu(a * b + c)
        return loss, {"a": a, "b": b, "c": c}

    passed &= check_expression("relu(a*b + c)", simple_scratch, simple_torch)

    def shared_scratch():
        a, b, c = Value(1.5), Value(2.0), Value(-1.0)
        loss = a * b + b * c
        return loss, {"a": a, "b": b, "c": c}

    def shared_torch():
        a = torch.tensor(1.5, requires_grad=True)
        b = torch.tensor(2.0, requires_grad=True)
        c = torch.tensor(-1.0, requires_grad=True)
        loss = a * b + b * c
        return loss, {"a": a, "b": b, "c": c}

    passed &= check_expression("a*b + b*c (shared b)", shared_scratch, shared_torch)

    if passed:
        print("\nPyTorch comparison: all PASSED.")
    else:
        print("\nPyTorch comparison: some FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
