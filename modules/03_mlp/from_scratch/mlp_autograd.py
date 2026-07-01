"""梯度一致性检查：numpy 矩阵 backprop vs Module 01 标量 autograd

在 2-2-3 小网络上，对单样本用 MSE-on-logits 损失（非 CE，便于标量图实现），
逐参数比较 mlp_numpy.MLP.backward 与 Value.backward 的梯度。

通过本脚本可确认：手写矩阵公式与展开计算图是同一套链式法则。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "modules/01_preliminaries_autograd/from_scratch"))
sys.path.insert(0, str(REPO_ROOT / "modules/03_mlp/from_scratch"))

from mlp_numpy import MLP, cross_entropy_grad, softmax
from value import Value


def scalar_mlp_loss(
    x1: float,
    x2: float,
    params: dict[str, float],
    n_hidden: int = 2,
    n_classes: int = 3,
    label: int = 0,
) -> float:
    """Forward pass for a single sample using scalar Values."""
    xv1 = Value(x1)
    xv2 = Value(x2)
    hidden: list[Value] = []
    for h in range(n_hidden):
        w1 = Value(params[f"W1_0_{h}"])
        w2 = Value(params[f"W1_1_{h}"])
        b = Value(params[f"b1_{h}"])
        hidden.append((xv1 * w1 + xv2 * w2 + b).relu())

    logits: list[Value] = []
    for c in range(n_classes):
        total = Value(0.0)
        for h in range(n_hidden):
            w = Value(params[f"W2_{h}_{c}"])
            total = total + hidden[h] * w
        total = total + Value(params[f"b2_{c}"])
        logits.append(total)

    max_logit = logits[0]
    for lg in logits[1:]:
        max_logit = max_logit if max_logit.data > lg.data else lg

    exp_sum = Value(0.0)
    for lg in logits:
        exp_sum = exp_sum + (lg - max_logit).relu()  # simplified; use exp via power
    # Use log-sum-exp trick with scalar ops: compute softmax prob for target class
    # For gradient check, use negative log prob of correct class
    shifted = [lg - max_logit for lg in logits]
    # approximate exp with e^x via Taylor is wrong; use power with fractional - skip
    # Instead compute MSE on logits vs one-hot for gradient check simplicity
    target = Value(1.0 if label == 0 else 0.0)
    loss = (logits[label] - target) ** 2
    for c in range(n_classes):
        if c != label:
            t = Value(0.0)
            loss = loss + (logits[c] - t) ** 2
    return loss


def flatten_params(model: MLP) -> dict[str, float]:
    params: dict[str, float] = {}
    for i in range(model.W1.shape[0]):
        for j in range(model.W1.shape[1]):
            params[f"W1_{i}_{j}"] = float(model.W1[i, j])
    for j in range(model.b1.shape[0]):
        params[f"b1_{j}"] = float(model.b1[j])
    for i in range(model.W2.shape[0]):
        for j in range(model.W2.shape[1]):
            params[f"W2_{i}_{j}"] = float(model.W2[i, j])
    for j in range(model.b2.shape[0]):
        params[f"b2_{j}"] = float(model.b2[j])
    return params


def autograd_grads_single_sample(
    x: np.ndarray,
    y: int,
    params: dict[str, float],
    n_hidden: int,
    n_classes: int,
) -> dict[str, float]:
    """Compute gradients via scalar autograd for MSE-on-logits loss."""
    x1, x2 = float(x[0]), float(x[1])
    nodes = {k: Value(v) for k, v in params.items()}
    xv1 = Value(x1)
    xv2 = Value(x2)

    hidden: list[Value] = []
    for h in range(n_hidden):
        w1, w2, b = nodes[f"W1_0_{h}"], nodes[f"W1_1_{h}"], nodes[f"b1_{h}"]
        hidden.append((xv1 * w1 + xv2 * w2 + b).relu())

    logits: list[Value] = []
    for c in range(n_classes):
        total = Value(0.0)
        for h in range(n_hidden):
            total = total + hidden[h] * nodes[f"W2_{h}_{c}"]
        logits.append(total + nodes[f"b2_{c}"])

    loss = Value(0.0)
    for c in range(n_classes):
        target = Value(1.0 if c == y else 0.0)
        diff = logits[c] - target
        loss = loss + diff * diff

    for n in nodes.values():
        n.zero_grad()
    loss.backward()  # 标量图逆拓扑反传
    return {k: nodes[k].grad for k in params}


def numpy_grads_single_sample(
    model: MLP,
    x: np.ndarray,
    y: int,
) -> dict[str, float]:
    x_batch = x.reshape(1, -1)
    logits = model.forward(x_batch)
    n_classes = logits.shape[1]
    grad_logits = np.zeros_like(logits)
    for c in range(n_classes):
        target = 1.0 if c == y else 0.0
        grad_logits[0, c] = 2.0 * (logits[0, c] - target)
    model.backward(grad_logits)
    grads = model.get_grads()
    flat: dict[str, float] = {}
    for i in range(model.W1.shape[0]):
        for j in range(model.W1.shape[1]):
            flat[f"W1_{i}_{j}"] = float(grads["W1"][i, j])
    for j in range(model.b1.shape[0]):
        flat[f"b1_{j}"] = float(grads["b1"][j])
    for i in range(model.W2.shape[0]):
        for j in range(model.W2.shape[1]):
            flat[f"W2_{i}_{j}"] = float(grads["W2"][i, j])
    for j in range(model.b2.shape[0]):
        flat[f"b2_{j}"] = float(grads["b2"][j])
    return flat


def compare_gradients(tol: float = 1e-4) -> bool:
    model = MLP(2, 2, 3, activation="relu", seed=7)
    params = flatten_params(model)
    x = np.array([0.5, -0.3])
    y = 1

    ag = autograd_grads_single_sample(x, y, params, n_hidden=2, n_classes=3)
    ng = numpy_grads_single_sample(model, x, y)

    all_ok = True
    for key in params:
        a, n = ag[key], ng[key]
        err = abs(a - n) / max(abs(a), abs(n), 1e-12)
        ok = err < tol
        if not ok:
            print(f"  FAIL {key}: autograd={a:.6f}, numpy={n:.6f}, err={err:.2e}")
            all_ok = False
    return all_ok


if __name__ == "__main__":
    ok = compare_gradients()
    if ok:
        print("Autograd vs numpy backprop: all gradients match.")
    else:
        print("Gradient mismatch detected.")
        sys.exit(1)
