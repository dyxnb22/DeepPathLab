"""LSTM cell from scratch (NumPy) — 单步前向与门控检查.

Implements the standard LSTM equations with forget-gate bias init (=1).
Cell state c_t provides a linear path for long-range gradient flow.

Run: python modules/07_lstm_gru/from_scratch/lstm_cell.py
"""

from __future__ import annotations

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def lstm_step(
    x_t: np.ndarray,
    h_prev: np.ndarray,
    c_prev: np.ndarray,
    W_xi: np.ndarray,
    W_hi: np.ndarray,
    bi: np.ndarray,
    W_xf: np.ndarray,
    W_hf: np.ndarray,
    bf: np.ndarray,
    W_xo: np.ndarray,
    W_ho: np.ndarray,
    bo: np.ndarray,
    W_xc: np.ndarray,
    W_hc: np.ndarray,
    bc: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Single LSTM timestep — 单时间步 LSTM 前向.

    Gates:
        i_t: input gate  — 控制写入多少新信息到 cell
        f_t: forget gate — 控制保留多少旧 cell（≈1 时长程记忆）
        o_t: output gate — 控制读出多少 cell 到 h_t

    Cell update (linear path for gradients):
        c_t = f_t * c_{t-1} + i_t * tanh(candidate)
        h_t = o_t * tanh(c_t)
    """
    i_t = sigmoid(W_xi @ x_t + W_hi @ h_prev + bi)
    f_t = sigmoid(W_xf @ x_t + W_hf @ h_prev + bf)
    o_t = sigmoid(W_xo @ x_t + W_ho @ h_prev + bo)
    c_tilde = np.tanh(W_xc @ x_t + W_hc @ h_prev + bc)
    c_t = f_t * c_prev + i_t * c_tilde  # 关键：cell 线性累加
    h_t = o_t * np.tanh(c_t)
    gates = {"input": i_t, "forget": f_t, "output": o_t, "cell": c_t}
    return h_t, c_t, gates


def lstm_forward(
    x_seq: np.ndarray,
    params: dict[str, np.ndarray],
    hidden_dim: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[dict[str, np.ndarray]]]:
    """Unroll LSTM over sequence x_seq: (T, input_dim)."""
    T = x_seq.shape[0]
    h = np.zeros(hidden_dim)
    c = np.zeros(hidden_dim)
    h_states, c_states, all_gates = [h.copy()], [c.copy()], []

    keys = ["W_xi", "W_hi", "bi", "W_xf", "W_hf", "bf", "W_xo", "W_ho", "bo", "W_xc", "W_hc", "bc"]
    p = params

    for t in range(T):
        h, c, gates = lstm_step(x_seq[t], h, c, *[p[k] for k in keys])
        h_states.append(h.copy())
        c_states.append(c.copy())
        all_gates.append(gates)

    return h_states, c_states, all_gates


def init_lstm_params(input_dim: int, hidden_dim: int, seed: int = 0) -> dict[str, np.ndarray]:
    """Initialize LSTM weights; forget bias = 1 so f_t starts near 1.

    遗忘门偏置初始化为 1 是常见技巧：训练初期倾向于保留 cell 状态。
    """
    rng = np.random.default_rng(seed)
    scale = 0.1
    params = {}
    for prefix in ("i", "f", "o", "c"):
        params[f"W_x{prefix}"] = rng.normal(scale=scale, size=(hidden_dim, input_dim))
        params[f"W_h{prefix}"] = rng.normal(scale=scale, size=(hidden_dim, hidden_dim))
        params[f"b{prefix}"] = np.zeros(hidden_dim)
        if prefix == "f":
            params["bf"] = np.ones(hidden_dim)  # forget bias init → f_t ≈ σ(1) ≈ 0.73
    return params


if __name__ == "__main__":
    input_dim, hidden_dim, T = 9, 16, 12
    rng = np.random.default_rng(42)
    x_seq = rng.normal(size=(T, input_dim)) * 0.1
    params = init_lstm_params(input_dim, hidden_dim)
    h_states, c_states, gates = lstm_forward(x_seq, params, hidden_dim)
    print(f"LSTM forward: T={T}, final h norm={np.linalg.norm(h_states[-1]):.4f}")
    print(f"Mean forget gate (last step): {gates[-1]['forget'].mean():.4f}")
