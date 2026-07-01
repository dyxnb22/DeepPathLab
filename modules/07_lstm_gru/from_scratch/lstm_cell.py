"""LSTM cell forward pass from scratch (numpy)."""

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
    """Single LSTM timestep.

    Returns h_t, c_t, and gate activations for inspection.
    """
    i_t = sigmoid(W_xi @ x_t + W_hi @ h_prev + bi)
    f_t = sigmoid(W_xf @ x_t + W_hf @ h_prev + bf)
    o_t = sigmoid(W_xo @ x_t + W_ho @ h_prev + bo)
    c_tilde = np.tanh(W_xc @ x_t + W_hc @ h_prev + bc)
    c_t = f_t * c_prev + i_t * c_tilde
    h_t = o_t * np.tanh(c_t)
    gates = {"input": i_t, "forget": f_t, "output": o_t, "cell": c_t}
    return h_t, c_t, gates


def lstm_forward(
    x_seq: np.ndarray,
    params: dict[str, np.ndarray],
    hidden_dim: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[dict[str, np.ndarray]]]:
    """Forward pass over sequence x_seq (T, input_dim)."""
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
    rng = np.random.default_rng(seed)
    scale = 0.1
    params = {}
    for prefix in ("i", "f", "o", "c"):
        params[f"W_x{prefix}"] = rng.normal(scale=scale, size=(hidden_dim, input_dim))
        params[f"W_h{prefix}"] = rng.normal(scale=scale, size=(hidden_dim, hidden_dim))
        params[f"b{prefix}"] = np.zeros(hidden_dim)
        if prefix == "f":
            params["bf"] = np.ones(hidden_dim)  # forget bias init to 1 (common trick)
    return params


if __name__ == "__main__":
    input_dim, hidden_dim, T = 9, 16, 12
    rng = np.random.default_rng(42)
    x_seq = rng.normal(size=(T, input_dim)) * 0.1
    params = init_lstm_params(input_dim, hidden_dim)
    h_states, c_states, gates = lstm_forward(x_seq, params, hidden_dim)
    print(f"LSTM forward: T={T}, final h norm={np.linalg.norm(h_states[-1]):.4f}")
    print(f"Mean forget gate (last step): {gates[-1]['forget'].mean():.4f}")
