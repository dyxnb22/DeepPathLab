"""GRU cell forward pass from scratch (numpy)."""

from __future__ import annotations

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def gru_step(
    x_t: np.ndarray,
    h_prev: np.ndarray,
    W_xr: np.ndarray,
    W_hr: np.ndarray,
    br: np.ndarray,
    W_xz: np.ndarray,
    W_hz: np.ndarray,
    bz: np.ndarray,
    W_xh: np.ndarray,
    W_hh: np.ndarray,
    bh: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Single GRU timestep: reset and update gates."""
    r_t = sigmoid(W_xr @ x_t + W_hr @ h_prev + br)
    z_t = sigmoid(W_xz @ x_t + W_hz @ h_prev + bz)
    h_tilde = np.tanh(W_xh @ x_t + W_hh @ (r_t * h_prev) + bh)
    h_t = (1 - z_t) * h_prev + z_t * h_tilde
    gates = {"reset": r_t, "update": z_t, "candidate": h_tilde}
    return h_t, gates


def gru_forward(
    x_seq: np.ndarray,
    params: dict[str, np.ndarray],
    hidden_dim: int,
) -> tuple[list[np.ndarray], list[dict[str, np.ndarray]]]:
    T = x_seq.shape[0]
    h = np.zeros(hidden_dim)
    h_states, all_gates = [h.copy()], []

    for t in range(T):
        h, gates = gru_step(
            x_seq[t],
            h,
            params["W_xr"], params["W_hr"], params["br"],
            params["W_xz"], params["W_hz"], params["bz"],
            params["W_xh"], params["W_hh"], params["bh"],
        )
        h_states.append(h.copy())
        all_gates.append(gates)

    return h_states, all_gates


def init_gru_params(input_dim: int, hidden_dim: int, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    scale = 0.1
    params = {}
    for name in ("r", "z"):
        params[f"W_x{name}"] = rng.normal(scale=scale, size=(hidden_dim, input_dim))
        params[f"W_h{name}"] = rng.normal(scale=scale, size=(hidden_dim, hidden_dim))
        params[f"b{name}"] = np.zeros(hidden_dim)
    params["W_xh"] = rng.normal(scale=scale, size=(hidden_dim, input_dim))
    params["W_hh"] = rng.normal(scale=scale, size=(hidden_dim, hidden_dim))
    params["bh"] = np.zeros(hidden_dim)
    return params


if __name__ == "__main__":
    input_dim, hidden_dim, T = 9, 16, 12
    rng = np.random.default_rng(42)
    x_seq = rng.normal(size=(T, input_dim)) * 0.1
    params = init_gru_params(input_dim, hidden_dim)
    h_states, gates = gru_forward(x_seq, params, hidden_dim)
    print(f"GRU forward: T={T}, final h norm={np.linalg.norm(h_states[-1]):.4f}")
    print(f"Mean update gate (last step): {gates[-1]['update'].mean():.4f}")
