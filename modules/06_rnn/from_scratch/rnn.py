"""Vanilla RNN from scratch — NumPy 前向与全序列 BPTT.

Implements:
  - rnn_forward: unrolled hidden states h_0 .. h_T
  - rnn_backward_full: BPTT with per-timestep cross-entropy loss
  - sequence_copy_task: synthetic task to expose vanishing gradients

Run: python modules/06_rnn/from_scratch/rnn.py
"""

from __future__ import annotations

import numpy as np


def rnn_forward(
    x_seq: np.ndarray,
    W_xh: np.ndarray,
    W_hh: np.ndarray,
    b_h: np.ndarray,
    W_hy: np.ndarray,
    b_y: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    """Forward pass for a single sequence (时间展开前向).

    Args:
        x_seq: (T, input_dim) — 输入序列，每步一个 one-hot 或特征向量
        W_xh, W_hh, b_h: 隐藏层参数
        W_hy, b_y: 输出层参数

    Returns:
        hidden_states: length T+1，含初始 h_0=0
        outputs: length T，每步 logits
        final_logits: 最后一步输出（便捷访问）
    """
    T = x_seq.shape[0]
    hidden_dim = W_hh.shape[0]
    h = np.zeros(hidden_dim)
    hidden_states = [h.copy()]
    outputs = []

    for t in range(T):
        # h_t = tanh(W_xh @ x_t + W_hh @ h_{t-1} + b_h)
        h = np.tanh(W_xh @ x_seq[t] + W_hh @ h + b_h)
        hidden_states.append(h.copy())
        y = W_hy @ h + b_y
        outputs.append(y)

    return hidden_states, outputs, outputs[-1]


def rnn_backward_full(
    x_seq: np.ndarray,
    hidden_states: list[np.ndarray],
    outputs: list[np.ndarray],
    targets: np.ndarray,
    W_xh: np.ndarray,
    W_hh: np.ndarray,
    W_hy: np.ndarray,
) -> tuple[dict[str, np.ndarray], float, float]:
    """Full BPTT with per-timestep cross-entropy loss.

    从 t=T-1 反向到 t=0：
      1. 输出层梯度 dy → 累积 dW_hy, db_y
      2. 隐藏层梯度 dh 沿时间回传：dh = W_hh.T @ (dh * (1-h^2))
      3. 每步累积 dW_xh, dW_hh, db_h

    Returns:
        grads: 各参数梯度
        loss: 平均交叉熵
        dh0_norm: ||dh_0|| — 回传到初始时刻的梯度范数，用于观察消失
    """
    T = x_seq.shape[0]
    hidden_dim = W_hh.shape[0]
    output_dim = W_hy.shape[0]

    dW_xh = np.zeros_like(W_xh)
    dW_hh = np.zeros_like(W_hh)
    db_h = np.zeros(hidden_dim)
    dW_hy = np.zeros_like(W_hy)
    db_y = np.zeros(output_dim)

    total_loss = 0.0
    dh = np.zeros(hidden_dim)  # 从后续时间步回传的隐藏梯度

    for t in reversed(range(T)):
        logits = outputs[t]
        probs = np.exp(logits - np.max(logits))
        probs /= probs.sum()
        target = int(targets[t])
        one_hot = np.zeros_like(probs)
        one_hot[target] = 1.0
        dy = probs - one_hot  # softmax + CE 的梯度
        total_loss += float(-np.log(probs[target] + 1e-12))

        dW_hy += np.outer(dy, hidden_states[t + 1])
        db_y += dy
        dh = W_hy.T @ dy + dh  # 累加来自 t+1 及 loss_t 的梯度

        h = hidden_states[t + 1]
        h_prev = hidden_states[t]
        dtanh = dh * (1 - h * h)  # tanh 局部导数
        dW_xh += np.outer(dtanh, x_seq[t])
        dW_hh += np.outer(dtanh, h_prev)
        db_h += dtanh
        dh = W_hh.T @ dtanh  # 继续向 t-1 回传（连乘 W_hh 的来源）

    grads = {"W_xh": dW_xh, "W_hh": dW_hh, "b_h": db_h, "W_hy": dW_hy, "b_y": db_y}
    return grads, total_loss / T, float(np.linalg.norm(dh))


def sequence_copy_task(seq_len: int = 5, vocab_size: int = 4, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Copy task: encode symbol at step 0, predict it at every step.

    首步 one-hot 编码待记忆符号，每步都预测同一符号。
    迫使 BPTT 将梯度从末步传回首步，适合观察 |dh_0| 随 seq_len 衰减。
    """
    rng = np.random.default_rng(seed)
    symbol = rng.integers(0, vocab_size)
    x_seq = np.zeros((seq_len, vocab_size))
    x_seq[0, symbol] = 1.0
    targets = np.full(seq_len, symbol)
    return x_seq, targets


if __name__ == "__main__":
    vocab_size = 4
    seq_len = 8
    input_dim = vocab_size
    hidden_dim = 16
    rng = np.random.default_rng(42)

    W_xh = rng.normal(scale=0.1, size=(hidden_dim, input_dim))
    W_hh = rng.normal(scale=0.1, size=(hidden_dim, hidden_dim))
    b_h = np.zeros(hidden_dim)
    W_hy = rng.normal(scale=0.1, size=(vocab_size, hidden_dim))
    b_y = np.zeros(vocab_size)

    x_seq, targets = sequence_copy_task(seq_len, vocab_size)
    hidden_states, outputs, _ = rnn_forward(x_seq, W_xh, W_hh, b_h, W_hy, b_y)
    grads, loss, dh0_norm = rnn_backward_full(
        x_seq, hidden_states, outputs, targets, W_xh, W_hh, W_hy
    )
    print(f"Sequence length: {seq_len}, initial loss: {loss:.4f}, |dh_0|={dh0_norm:.6f}")
    print(f"Gradient norms: W_hh={np.linalg.norm(grads['W_hh']):.4f}, W_xh={np.linalg.norm(grads['W_xh']):.4f}")
