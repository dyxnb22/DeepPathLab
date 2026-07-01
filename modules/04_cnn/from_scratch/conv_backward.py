"""卷积反向传播概念说明（Module 04 from_scratch）

本文件不实现完整训练级 conv backward，而是记录核心公式与直觉：
- ∂L/∂x：用翻转后的 kernel 对 grad_output 做 full convolution
- ∂L/∂W：各输出位置的 input patch 与 grad_output 外积累加

实际 LeNet 训练见 reproduce/lenet_fashion_mnist.py（PyTorch autograd）。
im2col 将卷积化为矩阵乘，是 GPU 高效实现的基础。
"""

from __future__ import annotations

# 详细推导见本模块 notes.md；运行本脚本可快速回顾要点。

CONV_BACKWARD_NOTES = """
Gradient w.r.t. input (simplified single-channel):
  pad grad_output, then convolve with rotated kernel.

Gradient w.r.t. kernel:
  For each output position, outer product of input patch and grad_output value.
"""

if __name__ == "__main__":
    print(CONV_BACKWARD_NOTES)
