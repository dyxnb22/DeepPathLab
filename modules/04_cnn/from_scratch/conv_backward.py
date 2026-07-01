"""Notes and reference for convolution backward pass.

Full conv backward is complex; this module documents the key ideas
rather than implementing a full training-grade conv backward.

Forward: y = conv(x, W)
Backward w.r.t. input: conv_backward on grad_output uses flipped kernels (full convolution).
Backward w.r.t. weights: correlate input with grad_output.

For training, we use PyTorch autograd in reproduce/lenet_fashion_mnist.py.
The from_scratch conv2d.py focuses on forward pass correctness.

Key insight (im2col):
Convolution can be expressed as matrix multiplication by unfolding
input patches into columns, enabling efficient GPU implementation.
"""

from __future__ import annotations

# This file serves as documentation for conv backward concepts.
# See notes.md in this module for the full explanation.

CONV_BACKWARD_NOTES = """
Gradient w.r.t. input (simplified single-channel):
  pad grad_output, then convolve with rotated kernel.

Gradient w.r.t. kernel:
  For each output position, outer product of input patch and grad_output value.
"""

if __name__ == "__main__":
    print(CONV_BACKWARD_NOTES)
