"""Shared utilities for DeepPath Lab modules.

Submodules:
  gradient_check — numerical gradient verification
  plotting       — save training curves to module outputs/
  synthetic_data — reproducible numpy datasets for early modules
"""

from lib import gradient_check, plotting, synthetic_data

__all__ = ["gradient_check", "plotting", "synthetic_data"]
