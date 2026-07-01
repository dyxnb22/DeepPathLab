#!/usr/bin/env python3
"""PyTorch matrix factorization baseline."""

from __future__ import annotations

import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).resolve().parents[1] / "experiments" / "baseline_comparison.py"))
