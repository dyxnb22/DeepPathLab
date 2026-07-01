#!/usr/bin/env python3
"""Entry point for optimizer benchmark."""

from __future__ import annotations

import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).resolve().parents[1] / "experiments" / "optimizer_comparison.py"))
