"""Local analysis package for project-level utilities.

This file makes the `analysis` folder a proper Python package so it can be
installed with pip (editable or system-wide installs).
"""
from .causal_shap import *

__all__ = [name for name in globals() if not name.startswith("_")]
