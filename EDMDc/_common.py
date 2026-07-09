"""Shared utilities for the EDMDc package.

Minimal helpers used by the collectors (collect_fossen, collect_isaac) and
the train/eval scripts. Add more utilities here as new modules need them.
"""
from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Repo root — one parent up from EDMDc/."""
    return Path(__file__).resolve().parents[1]


def default_yaml() -> Path:
    """Default vehicle YAML (BlueROVHeavy)."""
    return project_root() / "assets" / "BlueROVHeavy" / "BlueROVHeavy.yaml"
