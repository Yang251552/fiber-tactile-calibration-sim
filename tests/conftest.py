"""Pytest fixtures. Puts src/ on sys.path so `import fibercal` works."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fibercal.config import load_config  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def size_mm(cfg):
    return float(cfg["surface"]["size_mm"])
