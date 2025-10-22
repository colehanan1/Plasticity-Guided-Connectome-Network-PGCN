"""Test utilities for ensuring optional dependencies are present."""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _ensure_package(name: str) -> None:
    try:
        importlib.import_module(name)
    except ImportError:  # pragma: no cover - executed on CI nodes without deps
        subprocess.check_call([sys.executable, "-m", "pip", "install", name])


def pytest_sessionstart(session):  # type: ignore[unused-argument]
    for pkg in ("pandas", "pyarrow", "rich", "tenacity", "networkx"):
        _ensure_package(pkg)
