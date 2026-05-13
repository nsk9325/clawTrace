"""Shared test helpers. pytest auto-loads this and exposes ``load_module``
to every test in this directory via ``from conftest import load_module``."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any


def load_module(module_name: str, file_name: str) -> Any:
    """Load a top-level project module by file name (relative to the repo root)."""
    module_path = Path(__file__).resolve().parent.parent / file_name
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
