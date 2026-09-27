"""Configuration loading + merge helpers.

All experiment parameters live in config/*.yaml so experiments are
reproducible (every run stores a full config snapshot).
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(name: str) -> dict:
    return load_yaml(CONFIG_DIR / f"{name}.yaml")


def load_all_config() -> dict:
    return {
        "actions": load_config("actions"),
        "experiment": load_config("experiment"),
        "tracking": load_config("tracking"),
    }


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` into a deep copy of `base`."""
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def flatten(d: dict, prefix: str = "") -> dict[str, Any]:
    """Flatten nested dict to dotted keys (for manifest snapshot)."""
    items: dict[str, Any] = {}
    for k, v in (d or {}).items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            items.update(flatten(v, key))
        else:
            items[key] = v
    return items
