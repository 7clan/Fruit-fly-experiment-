"""Configuration loading for Phase 2.

Single sources of truth:
  - phase2/config/gate.yaml      pre-registered gate + analysis constants
  - phase2/config/tracking.yaml  detector / recording parameters (apparatus)

No analysis constant is duplicated here; the gate file is authoritative.
"""
from __future__ import annotations

from pathlib import Path

import yaml

PHASE2_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PHASE2_ROOT / "config"
DATA_DIR = PHASE2_ROOT / "data"
RESULTS_DIR = PHASE2_ROOT / "results"


def _load(name: str) -> dict:
    with open(CONFIG_DIR / f"{name}.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_gate_cfg() -> dict:
    """Pre-registered gate thresholds + analysis constants (do not edit)."""
    return _load("gate")


def load_tracking_cfg() -> dict:
    """Detector / recording parameters."""
    return _load("tracking")


def analysis_constants(gate_cfg: dict | None = None) -> dict:
    """Flat view of gate.yaml:analysis_constants (the frozen analysis params)."""
    g = gate_cfg if gate_cfg is not None else load_gate_cfg()
    return dict(g["analysis_constants"])
