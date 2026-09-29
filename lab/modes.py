"""Agent modes (FINAL_ARCHITECTURE.md §2). All three must remain available.

  PURE_FIXED_FLY          canonical fixed connectome brain, no helper
  BIOLOGICAL_PLASTICITY_FLY  frozen Gate-4 v2 experimental plasticity
                            system — DO NOT CONTINUE TUNING IT
  HYBRID_FLY              practical game-playing configuration (canonical
                          brain + CV + world model + episodic memory +
                          engineered value + planner + ability resolver)

Every mode is recorded in replay events and log lines. ENGINEERED helper
components are only active in HYBRID_FLY.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AgentMode(Enum):
    PURE_FIXED_FLY = "PURE_FIXED_FLY"
    BIOLOGICAL_PLASTICITY_FLY = "BIOLOGICAL_PLASTICITY_FLY"
    HYBRID_FLY = "HYBRID_FLY"


@dataclass
class ModeConfig:
    mode: AgentMode
    # Biological plasticity (Mode B only): frozen v2 configuration,
    # immutable by policy — no further tuning parameters are exposed.
    plasticity_frozen_config: str | None = None   # e.g. "gate4_v2_final"

    def to_dict(self) -> dict:
        return {
            "mode": self.mode.value,
            "plasticity_frozen_config": self.plasticity_frozen_config,
            "engineered_helper": self.mode is AgentMode.HYBRID_FLY,
        }


DEFAULT_MODE = ModeConfig(mode=AgentMode.HYBRID_FLY)


def validate_mode_config(cfg: ModeConfig) -> None:
    """Fail closed on mode-misuse."""
    if cfg.mode is AgentMode.BIOLOGICAL_PLASTICITY_FLY:
        if not cfg.plasticity_frozen_config:
            raise ValueError(
                "Mode B requires the frozen Gate-4 v2 config reference; "
                "biological plasticity is FROZEN — no tuning variants")
    if cfg.mode is AgentMode.PURE_FIXED_FLY and cfg.plasticity_frozen_config:
        raise ValueError("Mode A is the FIXED connectome; plasticity ref "
                         "must be None")
