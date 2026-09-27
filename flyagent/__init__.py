"""FlyAgent - Phase 1 apparatus for the hybrid biological game-playing agent.

Hierarchy (see docs/ROADMAP.md):
    BIOLOGICAL BEHAVIOR -> ACTION DECODER -> GAME CONTROL -> GAME PERCEPTION
    -> TASK SYSTEM -> LONG-TERM OBJECTIVE

Phase 1 implements everything up to GAME CONTROL against a 2D artificial
environment. Nothing in Phase 1 touches the real game.
"""
from .version import __version__, PHASE

__all__ = ["__version__", "PHASE"]
