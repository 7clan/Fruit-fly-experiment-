"""Low-rate semantic game coach for DigitalFlyLab.

The coach is ENGINEERED and explicitly separate from the biological fly brain.
It interprets game semantics and proposes high-level skills; it never emits
raw keyboard scan codes.
"""

from .semantic_coach import SemanticCoachWorker
from .local_smolvlm import LocalSmolVLMCoachWorker
from .local_smollm import LocalSmolLMCoachWorker

__all__ = ["SemanticCoachWorker", "LocalSmolVLMCoachWorker", "LocalSmolLMCoachWorker"]
