"""AI-only training/trajectory infrastructure.

This package is deliberately separate from the fruit-fly scientific stack.
It records gameplay trajectories, conservative rewards and human-teacher
actions for later imitation learning / supervised tuning.
"""

from .reward import OnlineReward
from .trajectory import TrajectoryRecorder
from .teacher_recorder import TeacherInputRecorder

__all__ = ["OnlineReward", "TrajectoryRecorder", "TeacherInputRecorder"]
