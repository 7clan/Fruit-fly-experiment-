"""Action layer [ENGINEERED]: AbilityRegistry + AbilityResolver +
MotorExecutor + input backends. Autonomy is HARD-GATED until the
corresponding experimental gate passes."""
from .ability_registry import AbilityRecord, AbilityRegistry  # noqa: F401
from .ability_resolver import AbilityResolver, ResolvedAbility  # noqa: F401
from .motor_executor import (ConcreteAction, InputBackend,  # noqa: F401
                             MotorExecutor, SafeNoopBackend,
                             create_windows_input_backend)
