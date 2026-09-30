"""MotorExecutor + input backends (PROCESS 6, TARGET 20–60 Hz).

FINAL_ARCHITECTURE §19/§20/§25:
  * The executor owns key-down/key-up, mouse movement, action duration,
    cooldown timers, action lock, held-key state. The brain need NOT
    recompute while merely holding W for 300 ms — repeated APPROACH
    intentions extend the hold, they don't re-trigger.
  * Every emitted input is timestamped and logged (replay).
  * URGENT REFLEX PATH: threat → fly tick → DEFEND/EVADE → resolver →
    input never waits on planner/OCR/memory/dashboard/LLM.
  * CV→input SHORTCUT IS FORBIDDEN (spec §25): the executor ONLY accepts
    (Intention, ResolvedAbility) from the brain path, never raw vision
    directives. Exceptions (human emergency stop, technical failsafe,
    app recovery) are separate, explicitly-logged mechanisms.
  * **AUTONOMY GATE**: no input is emitted unless `autonomy_enabled`
    is True, which the application sets ONLY when the corresponding
    experimental gate (6+ for movement, 7+ for combat) has PASSED and
    the user armed it. Gate-5 (passive) runs with autonomy DISABLED.

Backends:
  * SafeNoopBackend — emits nothing; used for Gate-5 passive runs, CI,
    and any state where autonomy is disabled. Still fully logs what
    WOULD have been emitted (shadow mode) — this is the honest way to
    bring up the pipeline before any input is legal.
  * WindowsInputBackend — SendInput-based key/mouse injection
    (WINDOWS-ONLY, guarded import in lab/action/windows_input.py).
    Includes the global emergency-stop hotkey that releases all held
    keys immediately.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from ..bus import Bus, StateChannel, StreamChannel
from ..schemas import Intention
from ..worker import Worker
from .ability_resolver import ResolvedAbility


# ---------------------------------------------------------------------------
# backends
# ---------------------------------------------------------------------------

class InputBackend(ABC):
    """Platform input injection interface."""

    name = "abstract"

    @abstractmethod
    def key_down(self, code: str) -> None: ...

    @abstractmethod
    def key_up(self, code: str) -> None: ...

    @abstractmethod
    def mouse_move(self, dx: int, dy: int) -> None: ...

    @abstractmethod
    def release_all(self) -> None: ...

    def backend_health(self) -> dict:
        return {"backend": self.name, "ok": True}


class SafeNoopBackend(InputBackend):
    """Emits NOTHING. Logs shadow decisions. Default until gates pass."""
    name = "safe_noop"

    def __init__(self):
        self.emitted = 0
        self._lock = threading.Lock()

    def key_down(self, code: str) -> None:
        with self._lock:
            self.emitted += 1

    def key_up(self, code: str) -> None:
        with self._lock:
            self.emitted += 1

    def mouse_move(self, dx: int, dy: int) -> None:
        with self._lock:
            self.emitted += 1

    def release_all(self) -> None:
        pass


def create_windows_input_backend() -> InputBackend:
    """Factory for the Windows SendInput backend (guarded)."""
    import platform
    if platform.system() != "Windows":
        raise RuntimeError("Windows input backend requires Windows; "
                           "use SafeNoopBackend elsewhere")
    from .windows_input import WindowsInputBackend
    return WindowsInputBackend()


class MovementOnlyBackend(InputBackend):
    """Hard safety wrapper for the first Gate-6 active-input run.

    Only W/A/D movement keys are forwarded. Mouse input and every other key
    are impossible here, even if an upstream bug requests them.
    """
    name = "windows_navigation_keys_only"

    ALLOWED = {"W", "A", "D"}

    def __init__(self, inner: InputBackend):
        self.inner = inner

    def key_down(self, code: str) -> None:
        if code.upper() in self.ALLOWED:
            self.inner.key_down(code)

    def key_up(self, code: str) -> None:
        if code.upper() in self.ALLOWED:
            self.inner.key_up(code)

    def mouse_move(self, dx: int, dy: int) -> None:
        # Gate-6 navigation no longer depends on Roblox mouse-capture state.
        # Steering is W+A / W+D only.
        return

    def release_all(self) -> None:
        self.inner.release_all()

    def backend_health(self) -> dict:
        h = dict(self.inner.backend_health())
        h["backend"] = self.name
        h["allowed_keys"] = sorted(self.ALLOWED)
        return h


def create_windows_movement_only_backend() -> InputBackend:
    return MovementOnlyBackend(create_windows_input_backend())


# ---------------------------------------------------------------------------
# concrete action schema (input binding expansion)
# ---------------------------------------------------------------------------

@dataclass
class ConcreteAction:
    """A resolved, executor-ready action: binding-level, no further
    decisions needed. Produced from (Intention, ResolvedAbility)."""
    intention: str
    ability_id: str
    bindings: list = field(default_factory=list)   # ["key:W"] etc.
    hold_s: float = 0.15
    mouse_dx: int = 0
    mouse_dy: int = 0
    notes: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"intention": self.intention, "ability_id": self.ability_id,
                "bindings": list(self.bindings), "hold_s": self.hold_s,
                "mouse_dx": self.mouse_dx, "mouse_dy": self.mouse_dy,
                "notes": dict(self.notes)}


# ---------------------------------------------------------------------------
# executor
# ---------------------------------------------------------------------------

class MotorExecutor(Worker):
    """Consumes brain.output (intention) + resolver decisions; emits
    inputs through the backend. 20–60 Hz; holds keys between brain ticks
    so the brain doesn't recompute for a held movement."""

    name = "motor_executor"
    TOPIC_IN = "brain.output"
    TOPIC_ACTION = "action.selected"        # state channel (dashboard)
    TOPIC_INPUT = "action.inputs"           # stream (replay: every input)

    def __init__(self, bus: Bus, target_hz: float = 40.0,
                 backend: Optional[InputBackend] = None,
                 autonomy_enabled: bool = False,
                 movement_only: bool = False):
        super().__init__(bus, target_hz=target_hz)
        self.brain_out: StateChannel = bus.state(self.TOPIC_IN)
        self.action_state: StateChannel = bus.state(self.TOPIC_ACTION)
        self.inputs: StreamChannel = bus.stream(self.TOPIC_INPUT, maxsize=256)
        self.backend = backend or SafeNoopBackend()
        self.autonomy_enabled = bool(autonomy_enabled)   # GATED (see module doc)
        self.movement_only = bool(movement_only)
        self._held: dict[str, float] = {}   # code -> hold-until ns
        self._lock = threading.Lock()
        self.action_lock_until_ns = 0
        self._last_sig: tuple = ()          # (intention, ability, bindings)
        self._last_hold_refresh_ns = 0
        self._last_brain_out_ts: int = -1
        self.stats.update({"inputs_emitted": 0, "shadow_only": 0,
                           "emergency_stops": 0, "hold_refreshes": 0})

    # -- autonomy gate ----------------------------------------------------
    def set_autonomy(self, enabled: bool, reason: str = "") -> None:
        self.autonomy_enabled = bool(enabled)
        if not enabled:
            self.emergency_stop(reason="autonomy_disabled")

    def emergency_stop(self, reason: str = "manual") -> None:
        """Release ALL held keys immediately; log separately (spec §25
        exception path)."""
        with self._lock:
            self.backend.release_all()
            self._held.clear()
            self.action_lock_until_ns = 0
        self.stats["emergency_stops"] += 1
        self.inputs.publish({
            "kind": "EMERGENCY_STOP", "reason": reason,
            "ts_ns": self.clock.now_ns(), "exception_path": True,
        })

    # -- main loop ------------------------------------------------------------
    def step(self) -> None:
        now = self.clock.now_ns()
        self._expire_holds(now)
        snap = self.brain_out.read()
        if snap is None:
            return
        out = snap.payload
        intention = Intention.from_dict(out["intention"])
        action = self._materialize(out, intention)
        brain_out_ts = int(out.get("ts_ns", now))
        self.action_state.write(action.to_dict() | {
            "ts_ns": now, "brain_out_ts_ns": brain_out_ts,
            "autonomy": self.autonomy_enabled,
            "backend": self.backend.name,
            "backend_health": self.backend.backend_health()}, ts_ns=now)
        # one decision event per brain output (exact latency chain in replay)
        new_brain_decision = brain_out_ts != self._last_brain_out_ts
        if new_brain_decision:
            self._last_brain_out_ts = brain_out_ts
            self.inputs.publish({
                "kind": "decision", "ts_ns": now,
                "brain_out_ts_ns": brain_out_ts,
                "brain_to_action_ms": round((now - brain_out_ts) / 1e6, 3),
                "intention": intention.name,
                "ability_id": action.ability_id,
                "shadow": not self.autonomy_enabled,
            }, ts_ns=now)
        self._execute(action, now, new_brain_decision=new_brain_decision)

    # -- action materialization -------------------------------------------
    def _materialize(self, brain_out: dict,
                     intention: Intention) -> ConcreteAction:
        """(Intention, resolver-free path) → ConcreteAction.

        For Gate-5/6 the action is basic movement derived from the
        intention directly (binding map below). From Gate-7 stages the
        AbilityResolver output (brain_out['resolved']) is used when
        present; its ability id overrides the default binding.
        """
        resolved = brain_out.get("resolved")
        if resolved:
            ra = ResolvedAbility(**{k: resolved[k] for k in
                                    ("intention", "ability_id", "score",
                                     "candidates", "resolver_config",
                                     "blocked_reason")})
            if ra.ability_id:
                return ConcreteAction(
                    intention=ra.intention, ability_id=ra.ability_id,
                    bindings=[], hold_s=0.2,
                    notes={"resolver": ra.resolver_config,
                           "score": ra.score})
        # First active Gate-6 run is deliberately movement-only.
        # RETREAT/ESCAPE/combat-like intentions fail closed to STOP.
        if self.movement_only:
            # TURN means camera/heading correction, not strafing. APPROACH is
            # a modest forward pulse long enough to be visible despite the
            # slow canonical brain cadence on the target laptop.
            if intention.name == "TURN_LEFT":
                return ConcreteAction(
                    intention="TURN_LEFT", ability_id="",
                    bindings=["key:W", "key:A"], hold_s=2.5,
                    mouse_dx=0, mouse_dy=0,
                    notes={"source": "gate6_navigation_keys",
                           "brain_intention": intention.name,
                           "locomotor_interpretation":
                               "P9 turn -> forward plus ipsiversive steering"})
            if intention.name == "TURN_RIGHT":
                return ConcreteAction(
                    intention="TURN_RIGHT", ability_id="",
                    bindings=["key:W", "key:D"], hold_s=2.5,
                    mouse_dx=0, mouse_dy=0,
                    notes={"source": "gate6_navigation_keys",
                           "brain_intention": intention.name,
                           "locomotor_interpretation":
                               "P9 turn -> forward plus ipsiversive steering"})
            if intention.name == "APPROACH":
                return ConcreteAction(
                    intention="APPROACH", ability_id="",
                    bindings=["key:W"], hold_s=2.5,
                    notes={"source": "gate6_navigation_keys",
                           "brain_intention": intention.name})
            return ConcreteAction(
                intention="STOP", ability_id="", bindings=[], hold_s=0.0,
                notes={"source": "gate6_navigation_failsafe",
                       "brain_intention": intention.name})

        # Passive/shadow development path retains the broader map.
        binding_map = {
            "STOP": ([], 0.0),
            "TURN_LEFT": (["key:A"], 0.12),
            "TURN_RIGHT": (["key:D"], 0.12),
            "APPROACH": (["key:W"], 0.30),
            "RETREAT": (["key:S"], 0.20),
            "ESCAPE": (["key:SPACE"], 0.10),
        }
        bindings, hold_s = binding_map.get(intention.name, ([], 0.0))
        return ConcreteAction(intention=intention.name, ability_id="",
                              bindings=bindings, hold_s=hold_s,
                              notes={"source": "basic_movement_map"})

    # -- execution ------------------------------------------------------------
    def _execute(self, action: ConcreteAction, now_ns: int,
                 new_brain_decision: bool = True) -> None:
        shadow = not self.autonomy_enabled
        sig = (action.intention, action.ability_id, tuple(action.bindings),
               int(action.mouse_dx), int(action.mouse_dy))
        # State channels are polled much faster than the canonical brain.
        # Re-reading the same brain.output is NOT a repeated biological
        # decision and must not refresh a key forever.
        if not new_brain_decision:
            return
        with self._lock:
            same_action = sig == self._last_sig
            if same_action and action.bindings:
                # HOLD REFRESH: a repeated biological intention extends held
                # keys. If a key expired since the previous brain chunk, it
                # is physically pressed again and that real output is logged.
                repressed = []
                for b in action.bindings:
                    if b.startswith("key:"):
                        code = b[4:]
                        if code not in self._held and not shadow:
                            self.backend.key_down(code)
                            repressed.append(code)
                        self._held[code] = now_ns + int(action.hold_s * 1e9)
                self._last_sig = sig
                self._last_hold_refresh_ns = now_ns
                self.stats["hold_refreshes"] += 1
                if repressed:
                    self.inputs.publish({
                        "kind": "input_refresh",
                        "ts_ns": now_ns,
                        "shadow": False,
                        "intention": action.intention,
                        "bindings": [f"key:{k}" for k in repressed],
                        "hold_s": action.hold_s,
                    }, ts_ns=now_ns)
                    self.stats["inputs_emitted"] += 1
                # No mouse steering exists in Gate-6F. A held-key refresh is
                # complete here.
                return
            if not shadow and now_ns < self.action_lock_until_ns:
                return
            if not same_action:
                for b in action.bindings:
                    if b.startswith("key:"):
                        code = b[4:]
                        if not shadow:
                            self.backend.key_down(code)
                        self._held[code] = now_ns + int(action.hold_s * 1e9)
            if action.mouse_dx or action.mouse_dy:
                if not shadow:
                    self.backend.mouse_move(action.mouse_dx, action.mouse_dy)
            if not shadow and (action.bindings
                               or action.mouse_dx or action.mouse_dy):
                # simple action lock while a movement command is active
                self.action_lock_until_ns = now_ns + int(
                    max(action.hold_s, 0.12) * 1e9 * 0.5)
            self._last_sig = sig
            self._last_hold_refresh_ns = now_ns
        ev = {
            "kind": "input", "ts_ns": now_ns, "shadow": shadow,
            "action": action.to_dict(),
            "intention": action.intention, "ability_id": action.ability_id,
            "bindings": action.bindings, "hold_s": action.hold_s,
        }
        self.inputs.publish(ev, ts_ns=now_ns)
        if shadow:
            self.stats["shadow_only"] += 1
        elif action.bindings or action.mouse_dx or action.mouse_dy:
            self.stats["inputs_emitted"] += 1
        self._pending_release(ev, action)

    def _pending_release(self, ev: dict, action: ConcreteAction) -> None:
        # records intended releases for replay completeness; actual key_up
        # happens in _expire_holds (kept accurate for held-key state)
        ev["release_at_ns"] = ev["ts_ns"] + int(action.hold_s * 1e9)

    def _expire_holds(self, now_ns: int) -> None:
        with self._lock:
            expired = [c for c, until in self._held.items() if now_ns >= until]
            for code in expired:
                if not self.autonomy_enabled:
                    pass  # shadow mode never pressed it down
                else:
                    self.backend.key_up(code)
                del self._held[code]
                self.inputs.publish({
                    "kind": "key_up", "ts_ns": now_ns, "code": code,
                    "shadow": not self.autonomy_enabled},
                    ts_ns=now_ns)

    def held_keys(self) -> list:
        with self._lock:
            return sorted(self._held.keys())
