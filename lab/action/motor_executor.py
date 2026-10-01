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

    def mouse_button_down(self, button: str) -> None:
        raise NotImplementedError

    def mouse_button_up(self, button: str) -> None:
        raise NotImplementedError

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

    def mouse_button_down(self, button: str) -> None:
        with self._lock:
            self.emitted += 1

    def mouse_button_up(self, button: str) -> None:
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

    def mouse_button_down(self, button: str) -> None:
        return

    def mouse_button_up(self, button: str) -> None:
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


class QuestingBackend(InputBackend):
    """Hard allowlist for autonomous quest/navigation + starter PvE."""

    name = "windows_questing_v1"
    ALLOWED_KEYS = {
        "W", "A", "S", "D", "SPACE", "CTRL", "Q",
        "F", "T", "E", "R",
    }
    ALLOWED_MOUSE = {"left"}

    def __init__(self, inner: InputBackend):
        self.inner = inner

    def key_down(self, code: str) -> None:
        if code.upper() in self.ALLOWED_KEYS:
            self.inner.key_down(code)

    def key_up(self, code: str) -> None:
        if code.upper() in self.ALLOWED_KEYS:
            self.inner.key_up(code)

    def mouse_move(self, dx: int, dy: int) -> None:
        dx = max(-90, min(90, int(dx)))
        dy = max(-35, min(35, int(dy)))
        if dx or dy:
            drag = getattr(self.inner, "camera_drag", None)
            if drag is not None:
                drag(dx, dy)

    def mouse_button_down(self, button: str) -> None:
        if str(button).lower() in self.ALLOWED_MOUSE:
            self.inner.mouse_button_down(button)

    def mouse_button_up(self, button: str) -> None:
        if str(button).lower() in self.ALLOWED_MOUSE:
            self.inner.mouse_button_up(button)

    def release_all(self) -> None:
        self.inner.release_all()

    def backend_health(self) -> dict:
        h = dict(self.inner.backend_health())
        h["backend"] = self.name
        h["allowed_keys"] = sorted(self.ALLOWED_KEYS)
        h["allowed_mouse"] = sorted(self.ALLOWED_MOUSE)
        h["camera_drag"] = True
        return h


def create_windows_questing_backend() -> InputBackend:
    return QuestingBackend(create_windows_input_backend())


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
        # Navigation-only fail-closed observation. It may STOP a stale or
        # contradictory neural movement, but it is never allowed to choose a
        # key or substitute a different direction.
        self.world_obs: StateChannel = bus.state("world.observation")
        self.action_state: StateChannel = bus.state(self.TOPIC_ACTION)
        self.inputs: StreamChannel = bus.stream(self.TOPIC_INPUT, maxsize=256)
        self.backend = backend or SafeNoopBackend()
        self.autonomy_enabled = bool(autonomy_enabled)   # GATED (see module doc)
        self.movement_only = bool(movement_only)
        self._held: dict[str, float] = {}   # keyboard code -> hold-until ns
        self._held_mouse: dict[str, float] = {}  # mouse button -> hold-until ns
        self._lock = threading.Lock()
        self.action_lock_until_ns = 0
        self._last_sig: tuple = ()          # (intention, ability, bindings)
        self._last_hold_refresh_ns = 0
        self._last_brain_out_ts: int = -1
        self.stats.update({"inputs_emitted": 0, "shadow_only": 0,
                           "emergency_stops": 0, "hold_refreshes": 0,
                           "navigation_vetoes": 0})

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
            self._held_mouse.clear()
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
        self._navigation_safety_for_held(now)
        snap = self.brain_out.read()
        if snap is None:
            return
        out = snap.payload
        intention = Intention.from_dict(out["intention"])
        action = self._materialize(out, intention)
        brain_out_ts = int(out.get("ts_ns", now))
        # one decision event per brain output (exact latency chain in replay)
        new_brain_decision = brain_out_ts != self._last_brain_out_ts
        if new_brain_decision:
            veto_reason = self._navigation_veto_reason(action, now)
            if veto_reason is not None:
                # Technical failsafe only: preserve the neural decision in
                # the decision event, but materialize STOP rather than
                # allowing stale vision to steer the opposite way. Vision
                # never selects an alternate movement.
                action = ConcreteAction(
                    intention="STOP", ability_id="", bindings=[], hold_s=0.0,
                    notes={"source": "navigation_safety_veto",
                           "brain_intention": intention.name,
                           "reason": veto_reason})
                self.stats["navigation_vetoes"] += 1
            self.action_state.write(action.to_dict() | {
                "ts_ns": now, "brain_out_ts_ns": brain_out_ts,
                "autonomy": self.autonomy_enabled,
                "backend": self.backend.name,
                "backend_health": self.backend.backend_health()}, ts_ns=now)
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
            fields = {
                "intention": resolved["intention"],
                "ability_id": resolved["ability_id"],
                "score": resolved["score"],
                "input_binding": resolved.get("input_binding", ""),
                "candidates": resolved.get("candidates", []),
                "resolver_config": resolved.get("resolver_config", ""),
                "blocked_reason": resolved.get("blocked_reason"),
            }
            ra = ResolvedAbility(**fields)
            if ra.ability_id:
                bindings = [ra.input_binding] if ra.input_binding else []
                return ConcreteAction(
                    intention=ra.intention, ability_id=ra.ability_id,
                    bindings=bindings, hold_s=0.12,
                    notes={"resolver": ra.resolver_config,
                           "score": ra.score,
                           "observed_binding": ra.input_binding})
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
                # The canonical brain is slower than wall-clock real time on
                # the target laptop. Let a neural APPROACH persist most of one
                # measured brain interval, but cap it and keep the live visual
                # safety veto active between chunks.
                try:
                    wall_s = float(brain_out.get("chunk_wall_s", 0.0))
                except (TypeError, ValueError):
                    wall_s = 0.0
                hold_s = max(2.5, min(5.0, wall_s * 0.75))
                return ConcreteAction(
                    intention="APPROACH", ability_id="",
                    bindings=["key:W"], hold_s=hold_s,
                    notes={"source": "gate6_navigation_keys",
                           "brain_intention": intention.name,
                           "adaptive_hold_from_chunk_wall_s": wall_s})
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
            "EVADE_LEFT": (["key:A", "key:Q"], 0.12),
            "EVADE_RIGHT": (["key:D", "key:Q"], 0.12),
            "DEFEND": (["key:F"], 0.25),
            "ATTACK_LIGHT": (["mouse:left"], 0.07),
            # Heavy/ranged/special attacks are loadout-specific. They are
            # executed only when AbilityResolver supplies an observed HUD
            # binding; guessing a fixed key here would be unsafe and stale.
            "ATTACK_HEAVY": ([], 0.0),
            "ATTACK_RANGED": ([], 0.0),
            "SPECIAL": ([], 0.0),
            "ESCAPE": (["key:SPACE"], 0.10),
        }
        bindings, hold_s = binding_map.get(intention.name, ([], 0.0))
        return ConcreteAction(intention=intention.name, ability_id="",
                              bindings=bindings, hold_s=hold_s,
                              notes={"source": "basic_movement_map"})

    # -- navigation safety -----------------------------------------------------
    def _navigation_veto_reason(self, action: ConcreteAction,
                                now_ns: int) -> str | None:
        """Return a STOP-only safety reason for movement-only navigation.

        Current vision is allowed to veto a stale/contradictory neural
        movement, never to select a key. This is important because a 50 ms
        canonical biological chunk currently takes several wall-seconds on
        the target laptop, so the visual waypoint may move significantly
        before the next neural decision arrives.
        """
        if not self.movement_only or not action.bindings:
            return None

        snap = self.world_obs.read()
        if snap is None:
            return "no_world_observation"
        obs = snap.payload or {}
        obs_ts = int(obs.get("ts_ns", snap.ts_ns))
        if now_ns - obs_ts > int(1.5e9):
            return "stale_world_observation"

        target = obs.get("target") or {}
        target_type = str(target.get("type") or "none")
        try:
            confidence = float(target.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        direction = target.get("direction")
        try:
            direction = float(direction)
        except (TypeError, ValueError):
            direction = None

        if (target_type not in {"recommended_quest_waypoint", "quest_marker"}
                or confidence < 0.60 or direction is None):
            return "navigation_target_not_visible"

        # Positive bearing = target right, negative = target left.
        deadband = 0.15  # ~8.6 degrees; ignore tiny detector jitter.
        if action.intention == "TURN_LEFT" and direction > deadband:
            return "turn_left_conflicts_with_target_right"
        if action.intention == "TURN_RIGHT" and direction < -deadband:
            return "turn_right_conflicts_with_target_left"
        if action.intention == "APPROACH" and abs(direction) > 0.60:
            return "approach_target_too_far_off_axis"
        return None

    def _navigation_safety_for_held(self, now_ns: int) -> None:
        """Release a held neural movement if fresh vision contradicts it.

        This never presses a key or chooses another direction. After a veto,
        movement stays released until a NEW brain decision arrives.
        """
        if (not self.movement_only or not self.autonomy_enabled
                or not self._held or not self._last_sig):
            return
        intention = str(self._last_sig[0])
        if intention not in {"TURN_LEFT", "TURN_RIGHT", "APPROACH"}:
            return
        action = ConcreteAction(
            intention=intention, ability_id="",
            bindings=[f"key:{k}" for k in sorted(self._held)])
        reason = self._navigation_veto_reason(action, now_ns)
        if reason is None:
            return
        with self._lock:
            self.backend.release_all()
            self._held.clear()
            self.action_lock_until_ns = 0
        self.stats["navigation_vetoes"] += 1
        self.inputs.publish({
            "kind": "navigation_veto",
            "ts_ns": now_ns,
            "reason": reason,
            "brain_intention": intention,
            "exception_path": True,
        }, ts_ns=now_ns)

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
                            repressed.append(f"key:{code}")
                        self._held[code] = now_ns + int(action.hold_s * 1e9)
                    elif b.startswith("mouse:"):
                        button = b[6:].lower()
                        if button not in self._held_mouse and not shadow:
                            self.backend.mouse_button_down(button)
                            repressed.append(f"mouse:{button}")
                        self._held_mouse[button] = (
                            now_ns + int(action.hold_s * 1e9))
                self._last_sig = sig
                self._last_hold_refresh_ns = now_ns
                self.stats["hold_refreshes"] += 1
                if repressed:
                    self.inputs.publish({
                        "kind": "input_refresh",
                        "ts_ns": now_ns,
                        "shadow": False,
                        "intention": action.intention,
                        "bindings": repressed,
                        "hold_s": action.hold_s,
                    }, ts_ns=now_ns)
                    self.stats["inputs_emitted"] += 1
                # No mouse steering exists in Gate-6F. A held-key refresh is
                # complete here.
                return
            if (not shadow and now_ns < self.action_lock_until_ns
                    and not self.movement_only):
                return
            if not same_action:
                desired = {
                    b[4:] for b in action.bindings if b.startswith("key:")
                }
                desired_mouse = {
                    b[6:].lower() for b in action.bindings
                    if b.startswith("mouse:")
                }
                # A new neural decision replaces the previous input state.
                for code in list(self._held):
                    if code not in desired:
                        if not shadow:
                            self.backend.key_up(code)
                        del self._held[code]
                        self.inputs.publish({
                            "kind": "key_up", "ts_ns": now_ns,
                            "code": code, "shadow": shadow,
                            "reason": "new_neural_action",
                        }, ts_ns=now_ns)
                for button in list(self._held_mouse):
                    if button not in desired_mouse:
                        if not shadow:
                            self.backend.mouse_button_up(button)
                        del self._held_mouse[button]
                        self.inputs.publish({
                            "kind": "mouse_up", "ts_ns": now_ns,
                            "button": button, "shadow": shadow,
                            "reason": "new_neural_action",
                        }, ts_ns=now_ns)
                for code in desired:
                    if code not in self._held and not shadow:
                        self.backend.key_down(code)
                    self._held[code] = now_ns + int(action.hold_s * 1e9)
                for button in desired_mouse:
                    if button not in self._held_mouse and not shadow:
                        self.backend.mouse_button_down(button)
                    self._held_mouse[button] = (
                        now_ns + int(action.hold_s * 1e9))
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
                if self.autonomy_enabled:
                    self.backend.key_up(code)
                del self._held[code]
                self.inputs.publish({
                    "kind": "key_up", "ts_ns": now_ns, "code": code,
                    "shadow": not self.autonomy_enabled},
                    ts_ns=now_ns)

            expired_mouse = [
                b for b, until in self._held_mouse.items()
                if now_ns >= until
            ]
            for button in expired_mouse:
                if self.autonomy_enabled:
                    self.backend.mouse_button_up(button)
                del self._held_mouse[button]
                self.inputs.publish({
                    "kind": "mouse_up", "ts_ns": now_ns,
                    "button": button,
                    "shadow": not self.autonomy_enabled},
                    ts_ns=now_ns)

    def held_keys(self) -> list:
        with self._lock:
            return sorted(self._held.keys())

    def held_inputs(self) -> dict:
        with self._lock:
            return {
                "keys": sorted(self._held.keys()),
                "mouse_buttons": sorted(self._held_mouse.keys()),
            }
