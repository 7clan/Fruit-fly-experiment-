from lab.action.motor_executor import MotorExecutor, SafeNoopBackend
from lab.bus import Bus
from lab.schemas import Intention


def _action(name):
    ex = MotorExecutor(
        Bus(), backend=SafeNoopBackend(), movement_only=True,
        autonomy_enabled=False)
    return ex._materialize({}, Intention(name=name))


def test_turn_left_uses_forward_plus_left_key_only():
    a = _action("TURN_LEFT")
    assert a.bindings == ["key:W", "key:A"]
    assert a.mouse_dx == 0 and a.mouse_dy == 0
    assert a.hold_s == 2.5


def test_turn_right_uses_forward_plus_right_key_only():
    a = _action("TURN_RIGHT")
    assert a.bindings == ["key:W", "key:D"]
    assert a.mouse_dx == 0 and a.mouse_dy == 0
    assert a.hold_s == 2.5


def test_approach_is_forward_only_and_combatish_intents_fail_closed():
    assert _action("APPROACH").bindings == ["key:W"]
    assert _action("RETREAT").bindings == []
    assert _action("ESCAPE").bindings == []
    assert _action("STOP").bindings == []


def _obs(target_type="recommended_quest_waypoint", direction=0.0,
         confidence=0.92, ts_ns=1):
    return {
        "ts_ns": ts_ns,
        "target": {
            "type": target_type,
            "direction": direction,
            "distance": 0.5,
            "confidence": confidence,
        },
        "player": {},
        "enemies": [],
        "ui": {},
        "abilities": [],
    }


def test_navigation_safety_veto_blocks_turn_opposite_fresh_target():
    bus = Bus()
    ex = MotorExecutor(
        bus, backend=SafeNoopBackend(), movement_only=True,
        autonomy_enabled=False)
    now = ex.clock.now_ns()
    bus.state("world.observation").write(
        _obs(direction=0.8, ts_ns=now), ts_ns=now)
    a = ex._materialize({}, Intention(name="TURN_LEFT"))
    assert ex._navigation_veto_reason(
        a, now) == "turn_left_conflicts_with_target_right"


def test_navigation_safety_veto_allows_matching_turn():
    bus = Bus()
    ex = MotorExecutor(
        bus, backend=SafeNoopBackend(), movement_only=True,
        autonomy_enabled=False)
    now = ex.clock.now_ns()
    bus.state("world.observation").write(
        _obs(direction=-0.8, ts_ns=now), ts_ns=now)
    a = ex._materialize({}, Intention(name="TURN_LEFT"))
    assert ex._navigation_veto_reason(a, now) is None


def test_navigation_safety_veto_fails_closed_when_target_missing():
    bus = Bus()
    ex = MotorExecutor(
        bus, backend=SafeNoopBackend(), movement_only=True,
        autonomy_enabled=False)
    now = ex.clock.now_ns()
    bus.state("world.observation").write(
        _obs(target_type="none", direction=None, confidence=0.0, ts_ns=now),
        ts_ns=now)
    a = ex._materialize({}, Intention(name="APPROACH"))
    assert ex._navigation_veto_reason(
        a, now) == "navigation_target_not_visible"


def test_approach_hold_adapts_to_slow_canonical_chunk():
    ex = MotorExecutor(
        Bus(), backend=SafeNoopBackend(), movement_only=True,
        autonomy_enabled=False)
    a = ex._materialize(
        {"chunk_wall_s": 6.8}, Intention(name="APPROACH"))
    assert a.hold_s == 5.0


class _RecordingBackend(SafeNoopBackend):
    name = "recording"

    def __init__(self):
        super().__init__()
        self.down = []
        self.up = []

    def key_down(self, code):
        self.down.append(code)

    def key_up(self, code):
        self.up.append(code)

    def release_all(self):
        for code in list(self.down):
            if code not in self.up:
                self.up.append(code)


def test_new_neural_action_releases_obsolete_steering_key():
    bus = Bus()
    backend = _RecordingBackend()
    ex = MotorExecutor(
        bus, backend=backend, movement_only=True,
        autonomy_enabled=True)
    now = ex.clock.now_ns()
    # First neural action holds W+D.
    ex._execute(
        ex._materialize({}, Intention(name="TURN_RIGHT")),
        now, new_brain_decision=True)
    assert "D" in ex.held_keys()
    # Next action becomes W+A before the old timeout. D must be released
    # immediately rather than overlapping with A.
    ex._execute(
        ex._materialize({}, Intention(name="TURN_LEFT")),
        now + 100_000_000, new_brain_decision=True)
    assert "D" not in ex.held_keys()
    assert "A" in ex.held_keys()
    assert "D" in backend.up
