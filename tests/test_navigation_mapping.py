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
