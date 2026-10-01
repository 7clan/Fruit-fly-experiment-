from lab.action.ability_registry import AbilityRegistry
from lab.action.ability_resolver import AbilityResolver
from lab.action.gpo_controls import (
    CORE_CONTROLS, binding_supported, control_map,
    validate_observed_ability_binding,
)
from lab.action.motor_executor import InputBackend, MotorExecutor
from lab.bus import Bus
from lab.schemas import Intention


def test_core_gpo_control_catalog_contains_all_generic_families():
    ids = set(control_map())
    required = {
        "move_forward", "move_backward", "move_left", "move_right",
        "jump", "sprint", "dash_forward", "dash_backward",
        "dash_left", "dash_right", "climb_or_dive", "geppo",
        "basic_attack", "m1_string", "uptilt_air_combo",
        "block", "perfect_block", "reload",
        "interact", "carry_downed", "grip_downed", "sit_ship",
        "menu", "buso_haki", "observation_haki", "evasive_contextual",
    }
    assert required <= ids


def test_every_concrete_catalog_binding_is_backend_supported():
    for control in CORE_CONTROLS:
        for binding in control.bindings:
            assert binding_supported(binding), (control.control_id, binding)


def test_observed_ability_bindings_are_normalized():
    assert validate_observed_ability_binding("key:e") == "key:E"
    assert validate_observed_ability_binding("key:N") == "key:N"
    assert validate_observed_ability_binding("key:7") == "key:7"
    assert validate_observed_ability_binding("mouse:LEFT") == "mouse:left"


def test_registry_accepts_current_loadout_move_without_hardcoded_name():
    reg = AbilityRegistry()
    rec = reg.register_observed_binding(
        "hud_slot_1", "Observed move", "key:e", ["ATTACK_HEAVY"],
        range="close", confidence=0.8)
    assert rec.input_binding == "key:E"
    assert reg.get("hud_slot_1").name == "Observed move"


def test_resolver_carries_observed_binding_to_executor():
    reg = AbilityRegistry()
    reg.register_observed_binding(
        "hud_m1_like", "Observed attack", "mouse:left", ["ATTACK_LIGHT"],
        range="close", confidence=1.0)
    resolver = AbilityResolver(reg)
    resolved = resolver.resolve(
        Intention(name="ATTACK_LIGHT"), distance=0.9, now_ns=1)
    assert resolved.ability_id == "hud_m1_like"
    assert resolved.input_binding == "mouse:left"

    ex = MotorExecutor(Bus(), autonomy_enabled=False)
    action = ex._materialize({"resolved": resolved.to_dict()},
                             Intention(name="ATTACK_LIGHT"))
    assert action.bindings == ["mouse:left"]


class _FakeBackend(InputBackend):
    name = "fake_full_gpo"

    def __init__(self):
        self.events = []

    def key_down(self, code): self.events.append(("key_down", code))
    def key_up(self, code): self.events.append(("key_up", code))
    def mouse_move(self, dx, dy): self.events.append(("mouse_move", dx, dy))
    def mouse_button_down(self, button):
        self.events.append(("mouse_down", button))
    def mouse_button_up(self, button):
        self.events.append(("mouse_up", button))
    def release_all(self): self.events.append(("release_all",))


def test_executor_can_emit_and_release_mouse_bound_move():
    backend = _FakeBackend()
    ex = MotorExecutor(Bus(), backend=backend, autonomy_enabled=True)
    action = ex._materialize(
        {"resolved": {
            "intention": "ATTACK_LIGHT",
            "ability_id": "m1",
            "score": 1.0,
            "input_binding": "mouse:left",
            "candidates": [],
            "resolver_config": "test",
            "blocked_reason": None,
        }},
        Intention(name="ATTACK_LIGHT"),
    )
    ex._execute(action, 1_000_000_000, new_brain_decision=True)
    assert ("mouse_down", "left") in backend.events
    ex._expire_holds(2_000_000_000)
    assert ("mouse_up", "left") in backend.events
