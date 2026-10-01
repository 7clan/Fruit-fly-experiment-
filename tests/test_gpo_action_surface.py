from lab.action.gpo_controls import (
    CORE_CONTROLS, OBSERVED_ABILITY_KEYS, binding_supported,
)
from lab.action.motor_executor import QuestingBackend


class _FakeBackend:
    name = "fake"

    def __init__(self):
        self.events = []

    def key_down(self, code):
        self.events.append(("down", code.upper()))

    def key_up(self, code):
        self.events.append(("up", code.upper()))

    def mouse_move(self, dx, dy):
        self.events.append(("move", int(dx), int(dy)))

    def mouse_button_down(self, button):
        self.events.append(("mdown", str(button).lower()))

    def mouse_button_up(self, button):
        self.events.append(("mup", str(button).lower()))

    def camera_drag(self, dx, dy=0):
        self.events.append(("camera", int(dx), int(dy)))

    def release_all(self):
        self.events.append(("release",))

    def backend_health(self):
        return {"ok": True}


def test_control_catalog_contains_core_gpo_primitives():
    ids = {c.control_id for c in CORE_CONTROLS}
    required = {
        "move_forward", "move_backward", "move_left", "move_right",
        "jump", "sprint", "dash_forward", "dash_backward",
        "dash_left", "dash_right", "climb_or_dive", "geppo",
        "basic_attack", "m1_string", "uptilt_air_combo",
        "block", "perfect_block", "interact", "camera_look",
        "buso_haki", "observation_haki",
    }
    assert required <= ids


def test_dynamic_skill_binding_surface_is_general_not_loadout_hardcoded():
    for key in OBSERVED_ABILITY_KEYS:
        assert binding_supported(f"key:{key}")
    assert binding_supported("mouse:left")


def test_quest_backend_physically_supports_known_movement_combat_skill_keys():
    fake = _FakeBackend()
    b = QuestingBackend(fake)
    for key in [
        "W", "A", "S", "D", "SPACE", "CTRL", "Q", "F", "T",
        "E", "R", "Z", "X", "C", "V", "B", "N", "G", "J",
        "1", "2", "3", "4", "5", "6", "7", "8", "9", "0",
    ]:
        b.key_down(key)
    downs = {e[1] for e in fake.events if e[0] == "down"}
    assert {"W", "A", "S", "D", "SPACE", "Q", "F", "T", "E", "R"} <= downs
    assert {"Z", "X", "C", "N", "G", "J", "1", "9", "0"} <= downs


def test_quest_backend_never_moves_or_drags_user_camera():
    fake = _FakeBackend()
    b = QuestingBackend(fake)
    b.mouse_move(999, -999)
    assert not any(e[0] in {"move", "camera"} for e in fake.events)
