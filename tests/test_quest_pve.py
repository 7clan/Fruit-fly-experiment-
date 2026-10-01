import numpy as np

from lab.action.motor_executor import (
    MotorExecutor, QuestingBackend, SafeNoopBackend,
)
from lab.action.quest_combat_supervisor import QuestCombatSupervisor
from lab.bus import Bus
from lab.perception.fast_vision import GPOHeuristicFastVision
from lab.schemas import Intention
from lab.world.value import ValueTable


class _FakeInput:
    name = "fake"

    def __init__(self):
        self.events = []

    def key_down(self, key):
        self.events.append(("key_down", key.upper()))

    def key_up(self, key):
        self.events.append(("key_up", key.upper()))

    def mouse_button_down(self, button):
        self.events.append(("mouse_down", button))

    def mouse_button_up(self, button):
        self.events.append(("mouse_up", button))

    def camera_drag(self, dx, dy=0):
        self.events.append(("camera", int(dx), int(dy)))

    def release_all(self):
        self.events.append(("release",))

    def backend_health(self):
        return {"ok": True}


def test_questing_backend_is_hard_allowlisted_to_known_gpo_action_surface():
    fake = _FakeInput()
    b = QuestingBackend(fake)
    requested = [
        "W", "A", "S", "D", "SPACE", "CTRL", "SHIFT", "Q",
        "F", "T", "E", "R", "Z", "X", "C", "V", "B", "N",
        "G", "J", "P", "M", "1", "5", "0",
        # unsupported scan-code aliases must still be dropped
        "DELETE", "HOME",
    ]
    for key in requested:
        b.key_down(key)
    b.mouse_button_down("left")
    b.mouse_button_down("right")
    b.mouse_move(120, 99)

    allowed = {e[1] for e in fake.events if e[0] == "key_down"}
    assert {"W", "A", "S", "D", "SPACE", "CTRL", "Q", "F", "T",
            "E", "R", "Z", "X", "C", "N", "G", "J", "1", "5", "0"} <= allowed
    assert "DELETE" not in allowed and "HOME" not in allowed
    assert ("mouse_down", "left") in fake.events
    assert ("mouse_down", "right") not in fake.events
    # Quest mode never takes over the user's mouse/camera.
    assert not any(e[0] == "camera" for e in fake.events)


def test_fly_questing_turn_uses_forward_steer_without_camera_drag():
    ex = MotorExecutor(
        Bus(), backend=SafeNoopBackend(), autonomy_enabled=False,
        questing=True)
    left = ex._materialize({}, Intention(name="TURN_LEFT"))
    right = ex._materialize({}, Intention(name="TURN_RIGHT"))
    assert left.bindings == ["key:W", "key:A"]
    assert left.mouse_dx == 0
    assert right.bindings == ["key:W", "key:D"]
    assert right.mouse_dx == 0


def test_yellow_quest_marker_emits_interact_semantic_command():
    bus = Bus()
    sup = QuestCombatSupervisor(bus, ValueTable())
    bus.state("action.meta").write({"autonomy": True})
    bus.state("world.observation").write({
        "target": {
            "type": "quest_marker", "distance": 0.85,
            "direction": 0.05, "confidence": 0.9,
        },
        "player": {
            "health": 1.0, "health_units": "fraction",
            "stamina": 1.0,
        },
        "notes": {
            "quest_marker_detected": True,
            "quest_marker_proximity": 0.85,
            "quest_marker_direction": 0.05,
        },
    })
    sup.step()
    cmd = bus.state("action.command").read().payload
    assert cmd["name"] == "INTERACT_QUEST"
    assert cmd["ENGINEERED"] is True


def test_damage_triggers_learnable_defense_not_agent_shutdown():
    bus = Bus()
    values = ValueTable()
    sup = QuestCombatSupervisor(bus, values)
    bus.state("action.meta").write({"autonomy": True})

    def obs(health):
        bus.state("world.observation").write({
            "target": {
                "type": "quest_enemy_marker", "distance": 0.45,
                "direction": 0.0, "confidence": 0.94,
            },
            "player": {
                "health": health, "health_units": "fraction",
                "stamina": 1.0,
            },
        })

    obs(1.0)
    sup.step()
    # Clear command cooldown without sleeping; this is a deterministic unit
    # test of the event rule, not wall-clock pacing.
    sup._last_command_ns = 0
    obs(0.94)
    sup.step()
    cmd = bus.state("action.command").read().payload
    assert cmd["name"] in {"BLOCK", "EVADE_BACK"}
    assert sup._pending_defense is not None


def test_stalled_fly_navigation_requests_jump_then_climb():
    sup = QuestCombatSupervisor(Bus(), ValueTable())
    t0 = 10_000_000_000
    assert sup._update_progress(
        "recommended_quest_waypoint", 0.2, t0, "APPROACH") is None
    # No progress for >3.5 s -> jump.
    assert sup._update_progress(
        "recommended_quest_waypoint", 0.2,
        t0 + 4_000_000_000, "APPROACH") == "JUMP"
    # Repeated stalls eventually escalate to climb.
    sup._last_command_ns = 0
    assert sup._update_progress(
        "recommended_quest_waypoint", 0.2,
        t0 + 8_000_000_000, "APPROACH") == "JUMP"
    sup._last_command_ns = 0
    assert sup._update_progress(
        "recommended_quest_waypoint", 0.2,
        t0 + 12_000_000_000, "APPROACH") == "CLIMB"


def test_red_circle_with_red_distance_support_becomes_quest_enemy_marker():
    import cv2

    img = np.zeros((360, 640, 3), dtype=np.uint8)
    # Bright avatar cue near the camera anchor.
    cv2.rectangle(img, (300, 170), (340, 240), (245, 245, 245), -1)
    # User-observed post-quest red objective dot + text support below.
    cv2.circle(img, (500, 155), 9, (0, 0, 255), -1)
    cv2.putText(
        img, "20m", (478, 185), cv2.FONT_HERSHEY_SIMPLEX,
        0.45, (0, 0, 255), 1, cv2.LINE_AA)

    det = GPOHeuristicFastVision(role_detection=False).detect(img)
    assert det["target"]["type"] == "quest_enemy_marker"
    assert det["_notes"]["quest_enemy_marker_detected"] is True
    assert len(det["enemies"]) == 1


def test_engineered_defense_values_persist(tmp_path):
    p = tmp_path / "values.json"
    first = ValueTable()
    first.observe(("gpo", "defense", "quest_enemy"), "block", True)
    first.observe(("gpo", "defense", "quest_enemy"), "block", False)
    first.save_file(p)

    second = ValueTable()
    assert second.load_file(p) is True
    snap = second.snapshot()["gpo|defense|quest_enemy"]["block"]
    assert snap["uses"] == 2
    assert snap["success"] == 1


def test_yellow_quest_cue_survives_green_primary_target():
    bus = Bus()
    sup = QuestCombatSupervisor(bus, ValueTable())
    bus.state("action.meta").write({"autonomy": True})
    bus.state("world.observation").write({
        "target": {
            "type": "recommended_quest_waypoint", "distance": 0.80,
            "direction": 0.02, "confidence": 0.92,
        },
        "player": {
            "health": 1.0, "health_units": "fraction", "stamina": 1.0,
        },
        "notes": {
            "quest_marker_detected": True,
            "quest_marker_proximity": 0.82,
            "quest_marker_direction": 0.03,
        },
    })
    sup.step()
    cmd = bus.state("action.command").read().payload
    assert cmd["name"] == "INTERACT_QUEST"


def test_quest_enemy_marker_is_valid_fly_navigation_target():
    bus = Bus()
    ex = MotorExecutor(
        bus, backend=SafeNoopBackend(), autonomy_enabled=False,
        questing=True)
    now = ex.clock.now_ns()
    bus.state("world.observation").write({
        "ts_ns": now,
        "target": {
            "type": "quest_enemy_marker",
            "direction": 0.4,
            "distance": 0.4,
            "confidence": 0.94,
        },
    }, ts_ns=now)
    action = ex._materialize({}, Intention(name="TURN_RIGHT"))
    assert ex._navigation_veto_reason(action, now) is None


def test_red_route_arrow_does_not_become_enemy_marker():
    import cv2

    img = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.rectangle(img, (300, 170), (340, 240), (245, 245, 245), -1)
    # Thin route arrow similar to the recommended-quest path indicator.
    pts = np.array([[400, 170], [485, 170], [470, 160],
                    [500, 175], [470, 190], [485, 178],
                    [400, 178]], dtype=np.int32)
    cv2.fillPoly(img, [pts], (0, 0, 255))

    det = GPOHeuristicFastVision(role_detection=False).detect(img)
    assert det["_notes"]["quest_enemy_marker_detected"] is False
    assert det["target"]["type"] != "quest_enemy_marker"


def test_current_gpo_control_catalog_covers_core_and_default_melee():
    from lab.action.gpo_controls import CORE_CONTROLS, loadout_controls

    core = {x.control_id: x for x in CORE_CONTROLS}
    assert core["jump"].bindings == ("key:SPACE",)
    assert core["block"].bindings == ("key:F",)
    assert core["interact"].bindings == ("key:T",)
    assert core["dash_forward"].bindings == ("key:W", "key:Q")
    assert core["climb_or_dive"].bindings == ("key:CTRL",)
    assert core["basic_attack"].bindings == ("mouse:left",)

    melee = {x.control_id: x for x in loadout_controls("default_melee")}
    assert melee["melee_gut_punch"].bindings == ("key:E",)
    assert melee["melee_ground_smash"].bindings == ("key:R",)


def test_camera_assist_does_not_recenter_while_fly_is_turning():
    bus = Bus()
    sup = QuestCombatSupervisor(bus, ValueTable())
    bus.state("action.meta").write({"autonomy": True})
    bus.state("brain.output").write({
        "intention": {"name": "TURN_RIGHT", "confidence": 1.0}
    })
    bus.state("world.observation").write({
        "target": {
            "type": "recommended_quest_waypoint",
            "distance": 0.3, "direction": 1.1, "confidence": 0.92,
        },
        "player": {
            "health": 1.0, "health_units": "fraction", "stamina": 1.0,
        },
        "notes": {},
    })
    sup.step()
    env = bus.state("action.command").read()
    assert env is None or (env.payload or {}).get("name") != "SEARCH_CAMERA"
