from lab.action.ai_only_supervisor import AIOnlyAutopilotSupervisor
from lab.action.motor_executor import (
    AIOnlyBackend, MotorExecutor, SafeNoopBackend,
)
from lab.bus import Bus
from lab.app import DigitalFlyLab
from lab.coach.ai_only import AIOnlyOllamaCoachWorker


def _armed_bus(target, plan):
    bus = Bus()
    bus.state("action.meta").write({
        "autonomy": True,
        "gpo_loadout": "default_melee",
    })
    bus.state("world.observation").write({
        "target": target,
        "player": {
            "health": 1.0,
            "health_units": "fraction",
            "stamina": 1.0,
        },
        "notes": {},
    })
    bus.state("coach.plan").write(plan)
    return bus


def test_ai_only_navigation_plan_is_realized_without_brain():
    bus = _armed_bus(
        {
            "type": "recommended_quest_waypoint",
            "direction": 0.72,
            "distance": 0.30,
            "confidence": 0.92,
        },
        {
            "plan_id": 1,
            "skill": "NAVIGATE_OBJECTIVE",
            "target": "green quest waypoint",
            "confidence": 0.95,
            "explanation": "follow the objective",
        },
    )
    sup = AIOnlyAutopilotSupervisor(bus)
    sup.step()
    cmd = bus.state("action.command").read()
    assert cmd is not None
    assert cmd.payload["name"] == "STEER_TARGET"
    assert cmd.payload["direction"] == 0.72
    assert cmd.payload["source"] == "ai_only_autopilot"
    assert cmd.payload["AI_ONLY"] is True


def test_ai_only_fight_plan_attacks_only_confirmed_close_quest_enemy():
    bus = _armed_bus(
        {
            "type": "quest_enemy_marker",
            "direction": 0.0,
            "distance": 0.82,
            "confidence": 0.94,
        },
        {
            "plan_id": 2,
            "skill": "FIGHT_QUEST_TARGET",
            "target": "Corrupt Marine",
            "confidence": 0.96,
            "explanation": "fight the quest enemy",
        },
    )
    sup = AIOnlyAutopilotSupervisor(bus)
    sup.step()
    cmd = bus.state("action.command").read()
    assert cmd is not None
    assert cmd.payload["name"] == "ATTACK_LIGHT"
    assert sup.stats["combat_commands"] == 1


def test_ai_only_fight_plan_approaches_confirmed_far_quest_enemy():
    bus = _armed_bus(
        {
            "type": "quest_enemy_marker",
            "direction": -0.65,
            "distance": 0.35,
            "confidence": 0.94,
        },
        {
            "plan_id": 3,
            "skill": "FIGHT_QUEST_TARGET",
            "target": "quest enemy",
            "confidence": 0.95,
            "explanation": "close distance to quest enemy",
        },
    )
    sup = AIOnlyAutopilotSupervisor(bus)
    sup.step()
    cmd = bus.state("action.command").read()
    assert cmd is not None
    assert cmd.payload["name"] == "STEER_TARGET"
    assert cmd.payload["direction"] == -0.65


def test_ai_only_take_quest_uses_yellow_marker_then_t():
    bus = _armed_bus(
        {
            "type": "recommended_quest_waypoint",
            "direction": 0.0,
            "distance": 0.50,
            "confidence": 0.92,
        },
        {
            "plan_id": 4,
            "skill": "TAKE_QUEST",
            "target": "quest giver",
            "confidence": 0.95,
            "explanation": "accept the quest",
        },
    )
    bus.state("world.observation").write({
        "target": {
            "type": "recommended_quest_waypoint",
            "direction": 0.0,
            "distance": 0.50,
            "confidence": 0.92,
        },
        "player": {
            "health": 1.0,
            "health_units": "fraction",
            "stamina": 1.0,
        },
        "notes": {
            "quest_marker_detected": True,
            "quest_marker_direction": 0.05,
            "quest_marker_proximity": 0.82,
        },
    })
    sup = AIOnlyAutopilotSupervisor(bus)
    sup.step()
    cmd = bus.state("action.command").read()
    assert cmd is not None
    assert cmd.payload["name"] == "INTERACT_QUEST"


def test_command_only_executor_does_not_consume_brain_output():
    bus = Bus()
    ex = MotorExecutor(
        bus,
        backend=SafeNoopBackend(),
        autonomy_enabled=False,
        questing=True,
        command_only=True,
    )
    bus.state("brain.output").write({
        "ts_ns": ex.clock.now_ns(),
        "intention": {"name": "APPROACH", "confidence": 1.0},
    })
    ex.step()
    assert ex.command_only is True
    assert ex._last_brain_out_ts == -1
    assert bus.state("action.selected").read() is None


def test_ai_only_prompt_declares_cloud_ai_as_sole_controller():
    bus = Bus()
    coach = AIOnlyOllamaCoachWorker(
        bus,
        model="gemma4:31b",
        api_key="test-only",
    )
    prompt = coach._prompt(
        {
            "target": {
                "type": "recommended_quest_waypoint",
                "direction": 0.2,
                "distance": 0.4,
                "confidence": 0.92,
            },
            "player": {"health": 1.0, "stamina": 1.0},
            "notes": {},
        },
        {"phase": "travel"},
        {},
        {"controls": []},
        {"autonomy": True, "gpo_loadout": "default_melee"},
        {},
    )
    assert "SOLE GAMEPLAY CONTROLLER" in prompt
    assert "NO fruit-fly controller" in prompt
    assert "NAVIGATE_OBJECTIVE" in prompt
    assert coach.stats["decision_owner"] == "cloud_ai"
    assert coach.stats["fruit_fly_control"] is False



class _FakeCameraInput:
    def __init__(self):
        self.events = []

    def mouse_move(self, dx, dy):
        self.events.append(("mouse_move", int(dx), int(dy)))

    def key_down(self, key):
        self.events.append(("key_down", key))

    def key_up(self, key):
        self.events.append(("key_up", key))

    def mouse_button_down(self, button):
        self.events.append(("mouse_down", button))

    def mouse_button_up(self, button):
        self.events.append(("mouse_up", button))

    def release_all(self):
        self.events.append(("release",))

    def backend_health(self):
        return {"ok": True}


def test_ai_only_explicit_look_plan_becomes_bounded_camera_command():
    bus = _armed_bus(
        {"type": "none", "direction": None, "distance": None,
         "confidence": 0.0},
        {
            "plan_id": 5,
            "skill": "LOOK_RIGHT",
            "target": "search right",
            "confidence": 0.91,
            "explanation": "objective is off-screen to the right",
        },
    )
    sup = AIOnlyAutopilotSupervisor(bus)
    sup.step()
    cmd = bus.state("action.command").read()
    assert cmd is not None
    assert cmd.payload["name"] == "SEARCH_CAMERA"
    assert cmd.payload["dx"] == 90


def test_ai_only_backend_forwards_relative_camera_motion():
    fake = _FakeCameraInput()
    backend = AIOnlyBackend(fake)
    backend.mouse_move(-90, 0)
    assert ("mouse_move", -90, 0) in fake.events
    health = backend.backend_health()
    assert health["camera_drag"] is True
    assert health["mouse_move_enabled"] is True



def test_ai_only_lab_excludes_fly_encoder_and_brain_workers():
    lab = DigitalFlyLab(
        capture_kind="synthetic",
        runtime_kind="mock",
        autonomy=False,
        ai_only=True,
        semantic_coach=True,
        coach_provider="ollama_cloud",
        dashboard=False,
    )
    assert lab.ai_only is True
    assert lab.executor.command_only is True
    assert lab.brain not in lab.workers
    assert lab.encoder not in lab.workers
    assert lab.quest_supervisor is None
    assert lab.ai_only_supervisor in lab.workers
    assert lab.semantic_coach in lab.workers
    assert lab.semantic_coach.stats["decision_owner"] == "cloud_ai"
