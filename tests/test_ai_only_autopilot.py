from lab.action.ai_only_supervisor import AIOnlyAutopilotSupervisor
from lab.action.gpo_controls import control_map
from lab.action.motor_executor import (
    AIOnlyBackend, MotorExecutor, SafeNoopBackend,
)
from lab.app import DigitalFlyLab
from lab.bus import Bus
from lab.coach.ai_only import AIOnlyOllamaCoachWorker
from lab.coach.ollama_cloud_probe import _candidate_models, _normalize_model


def _armed_bus(target, plan, notes=None):
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
        "notes": notes or {},
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
    assert cmd.payload["camera_align"] is True
    assert cmd.payload["source"] == "ai_only_autopilot"
    assert cmd.payload["AI_ONLY"] is True


def test_red_objective_marker_alone_never_becomes_melee_click():
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
            "target": "Corrupt Marine objective",
            "confidence": 0.96,
            "explanation": "fight the quest enemy",
        },
        notes={"quest_enemy_marker_detected": True},
    )
    sup = AIOnlyAutopilotSupervisor(bus)
    sup.step()
    cmd = bus.state("action.command").read()
    assert cmd is not None
    assert cmd.payload["name"] == "STEER_TARGET"
    assert sup.stats["combat_commands"] == 0


def test_confirmed_close_quest_enemy_actor_is_attacked():
    actor = {
        "track_id": 7,
        "bbox": [0.46, 0.34, 0.55, 0.63],
        "direction": 0.05,
        "distance": 0.66,
        "confidence": 0.90,
        "association_to_objective": 0.06,
    }
    bus = _armed_bus(
        {
            "type": "quest_enemy_actor",
            "direction": actor["direction"],
            "distance": actor["distance"],
            "confidence": actor["confidence"],
        },
        {
            "plan_id": 3,
            "skill": "FIGHT_QUEST_TARGET",
            "target": "Corrupt Marine",
            "confidence": 0.96,
            "explanation": "fight the tracked quest NPC",
        },
        notes={
            "quest_enemy_marker_detected": True,
            "quest_enemy_actor_visible": True,
            "quest_enemy_actor": actor,
        },
    )
    sup = AIOnlyAutopilotSupervisor(bus)
    sup.step()
    cmd = bus.state("action.command").read()
    assert cmd is not None
    assert cmd.payload["name"] == "ATTACK_LIGHT"
    assert cmd.payload["target_type"] == "quest_enemy_actor"
    assert sup.stats["combat_commands"] == 1


def test_far_quest_enemy_actor_is_approached_before_attack():
    actor = {
        "track_id": 8,
        "bbox": [0.25, 0.28, 0.30, 0.40],
        "direction": -0.65,
        "distance": 0.31,
        "confidence": 0.88,
        "association_to_objective": 0.10,
    }
    bus = _armed_bus(
        {
            "type": "quest_enemy_actor",
            "direction": actor["direction"],
            "distance": actor["distance"],
            "confidence": actor["confidence"],
        },
        {
            "plan_id": 4,
            "skill": "FIGHT_QUEST_TARGET",
            "target": "quest enemy",
            "confidence": 0.95,
            "explanation": "close distance to quest enemy",
        },
        notes={
            "quest_enemy_marker_detected": True,
            "quest_enemy_actor_visible": True,
            "quest_enemy_actor": actor,
        },
    )
    sup = AIOnlyAutopilotSupervisor(bus)
    sup.step()
    cmd = bus.state("action.command").read()
    assert cmd is not None
    assert cmd.payload["name"] == "STEER_TARGET"
    assert cmd.payload["direction"] == -0.65





def test_close_stable_red_marker_can_fallback_to_m1_when_body_tracker_misses():
    bus = _armed_bus(
        {
            "type": "quest_enemy_marker",
            "direction": 0.05,
            "distance": 0.86,
            "confidence": 0.94,
        },
        {
            "plan_id": 31,
            "skill": "FIGHT_QUEST_TARGET",
            "target": "red-marked quest NPC",
            "confidence": 0.94,
            "explanation": "fight the close quest objective",
        },
        notes={"quest_enemy_marker_detected": True},
    )
    sup = AIOnlyAutopilotSupervisor(bus)
    sup._enemy_marker_since_ns = sup.clock.now_ns() - int(1.0e9)
    sup.step()
    cmd = bus.state("action.command").read()
    assert cmd is not None
    assert cmd.payload["name"] == "ATTACK_LIGHT"
    assert cmd.payload["target_type"] == "quest_enemy_marker_close_fallback"


def test_navigation_servo_does_not_reissue_same_command_every_worker_tick():
    bus = _armed_bus(
        {
            "type": "recommended_quest_waypoint",
            "direction": 0.25,
            "distance": 0.35,
            "confidence": 0.92,
        },
        {
            "plan_id": 32,
            "skill": "NAVIGATE_OBJECTIVE",
            "target": "green waypoint",
            "confidence": 0.94,
            "explanation": "follow waypoint",
        },
    )
    sup = AIOnlyAutopilotSupervisor(bus)
    sup.step()
    first = bus.state("action.command").read()
    assert first is not None
    first_id = first.payload["command_id"]
    sup.step()
    second = bus.state("action.command").read()
    assert second.payload["command_id"] == first_id


def test_ai_only_take_quest_interacts_at_moderate_close_range_once():
    plan = {
        "plan_id": 5,
        "skill": "TAKE_QUEST",
        "target": "Robert",
        "confidence": 0.95,
        "explanation": "accept the quest",
    }
    bus = _armed_bus(
        {
            "type": "quest_marker",
            "direction": 0.10,
            "distance": 0.56,
            "confidence": 0.82,
        },
        plan,
        notes={
            "quest_marker_detected": True,
            "quest_marker_direction": 0.10,
            "quest_marker_proximity": 0.56,
        },
    )
    sup = AIOnlyAutopilotSupervisor(bus)
    sup.step()
    first = bus.state("action.command").read()
    assert first is not None
    assert first.payload["name"] == "INTERACT_QUEST"

    state = bus.state("quest.state").read().payload
    assert state["quest_status"] == "pending_accept"
    assert state["awaiting_quest_confirmation"] is True

    # Same plan cannot spam T while waiting for quest confirmation.
    first_id = first.payload["command_id"]
    sup.step()
    second = bus.state("action.command").read()
    assert second.payload["command_id"] == first_id
    assert sup.stats["quest_interact_suppressed"] >= 1


def test_active_quest_blocks_retake_even_if_yellow_marker_is_visible():
    bus = _armed_bus(
        {
            "type": "quest_enemy_marker",
            "direction": 0.2,
            "distance": 0.4,
            "confidence": 0.94,
        },
        {
            "plan_id": 6,
            "skill": "TAKE_QUEST",
            "target": "quest giver",
            "confidence": 0.95,
            "explanation": "take a quest",
        },
        notes={
            "quest_enemy_marker_detected": True,
            "quest_marker_detected": True,
            "quest_marker_direction": 0.0,
            "quest_marker_proximity": 0.9,
        },
    )
    sup = AIOnlyAutopilotSupervisor(bus)
    sup.step()
    assert bus.state("action.command").read() is None
    state = bus.state("quest.state").read().payload
    assert state["quest_status"] == "active"
    assert state["quest_active"] is True


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


def test_ai_only_prompt_contains_state_contract_and_no_safezone_excuse():
    bus = Bus()
    coach = AIOnlyOllamaCoachWorker(
        bus,
        model="gemma4:cloud",
        api_key="test-only",
    )
    prompt = coach._prompt(
        {
            "target": {
                "type": "quest_enemy_marker",
                "direction": 0.2,
                "distance": 0.4,
                "confidence": 0.92,
            },
            "player": {"health": 1.0, "stamina": 1.0},
            "notes": {"quest_enemy_marker_detected": True},
        },
        {
            "phase": "active",
            "quest_status": "active",
            "quest_enemy_actor_visible": False,
            "stuck": False,
            "circling": False,
        },
        {},
        {"controls": []},
        {"autonomy": True, "gpo_loadout": "default_melee"},
        {},
    )
    assert "SOLE GAMEPLAY CONTROLLER" in prompt
    assert "NO fruit-fly controller" in prompt
    assert '"quest_status": "active"' in prompt
    assert "SAFEZONE / PROTECTED" in prompt
    assert "reason to avoid" in prompt
    assert "quest_enemy_actor_visible" in prompt
    assert coach.stats["decision_owner"] == "cloud_ai"
    assert coach.stats["fruit_fly_control"] is False


class _FakeCameraInput:
    def __init__(self):
        self.events = []

    def mouse_move(self, dx, dy):
        self.events.append(("mouse_move", int(dx), int(dy)))

    def camera_drag(self, dx, dy):
        self.events.append(("camera_drag", int(dx), int(dy)))

    def key_down(self, key):
        self.events.append(("key_down", key))

    def key_up(self, key):
        self.events.append(("key_up", key))

    def mouse_button_down(self, button):
        self.events.append(("mouse_down", button))

    def mouse_button_up(self, button):
        self.events.append(("mouse_up", button))

    def ui_click(self, x, y, button="left", restore_cursor=True):
        self.events.append(("ui_click", float(x), float(y), button))

    def release_all(self):
        self.events.append(("release",))

    def backend_health(self):
        return {"ok": True}


def test_ai_only_explicit_look_plan_becomes_bounded_camera_command():
    bus = _armed_bus(
        {"type": "none", "direction": None, "distance": None,
         "confidence": 0.0},
        {
            "plan_id": 7,
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
    assert cmd.payload["dx"] == 105


def test_ai_only_backend_uses_roblox_camera_drag():
    fake = _FakeCameraInput()
    backend = AIOnlyBackend(fake)
    backend.mouse_move(-90, 0)
    assert ("camera_drag", -90, 0) in fake.events
    health = backend.backend_health()
    assert health["camera_drag"] is True
    assert health["mouse_move_enabled"] is True


def test_climb_macro_uses_jump_forward_and_ctrl():
    bus = Bus()
    fake = _FakeCameraInput()
    backend = AIOnlyBackend(fake)
    ex = MotorExecutor(
        bus,
        backend=backend,
        autonomy_enabled=True,
        questing=True,
        command_only=True,
    )
    now = ex.clock.now_ns()
    bus.state("action.command").write({
        "command_id": 1,
        "name": "CLIMB",
        "source": "test",
        "hold_s": 1.1,
        "ts_ns": now,
        "expires_ns": now + int(2e9),
    })
    ex.step()
    assert ("key_down", "SPACE") in fake.events
    assert ("key_down", "W") in fake.events
    assert ("key_down", "CTRL") in fake.events


def test_verified_default_melee_abilities_are_executable_controls():
    controls = control_map()
    assert "melee_gut_punch" in controls
    assert controls["melee_gut_punch"].bindings == ("key:E",)
    assert "melee_ground_smash" in controls
    assert controls["melee_ground_smash"].bindings == ("key:R",)


def test_cloud_alias_is_not_stripped_from_model_id():
    assert _normalize_model("gemma4:cloud") == "gemma4:cloud"
    assert _normalize_model("gemma4:31b-cloud") == "gemma4:31b-cloud"


def test_ai_only_lab_excludes_fly_workers_but_enables_role_detection():
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
    assert lab.fast_vision.role_detection is True
    assert lab.brain not in lab.workers
    assert lab.encoder not in lab.workers
    assert lab.planner not in lab.workers
    assert lab.quest_supervisor is None
    assert lab.ai_only_supervisor in lab.workers
    assert lab.semantic_coach in lab.workers
    assert lab.semantic_coach.stats["decision_owner"] == "cloud_ai"



def test_ai_only_signature_ignores_screen_jitter_but_replans_on_events():
    bus = Bus()
    coach = AIOnlyOllamaCoachWorker(
        bus, model="gemma4:cloud", api_key="test-only")
    q = {
        "quest_status": "active",
        "quest_enemy_actor_visible": False,
        "quest_enemy_objective_visible": True,
        "quest_giver_visible": False,
        "recommended_waypoint_visible": False,
        "awaiting_quest_confirmation": False,
        "stuck": False,
        "circling": False,
        "action_epoch": 0,
        "ui_epoch": 0,
    }
    a = {
        "target": {
            "type": "quest_enemy_marker",
            "direction": -1.2,
            "distance": 0.20,
        },
        "player": {"health": 1.0},
        "notes": {"quest_enemy_marker_detected": True},
    }
    b = {
        "target": {
            "type": "quest_enemy_marker",
            "direction": 0.9,
            "distance": 0.71,
        },
        "player": {"health": 1.0},
        "notes": {"quest_enemy_marker_detected": True},
    }
    assert coach._signature(a, q, {}) == coach._signature(b, q, {})
    q2 = dict(q)
    q2["stuck"] = True
    assert coach._signature(a, q, {}) != coach._signature(a, q2, {})


def test_ai_visible_interact_prompt_can_trigger_t_despite_noisy_geometry():
    bus = _armed_bus(
        {
            "type": "quest_marker",
            "direction": 0.95,
            "distance": 0.30,
            "confidence": 0.82,
        },
        {
            "plan_id": 80,
            "skill": "TAKE_QUEST",
            "target": "Robert",
            "control_id": "interact",
            "confidence": 0.96,
            "explanation": "T prompt is visible",
            "perception": {
                "quest_state": "available",
                "interaction_prompt_visible": True,
            },
        },
        notes={
            "quest_marker_detected": True,
            "quest_marker_direction": 0.95,
            "quest_marker_proximity": 0.30,
        },
    )
    sup = AIOnlyAutopilotSupervisor(bus)
    sup.step()
    cmd = bus.state("action.command").read()
    assert cmd is not None
    assert cmd.payload["name"] == "INTERACT_QUEST"
    state = bus.state("quest.state").read().payload
    assert state["action_epoch"] == 1


def test_ai_visual_enemy_confirmation_relaxes_marker_melee_gate():
    bus = _armed_bus(
        {
            "type": "quest_enemy_marker",
            "direction": 0.20,
            "distance": 0.66,
            "confidence": 0.94,
        },
        {
            "plan_id": 81,
            "skill": "FIGHT_QUEST_TARGET",
            "target": "Corrupt Marine",
            "confidence": 0.95,
            "explanation": "NPC is visibly in front",
            "perception": {
                "quest_state": "active",
                "enemy_actor_visible": True,
            },
        },
        notes={"quest_enemy_marker_detected": True},
    )
    sup = AIOnlyAutopilotSupervisor(bus)
    sup._enemy_marker_since_ns = sup.clock.now_ns() - int(1.0e9)
    sup.step()
    cmd = bus.state("action.command").read()
    assert cmd is not None
    assert cmd.payload["name"] in {"ATTACK_LIGHT", "ATTACK_ADVANCE"}
    assert sup.stats["marker_melee_fallbacks"] == 1


def test_jump_is_one_shot_per_ai_plan_not_spammed():
    bus = _armed_bus(
        {"type": "quest_enemy_marker", "direction": 0.0,
         "distance": 0.3, "confidence": 0.9},
        {
            "plan_id": 82,
            "skill": "JUMP",
            "target": "ledge",
            "confidence": 0.9,
            "explanation": "clear ledge",
        },
    )
    sup = AIOnlyAutopilotSupervisor(bus)
    sup.step()
    first = bus.state("action.command").read()
    assert first is not None
    first_id = first.payload["command_id"]
    sup.step()
    second = bus.state("action.command").read()
    assert second.payload["command_id"] == first_id


def test_fast_model_candidates_precede_configured_model():
    models = _candidate_models("gemma4:31b", prefer_fast=True)
    assert models[0] == "qwen3-vl:4b"
    assert "gemma4:31b" in models



def test_attack_advance_physically_combines_w_and_m1():
    bus = Bus()
    fake = _FakeCameraInput()
    backend = AIOnlyBackend(fake)
    ex = MotorExecutor(
        bus,
        backend=backend,
        autonomy_enabled=True,
        questing=True,
        command_only=True,
    )
    now = ex.clock.now_ns()
    bus.state("action.command").write({
        "command_id": 99,
        "name": "ATTACK_ADVANCE",
        "source": "test",
        "ts_ns": now,
        "expires_ns": now + int(2e9),
    })
    ex.step()
    assert ("key_down", "W") in fake.events
    assert ("mouse_down", "left") in fake.events
    assert ("mouse_up", "left") in fake.events
