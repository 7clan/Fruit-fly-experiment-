import numpy as np

from lab.action.ai_only_supervisor import AIOnlyAutopilotSupervisor
from lab.action.gpo_controls import control_map
from lab.action.motor_executor import (
    AIOnlyBackend, MotorExecutor, SafeNoopBackend,
)
from lab.app import DigitalFlyLab
from lab.bus import Bus
from lab.coach.ai_only import (
    AIOnlyGeminiCoachWorker, AIOnlyOllamaCoachWorker,
)
from lab.coach.ollama_cloud_probe import _candidate_models, _normalize_model
from lab.perception.ai_visual_tracker import AIVisualTracker


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
    assert models[0] == "qwen3.5:4b"
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



def test_ai_visual_quest_giver_can_trigger_interact_without_yellow_cv():
    bus = _armed_bus(
        {"type": "none", "direction": None, "distance": None,
         "confidence": 0.0},
        {
            "plan_id": 120,
            "skill": "TAKE_QUEST",
            "target": "Robert",
            "control_id": "interact",
            "confidence": 0.97,
            "explanation": "T Interact is visibly on screen",
            "perception": {
                "quest_state": "available",
                "interaction_prompt_visible": True,
                "dialogue_visible": False,
                "enemy_actor_visible": False,
            },
            "visual_target": {
                "kind": "quest_giver",
                "x_norm": 0.51,
                "y_norm": 0.45,
                "confidence": 0.96,
                "melee_ready": False,
            },
        },
        notes={},
    )
    sup = AIOnlyAutopilotSupervisor(bus)
    sup.step()
    cmd = bus.state("action.command").read()
    assert cmd is not None
    assert cmd.payload["name"] == "INTERACT_QUEST"
    assert cmd.payload["target_type"] == "quest_giver_ai_prompt"


def test_ai_visual_enemy_can_attack_when_local_body_tracker_misses():
    bus = _armed_bus(
        {
            "type": "quest_enemy_marker",
            "direction": 0.15,
            "distance": 0.52,
            "confidence": 0.93,
        },
        {
            "plan_id": 121,
            "skill": "FIGHT_QUEST_TARGET",
            "target": "Corrupt Marine",
            "confidence": 0.96,
            "explanation": "Corrupt Marine body is visibly in melee range",
            "perception": {
                "quest_state": "active",
                "interaction_prompt_visible": False,
                "dialogue_visible": False,
                "enemy_actor_visible": True,
            },
            "visual_target": {
                "kind": "quest_enemy_actor",
                "x_norm": 0.52,
                "y_norm": 0.49,
                "confidence": 0.95,
                "melee_ready": True,
            },
        },
        notes={"quest_enemy_marker_detected": True},
    )
    sup = AIOnlyAutopilotSupervisor(bus)
    sup.step()
    cmd = bus.state("action.command").read()
    assert cmd is not None
    assert cmd.payload["name"] == "ATTACK_LIGHT"
    assert cmd.payload["target_type"] == "quest_enemy_actor_ai_visual"
    assert sup.stats["combat_commands"] == 1


def test_ai_visual_enemy_off_axis_steers_instead_of_clicking_empty_space():
    bus = _armed_bus(
        {
            "type": "quest_enemy_marker",
            "direction": 0.10,
            "distance": 0.50,
            "confidence": 0.93,
        },
        {
            "plan_id": 122,
            "skill": "FIGHT_QUEST_TARGET",
            "target": "Corrupt Marine",
            "confidence": 0.95,
            "explanation": "enemy is visible on the right",
            "perception": {
                "quest_state": "active",
                "enemy_actor_visible": True,
            },
            "visual_target": {
                "kind": "quest_enemy_actor",
                "x_norm": 0.91,
                "y_norm": 0.50,
                "confidence": 0.94,
                "melee_ready": False,
            },
        },
        notes={"quest_enemy_marker_detected": True},
    )
    sup = AIOnlyAutopilotSupervisor(bus)
    sup.step()
    cmd = bus.state("action.command").read()
    assert cmd is not None
    assert cmd.payload["name"] == "STEER_TARGET"
    assert cmd.payload["target_type"] == "quest_enemy_actor_ai_visual"


def test_ambiguous_edge_quest_dialogue_click_is_suppressed():
    bus = _armed_bus(
        {
            "type": "quest_marker",
            "direction": 0.0,
            "distance": 0.6,
            "confidence": 0.9,
        },
        {
            "plan_id": 123,
            "skill": "UI_CLICK",
            "target": "quest dialogue",
            "confidence": 0.95,
            "explanation": "click confirmation",
            "perception": {
                "quest_state": "pending_accept",
                "dialogue_visible": True,
            },
            "ui_click": {
                "needed": True,
                "x_norm": 0.05,
                "y_norm": 0.44,
                "label": "QUIT/Accept",
            },
        },
        notes={"quest_marker_detected": True},
    )
    sup = AIOnlyAutopilotSupervisor(bus)
    sup.step()
    assert bus.state("action.command").read() is None
    assert sup.stats["ui_clicks_suppressed"] == 1


def test_ai_only_plan_validation_keeps_visual_target_grounding():
    bus = Bus()
    coach = AIOnlyOllamaCoachWorker(
        bus, model="gemma4:cloud", api_key="test-only")
    plan = coach._validate_plan({
        "scene": "combat",
        "objective": "fight",
        "target": "Corrupt Marine",
        "skill": "FIGHT_QUEST_TARGET",
        "control_id": "",
        "observed_ability": {"binding": "", "label": ""},
        "visual_target": {
            "kind": "quest_enemy_actor",
            "x_norm": 0.62,
            "y_norm": 0.48,
            "confidence": 0.93,
            "melee_ready": True,
        },
        "ui_click": {
            "needed": False, "x_norm": 0, "y_norm": 0, "label": ""},
        "confidence": 0.95,
        "explanation": "visible quest NPC",
        "next_after_success": "continue",
        "perception": {
            "quest_state": "active",
            "enemy_actor_visible": True,
        },
        "knowledge_query": "",
        "memory_updates": [],
    }, {"controls": []})
    assert plan["visual_target"]["kind"] == "quest_enemy_actor"
    assert plan["visual_target"]["x_norm"] == 0.62
    assert plan["visual_target"]["melee_ready"] is True



def test_ai_visual_navigation_fallback_works_when_local_cv_has_no_target():
    bus = _armed_bus(
        {"type": "none", "direction": None, "distance": None,
         "confidence": 0.0},
        {
            "plan_id": 124,
            "skill": "NAVIGATE_OBJECTIVE",
            "target": "red quest objective",
            "confidence": 0.94,
            "explanation": "objective is visible on the right",
            "visual_target": {
                "kind": "quest_objective",
                "x_norm": 0.78,
                "y_norm": 0.42,
                "confidence": 0.93,
                "melee_ready": False,
            },
        },
        notes={},
    )
    sup = AIOnlyAutopilotSupervisor(bus)
    sup.step()
    cmd = bus.state("action.command").read()
    assert cmd is not None
    assert cmd.payload["name"] == "STEER_TARGET"
    assert cmd.payload["target_type"] == "ai_visual_quest_objective"


def test_stuck_navigation_waits_for_recovery_replan_instead_of_driving_wall():
    bus = _armed_bus(
        {
            "type": "quest_enemy_marker",
            "direction": 0.0,
            "distance": 0.25,
            "confidence": 0.9,
        },
        {
            "plan_id": 125,
            "skill": "NAVIGATE_OBJECTIVE",
            "target": "quest objective",
            "confidence": 0.95,
            "explanation": "go to marker",
        },
        notes={"quest_enemy_marker_detected": True},
    )
    sup = AIOnlyAutopilotSupervisor(bus)
    # Seed the same target as already being pursued without progress long
    # enough for _observe() to derive the real stuck state.
    sup._progress_target = "quest_enemy_marker"
    sup._best_proximity = 0.25
    sup._last_progress_ns = sup.clock.now_ns() - int(5.0e9)
    sup.step()
    assert sup._stuck is True
    assert bus.state("action.command").read() is None



def test_active_quest_latches_through_brief_marker_loss():
    bus = _armed_bus(
        {
            "type": "quest_enemy_marker",
            "direction": 0.2,
            "distance": 0.4,
            "confidence": 0.9,
        },
        {
            "plan_id": 126,
            "skill": "NAVIGATE_OBJECTIVE",
            "target": "quest objective",
            "confidence": 0.95,
            "explanation": "follow active quest",
            "perception": {"quest_state": "active"},
        },
        notes={"quest_enemy_marker_detected": True},
    )
    sup = AIOnlyAutopilotSupervisor(bus)
    sup.step()
    assert sup._quest_active_latched is True

    # Marker leaves the camera; active quest must not become "available"
    # just because the yellow giver happens to be visible again.
    bus.state("world.observation").write({
        "target": {
            "type": "quest_marker",
            "direction": 0.0,
            "distance": 0.7,
            "confidence": 0.9,
        },
        "player": {
            "health": 1.0,
            "health_units": "fraction",
            "stamina": 1.0,
        },
        "notes": {
            "quest_marker_detected": True,
            "quest_marker_direction": 0.0,
            "quest_marker_proximity": 0.7,
        },
    })
    sup.step()
    state = bus.state("quest.state").read().payload
    assert state["quest_status"] == "active"
    assert state["quest_active_latched"] is True


def test_ai_completed_state_clears_active_quest_latch():
    bus = _armed_bus(
        {"type": "none", "direction": None, "distance": None,
         "confidence": 0.0},
        {
            "plan_id": 127,
            "skill": "REOBSERVE",
            "target": "none",
            "confidence": 0.95,
            "explanation": "quest reward/completion is visible",
            "perception": {"quest_state": "completed"},
        },
        notes={},
    )
    sup = AIOnlyAutopilotSupervisor(bus)
    sup._quest_active_latched = True
    sup._quest_status = "active"
    sup.step()
    state = bus.state("quest.state").read().payload
    assert state["quest_status"] == "completed"
    assert state["quest_active_latched"] is False



def test_ai_only_plan_validation_keeps_equipment_perception():
    bus = Bus()
    coach = AIOnlyOllamaCoachWorker(
        bus, model="qwen3.5:4b", api_key="test-only")
    plan = coach._validate_plan({
        "scene": "combat prep",
        "objective": "equip melee",
        "target": "hotbar",
        "skill": "EQUIP_SLOT",
        "control_id": "equip_slot_1",
        "observed_ability": {"binding": "", "label": ""},
        "visual_target": {
            "kind": "ui", "x_norm": 0.15, "y_norm": 0.92,
            "confidence": 0.9, "melee_ready": False,
        },
        "ui_click": {
            "needed": False, "x_norm": 0, "y_norm": 0, "label": ""},
        "confidence": 0.95,
        "explanation": "slot 1 is visibly the melee tool",
        "next_after_success": "fight",
        "perception": {
            "quest_state": "active",
            "dialogue_visible": False,
            "interaction_prompt_visible": False,
            "enemy_actor_visible": True,
            "player_dead": False,
            "safezone_visible": True,
            "equipped_slot_visible": "1",
            "melee_ready_visible": True,
        },
        "knowledge_query": "",
        "memory_updates": [],
    }, {"controls": [{"control_id": "equip_slot_1"}]})
    assert plan["perception"]["equipped_slot_visible"] == "1"
    assert plan["perception"]["melee_ready_visible"] is True


def test_supervisor_accepts_visually_confirmed_equipped_slot():
    bus = _armed_bus(
        {
            "type": "quest_enemy_marker",
            "direction": 0.0,
            "distance": 0.3,
            "confidence": 0.9,
        },
        {
            "plan_id": 130,
            "skill": "WAIT",
            "target": "none",
            "confidence": 0.95,
            "explanation": "melee is visibly equipped",
            "perception": {
                "quest_state": "active",
                "equipped_slot_visible": "1",
                "melee_ready_visible": True,
            },
        },
        notes={"quest_enemy_marker_detected": True},
    )
    sup = AIOnlyAutopilotSupervisor(bus)
    sup.step()
    state = bus.state("quest.state").read().payload
    assert state["last_equipped_slot"] == "1"
    assert state["melee_ready_visible"] is True



def test_ai_only_gemini_controller_reuses_same_ai_only_contract():
    bus = Bus()
    coach = AIOnlyGeminiCoachWorker(
        bus, model="gemini-3.5-flash-lite", api_key="test-only")
    assert coach.provider == "gemini_ai_only"
    assert coach.stats["decision_owner"] == "cloud_ai"
    assert coach.stats["fruit_fly_control"] is False
    assert coach.max_output_tokens == 320
    prompt = coach._prompt(
        {"target": {"type": "none"}, "player": {}, "notes": {}},
        {"quest_status": "unknown"},
        {},
        {"controls": []},
        {"autonomy": True, "gpo_loadout": "default_melee"},
        {},
    )
    assert "SOLE GAMEPLAY CONTROLLER" in prompt
    assert "bbox_norm" in prompt


def test_visual_target_validation_keeps_trackable_bbox():
    bus = Bus()
    coach = AIOnlyOllamaCoachWorker(
        bus, model="gemma4:cloud", api_key="test-only")
    plan = coach._validate_plan({
        "scene": "combat",
        "objective": "fight",
        "target": "Corrupt Marine",
        "skill": "FIGHT_QUEST_TARGET",
        "control_id": "",
        "observed_ability": {"binding": "", "label": ""},
        "visual_target": {
            "kind": "quest_enemy_actor",
            "x_norm": 0.55,
            "y_norm": 0.48,
            "bbox_norm": [0.42, 0.22, 0.66, 0.74],
            "confidence": 0.95,
            "melee_ready": False,
        },
        "ui_click": {
            "needed": False, "x_norm": 0, "y_norm": 0, "label": ""},
        "confidence": 0.95,
        "explanation": "visible NPC",
        "next_after_success": "fight",
        "perception": {
            "quest_state": "active",
            "enemy_actor_visible": True,
        },
        "knowledge_query": "",
        "memory_updates": [],
    }, {"controls": []})
    assert plan["visual_target"]["bbox_norm"] == [0.42, 0.22, 0.66, 0.74]


def test_ai_visual_tracker_follows_ai_selected_patch_between_cloud_calls():
    bus = Bus()
    tracker = AIVisualTracker(bus, target_hz=8.0, max_width=320)

    img1 = np.zeros((200, 320, 3), dtype=np.uint8)
    # Non-uniform target patch: deterministic stripes/corners prevent
    # zero-variance template matching.
    img1[60:140, 90:145, :] = 45
    img1[65:100, 96:118, 1] = 230
    img1[104:134, 120:141, 2] = 210
    img1[82:91, 92:143, 0] = 180

    bus.state("coach.plan").write({
        "plan_id": 501,
        "visual_target": {
            "kind": "quest_enemy_actor",
            "x_norm": (90 + 145) / 2 / 320,
            "y_norm": (60 + 140) / 2 / 200,
            "bbox_norm": [90/320, 60/200, 145/320, 140/200],
            "confidence": 0.95,
        },
    })
    bus.state("capture.frames.latest").write({"data_ref": img1})
    tracker.step()
    first = bus.state("ai.visual.track").read()
    assert first is not None
    x1 = first.payload["x_norm"]

    img2 = np.zeros_like(img1)
    img2[60:140, 110:165, :] = 45
    img2[65:100, 116:138, 1] = 230
    img2[104:134, 140:161, 2] = 210
    img2[82:91, 112:163, 0] = 180
    bus.state("capture.frames.latest").write({"data_ref": img2})
    tracker.step()
    second = bus.state("ai.visual.track").read()
    assert second is not None
    assert second.payload["plan_id"] == 501
    assert second.payload["kind"] == "quest_enemy_actor"
    assert second.payload["x_norm"] > x1 + 0.025
    assert second.payload["confidence"] >= 0.42


def test_fight_macro_can_equip_ai_selected_slot_then_continue_same_plan():
    bus = _armed_bus(
        {
            "type": "quest_enemy_actor",
            "direction": 0.05,
            "distance": 0.70,
            "confidence": 0.92,
        },
        {
            "plan_id": 502,
            "skill": "FIGHT_QUEST_TARGET",
            "target": "Corrupt Marine",
            "control_id": "equip_slot_1",
            "confidence": 0.96,
            "explanation": "equip melee and fight",
            "perception": {
                "quest_state": "active",
                "enemy_actor_visible": True,
                "melee_ready_visible": False,
            },
        },
        notes={
            "quest_enemy_marker_detected": True,
            "quest_enemy_actor_visible": True,
            "quest_enemy_actor": {
                "direction": 0.05,
                "distance": 0.70,
                "confidence": 0.92,
            },
        },
    )
    sup = AIOnlyAutopilotSupervisor(bus)
    sup.step()
    first = bus.state("action.command").read()
    assert first is not None
    assert first.payload["name"] == "EQUIP_SLOT"
    assert first.payload["slot"] == "1"
    assert sup._last_fight_equip_plan_id == 502

    # Same cloud plan is still live; after the equipment precondition the
    # adapter is allowed to execute the AI's fight macro.
    sup._last_emit_ns = 0
    sup.step()
    second = bus.state("action.command").read()
    assert second.payload["name"] in {"ATTACK_LIGHT", "ATTACK_ADVANCE"}


def test_take_quest_macro_can_click_visible_dialog_without_second_ai_call():
    bus = _armed_bus(
        {
            "type": "quest_marker",
            "direction": 0.0,
            "distance": 0.65,
            "confidence": 0.9,
        },
        {
            "plan_id": 503,
            "skill": "TAKE_QUEST",
            "target": "quest giver dialogue",
            "confidence": 0.95,
            "explanation": "accept visible quest dialogue",
            "perception": {
                "quest_state": "available",
                "dialogue_visible": True,
                "interaction_prompt_visible": False,
            },
            "ui_click": {
                "needed": True,
                "x_norm": 0.62,
                "y_norm": 0.76,
                "label": "Accept",
            },
        },
        notes={
            "quest_marker_detected": True,
            "quest_marker_direction": 0.0,
            "quest_marker_proximity": 0.65,
        },
    )
    sup = AIOnlyAutopilotSupervisor(bus)
    sup.step()
    cmd = bus.state("action.command").read()
    assert cmd is not None
    assert cmd.payload["name"] == "UI_CLICK"
    assert cmd.payload["ui_label"] == "Accept"
    assert sup._quest_status == "pending_accept"


def test_ai_only_backend_ui_click_never_restores_cursor_outside_game():
    fake = _FakeCameraInput()
    backend = AIOnlyBackend(fake)
    backend.ui_click(0.55, 0.72)
    assert ("ui_click", 0.55, 0.72, "left") in fake.events
