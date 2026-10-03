from lab.bus import Bus
from lab.coach.semantic_coach import SemanticCoachWorker
from lab.coach.local_smolvlm import LocalSmolVLMCoachWorker
from lab.coach.local_smollm import LocalSmolLMCoachWorker
from lab.coach.llama_api import LlamaApiCoachWorker
from lab.coach.meta_model_api import MetaModelApiCoachWorker
from lab.coach.ollama_cloud import OllamaCloudCoachWorker
from lab.coach.meta_probe import choose_model as choose_meta_model
from lab.coach.llama_probe import choose_model as choose_llama_model
from lab.coach.gpo_skills import (
    select_skill_cards, render_skill_cards, procedural_skill_plan,
)
from lab.coach.probe import (
    choose_model, sanitize_api_key, valid_api_key_shape,
)
from lab.action.quest_combat_supervisor import QuestCombatSupervisor
from lab.world.value import ValueTable


def _catalog():
    return {
        "controls": [
            {"control_id": "interact", "name": "Interact"},
            {"control_id": "move_forward", "name": "Move forward"},
            {"control_id": "buso_haki", "name": "Busoshoku Haki"},
        ]
    }


def test_coach_validation_rejects_unknown_skill_and_unknown_control():
    bus = Bus()
    coach = SemanticCoachWorker(bus, api_key="")
    plan = coach._validate_plan({
        "scene": "test",
        "objective": "test",
        "target": "none",
        "skill": "PRESS_RANDOM_KEY",
        "control_id": "totally_fake",
        "confidence": 0.99,
        "ui_click": {"needed": False, "x_norm": 0.0, "y_norm": 0.0},
        "explanation": "test",
        "next_after_success": "",
    }, _catalog())
    assert plan["skill"] == "REOBSERVE"
    assert plan["control_id"] == ""


def test_coach_validation_requires_high_confidence_for_ui_click():
    bus = Bus()
    coach = SemanticCoachWorker(bus, api_key="")
    plan = coach._validate_plan({
        "scene": "shop",
        "objective": "buy",
        "target": "button",
        "skill": "UI_CLICK",
        "control_id": "",
        "confidence": 0.7,
        "ui_click": {
            "needed": True, "x_norm": 0.5, "y_norm": 0.5,
            "label": "Buy",
        },
        "explanation": "visible buy button",
        "next_after_success": "verify",
    }, _catalog())
    assert plan["skill"] == "REOBSERVE"
    assert plan["ui_click"]["needed"] is False


def test_coach_validation_keeps_verified_control_id():
    bus = Bus()
    coach = SemanticCoachWorker(bus, api_key="")
    plan = coach._validate_plan({
        "scene": "npc",
        "objective": "talk",
        "target": "quest npc",
        "skill": "EXEC_CONTROL",
        "control_id": "interact",
        "confidence": 0.95,
        "ui_click": {"needed": False, "x_norm": 0, "y_norm": 0},
        "explanation": "talk to quest NPC",
        "next_after_success": "verify quest",
    }, _catalog())
    assert plan["skill"] == "EXEC_CONTROL"
    assert plan["control_id"] == "interact"


def _armed_supervisor(target):
    bus = Bus()
    sup = QuestCombatSupervisor(bus, ValueTable())
    bus.state("action.meta").write({"autonomy": True})
    bus.state("world.observation").write({
        "target": target,
        "player": {
            "health": 1.0, "health_units": "fraction", "stamina": 1.0,
        },
        "notes": {},
    })
    return bus, sup


def test_coach_can_attack_only_confirmed_quest_enemy():
    bus, sup = _armed_supervisor({
        "type": "quest_enemy_marker",
        "distance": 0.8,
        "direction": 0.0,
        "confidence": 0.95,
    })
    bus.state("coach.plan").write({
        "plan_id": 1,
        "ts_ns": sup.clock.now_ns(),
        "skill": "FIGHT_QUEST_TARGET",
        "confidence": 0.95,
        "explanation": "red quest enemy in melee range",
    })
    sup.step()
    cmd = bus.state("action.command").read()
    assert cmd is not None
    assert cmd.payload["name"] == "ATTACK_LIGHT"
    assert cmd.payload["source"] == "semantic_coach"


def test_coach_does_not_attack_unmarked_humanoid():
    bus, sup = _armed_supervisor({
        "type": "humanoid_unknown",
        "distance": 0.9,
        "direction": 0.0,
        "confidence": 0.95,
    })
    bus.state("coach.plan").write({
        "plan_id": 1,
        "ts_ns": sup.clock.now_ns(),
        "skill": "FIGHT_QUEST_TARGET",
        "confidence": 0.99,
        "explanation": "person nearby",
    })
    sup.step()
    cmd = bus.state("action.command").read()
    assert cmd is None


def test_coach_purchase_becomes_only_high_confidence_visible_ui_click():
    bus, sup = _armed_supervisor({
        "type": "none",
        "distance": 0.0,
        "direction": 0.0,
        "confidence": 0.0,
    })
    bus.state("coach.plan").write({
        "plan_id": 1,
        "ts_ns": sup.clock.now_ns(),
        "skill": "BUY_ITEM",
        "confidence": 0.92,
        "explanation": "visible low-risk progression purchase",
        "ui_click": {
            "needed": True,
            "x_norm": 0.63,
            "y_norm": 0.71,
            "label": "Buy",
        },
    })
    sup.step()
    cmd = bus.state("action.command").read()
    assert cmd is not None
    assert cmd.payload["name"] == "UI_CLICK"
    assert cmd.payload["ui_context"] is True
    assert cmd.payload["purchase_intent"] is True
    assert 0.0 <= cmd.payload["x_norm"] <= 1.0
    assert 0.0 <= cmd.payload["y_norm"] <= 1.0


def test_coach_persistent_memory_accepts_only_high_confidence_observed_facts():
    bus = Bus()
    coach = SemanticCoachWorker(bus, api_key="")
    updates = coach._validated_memory_updates([
        {
            "key": "level", "value": 20,
            "confidence": 0.97, "evidence": "visible",
        },
        {
            "key": "fruit", "value": "Pika",
            "confidence": 0.99, "evidence": "inferred",
        },
        {
            "key": "totally_unknown", "value": "x",
            "confidence": 1.0, "evidence": "visible",
        },
    ])
    assert updates == [{
        "key": "level",
        "value": 20,
        "confidence": 0.97,
        "evidence": "visible",
    }]


def test_coach_live_ability_requires_visible_binding_and_label():
    bus = Bus()
    coach = SemanticCoachWorker(bus, api_key="")
    good = coach._validate_plan({
        "scene": "combat",
        "objective": "use observed move",
        "target": "quest enemy",
        "skill": "USE_OBSERVED_ABILITY",
        "control_id": "",
        "observed_ability": {"binding": "E", "label": "Gut Punch"},
        "confidence": 0.91,
        "ui_click": {"needed": False, "x_norm": 0, "y_norm": 0},
        "explanation": "move is visible on current HUD",
        "next_after_success": "reobserve",
        "knowledge_query": "",
        "memory_updates": [],
    }, _catalog())
    assert good["skill"] == "USE_OBSERVED_ABILITY"
    assert good["observed_ability"] == {
        "binding": "E", "label": "Gut Punch"
    }

    bad = coach._validate_plan({
        "scene": "combat",
        "objective": "guess a move",
        "target": "quest enemy",
        "skill": "USE_OBSERVED_ABILITY",
        "control_id": "",
        "observed_ability": {"binding": "Z", "label": ""},
        "confidence": 0.99,
        "ui_click": {"needed": False, "x_norm": 0, "y_norm": 0},
        "explanation": "guess",
        "next_after_success": "",
        "knowledge_query": "",
        "memory_updates": [],
    }, _catalog())
    assert bad["skill"] == "REOBSERVE"


def test_probe_prefers_requested_when_available():
    available = {
        "gemini-3.5-flash-lite",
        "gemini-3.1-flash-lite",
    }
    assert choose_model("gemini-3.1-flash-lite", available) == (
        "gemini-3.1-flash-lite"
    )


def test_probe_falls_back_to_current_flash_lite():
    available = {
        "gemini-3.5-flash-lite",
        "gemini-3.8-flash",
    }
    assert choose_model("gemini-3.1-flash-lite", available) == (
        "gemini-3.5-flash-lite"
    )


def test_local_smolvlm_coach_is_offline_and_low_duty_cycle():
    bus = Bus()
    coach = LocalSmolVLMCoachWorker(bus)
    assert coach.provider == "local_smolvlm2_llamacpp"
    assert coach.allow_remote_wiki is False
    assert coach.min_call_interval_s >= 18.0
    assert coach.stats["local_only_inference"] is True


def test_local_smolvlm_prompt_stays_compact_for_tiny_model():
    bus = Bus()
    coach = LocalSmolVLMCoachWorker(bus)
    prompt = coach._prompt(
        {
            "target": {
                "type": "recommended_quest_waypoint",
                "direction": 0.3,
                "distance": 0.4,
            },
            "player": {"health": 1.0, "stamina": 1.0},
            "notes": {"quest_marker_detected": True},
        },
        {"phase": "travel"},
        {"intention": {"name": "APPROACH"}},
        _catalog(),
        {"gpo_loadout": "default_melee"},
        {},
    )
    assert "SMALL LOCAL VISUAL COACH" in prompt
    assert "NAVIGATE_OBJECTIVE" in prompt
    assert len(prompt) < 18000



def test_skill_cards_select_quest_accept_for_yellow_marker():
    cards = select_skill_cards(
        '{"notes":{"yellow_quest":true},"target":{"type":"quest_marker"}}')
    ids = [card.skill_id for card in cards]
    assert "quest_accept" in ids


def test_skill_cards_select_combat_for_red_quest_enemy():
    cards = select_skill_cards(
        '{"target":{"type":"quest_enemy_marker"},"health":0.8}')
    ids = [card.skill_id for card in cards]
    assert "quest_combat" in ids


def test_skill_cards_include_failure_recovery_not_just_action_names():
    text = render_skill_cards(
        '{"target":{"type":"recommended_quest_waypoint"},"stuck":true}')
    assert "IF FAIL:" in text
    assert "obstacle_recovery" in text
    assert "quest_travel" in text


def test_procedural_skill_router_handles_green_waypoint_without_vlm():
    plan = procedural_skill_plan({
        "target": {
            "type": "recommended_quest_waypoint",
            "distance": 0.35,
            "direction": 0.8,
        },
        "player": {"health": 1.0},
        "ui": {"dialogue": False, "menu": False},
        "notes": {},
    }, {"phase": "travel"})
    assert plan is not None
    assert plan["skill"] == "NAVIGATE_OBJECTIVE"
    assert plan["skill_card"] == "quest_travel"


def test_procedural_skill_router_handles_close_red_enemy_without_vlm():
    plan = procedural_skill_plan({
        "target": {
            "type": "quest_enemy_marker",
            "distance": 0.80,
            "direction": 0.0,
        },
        "player": {"health": 0.9},
        "ui": {"dialogue": False, "menu": False},
        "notes": {},
    }, {"phase": "combat"})
    assert plan is not None
    assert plan["skill"] == "FIGHT_QUEST_TARGET"
    assert plan["skill_card"] == "quest_combat"


def test_local_coach_common_quest_step_uses_text_model_skill_selector():
    bus = Bus()
    coach = LocalSmolVLMCoachWorker(bus)

    def fake_local_action(prompt, allowed):
        assert "GPO skill selector" in prompt
        assert "NAVIGATE_OBJECTIVE" in allowed
        return "NAVIGATE_OBJECTIVE"

    coach._call_text_action = fake_local_action
    bus.state("action.meta").write({
        "autonomy": True,
        "gpo_loadout": "default_melee",
    })
    bus.state("action.control_catalog").write(_catalog())
    bus.state("quest.state").write({"phase": "travel"})
    bus.state("world.observation").write({
        "target": {
            "type": "recommended_quest_waypoint",
            "distance": 0.30,
            "direction": -0.5,
            "confidence": 0.92,
        },
        "player": {"health": 1.0, "stamina": 1.0},
        "ui": {"dialogue": False, "menu": False},
        "notes": {"recommended_waypoint_detected": True},
    })
    coach.step()
    env = bus.state("coach.plan").read()
    assert env is not None
    assert env.payload["skill"] == "NAVIGATE_OBJECTIVE"
    assert env.payload["provider"] == "local_smolvlm2_text_skill"
    assert env.payload["local_vlm_used"] is True
    assert coach.stats["calls"] == 1
    assert coach.stats["text_skill_calls"] == 1


def test_procedural_candidate_prioritizes_yellow_quest_giver_over_red_npc():
    plan = procedural_skill_plan({
        "target": {
            "type": "quest_enemy_marker",
            "distance": 0.75,
            "direction": -1.2,
            "confidence": 0.94,
        },
        "player": {"health": 1.0},
        "ui": {"dialogue": False, "menu": False},
        "notes": {
            "quest_marker_detected": True,
            "quest_marker_proximity": 0.45,
            "quest_marker_direction": 0.35,
        },
    }, {"phase": "observe"})
    assert plan is not None
    assert plan["skill_card"] == "quest_accept"
    assert plan["skill"] == "NAVIGATE_OBJECTIVE"


def test_local_text_selector_prompt_is_tiny_and_label_only():
    bus = Bus()
    coach = LocalSmolVLMCoachWorker(bus)
    candidate = {
        "scene": "quest_travel",
        "skill": "NAVIGATE_OBJECTIVE",
        "skill_card": "quest_travel",
    }
    prompt = coach._text_skill_prompt(
        candidate,
        {
            "target": {
                "type": "recommended_quest_waypoint",
                "distance": 0.4,
                "direction": -0.2,
                "confidence": 0.92,
            },
            "player": {"health": 1.0, "stamina": 1.0},
            "notes": {"quest_marker_detected": False},
        },
        {"phase": "travel"},
        _catalog(),
    )
    assert "ANSWER=" in prompt
    assert "ALLOWED=" in prompt
    assert len(prompt) < 900



def test_smollm_135m_coach_is_text_only_and_offline():
    bus = Bus()
    coach = LocalSmolLMCoachWorker(bus)
    assert coach.provider == "local_smollm2_135m_llamacpp"
    assert coach.allow_remote_wiki is False
    assert coach.stats["text_only"] is True
    assert coach.stats["vision_encoder"] is False
    assert coach.stats["model_parameters"] == "135M"


def test_smollm_135m_prompt_is_tiny_structured_state():
    bus = Bus()
    coach = LocalSmolLMCoachWorker(bus)
    candidate = {
        "scene": "quest_travel",
        "skill": "NAVIGATE_OBJECTIVE",
        "skill_card": "quest_travel",
    }
    prompt = coach._skill_prompt(
        candidate,
        {
            "target": {
                "type": "recommended_quest_waypoint",
                "distance": 0.4,
                "direction": -0.2,
                "confidence": 0.92,
            },
            "player": {"health": 1.0, "stamina": 1.0},
            "notes": {"quest_marker_detected": False},
        },
        {"phase": "travel"},
        ["NAVIGATE_OBJECTIVE", "REOBSERVE", "SPRINT"],
    )
    assert "NAVIGATE_OBJECTIVE" in prompt
    assert "CHOICES=" in prompt
    assert "INDEX=" in prompt
    assert len(prompt) < 800


def test_smollm_135m_selects_skill_without_any_image():
    bus = Bus()
    coach = LocalSmolLMCoachWorker(bus)

    def fake_skill(prompt, allowed):
        assert "NAVIGATE_OBJECTIVE" in allowed
        assert "image" not in prompt.lower()
        return "NAVIGATE_OBJECTIVE"

    coach._call_skill = fake_skill
    bus.state("action.meta").write({
        "autonomy": True,
        "gpo_loadout": "default_melee",
    })
    bus.state("action.control_catalog").write(_catalog())
    bus.state("quest.state").write({"phase": "travel"})
    bus.state("world.observation").write({
        "target": {
            "type": "recommended_quest_waypoint",
            "distance": 0.30,
            "direction": -0.5,
            "confidence": 0.92,
        },
        "player": {"health": 1.0, "stamina": 1.0},
        "ui": {"dialogue": False, "menu": False},
        "notes": {"recommended_waypoint_detected": True},
    })
    coach.step()
    env = bus.state("coach.plan").read()
    assert env is not None
    assert env.payload["skill"] == "NAVIGATE_OBJECTIVE"
    assert env.payload["provider"] == "local_smollm2_135m_skill"
    assert env.payload["local_ai_used"] is True
    assert env.payload["text_only"] is True
    assert coach.stats["text_skill_calls"] == 1



def test_smollm_repeated_recovery_excludes_same_skill_after_two_uses():
    bus = Bus()
    coach = LocalSmolLMCoachWorker(bus)
    seen_choices = []

    def fake_skill(prompt, choices):
        seen_choices.append(list(choices))
        return choices[0]

    coach._call_skill = fake_skill
    bus.state("action.meta").write({
        "autonomy": True,
        "gpo_loadout": "default_melee",
    })
    bus.state("action.control_catalog").write(_catalog())
    bus.state("quest.state").write({"phase": "stuck"})
    bus.state("world.observation").write({
        "target": {"type": "none"},
        "player": {"health": 1.0, "stamina": 1.0},
        "ui": {"dialogue": False, "menu": False},
        "notes": {},
    })
    # Direct selector calls isolate the anti-repeat policy from rate limiting.
    candidate = procedural_skill_plan(
        bus.state("world.observation").read().payload,
        bus.state("quest.state").read().payload,
    )
    catalog = _catalog()
    obs = bus.state("world.observation").read().payload
    quest = bus.state("quest.state").read().payload
    for i in range(3):
        coach._select_with_model(
            candidate, obs, quest, catalog, coach.clock.now_ns() + i + 1)
    assert len(seen_choices) == 3
    repeated = seen_choices[0][0]
    assert repeated in seen_choices[0]
    assert repeated in seen_choices[1]
    assert repeated not in seen_choices[2]



def test_llama_probe_prefers_maverick_then_scout():
    available = {
        "Llama-4-Maverick-17B-128E-Instruct-FP8",
        "Llama-4-Scout-17B-16E-Instruct-FP8",
    }
    assert choose_llama_model("auto", available) == (
        "Llama-4-Maverick-17B-128E-Instruct-FP8"
    )
    assert choose_llama_model(
        "Llama-4-Scout-17B-16E-Instruct-FP8", available
    ) == "Llama-4-Scout-17B-16E-Instruct-FP8"


def test_llama_cloud_coach_marks_ai_and_vision():
    bus = Bus()
    coach = LlamaApiCoachWorker(
        bus,
        model="Llama-4-Scout-17B-16E-Instruct-FP8",
        api_key="test-only",
    )
    assert coach.provider == "meta_llama_api"
    assert coach.supports_vision is True
    plan = coach._validate_plan({
        "scene": "quest_travel",
        "objective": "follow objective",
        "target": "green waypoint",
        "skill": "NAVIGATE_OBJECTIVE",
        "control_id": "",
        "observed_ability": {"binding": "", "label": ""},
        "ui_click": {
            "needed": False,
            "x_norm": 0.0,
            "y_norm": 0.0,
            "label": "",
        },
        "confidence": 0.91,
        "explanation": "green objective is visible",
        "next_after_success": "reobserve",
        "knowledge_query": "",
        "memory_updates": [],
    }, _catalog())
    assert plan["ai_used"] is True
    assert plan["cloud_ai_used"] is True
    assert plan["provider"] == "meta_llama_api"



def test_meta_probe_prefers_current_muse_spark():
    available = {
        "muse-spark-1.3",
        "muse-spark-1.2",
    }
    assert choose_meta_model("auto", available) == "muse-spark-1.3"


def test_meta_cloud_coach_marks_ai_and_vision():
    bus = Bus()
    coach = MetaModelApiCoachWorker(
        bus,
        model="muse-spark-1.3",
        api_key="test-only",
    )
    assert coach.provider == "meta_model_api"
    assert coach.supports_vision is True
    plan = coach._validate_plan({
        "scene": "quest_travel",
        "objective": "follow objective",
        "target": "green waypoint",
        "skill": "NAVIGATE_OBJECTIVE",
        "control_id": "",
        "observed_ability": {"binding": "", "label": ""},
        "ui_click": {
            "needed": False,
            "x_norm": 0.0,
            "y_norm": 0.0,
            "label": "",
        },
        "confidence": 0.91,
        "explanation": "green objective is visible",
        "next_after_success": "reobserve",
        "knowledge_query": "",
        "memory_updates": [],
    }, _catalog())
    assert plan["ai_used"] is True
    assert plan["cloud_ai_used"] is True
    assert plan["provider"] == "meta_model_api"



def test_ollama_cloud_coach_marks_ai_and_vision():
    bus = Bus()
    coach = OllamaCloudCoachWorker(
        bus,
        model="deepseek-v4.1-flash",
        api_key="test-only",
    )
    assert coach.provider == "ollama_cloud"
    assert coach.supports_vision is True
    assert coach.allow_remote_wiki is True
    plan = coach._validate_plan({
        "scene": "quest_combat",
        "objective": "defeat quest enemy",
        "target": "red quest enemy",
        "skill": "FIGHT_QUEST_TARGET",
        "control_id": "",
        "observed_ability": {"binding": "", "label": ""},
        "ui_click": {
            "needed": False,
            "x_norm": 0.0,
            "y_norm": 0.0,
            "label": "",
        },
        "confidence": 0.93,
        "explanation": "quest-marked enemy is in melee range",
        "next_after_success": "reobserve",
        "knowledge_query": "",
        "memory_updates": [],
    }, _catalog())
    assert plan["cloud_ai_used"] is True
    assert plan["local_ai_used"] is False
    assert plan["provider"] == "ollama_cloud"
    assert plan["skill"] == "FIGHT_QUEST_TARGET"


def test_ollama_cloud_prompt_explicitly_supports_equip_then_m1_combat():
    bus = Bus()
    coach = OllamaCloudCoachWorker(
        bus,
        model="deepseek-v4.1-flash",
        api_key="test-only",
    )
    prompt = coach._prompt(
        {
            "target": {
                "type": "quest_enemy_marker",
                "distance": 0.82,
                "direction": 0.0,
                "confidence": 0.95,
            },
            "player": {"health": 1.0, "stamina": 1.0},
            "notes": {"quest_enemy_marker_detected": True},
        },
        {"phase": "combat"},
        {"intention": {"name": "APPROACH"}},
        {
            "controls": [
                {"control_id": "equip_slot_1", "name": "Equip slot 1"},
                {"control_id": "basic_attack", "name": "Basic attack"},
            ],
        },
        {"autonomy": True, "gpo_loadout": "default_melee"},
        {},
    )
    assert "EQUIP_SLOT" in prompt
    assert "left-mouse clicks (M1)" in prompt
    assert "FIGHT_QUEST_TARGET" in prompt


def test_coach_plan_waits_for_actionable_geometry_before_consuming():
    bus, sup = _armed_supervisor({
        "type": "quest_enemy_marker",
        "distance": 0.40,
        "direction": 0.0,
        "confidence": 0.95,
    })
    now = sup.clock.now_ns()
    bus.state("coach.plan").write({
        "plan_id": 77,
        "ts_ns": now,
        "skill": "FIGHT_QUEST_TARGET",
        "confidence": 0.95,
        "explanation": "fight the confirmed quest enemy",
    }, ts_ns=now)

    # Too far away: keep this plan pending instead of throwing it away.
    sup.step()
    assert sup._last_coach_plan_id != 77
    assert sup.stats["coach_commands"] == 0

    # Fresh geometry becomes actionable while the same AI plan is still valid.
    bus.state("world.observation").write({
        "target": {
            "type": "quest_enemy_marker",
            "distance": 0.82,
            "direction": 0.0,
            "confidence": 0.95,
        },
        "player": {
            "health": 1.0,
            "health_units": "fraction",
            "stamina": 1.0,
        },
        "notes": {},
    }, ts_ns=sup.clock.now_ns())
    sup.step()
    cmd = bus.state("action.command").read()
    assert cmd is not None
    assert cmd.payload["name"] == "ATTACK_LIGHT"
    assert cmd.payload["source"] == "semantic_coach"
    assert sup._last_coach_plan_id == 77
    assert sup.stats["coach_commands"] == 1
    assert sup.stats["coach_plans_seen"] == 1


def test_gemini_key_sanitizer_handles_common_copy_paste_forms():
    key = "AIza" + "x" * 35
    assert sanitize_api_key(key) == key
    assert sanitize_api_key('"' + key + '"') == key
    assert sanitize_api_key("GEMINI_API_KEY=" + key) == key
    assert sanitize_api_key("GOOGLE_API_KEY='" + key + "'") == key
    assert sanitize_api_key("x-goog-api-key: " + key + "\r\n") == key



def test_gemini_key_shape_rejects_masked_and_truncated_values():
    assert valid_api_key_shape("*") is False
    assert valid_api_key_shape("A") is False
    assert valid_api_key_shape("AIza-short") is False
    assert valid_api_key_shape("AQ.short") is False


def test_gemini_key_shape_accepts_standard_and_auth_key_forms():
    assert valid_api_key_shape(
        "AIza" + "x" * 32
    ) is True
    assert valid_api_key_shape(
        "AQ." + "y" * 40
    ) is True
