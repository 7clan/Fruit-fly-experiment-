from lab.bus import Bus
from lab.coach.semantic_coach import SemanticCoachWorker
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
