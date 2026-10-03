from __future__ import annotations

import json

from lab.action.ai_only_supervisor import AIOnlyAutopilotSupervisor
from lab.bus import Bus
from lab.clock import SHARED_CLOCK
from lab.skills import SkillLibrary
from lab.training.reward import OnlineReward
from lab.training.trajectory import TrajectoryRecorder


def test_online_reward_detects_progress_completion_and_death():
    r = OnlineReward()

    total, parts = r.update(
        quest={"quest_status": "active", "quest_progress_text": "Defeat 1/8"},
        plan={"perception": {"player_dead": False}},
        command={},
        world={"player": {"health": 1.0, "health_units": "fraction"}},
    )
    assert total == 0.0
    assert parts == []

    total, parts = r.update(
        quest={"quest_status": "active", "quest_progress_text": "Defeat 2/8"},
        plan={"perception": {"player_dead": False}},
        command={
            "command_id": 1,
            "reason": "ai_only:test:micro_recovery_1",
        },
        world={"player": {"health": 0.8, "health_units": "fraction"}},
    )
    names = {p["name"] for p in parts}
    assert "quest_progress" in names
    assert "generic_micro_recovery" in names
    assert total == 9.5

    total, parts = r.update(
        quest={"quest_status": "completed", "quest_progress_text": "Defeat 8/8"},
        plan={"perception": {"player_dead": True}},
        command={"command_id": 2, "reason": "done"},
        world={"player": {"health": 0.0, "health_units": "fraction"}},
    )
    names = {p["name"] for p in parts}
    assert "quest_completed" in names
    assert "quest_progress" in names
    assert "player_death" in names
    assert total == 140.0


def test_skill_library_persists_measured_outcomes(tmp_path):
    path = tmp_path / "skills.json"
    lib = SkillLibrary(path)
    lib.record_segment("FIGHT_QUEST_TARGET", 25.0, ts_ns=123)
    lib.record_segment("FIGHT_QUEST_TARGET", -20.0, ts_ns=456)
    lib.save()

    loaded = SkillLibrary(path)
    rec = loaded.snapshot()["skills"]["FIGHT_QUEST_TARGET"]
    assert rec["attempts"] == 2
    assert rec["positive_outcomes"] == 1
    assert rec["negative_outcomes"] == 1
    assert rec["last_seen_ns"] == 456


def test_trajectory_recorder_writes_persistent_training_data(tmp_path):
    bus = Bus()
    lib = SkillLibrary(tmp_path / "persistent" / "skills.json")
    rec = TrajectoryRecorder(
        bus,
        session_dir=tmp_path / "runs" / "lab_test",
        persistent_root=tmp_path / "persistent",
        skill_library=lib,
        target_hz=5.0,
    )

    bus.state("world.observation").write({
        "ts_ns": 100,
        "player": {"health": 1.0, "health_units": "fraction"},
    }, ts_ns=100)
    bus.state("quest.state").write({
        "ts_ns": 100,
        "quest_status": "active",
        "quest_progress_text": "Defeat 1/2",
    }, ts_ns=100)
    bus.state("coach.plan").write({
        "plan_id": 1,
        "ts_ns": 100,
        "skill": "FIGHT_QUEST_TARGET",
        "perception": {"player_dead": False},
    }, ts_ns=100)
    bus.state("action.command").write({
        "command_id": 1,
        "ts_ns": 100,
        "name": "COMBAT_BUNDLE",
        "reason": "ai_only:fight",
    }, ts_ns=100)
    rec.step()

    bus.state("world.observation").write({
        "ts_ns": 200,
        "player": {"health": 0.8, "health_units": "fraction"},
    }, ts_ns=200)
    bus.state("quest.state").write({
        "ts_ns": 200,
        "quest_status": "completed",
        "quest_progress_text": "Defeat 2/2",
    }, ts_ns=200)
    rec.step()
    rec.on_stop()

    rows = [
        json.loads(line)
        for line in rec.local_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 2
    assert rows[-1]["reward"]["total"] == 110.0
    assert rec.persistent_path.exists()
    assert rec.summary_path.exists()

    skill = SkillLibrary(tmp_path / "persistent" / "skills.json")
    fight = skill.snapshot()["skills"]["FIGHT_QUEST_TARGET"]
    assert fight["attempts"] == 1
    assert fight["positive_outcomes"] == 1


def _combat_bus(target_type: str, distance: float, direction: float):
    bus = Bus()
    now = SHARED_CLOCK.now_ns()
    bus.state("action.meta").write({"autonomy": True}, ts_ns=now)
    notes = {
        "quest_enemy_marker_detected": True,
        "quest_enemy_marker_direction": direction,
        "quest_enemy_marker_proximity": distance,
    }
    if target_type == "quest_enemy_actor":
        notes["quest_enemy_actor_visible"] = True
        notes["quest_enemy_actor"] = {
            "direction": direction,
            "distance": distance,
            "confidence": 0.95,
        }
    bus.state("world.observation").write({
        "ts_ns": now,
        "target": {
            "type": target_type,
            "direction": direction,
            "distance": distance,
            "confidence": 0.95,
        },
        "notes": notes,
    }, ts_ns=now)
    bus.state("coach.plan").write({
        "plan_id": 1,
        "ts_ns": now,
        "skill": "FIGHT_QUEST_TARGET",
        "confidence": 0.95,
        "explanation": "fight target",
        "perception": {
            "quest_state": "active",
            "quest_hud_visible": True,
            "enemy_actor_visible": target_type == "quest_enemy_actor",
        },
        "action_channels": {
            "locomotion": "orbit_right",
            "offense": "m1",
            "defense": "guard_between_attacks",
            "camera": "track_target",
            "equip_control_id": "",
        },
        "visual_target": {
            "kind": "quest_enemy_actor" if target_type == "quest_enemy_actor" else "none",
            "x_norm": 0.5,
            "y_norm": 0.5,
            "bbox_norm": None,
            "confidence": 0.95 if target_type == "quest_enemy_actor" else 0.0,
            "melee_ready": target_type == "quest_enemy_actor",
        },
    }, ts_ns=now)
    return bus


def test_visible_melee_enemy_does_not_trigger_navigation_recovery():
    bus = _combat_bus("quest_enemy_actor", 0.62, 0.05)
    sup = AIOnlyAutopilotSupervisor(bus)
    sup._stuck = True
    sup._circling = True
    sup._progress_target = "quest_enemy_actor"
    sup._last_progress_ns = sup.clock.now_ns() - int(5e9)
    sup.step()

    cmd = bus.state("action.command").read()
    assert cmd is not None
    assert cmd.payload["name"] == "COMBAT_BUNDLE"
    assert sup.stats["micro_recoveries"] == 0


def test_lost_combat_target_uses_reacquire_not_climb_jump():
    bus = _combat_bus("quest_enemy_marker", 0.40, 0.30)
    sup = AIOnlyAutopilotSupervisor(bus)
    sup._progress_target = "quest_enemy_marker"
    sup._best_proximity = 0.40
    sup._last_progress_ns = sup.clock.now_ns() - int(5e9)
    sup.step()

    cmd = bus.state("action.command").read()
    assert cmd is not None
    assert cmd.payload["name"] in {"SEARCH_CAMERA", "STEER_TARGET", "DASH_BACK"}
    assert cmd.payload["name"] not in {"JUMP", "CLIMB"}
    assert sup.stats["combat_recoveries"] == 1
