from __future__ import annotations

import json

from lab.action.ai_only_supervisor import AIOnlyAutopilotSupervisor
from lab.action.motor_executor import MotorExecutor
from lab.bus import Bus
from lab.clock import SHARED_CLOCK
from lab.evidence import EvidenceRecorder
from lab.skills import SkillLibrary
from lab.training.reward import OnlineReward
from lab.training.trajectory import TrajectoryRecorder
from lab.training.teacher_miner import mine_teacher_priors


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


def test_teacher_evidence_is_compact_raw_only_and_aligned(tmp_path):
    import numpy as np

    bus = Bus()
    now = SHARED_CLOCK.now_ns()
    bus.state("capture.frames.latest").write({
        "frame_id": 42,
        "ts_ns": now,
        "width": 640,
        "height": 360,
        "format": "bgr",
        "copy_count": 1,
        "data_ref": np.zeros((360, 640, 3), dtype=np.uint8),
    }, ts_ns=now)
    bus.state("world.observation").write({
        "ts_ns": now,
        "notes": {},
    }, ts_ns=now)
    bus.state("teacher.action").write({
        "ts_ns": now,
        "enabled": True,
        "focused": True,
        "actions": ["m1"],
    }, ts_ns=now)

    rec = EvidenceRecorder(
        bus, tmp_path, target_hz=1.0,
        save_raw=True, save_annotated=False,
        max_width=320, jpeg_quality=68)
    rec.step()
    rec.on_stop()

    files = sorted((tmp_path / "evidence").glob("*.jpg"))
    assert len(files) == 1
    assert files[0].name == "teacher_frame_00000042_raw.jpg"
    manifest = [
        json.loads(line)
        for line in (tmp_path / "evidence" / "manifest.jsonl")
        .read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert manifest[0]["frame_id"] == 42
    assert manifest[0]["teacher_actions"] == ["m1"]
    assert manifest[0]["width"] == 320
    assert manifest[0]["annotated_file"] is None


def test_teacher_prior_miner_gates_defense_and_learns_m1_timing(tmp_path):
    path = tmp_path / "teacher.jsonl"
    rows = []
    ts = 1_000_000_000
    # 120 short M1 presses, ~0.20 s onset cadence inside bursts.
    for i in range(120):
        rows.append({
            "ts_ns": ts,
            "teacher_action": {
                "enabled": True, "focused": True, "actions": ["m1"],
            },
        })
        ts += 100_000_000
        rows.append({
            "ts_ns": ts,
            "teacher_action": {
                "enabled": True, "focused": True, "actions": [],
            },
        })
        ts += 100_000_000
    path.write_text(
        "\n".join(json.dumps(x) for x in rows) + "\n",
        encoding="utf-8")

    priors = mine_teacher_priors([path])
    assert priors["readiness"]["attack_timing"] is True
    assert priors["readiness"]["defense"] is False
    assert priors["coverage"]["onsets"]["m1"] == 120
    assert 0.19 <= priors["timing"]["m1_inter_onset_median_s"] <= 0.21
    assert any("Defense is not training-ready" in x
               for x in priors["warnings"])


def test_motor_executor_applies_only_training_ready_attack_timing():
    bus = Bus()
    good = {
        "readiness": {"attack_timing": True},
        "coverage": {"onsets": {"m1": 414}},
        "timing": {"m1_inter_onset_median_s": 0.203},
    }
    ex = MotorExecutor(bus, combat_priors=good)
    assert ex.stats["teacher_attack_timing_applied"] is True
    assert abs(ex.stats["bundle_attack_interval_s"] - 0.203) < 1e-9

    bus2 = Bus()
    insufficient = {
        "readiness": {"attack_timing": False},
        "coverage": {"onsets": {"m1": 20}},
        "timing": {"m1_inter_onset_median_s": 0.19},
    }
    ex2 = MotorExecutor(bus2, combat_priors=insufficient)
    assert ex2.stats["teacher_attack_timing_applied"] is False
    assert ex2.stats["bundle_attack_interval_s"] == 0.26


def test_navigation_skill_policy_upgrade_retires_old_buggy_rewards(tmp_path):
    path = tmp_path / "skills.json"
    path.write_text(json.dumps({
        "schema_version": 1,
        "ENGINEERED": True,
        "skills": {
            "NAVIGATE_OBJECTIVE": {
                "skill_id": "NAVIGATE_OBJECTIVE",
                "category": "navigation",
                "preconditions": ["objective_or_waypoint_visible"],
                "success_conditions": ["objective_reached"],
                "failure_conditions": ["no_progress"],
                "attempts": 3,
                "positive_outcomes": 0,
                "negative_outcomes": 2,
                "neutral_outcomes": 1,
                "reward_sum": -144.5,
                "last_reward": -85.5,
                "last_seen_ns": 123,
                "policy_version": 1
            }
        }
    }), encoding="utf-8")

    lib = SkillLibrary(path)
    nav = lib.snapshot()["skills"]["NAVIGATE_OBJECTIVE"]
    assert nav["policy_version"] == 2
    assert nav["attempts"] == 0
    assert nav["negative_outcomes"] == 0
    assert nav["reward_sum"] == 0.0
    assert nav["retired_versions"][-1]["policy_version"] == 1
    assert nav["retired_versions"][-1]["reward_sum"] == -144.5
