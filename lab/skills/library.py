"""Persistent Voyager-style skill registry for AI-only gameplay.

This is an ENGINEERED memory layer. It does not execute raw inputs and does
not replace Gemini as the high-level decision owner. It records reusable skill
contracts and measured outcomes across sessions so later local policies and
fine-tuning have stable identifiers.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path


_DEFAULT_SKILLS = {
    "NAVIGATE_OBJECTIVE": {
        "category": "navigation",
        "preconditions": ["objective_or_waypoint_visible"],
        "success": ["objective_reached", "target_context_changes"],
        "failure": ["no_progress", "circling"],
    },
    "TAKE_QUEST": {
        "category": "quest",
        "preconditions": ["quest_giver_or_dialogue_visible"],
        "success": ["quest_status_active"],
        "failure": ["quest_status_not_changed"],
    },
    "FIGHT_QUEST_TARGET": {
        "category": "combat",
        "preconditions": ["quest_active", "quest_target_grounded"],
        "success": ["quest_progress_increment", "quest_completed"],
        "failure": ["player_death", "target_lost_too_long"],
    },
    "BLOCK": {
        "category": "combat_defense",
        "preconditions": ["incoming_attack_or_defensive_plan"],
        "success": ["damage_avoided"],
        "failure": ["damage_taken"],
    },
    "EVADE": {
        "category": "combat_defense",
        "preconditions": ["incoming_attack_or_disengage_plan"],
        "success": ["damage_avoided", "distance_created"],
        "failure": ["damage_taken"],
    },
    "CLIMB": {
        "category": "navigation_recovery",
        "preconditions": ["explicit_wall_or_climb_evidence"],
        "success": ["progress_restored"],
        "failure": ["still_stuck"],
    },
    "JUMP": {
        "category": "navigation_recovery",
        "preconditions": ["explicit_low_obstacle_evidence"],
        "success": ["progress_restored"],
        "failure": ["still_stuck"],
    },
    "REACQUIRE_COMBAT_TARGET": {
        "category": "combat_recovery",
        "preconditions": ["fight_goal_active", "target_temporarily_lost"],
        "success": ["target_reacquired"],
        "failure": ["target_still_lost"],
    },
}


class SkillLibrary:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.RLock()
        self.skills = {}
        self._load()
        self._ensure_defaults()

    def _load(self):
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                self.skills = dict(data.get("skills") or {})
        except Exception:
            self.skills = {}

    def _ensure_defaults(self):
        with self._lock:
            for sid, spec in _DEFAULT_SKILLS.items():
                rec = self.skills.setdefault(sid, {})
                rec.setdefault("skill_id", sid)
                rec.setdefault("category", spec["category"])
                rec.setdefault("preconditions", list(spec["preconditions"]))
                rec.setdefault("success_conditions", list(spec["success"]))
                rec.setdefault("failure_conditions", list(spec["failure"]))
                rec.setdefault("attempts", 0)
                rec.setdefault("positive_outcomes", 0)
                rec.setdefault("negative_outcomes", 0)
                rec.setdefault("neutral_outcomes", 0)
                rec.setdefault("reward_sum", 0.0)
                rec.setdefault("last_reward", 0.0)
                rec.setdefault("last_seen_ns", 0)
                rec.setdefault("policy_version", 1)

    def record_segment(self, skill_id: str, reward: float,
                       ts_ns: int = 0) -> None:
        sid = str(skill_id or "UNKNOWN").upper()
        if sid not in self.skills:
            self.skills[sid] = {
                "skill_id": sid,
                "category": "discovered",
                "preconditions": [],
                "success_conditions": [],
                "failure_conditions": [],
                "attempts": 0,
                "positive_outcomes": 0,
                "negative_outcomes": 0,
                "neutral_outcomes": 0,
                "reward_sum": 0.0,
                "last_reward": 0.0,
                "last_seen_ns": 0,
                "policy_version": 1,
            }
        with self._lock:
            rec = self.skills[sid]
            rec["attempts"] = int(rec.get("attempts", 0)) + 1
            rec["reward_sum"] = round(
                float(rec.get("reward_sum", 0.0)) + float(reward), 4)
            rec["last_reward"] = round(float(reward), 4)
            rec["last_seen_ns"] = int(ts_ns or 0)
            if reward >= 10.0:
                rec["positive_outcomes"] = int(
                    rec.get("positive_outcomes", 0)) + 1
            elif reward <= -10.0:
                rec["negative_outcomes"] = int(
                    rec.get("negative_outcomes", 0)) + 1
            else:
                rec["neutral_outcomes"] = int(
                    rec.get("neutral_outcomes", 0)) + 1

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "schema_version": 1,
                "ENGINEERED": True,
                "skills": json.loads(json.dumps(self.skills)),
            }

    def save(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(self.snapshot(), indent=1),
            encoding="utf-8")
        tmp.replace(self.path)
        return self.path
