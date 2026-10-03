"""Persistent AI gameplay trajectory recorder.

Records compact, training-oriented state/action/outcome snapshots without
blocking combat. Each session writes both a local copy under runs/<session>/
and a persistent copy under runtime_state/training/trajectories/.

The recorder does not train a model in-process. It creates stable data for
behavior cloning, DAgger corrections, Gemini SFT/RLFT export and offline
analysis.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..worker import Worker
from .reward import OnlineReward


class TrajectoryRecorder(Worker):
    name = "trajectory_recorder"

    def __init__(self, bus, session_dir: Path, persistent_root: Path,
                 skill_library=None, target_hz: float = 5.0):
        super().__init__(bus, target_hz=target_hz)
        self.world = bus.state("world.observation")
        self.quest = bus.state("quest.state")
        self.plan = bus.state("coach.plan")
        self.command = bus.state("action.command")
        self.meta = bus.state("action.meta")
        self.ai_track = bus.state("ai.visual.track")
        self.teacher = bus.state("teacher.action")

        self.session_dir = Path(session_dir)
        self.persistent_root = Path(persistent_root)
        self.local_dir = self.session_dir / "training"
        self.local_dir.mkdir(parents=True, exist_ok=True)
        self.persistent_dir = self.persistent_root / "trajectories"
        self.persistent_dir.mkdir(parents=True, exist_ok=True)

        self.session_id = self.session_dir.name
        self.local_path = self.local_dir / "trajectory.jsonl"
        self.persistent_path = self.persistent_dir / (
            self.session_id + ".jsonl")
        self.summary_path = self.local_dir / "trajectory_summary.json"
        self.persistent_summary_path = self.persistent_dir / (
            self.session_id + ".summary.json")

        self._local_fh = open(self.local_path, "a", encoding="utf-8")
        self._persistent_fh = open(
            self.persistent_path, "a", encoding="utf-8")
        self.reward = OnlineReward()
        self.skill_library = skill_library

        self._sample_id = 0
        self._last_plan_id = None
        self._last_command_id = None
        self._active_skill = None
        self._active_skill_reward = 0.0
        self._last_ts_ns = 0
        self.stats.update({
            "samples": 0,
            "reward_total": 0.0,
            "reward_events": 0,
            "plan_changes": 0,
            "command_changes": 0,
            "teacher_samples": 0,
            "quest_completions": 0,
            "player_deaths": 0,
            "progress_increments": 0,
        })

    @staticmethod
    def _payload(ch):
        env = ch.read()
        return (env.payload or {}) if env is not None else {}

    def _finalize_skill_segment(self, ts_ns: int):
        if self._active_skill and self.skill_library is not None:
            self.skill_library.record_segment(
                self._active_skill,
                self._active_skill_reward,
                ts_ns=ts_ns,
            )
        self._active_skill_reward = 0.0

    def _write(self, rec: dict):
        line = json.dumps(rec, default=str, separators=(",", ":"))
        self._local_fh.write(line + "\n")
        self._persistent_fh.write(line + "\n")

    def step(self) -> None:
        world = self._payload(self.world)
        quest = self._payload(self.quest)
        plan = self._payload(self.plan)
        command = self._payload(self.command)
        meta = self._payload(self.meta)
        ai_track = self._payload(self.ai_track)
        teacher = self._payload(self.teacher)

        now_ns = int(
            world.get("ts_ns")
            or quest.get("ts_ns")
            or plan.get("ts_ns")
            or command.get("ts_ns")
            or self.clock.now_ns()
        )
        self._last_ts_ns = now_ns

        try:
            plan_id = int(plan.get("plan_id", -1))
        except (TypeError, ValueError):
            plan_id = -1
        try:
            command_id = int(command.get("command_id", -1))
        except (TypeError, ValueError):
            command_id = -1

        if plan_id != self._last_plan_id and plan_id >= 0:
            self.stats["plan_changes"] += 1
            new_skill = str(plan.get("skill") or "UNKNOWN").upper()
            if self._active_skill is not None and new_skill != self._active_skill:
                self._finalize_skill_segment(now_ns)
            self._active_skill = new_skill
            self._last_plan_id = plan_id

        if command_id != self._last_command_id and command_id >= 0:
            self.stats["command_changes"] += 1
            self._last_command_id = command_id

        reward, components = self.reward.update(
            quest=quest, plan=plan, command=command, world=world)
        if reward:
            self.stats["reward_total"] = round(
                float(self.stats["reward_total"]) + float(reward), 4)
            self._active_skill_reward += float(reward)
        if components:
            self.stats["reward_events"] += len(components)
            for comp in components:
                name = comp.get("name")
                if name == "quest_completed":
                    self.stats["quest_completions"] += 1
                elif name == "player_death":
                    self.stats["player_deaths"] += 1
                elif name == "quest_progress":
                    self.stats["progress_increments"] += int(
                        comp.get("delta", 1))

        focused = bool(teacher.get("focused"))
        if focused and teacher.get("actions"):
            self.stats["teacher_samples"] += 1

        self._sample_id += 1
        rec = {
            "schema_version": 1,
            "session_id": self.session_id,
            "sample_id": self._sample_id,
            "ts_ns": now_ns,
            "mode": (
                "teacher" if teacher.get("enabled")
                else "ai_only" if plan else "observation"),
            "world": world,
            "quest": quest,
            "plan": plan,
            "command": command,
            "ai_track": ai_track,
            "teacher_action": teacher,
            "action_meta": {
                "autonomy": meta.get("autonomy"),
                "mode": meta.get("mode"),
            },
            "reward": {
                "total": reward,
                "components": components,
            },
        }
        self._write(rec)
        self.stats["samples"] += 1

        # Small periodic flushes keep data durable without entering the
        # motor/capture path (this worker has its own thread).
        if self._sample_id % 20 == 0:
            self._local_fh.flush()
            self._persistent_fh.flush()

    def on_stop(self) -> None:
        self._finalize_skill_segment(self._last_ts_ns or self.clock.now_ns())
        if self.skill_library is not None:
            self.skill_library.save()
        self._local_fh.flush()
        self._persistent_fh.flush()
        self._local_fh.close()
        self._persistent_fh.close()
        summary = {
            "schema_version": 1,
            "session_id": self.session_id,
            "trajectory": str(self.local_path),
            "persistent_trajectory": str(self.persistent_path),
            "stats": dict(self.stats),
        }
        text = json.dumps(summary, indent=1)
        self.summary_path.write_text(text, encoding="utf-8")
        self.persistent_summary_path.write_text(text, encoding="utf-8")
