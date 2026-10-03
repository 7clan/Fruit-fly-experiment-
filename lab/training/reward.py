"""Conservative online reward signals for AI-only gameplay training.

These rewards are ENGINEERED training labels, not biological reward and not
claims about hidden game state. Large rewards are emitted only for strong,
observable events. Small shaping penalties are explicitly labeled.
"""

from __future__ import annotations

import re


_PROGRESS_RE = re.compile(r"(?<!\d)(\d+)\s*/\s*(\d+)(?!\d)")


class OnlineReward:
    """Stateful event scorer used by TrajectoryRecorder.

    Reward components:
      +100 observed quest completion
       +10 observed quest counter increment
       -20 observed player death
      -0.5 generic micro-recovery command (shaping)
      -1.0 repeated combat recovery command (shaping)

    The recorder stores every component separately so future training can
    reweight or ignore shaping without regenerating trajectories.
    """

    def __init__(self):
        self.prev_quest_status = None
        self.prev_progress = None
        self.prev_dead = False
        self.last_command_id = -1

    @staticmethod
    def _progress(text):
        m = _PROGRESS_RE.search(str(text or ""))
        if not m:
            return None
        cur, total = int(m.group(1)), int(m.group(2))
        if total <= 0:
            return None
        return cur, total

    def update(self, *, quest: dict, plan: dict, command: dict,
               world: dict) -> tuple[float, list[dict]]:
        components = []

        qstatus = str((quest or {}).get("quest_status") or "")
        if (self.prev_quest_status == "active"
                and qstatus == "completed"):
            components.append({
                "name": "quest_completed",
                "value": 100.0,
                "confidence": 1.0,
                "kind": "outcome",
            })
        self.prev_quest_status = qstatus or self.prev_quest_status

        progress_text = (
            (quest or {}).get("quest_progress_text")
            or ((plan or {}).get("perception") or {}).get(
                "quest_progress_text")
        )
        progress = self._progress(progress_text)
        if (progress is not None and self.prev_progress is not None
                and progress[1] == self.prev_progress[1]
                and progress[0] > self.prev_progress[0]):
            delta = progress[0] - self.prev_progress[0]
            components.append({
                "name": "quest_progress",
                "value": 10.0 * float(delta),
                "confidence": 0.95,
                "kind": "outcome",
                "delta": int(delta),
            })
        if progress is not None:
            self.prev_progress = progress

        player = (world or {}).get("player") or {}
        dead = bool(
            ((plan or {}).get("perception") or {}).get("player_dead"))
        try:
            hp = float(player.get("health"))
            if player.get("health_units") == "fraction" and hp <= 0.01:
                dead = True
        except (TypeError, ValueError):
            pass
        if dead and not self.prev_dead:
            components.append({
                "name": "player_death",
                "value": -20.0,
                "confidence": 0.98,
                "kind": "outcome",
            })
        self.prev_dead = dead

        try:
            cid = int((command or {}).get("command_id", -1))
        except (TypeError, ValueError):
            cid = -1
        if cid >= 0 and cid != self.last_command_id:
            reason = str((command or {}).get("reason") or "")
            if "micro_recovery_" in reason:
                components.append({
                    "name": "generic_micro_recovery",
                    "value": -0.5,
                    "confidence": 1.0,
                    "kind": "shaping",
                })
            elif ":combat_recovery_" in reason:
                components.append({
                    "name": "combat_recovery",
                    "value": -1.0,
                    "confidence": 1.0,
                    "kind": "shaping",
                })
            self.last_command_id = cid

        total = float(sum(float(x["value"]) for x in components))
        return total, components
