"""The Phase-1 artificial game: a 2D arena with player, target, obstacles.

Design mirrors the real-game control loop at a trivial scale:
  actions in {FORWARD, BACKWARD, LEFT, RIGHT, STOP}
    -> persistent player velocity (like holding a key)
    -> step physics -> events -> reward

The player is moved ONLY by actions from whichever Controller drives it
(fly / random / fixed / keyboard), which is what makes the control-group
comparison meaningful.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

ACTION_VECTORS = {
    "FORWARD": (0.0, -1.0),   # screen up
    "BACKWARD": (0.0, 1.0),
    "LEFT": (-1.0, 0.0),
    "RIGHT": (1.0, 0.0),
    "STOP": (0.0, 0.0),
}
ACTIONS = list(ACTION_VECTORS)


@dataclass
class GameEvent:
    kind: str          # ACTION | TARGET_REACHED | TIMEOUT | BLOCKED | SPAWN
    t: float
    payload: dict = field(default_factory=dict)


class Arena2D:
    def __init__(self, game_cfg: dict, rng: np.random.Generator):
        self.cfg = dict(game_cfg or {})
        self.rng = rng
        self.w = int(self.cfg.get("width", 640))
        self.h = int(self.cfg.get("height", 480))
        self.player_radius = float(self.cfg.get("player_radius", 10))
        self.speed = float(self.cfg.get("player_speed", 85))
        self.target_radius = float(self.cfg.get("target_radius", 18))
        self.min_dist = float(self.cfg.get("spawn_min_dist", 260))
        self.max_dist = float(self.cfg.get("spawn_max_dist", 420))
        self.timeout_s = float(self.cfg.get("timeout_s", 20.0))
        self.scenario = self.cfg.get("scenario", "open_field")
        self.obstacles = [tuple(o) for o in self.cfg.get("obstacles", [])]
        self.events: list[GameEvent] = []
        self.t = 0.0
        self.path_length = 0.0
        self.n_actions = 0
        self.n_action_changes = 0
        self.action_times: list[float] = []
        self._last_action: str | None = None
        self.reset()

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self.t = 0.0
        self.path_length = 0.0
        self.n_actions = 0
        self.n_action_changes = 0
        self.action_times = []
        self._last_action = None
        self.events = []
        self._blocked_last = False
        self.velocity = (0.0, 0.0)

        m = self.player_radius + 30
        self.player = (float(self.rng.uniform(m, self.w - m)),
                       float(self.rng.uniform(m, self.h - m)))
        # rejection-sample the target at a controlled distance
        for _ in range(1000):
            tx = float(self.rng.uniform(3 * self.target_radius,
                                        self.w - 3 * self.target_radius))
            ty = float(self.rng.uniform(3 * self.target_radius,
                                        self.h - 3 * self.target_radius))
            d = math.hypot(tx - self.player[0], ty - self.player[1])
            if self.min_dist <= d <= self.max_dist and not self._inside_obstacle(tx, ty, self.target_radius):
                break
        self.target = (tx, ty)
        self.start = self.player
        self.straight_dist = math.hypot(tx - self.player[0], ty - self.player[1])
        self.events.append(GameEvent("SPAWN", 0.0,
                                     {"target": self.target, "player": self.player}))

    # ------------------------------------------------------------------
    def _inside_obstacle(self, x: float, y: float, r: float) -> bool:
        for (ox, oy, ow, oh) in self.obstacles:
            if self.scenario != "obstacles":
                continue
            cx = min(max(x, ox), ox + ow)
            cy = min(max(y, oy), oy + oh)
            if math.hypot(x - cx, y - cy) < r:
                return True
        return False

    def set_action(self, action: str) -> None:
        if action not in ACTION_VECTORS:
            raise ValueError(f"unknown action {action!r}")
        vx, vy = ACTION_VECTORS[action]
        self.velocity = (vx * self.speed, vy * self.speed)
        self.n_actions += 1
        if action != self._last_action:
            self.n_action_changes += 1
        self._last_action = action
        self.action_times.append(self.t)
        self.events.append(GameEvent("ACTION", self.t, {"action": action}))

    def step(self, dt: float) -> None:
        px, py = self.player
        nx = px + self.velocity[0] * dt
        ny = py + self.velocity[1] * dt
        r = self.player_radius
        nx = min(max(nx, r), self.w - r)
        ny = min(max(ny, r), self.h - r)

        blocked = self._inside_obstacle(nx, ny, r)
        if blocked:
            if not self._blocked_last:
                self.events.append(GameEvent("BLOCKED", self.t, {}))
            nx, ny = px, py          # hold position (simple resolution)
        self._blocked_last = blocked

        self.path_length += math.hypot(nx - px, ny - py)
        self.player = (nx, ny)
        self.t += dt

        if self.reached():
            self.events.append(GameEvent("TARGET_REACHED", self.t, {}))

    # ------------------------------------------------------------------
    def reached(self) -> bool:
        return math.hypot(self.player[0] - self.target[0],
                          self.player[1] - self.target[1]) <= self.target_radius

    def timed_out(self) -> bool:
        return self.t >= self.timeout_s

    @property
    def distance_to_target(self) -> float:
        return math.hypot(self.player[0] - self.target[0],
                          self.player[1] - self.target[1])

    def desired_action(self) -> str:
        """Axis-dominant greedy direction to target.

        Used by (a) the FixedController ceiling baseline and (b) cue
        placement for the fly. This is COMPUTER-side planning; the fly never
        sees the game, only its stimulus.
        """
        dx = self.target[0] - self.player[0]
        dy = self.target[1] - self.player[1]
        if abs(dx) >= abs(dy):
            return "RIGHT" if dx >= 0 else "LEFT"
        return "BACKWARD" if dy >= 0 else "FORWARD"

    def state(self) -> dict:
        """What a screen-perception system would eventually extract."""
        return {
            "player": self.player,
            "target": self.target,
            "distance": self.distance_to_target,
            "desired_action": self.desired_action(),
            "t": self.t,
            "action": self._last_action,
        }

    @property
    def path_efficiency(self) -> float:
        if self.path_length <= 0:
            return 0.0
        return float(min(1.0, self.straight_dist / self.path_length))

    @property
    def mean_inter_action_s(self) -> float:
        if len(self.action_times) < 2:
            return 0.0
        d = np.diff(self.action_times)
        return float(np.mean(d)) if len(d) else 0.0
