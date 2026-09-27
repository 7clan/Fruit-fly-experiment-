"""FlyController: the biological agent's entry point into the game loop.

Per frame:
  1. observe(game_state): the COMPUTER computes the axis-dominant direction
     to the target and presents the corresponding benign visual cue in the
     fly arena (goal mode). In shuffled mode the cue is random - the key
     control condition that dissociates fly taxis from computer planning.
     In none mode no cue is presented.
  2. One camera/video frame is pulled and pushed through the REAL pipeline:
     tracker -> calibration -> trajectory -> classifier -> decoder.
  3. act(): whatever action the decoder emitted (if any) is returned.

Attribution note (scientific honesty): goal selection is done by the
COMPUTER (cue placement). The FLY contributes the movement behavior that
selects among the presented options. Performance differences between
cue modes measure exactly that contribution.
"""
from __future__ import annotations

from .base import Controller
from ..fly_tracker.tracker import FlyTracker
from ..fly_tracker.trajectory import TrajectoryBuffer
from ..fly_tracker.calibration import Calibrator
from ..biological_agent.behavior_classifier import BehaviorClassifier
from ..biological_agent.action_decoder import ActionDecoder


class FlyController(Controller):
    name = "fly"

    def __init__(self, source, tracker: FlyTracker, calib: Calibrator,
                 trajectory: TrajectoryBuffer,
                 classifier: BehaviorClassifier, decoder: ActionDecoder,
                 cue_mode: str = "goal", frame_dt: float = 1.0 / 30.0):
        self.source = source
        self.tracker = tracker
        self.calib = calib
        self.trajectory = trajectory
        self.classifier = classifier
        self.decoder = decoder
        self.cue_mode = cue_mode          # goal | shuffled | none
        self.frame_dt = float(frame_dt)
        self.behavior = None
        self.last_frame = None     # cached for the live dashboard (display only)
        self._pending: str | None = None
        self._cue_axis: str | None = None
        self._cue_hysteresis_px = 25.0
        self._shuffled_side = "LEFT"
        self._shuffled_timer = 0.0
        self._rng = None
        self.n_frames = 0
        self.n_lost = 0

    # ------------------------------------------------------------------
    def reset(self, ctx: dict) -> None:
        self._rng = ctx["rng"]
        self._pending = None
        self._cue_axis = None
        self.decoder.reset()
        self.trajectory.reset()   # fresh time base per trial (t restarts at 0)

    def _current_cue_action(self, desired: str, game_state: dict) -> str | None:
        if self.cue_mode == "none":
            return None
        if self.cue_mode == "goal":
            # Hysteresis with overshoot reversal: keep the current cue axis
            # until another direction CLEARLY dominates (+- margin). Without
            # reversal, a player that overshoots past the target into a wall
            # keeps getting pulled into that wall forever.
            if self._cue_axis is None:
                self._cue_axis = desired
            else:
                hyst = self._cue_hysteresis_px
                p = game_state["player"]
                t = game_state["target"]
                dx, dy = t[0] - p[0], t[1] - p[1]
                ax = self._cue_axis
                if ax in ("LEFT", "RIGHT"):
                    if ax == "LEFT" and dx > hyst:
                        self._cue_axis = "RIGHT"
                    elif ax == "RIGHT" and dx < -hyst:
                        self._cue_axis = "LEFT"
                    elif abs(dy) > abs(dx) + hyst:
                        self._cue_axis = "BACKWARD" if dy >= 0 else "FORWARD"
                else:
                    if ax == "FORWARD" and dy > hyst:
                        self._cue_axis = "BACKWARD"
                    elif ax == "BACKWARD" and dy < -hyst:
                        self._cue_axis = "FORWARD"
                    elif abs(dx) > abs(dy) + hyst:
                        self._cue_axis = "RIGHT" if dx >= 0 else "LEFT"
            return self._cue_axis
        # shuffled: random side, re-rolled every ~2 s
        self._shuffled_timer -= self.frame_dt
        if self._shuffled_timer <= 0:
            self._shuffled_timer = 2.0
            self._shuffled_side = str(self._rng.choice(
                ["FORWARD", "BACKWARD", "LEFT", "RIGHT"]))
        return self._shuffled_side

    def observe(self, game_state: dict, t_ms: int) -> None:
        desired = game_state["desired_action"]
        self.source.set_cue(self._current_cue_action(desired, game_state))

        idx, frame = self.source.read()
        if frame is None:
            return
        self.last_frame = frame
        self.n_frames += 1
        t = t_ms / 1000.0
        tp = self.tracker.update(frame)
        if tp.found:
            xn, yn = self.calib.to_normalized(tp.x, tp.y)
            kin = self.trajectory.update(t, xn, yn, True, tp.confidence,
                                         self.calib)
        else:
            self.n_lost += 1
            kin = self.trajectory.update(t, 0.0, 0.0, False, 0.0, self.calib)
        self.behavior = self.classifier.classify(kin)
        action = self.decoder.update(self.behavior)
        if action is not None:
            self._pending = action

    def act(self) -> str | None:
        a, self._pending = self._pending, None
        return a

    @property
    def lost_rate(self) -> float:
        return (self.n_lost / self.n_frames) if self.n_frames else 0.0
