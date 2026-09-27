"""Controller factory: builds a controller + its input bundle from config."""
from __future__ import annotations

import numpy as np

from ..config import load_config
from ..fly_tracker.source import build_source
from ..fly_tracker.tracker import FlyTracker
from ..fly_tracker.calibration import Calibrator
from ..fly_tracker.trajectory import TrajectoryBuffer
from ..biological_agent.behavior_classifier import BehaviorClassifier
from ..biological_agent.action_decoder import ActionDecoder
from .base import Controller
from .baselines import RandomController, FixedController, KeyboardController
from .fly_controller import FlyController


class FlyBundle:
    """Everything needed to observe + decode one fly."""

    def __init__(self, tracking_cfg: dict, actions_cfg: dict,
                 fly_cfg: dict | None, rng: np.random.Generator):
        self.source = build_source(tracking_cfg, fly_cfg or {}, rng)
        ax, ay, aw, ah = self.source.arena_rect
        self.tracker = FlyTracker(tracking_cfg.get("tracker", {}), roi=(ax, ay, aw, ah))

        calib_cfg = tracking_cfg.get("calibration", {})
        cal_file = calib_cfg.get("file") or ""
        if cal_file:
            self.calib = Calibrator.load(cal_file)
        elif tracking_cfg["source"]["type"] == "synthetic":
            self.calib = Calibrator.from_rect(
                (ax, ay, aw, ah),
                flip_x=calib_cfg.get("flip_x", False),
                flip_y=calib_cfg.get("flip_y", False),
                arena_mm=tuple(tracking_cfg["source"].get("arena_mm", [90, 60])))
        else:
            raise ValueError(
                "A calibration file is required for webcam/video sources. "
                "Run scripts/calibrate_arena.py first (see README).")

        self.trajectory = TrajectoryBuffer()
        self.classifier = BehaviorClassifier(actions_cfg)
        self.decoder = ActionDecoder(actions_cfg)

    def close(self) -> None:
        self.source.close()


def build_controller(name: str, cfg: dict,
                     rng: np.random.Generator) -> tuple[Controller, FlyBundle | None]:
    """name: random | fixed | fly | keyboard.

    cfg: full config dict {actions, experiment, tracking}.
    Fly variants (cue_mode / cue_bias) are applied by the caller before
    calling this (see experiments/runner.py).
    """
    actions_cfg = cfg["actions"]
    exp_cfg = cfg["experiment"]

    if name == "random":
        c = RandomController(exp_cfg.get("controllers", {}).get("random", {}),
                             rng, actions_cfg.get("action_vocabulary"))
        return c, None

    if name == "fixed":
        c = FixedController(exp_cfg.get("controllers", {}).get("fixed", {}))
        return c, None

    if name == "keyboard":
        return KeyboardController(), None

    if name == "fly":
        variant = cfg.get("_fly_variant", {})  # cue_mode, cue_bias override
        tracking_cfg = cfg["tracking"]
        fly_cfg = dict(exp_cfg.get("synthetic_fly", {}))
        if "cue_bias" in variant:
            fly_cfg["cue_bias"] = variant["cue_bias"]
        bundle = FlyBundle(tracking_cfg, actions_cfg, fly_cfg, rng)
        ctrl = FlyController(
            source=bundle.source, tracker=bundle.tracker, calib=bundle.calib,
            trajectory=bundle.trajectory, classifier=bundle.classifier,
            decoder=bundle.decoder,
            cue_mode=variant.get("cue_mode", "goal"))
        return ctrl, bundle

    raise ValueError(f"unknown controller: {name}")
