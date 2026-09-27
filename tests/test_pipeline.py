"""Phase-1 pipeline tests (plain asserts; run: python -m tests.test_pipeline).

Covers the pieces that MUST work before any animal is involved:
  1. tracker recovers the synthetic fly from rendered frames (CV path)
  2. zone classifier + dwell decoder emit the right actions
  3. arena physics + sparse/shaped reward
  4. SQLite roundtrip
  5. calibration homography
"""
from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np

from flyagent.config import load_all_config
from flyagent.fly_tracker.source import SyntheticSource
from flyagent.fly_tracker.tracker import FlyTracker
from flyagent.fly_tracker.calibration import Calibrator
from flyagent.fly_tracker.trajectory import TrajectoryBuffer
from flyagent.biological_agent.behavior_classifier import BehaviorClassifier
from flyagent.biological_agent.action_decoder import ActionDecoder
from flyagent.environment.arena2d import Arena2D
from flyagent.learning.reward import RewardFunction
from flyagent.data.database import Database

CFG = load_all_config()


def make_fly_bundle(cue_bias=0.0, seed=42):
    fly_cfg = dict(CFG["experiment"]["synthetic_fly"])
    fly_cfg["cue_bias"] = cue_bias
    rng = np.random.default_rng(seed)
    src = SyntheticSource(CFG["tracking"]["source"], fly_cfg, rng,
                          warmup_frames=CFG["tracking"]["tracker"]["warmup_frames"])
    ax, ay, aw, ah = src.arena_rect
    tracker = FlyTracker(CFG["tracking"]["tracker"], roi=(ax, ay, aw, ah))
    calib = Calibrator.from_rect((ax, ay, aw, ah),
                                 arena_mm=tuple(CFG["tracking"]["source"]["arena_mm"]))
    return src, tracker, calib


def test_tracker_recovers_synthetic_fly():
    src, tracker, calib = make_fly_bundle()
    errors, found_n = [], 0
    for i in range(150):
        _, frame = src.read()
        tp = tracker.update(frame)
        if i < CFG["tracking"]["tracker"]["warmup_frames"] + 5:
            continue
        if tp.found:
            found_n += 1
            xn, yn = calib.to_normalized(tp.x, tp.y)
            tx, ty = src.true_position()
            errors.append(math.hypot(xn - tx, yn - ty))
    src.close()
    mean_err = float(np.mean(errors)) if errors else 1.0
    track_rate = found_n / (150 - CFG["tracking"]["tracker"]["warmup_frames"] - 5)
    assert track_rate > 0.90, f"tracking rate too low: {track_rate:.2f}"
    assert mean_err < 0.05, f"mean tracking error too high: {mean_err:.4f}"
    print(f"PASS tracker: rate={track_rate:.2f} mean_err={mean_err:.4f} (normalized)")


def test_decoder_zone_dwell():
    # explicit zone-dwell config (the default scheme is motion_direction)
    zone_cfg = dict(CFG["actions"])
    zone_cfg["scheme"] = "zone_dwell"
    classifier = BehaviorClassifier(CFG["actions"])
    decoder = ActionDecoder(zone_cfg)
    buf = TrajectoryBuffer()

    class _C:  # minimal calibrator stand-in with 90x60 mm arena
        arena_mm = (90.0, 60.0)
        def to_normalized(self, x, y):
            return x, y
        speed_mm_s = Calibrator.speed_mm_s
        heading_rad = Calibrator.heading_rad

    calib = _C()

    def feed(x, y, dt=1 / 30):
        kin = buf.update(buf.buf[-1][2] + dt if buf.buf else 0.0, x, y,
                         True, 1.0, calib)
        return decoder.update(classifier.classify(kin))

    # walk THROUGH the left zone (MOVING): expect LEFT, then cooldown silence
    actions = [feed(0.08 + 0.01 * i, 0.5) for i in range(12)]
    assert "LEFT" in actions, f"LEFT not emitted: {actions}"
    # center zone -> STOP region; pause (zero speed) -> STOP
    actions2 = [feed(0.5, 0.5) for _ in range(2)]
    stopped = None
    for _ in range(20):  # same position => PAUSED
        kin = buf.update(buf.buf[-1][2] + 1 / 30, 0.5, 0.5, True, 1.0, calib)
        a = decoder.update(classifier.classify(kin))
        if a == "STOP":
            stopped = a
            break
    assert stopped == "STOP", "STOP not emitted after sustained pause"
    # right zone while walking -> RIGHT (25 feeds: the STOP emission just
    # before sets a 20-frame cooldown which must fully elapse first)
    acts3 = [feed(0.75 + 0.006 * i, 0.5) for i in range(25)]
    assert "RIGHT" in acts3, f"RIGHT not emitted: {acts3}"
    print("PASS decoder: zone dwell emits LEFT/RIGHT, pause emits STOP")


def test_arena_and_reward():
    rng = np.random.default_rng(3)
    arena = Arena2D({**CFG["experiment"]["game"], "timeout_s": 15}, rng)
    reward = RewardFunction({"mode": "both", **CFG["experiment"]["reward"]})
    total = 0.0
    for _ in range(600):
        arena.set_action(arena.desired_action())
        n0 = len(arena.events)
        arena.step(1 / 30)
        kinds = [e.kind for e in arena.events[n0:]]
        c = reward.step(arena.distance_to_target, kinds, 1 / 30, arena.t)
        total += c.total
        if arena.reached():
            break
    assert arena.reached(), "greedy controller failed to reach target"
    assert arena.path_efficiency > 0.3
    assert total > 0.5, f"expected positive total reward, got {total}"
    print(f"PASS arena+reward: reached in {arena.t:.1f}s, "
          f"pathEff={arena.path_efficiency:.2f}, reward={total:.2f}")


def test_database_roundtrip():
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "t.db")
        db.insert_experiment("exp1", "2026-01-01T00:00:00", "fly", "fly_cue",
                             {"a": 1}, "0.1.0", 42)
        tid = db.insert_trial("exp1", {
            "trial_number": 1, "started_at_ms": 0, "duration_s": 5.0,
            "outcome": "SUCCESS", "time_to_target_s": 5.0,
            "path_length": 300, "straight_dist": 280,
            "path_efficiency": 0.93, "n_actions": 7, "n_action_changes": 5,
            "mean_inter_action_s": 0.7, "reward_total": 1.2,
            "target_x": 1, "target_y": 2, "start_x": 3, "start_y": 4})
        db.insert_event(tid, 1500, "ACTION", {"action": "LEFT"})
        db.insert_frame(tid, 1500, {"x": 0.1, "y": 0.2, "speed": 12.0,
                                    "heading": 0.0, "state": "MOVING",
                                    "zone": "middle_left", "action": "LEFT",
                                    "confidence": 0.9},
                        (10.0, 20.0), 250.0, 0.01)
        db.close()
        rows = Database.fetch_trials(Path(td) / "t.db")
        assert len(rows) == 1 and rows[0]["outcome"] == "SUCCESS"
    print("PASS database: roundtrip experiment/trial/event/frame")


def test_calibration():
    c = Calibrator.from_rect((20, 20, 320, 240), arena_mm=(90, 60))

    def close(p, ex, ey, tol=1e-4):
        return abs(p[0] - ex) < tol and abs(p[1] - ey) < tol

    assert close(c.to_normalized(20, 20), 0.0, 0.0)
    assert close(c.to_normalized(340, 20), 1.0, 0.0)
    assert close(c.to_normalized(340, 260), 1.0, 1.0)
    assert close(c.to_normalized(180, 140), 0.5, 0.5)
    assert abs(c.speed_mm_s(1.0, 0.0) - 90.0) < 1e-9
    assert abs(c.speed_mm_s(0.0, 1.0) - 60.0) < 1e-9
    print("PASS calibration: corners map to unit square; mm/s conversion")


if __name__ == "__main__":
    test_tracker_recovers_synthetic_fly()
    test_decoder_zone_dwell()
    test_arena_and_reward()
    test_database_roundtrip()
    test_calibration()
    print("\nALL TESTS PASSED")
