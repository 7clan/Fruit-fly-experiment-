"""Fast perception tier (TARGET 20–30 Hz; FINAL_ARCHITECTURE §5, §11).

Handles ONLY what is needed immediately:
  player position · target position · enemy position / approach ·
  motion · attack wind-up cues · health changes · large UI-state
  transitions · looming/threat · distance estimates.

NEVER waits for OCR or heavy detection. Consumes the newest capture
frame (drops obsolete frames — NEWER DATA WINS), emits a compact
WorldObservation on the "world.observation" state channel + a replay
stream event with detection latency measured honestly.

Backends:
  * HeuristicFastVision — dependency-free color/brightness blob analysis
    (works on SyntheticCapture frames; deterministic; used for pipeline
    bring-up, benchmark dry-runs, and as the CPU fallback detector).
  * ONNXFastVision — placeholder interface for the Windows ONNX Runtime /
    WinML route (guarded import; model pre-loaded at on_start(), fixed
    input size; NEVER loaded during combat). Wired on the Windows box.

The detector implementation is ENGINEERED computer vision; the brain
only ever receives FlyChannels derived from the WorldObservation.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..bus import Bus, StateChannel, StreamChannel
from ..schemas import (AbilityAvailability, EnemyState, PlayerState,
                       TargetState, UIState, WorldObservation)
from ..worker import Worker


class FastVisionWorker(Worker):
    name = "fast_vision"
    TOPIC_OBS = "world.observation"
    TOPIC_EVT = "perception.fast.events"

    def __init__(self, bus: Bus, target_hz: float = 24.0,
                 frames_channel: str = "capture.frames",
                 max_detect_width: int = 640):
        super().__init__(bus, target_hz=target_hz)
        self.frames: StreamChannel = bus.stream(frames_channel, maxsize=4)
        self.obs_state: StateChannel = bus.state(self.TOPIC_OBS)
        self.events: StreamChannel = bus.stream(self.TOPIC_EVT, maxsize=64)
        self.detector = HeuristicFastVision()
        self.max_detect_width = int(max_detect_width)

    def on_start(self) -> None:
        self.detector.warmup()

    def step(self) -> None:
        env = self.frames.newest()          # drop obsolete frames
        if env is None:
            return
        payload = env.payload
        img = payload.get("data_ref")
        if img is None:
            return
        # Full-resolution 1920x1030 frames are unnecessarily expensive
        # for the current fast detector on the target 2-core laptop. Resize
        # once in native OpenCV code before the repeated full-frame masks.
        # Geometry remains normalized, so downstream bearings/distances keep
        # the same coordinate semantics.
        detect_img = img
        resized_for_detect = False
        if (self.max_detect_width > 0 and img.shape[1] > self.max_detect_width):
            import cv2
            scale = self.max_detect_width / float(img.shape[1])
            detect_img = cv2.resize(
                img,
                (self.max_detect_width,
                 max(1, int(round(img.shape[0] * scale)))),
                interpolation=cv2.INTER_AREA,
            )
            resized_for_detect = True

        t0 = self.clock.now_ns()
        det = self.detector.detect(detect_img)
        detect_ms = self.clock.elapsed_ms(t0)
        obs = WorldObservation(
            ts_ns=self.clock.now_ns(),
            frame_ref={"frame_id": payload.get("frame_id"),
                       "capture_ts_ns": payload.get("capture_ts_ns"),
                       "width": payload.get("width"),
                       "height": payload.get("height")},
            source=self.name,
            player=PlayerState(**det["player"]),
            target=TargetState(**det["target"]),
            enemies=[EnemyState(**e) for e in det["enemies"]],
            ui=UIState(**det["ui"]),
            abilities=[AbilityAvailability(**a) for a in det["abilities"]],
            notes={"detect_ms": round(detect_ms, 3),
                   "detector": self.detector.name,
                   "source_shape": list(img.shape),
                   "detect_shape": list(detect_img.shape),
                   "resized_for_detect": resized_for_detect},
        )
        self.obs_state.write(obs.to_dict(), ts_ns=obs.ts_ns)
        # replay event (bounded stream; dropped under pressure by design)
        self.events.publish({
            "kind": "fast_vision",
            "frame_id": payload.get("frame_id"),
            "capture_ts_ns": payload.get("capture_ts_ns"),
            "detect_done_ts_ns": obs.ts_ns,
            "capture_to_detect_ms": round(
                (obs.ts_ns - payload.get("capture_ts_ns", obs.ts_ns)) / 1e6, 3),
            "detect_ms": round(detect_ms, 3),
            "observation": obs.to_dict(),
        }, ts_ns=obs.ts_ns)


# ---------------------------------------------------------------------------

class HeuristicFastVision:
    """Deterministic color-blob detector for synthetic/low-dependency use.

    Marker contract (mirrors SyntheticCapture):
      player  = green block near (0.5w, 0.62h)
      target  = orange block (quest marker analogue)
      enemy   = blue/red block; red = attack wind-up
    Produces normalized distances in [0,1] (1 = touching) and bearings
    relative to the player heading (up = 0 rad, clockwise positive).
    """

    name = "heuristic_v1"
    WARMUP_SHAPE = (360, 640, 3)

    def warmup(self) -> None:
        _ = self.detect(np.zeros(self.WARMUP_SHAPE, dtype=np.uint8))

    def detect(self, img: np.ndarray) -> dict:
        h, w = img.shape[:2]
        px, py = self._find_blob(img, (40, 140, 40), (90, 230, 90))
        tx, ty = self._find_blob(img, (40, 140, 200), (90, 220, 255))
        ex, ey, windup = self._find_enemy(img)

        player = {"position": [px / w, py / h] if px is not None else None,
                  "heading": 0.0, "health": None, "stamina": None,
                  "health_units": "unknown"}

        if tx is not None and px is not None:
            bearing = self._bearing(px, py, tx, ty, w, h)
            dist = self._norm_dist(px, py, tx, ty, w, h)
            target = {"type": "quest_marker", "direction": bearing,
                      "distance": dist, "confidence": 0.9}
        else:
            target = {"type": "none", "direction": None, "distance": None,
                      "confidence": 0.0}

        enemies = []
        if ex is not None and px is not None:
            bearing = self._bearing(px, py, ex, ey, w, h)
            dist = self._norm_dist(px, py, ex, ey, w, h)
            threat = max(0.0, 1.0 - dist / 0.6) * (1.0 if windup else 0.4)
            enemies.append({"direction": bearing, "distance": dist,
                            "attacking": bool(windup),
                            "threat": round(threat, 3), "confidence": 0.85,
                            "type": "melee"})
        ui = {"loading": False, "dialogue": False, "menu": False,
              "combat": len(enemies) > 0 and enemies[0]["distance"] < 0.5}
        return {"player": player, "target": target, "enemies": enemies,
                "ui": ui, "abilities": []}

    # -- internals ---------------------------------------------------------
    @staticmethod
    def _mask(img: np.ndarray, lo, hi) -> np.ndarray:
        return ((img[..., 0] >= lo[0]) & (img[..., 0] <= hi[0]) &
                (img[..., 1] >= lo[1]) & (img[..., 1] <= hi[1]) &
                (img[..., 2] >= lo[2]) & (img[..., 2] <= hi[2]))

    def _find_blob(self, img, lo, hi):
        m = self._mask(img, lo, hi)
        if not m.any():
            return None, None
        ys, xs = np.nonzero(m)
        return float(xs.mean()), float(ys.mean())

    def _find_enemy(self, img):
        # blue (idle) or red (wind-up) enemy marker
        for lo, hi, windup in (((60, 40, 180), (110, 90, 255), False),
                               ((180, 40, 40), (255, 110, 110), True)):
            ex, ey = self._find_blob(img, lo, hi)
            if ex is not None:
                return ex, ey, windup
        return None, None, False

    @staticmethod
    def _bearing(px, py, qx, qy, w, h):
        """Bearing of q relative to player 'up' axis, radians in [-pi, pi]."""
        import math
        dx = (qx - px) / w
        dy = (py - qy) / h          # up = forward
        return math.atan2(dx, max(dy, 1e-9) if dy >= 0 else dy)

    @staticmethod
    def _norm_dist(px, py, qx, qy, w, h):
        d = np.hypot((qx - px) / w, (qy - py) / h)
        return float(min(1.0, d * 1.4))


class ONNXFastVision:
    """Windows ONNX Runtime / WinML detector — interface placeholder.

    Contract (binding when wired on the Windows box):
      * model pre-loaded in warmup() (NEVER during combat)
      * fixed input size (letterbox resize)
      * provider auto-detection (WinML/CUDA/CPU) with the chosen provider
        reported in every observation's notes
      * outputs mapped into the same detect() dict contract as the
        heuristic backend (drop-in)
    """

    name = "onnx_unwired"
    AVAILABLE = False

    def warmup(self) -> None:
        raise NotImplementedError("ONNX backend is wired on the Windows box")

    def detect(self, img: np.ndarray) -> dict:
        raise NotImplementedError("ONNX backend is wired on the Windows box")
