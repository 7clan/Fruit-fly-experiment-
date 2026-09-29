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
from .entity_tracker import SimpleTrackletTracker


class FastVisionWorker(Worker):
    name = "fast_vision"
    TOPIC_OBS = "world.observation"
    TOPIC_EVT = "perception.fast.events"

    def __init__(self, bus: Bus, target_hz: float = 24.0,
                 frames_channel: str = "capture.frames",
                 max_detect_width: int = 640):
        super().__init__(bus, target_hz=target_hz)
        self.frames: StateChannel = bus.state(frames_channel + ".latest")
        self._last_frame_id = None
        self.obs_state: StateChannel = bus.state(self.TOPIC_OBS)
        self.events: StreamChannel = bus.stream(self.TOPIC_EVT, maxsize=64)
        self.detector = HeuristicFastVision()
        self.gpo_detector = GPOHeuristicFastVision()
        self.max_detect_width = int(max_detect_width)

    def on_start(self) -> None:
        self.detector.warmup()

    def step(self) -> None:
        snap = self.frames.read()
        if snap is None:
            return
        payload = snap.payload
        frame_id = payload.get("frame_id")
        if frame_id == self._last_frame_id:
            return
        self._last_frame_id = frame_id
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

        detector = (self.gpo_detector
                    if payload.get("source") == "windows_graphics_capture"
                    else self.detector)
        t0 = self.clock.now_ns()
        det = detector.detect(detect_img)
        detect_ms = self.clock.elapsed_ms(t0)
        detector_notes = dict(det.get("_notes", {}))
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
                   "detector": detector.name,
                   "source_shape": list(img.shape),
                   "detect_shape": list(detect_img.shape),
                   "resized_for_detect": resized_for_detect,
                   **detector_notes},
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


class GPOHeuristicFastVision:
    """First real-game fast detector, calibrated from live GPO frames.

    This deliberately detects only features supported by direct observation:
      * player/camera anchor from the bright central avatar silhouette
      * yellow QUEST/! marker as a navigation target
      * bottom-left red health and cyan stamina bar fill

    It does NOT invent enemy/combat detections. Those remain unknown until
    combat examples are collected and validated. Geometry and HUD ROIs use
    normalized coordinates so the detector survives moderate resolution
    changes. Output distance is screen-proximity in [0,1], where 1 means
    visually coincident/touching, matching the fly-channel contract.
    """

    name = "gpo_heuristic_v2"

    def __init__(self):
        # Conservative association: false positives are more dangerous than
        # missed detections for the first autonomous movement gate.
        self._tracker = SimpleTrackletTracker(max_misses=3, center_gate=0.055)

    def warmup(self) -> None:
        # OpenCV is optional outside the Windows/live environment; avoid
        # forcing it into synthetic/CI startup.
        return

    @staticmethod
    def _bearing(px, py, qx, qy, w, h):
        import math
        dx = (qx - px) / w
        dy = (py - qy) / h
        return math.atan2(dx, dy if abs(dy) > 1e-9 else 1e-9)

    @staticmethod
    def _bar_fraction(mask, w, h):
        import cv2
        roi = mask.copy()
        roi[:int(0.78 * h)] = 0
        roi[:, int(0.28 * w):] = 0
        n, _labels, stats, _centroids = cv2.connectedComponentsWithStats(roi)
        candidates = []
        min_area = max(12, int(w * h * 0.00015))
        for i in range(1, n):
            x, y, ww, hh, area = stats[i]
            if area >= min_area and ww > max(8, hh * 6):
                candidates.append((int(ww), int(area), int(x), int(y), int(hh)))
        if not candidates:
            return None
        best_width = max(candidates)[0]
        calibrated_full_width = max(1.0, 0.153 * w)
        return float(max(0.0, min(1.0, best_width / calibrated_full_width)))

    def _humanoid_proposals(self, img, player_xy, quest_xy=None):
        """Conservative avatar candidates with explicit role evidence.

        This remains engineered CV, not a learned detector. The goal here is
        to stop treating high-contrast scenery (hedges, walls, windows) as
        people. Quest NPCs may be anchored directly from the observed QUEST
        marker. A non-quest candidate is only called hostile_candidate when a
        compact saturated-red overhead marker is visible above it. Everything
        else stays humanoid_unknown.
        """
        import cv2

        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        edge = cv2.Canny(gray, 55, 130).astype(np.float32) / 255.0
        gx = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
        gy = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))

        roi = np.zeros((h, w), dtype=np.uint8)
        roi[int(0.22 * h):int(0.67 * h),
            int(0.05 * w):int(0.95 * w)] = 1

        scale = w / 640.0
        sizes = [
            (max(10, int(14 * scale)), max(22, int(30 * scale))),
            (max(12, int(18 * scale)), max(28, int(38 * scale))),
            (max(14, int(22 * scale)), max(34, int(46 * scale))),
        ]
        raw = []
        peak_kernel = np.ones((15, 15), dtype=np.uint8)
        for ww, hh in sizes:
            density = cv2.boxFilter(
                edge, -1, (ww, hh), normalize=True,
                borderType=cv2.BORDER_REPLICATE)
            local_max = cv2.dilate(density, peak_kernel)
            mask = ((density >= 0.205)
                    & (density >= local_max - 1e-6)
                    & (roi > 0))
            ys, xs = np.where(mask)
            if len(xs) == 0:
                continue
            vals = density[ys, xs]
            order = np.argsort(vals)[::-1][:24]
            for j in order:
                cx, cy = float(xs[j]), float(ys[j])
                if (abs(cx - player_xy[0]) < 0.11 * w
                        and abs(cy - player_xy[1]) < 0.17 * h):
                    continue
                x1 = max(0.0, cx - ww * 0.5)
                y1 = max(0.0, cy - hh * 0.5)
                x2 = min(float(w), cx + ww * 0.5)
                y2 = min(float(h), cy + hh * 0.5)
                raw.append((float(vals[j]), cx, cy, x1, y1, x2, y2))

        # Center-distance NMS.
        selected = []
        min_sep2 = (0.038 * w) ** 2
        for score, cx, cy, x1, y1, x2, y2 in sorted(raw, reverse=True):
            if any((cx - q[1]) ** 2 + (cy - q[2]) ** 2 < min_sep2
                   for q in selected):
                continue
            selected.append((score, cx, cy, x1, y1, x2, y2))
            if len(selected) >= 10:
                break

        # Red mask is only role evidence when it is compact and immediately
        # above a plausible avatar candidate.
        red = (
            cv2.inRange(hsv, np.array([0, 150, 130], dtype=np.uint8),
                        np.array([9, 255, 255], dtype=np.uint8))
            | cv2.inRange(hsv, np.array([171, 150, 130], dtype=np.uint8),
                          np.array([179, 255, 255], dtype=np.uint8))
        )

        out = []
        for score, cx, cy, x1, y1, x2, y2 in selected:
            ix1, iy1 = int(max(0, x1)), int(max(0, y1))
            ix2, iy2 = int(min(w, x2)), int(min(h, y2))
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            patch = hsv[iy1:iy2, ix1:ix2]
            pgray = gray[iy1:iy2, ix1:ix2]
            pedge = edge[iy1:iy2, ix1:ix2]
            pgx = gx[iy1:iy2, ix1:ix2]
            pgy = gy[iy1:iy2, ix1:ix2]
            if patch.size == 0:
                continue

            # Grass/hedges caused most of the observed false positives.
            hue = patch[..., 0]
            sat = patch[..., 1]
            val = patch[..., 2]
            green_frac = float(np.mean(
                (hue >= 32) & (hue <= 88) & (sat >= 55) & (val >= 35)))
            texture = float(np.std(pgray))
            edge_density = float(np.mean(pedge))
            ex = float(np.mean(pgx))
            ey = float(np.mean(pgy))
            vertical_ratio = ex / max(ex + ey, 1e-6)

            # Quest proximity may rescue an otherwise green/low-contrast NPC
            # because the independently observed QUEST marker is strong role
            # evidence. Ordinary candidates must pass conservative appearance
            # checks.
            near_quest = False
            if quest_xy is not None:
                qdx = abs(cx - quest_xy[0]) / max(w, 1)
                qdy = (cy - quest_xy[1]) / max(h, 1)
                near_quest = qdx < 0.075 and -0.01 <= qdy < 0.23

            if not near_quest:
                if green_frac > 0.48:
                    continue
                if texture < 22.0:
                    continue
                if not (0.11 <= edge_density <= 0.48):
                    continue
                if vertical_ratio < 0.36:
                    continue

            kind = "quest_npc" if near_quest else "humanoid_unknown"
            confidence = max(
                0.20, min(0.72, 0.30 + (score - 0.20) * 2.4
                          + min(texture / 180.0, 0.18)))
            if near_quest:
                confidence = max(confidence, 0.76)

            # Hostile evidence: a compact red marker/name indicator directly
            # above the candidate. This is deliberately not promoted to
            # EnemyState yet; the next passive run validates it.
            if kind != "quest_npc":
                bw = max(2, ix2 - ix1)
                bh = max(2, iy2 - iy1)
                rx1 = max(0, int(cx - 0.75 * bw))
                rx2 = min(w, int(cx + 0.75 * bw))
                ry1 = max(0, int(y1 - 0.55 * bh))
                ry2 = min(h, int(y1 + 0.10 * bh))
                rpatch = red[ry1:ry2, rx1:rx2]
                if rpatch.size:
                    nred = int(np.count_nonzero(rpatch))
                    red_frac = nred / float(rpatch.size)
                    if nred >= 3 and red_frac >= 0.008:
                        kind = "hostile_candidate"
                        confidence = max(confidence, 0.70)

            out.append({
                "bbox": [x1 / w, y1 / h, x2 / w, y2 / h],
                "kind": kind,
                "confidence": confidence,
            })

        # Strong QUEST evidence gets one explicit NPC anchor even if generic
        # edge proposals miss the body. This prevents quest NPC count=0 when
        # the yellow marker is clearly detected.
        if quest_xy is not None:
            qx, qy = quest_xy
            bw = 0.060 * w
            bh = 0.155 * h
            cx = float(qx)
            cy = float(min(h - bh * 0.5, qy + 0.095 * h))
            qbox = [
                max(0.0, cx - bw * 0.5) / w,
                max(0.0, cy - bh * 0.5) / h,
                min(float(w), cx + bw * 0.5) / w,
                min(float(h), cy + bh * 0.5) / h,
            ]
            # Avoid a duplicate quest proposal if one already overlaps it.
            qcx = (qbox[0] + qbox[2]) * 0.5
            qcy = (qbox[1] + qbox[3]) * 0.5
            duplicate = False
            for det in out:
                bx = (det["bbox"][0] + det["bbox"][2]) * 0.5
                by = (det["bbox"][1] + det["bbox"][3]) * 0.5
                if (bx - qcx) ** 2 + (by - qcy) ** 2 < 0.035 ** 2:
                    det["kind"] = "quest_npc"
                    det["confidence"] = max(
                        float(det.get("confidence", 0.0)), 0.82)
                    duplicate = True
                    break
            if not duplicate:
                out.append({
                    "bbox": qbox,
                    "kind": "quest_npc",
                    "confidence": 0.82,
                })

        return out

    def detect(self, img: np.ndarray) -> dict:
        import cv2
        import math

        h, w = img.shape[:2]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Player: choose a sufficiently large bright/low-saturation component
        # in the central lower gameplay region. If appearance changes and no
        # silhouette survives, use the third-person camera anchor explicitly
        # as a fallback rather than fabricating a visual detection.
        white = cv2.inRange(
            hsv, np.array([0, 0, 205], dtype=np.uint8),
            np.array([179, 85, 255], dtype=np.uint8))
        player_roi = np.zeros_like(white)
        y0, y1 = int(0.34 * h), int(0.72 * h)
        x0, x1 = int(0.40 * w), int(0.60 * w)
        player_roi[y0:y1, x0:x1] = white[y0:y1, x0:x1]
        n, _labels, stats, centroids = cv2.connectedComponentsWithStats(
            player_roi)
        player_candidates = []
        min_player_area = max(12, int(w * h * 0.00008))
        for i in range(1, n):
            x, y, ww, hh, area = stats[i]
            if area < min_player_area or hh < max(3, int(0.015 * h)):
                continue
            cx, cy = centroids[i]
            center_penalty = abs(cx / w - 0.5) * 2.0 + abs(cy / h - 0.52)
            player_candidates.append(
                (center_penalty, -int(area), float(cx), float(cy)))
        if player_candidates:
            _pen, _neg_area, px, py = min(player_candidates)
            player_mode = "bright_avatar"
        else:
            px, py = 0.5 * w, 0.52 * h
            player_mode = "camera_anchor_fallback"

        # QUEST target: saturated yellow components only in the gameplay
        # region. Bottom-left EXP/HUD yellow and top overlays are excluded.
        yellow = cv2.inRange(
            hsv, np.array([18, 140, 140], dtype=np.uint8),
            np.array([42, 255, 255], dtype=np.uint8))
        scene = np.zeros_like(yellow)
        sy0, sy1 = int(0.25 * h), int(0.72 * h)
        sx0, sx1 = int(0.08 * w), int(0.92 * w)
        scene[sy0:sy1, sx0:sx1] = yellow[sy0:sy1, sx0:sx1]
        n, _labels, stats, centroids = cv2.connectedComponentsWithStats(scene)
        comps = []
        min_yellow_area = max(3, int(w * h * 0.000003))
        for i in range(1, n):
            x, y, ww, hh, area = stats[i]
            if area >= min_yellow_area:
                cx, cy = centroids[i]
                comps.append((int(area), int(x), int(y), int(ww), int(hh),
                              float(cx), float(cy)))

        target_found = False
        quest_xy = None
        if comps:
            main = max(comps, key=lambda q: q[0])
            mcx, mcy = main[5], main[6]
            cluster = [
                q for q in comps
                if abs(q[5] - mcx) <= 0.06 * w
                and abs(q[6] - mcy) <= 0.12 * h
            ]
            total_area = float(sum(q[0] for q in cluster))
            tx = sum(q[5] * q[0] for q in cluster) / total_area
            # Use the bottom of the QUEST/! cluster: it points toward the NPC.
            ty = float(max(q[2] + q[4] for q in cluster))
            bearing = self._bearing(px, py, tx, ty, w, h)
            screen_d = math.hypot((tx - px) / w, (ty - py) / h)
            proximity = max(0.0, min(1.0, 1.0 - screen_d / 0.60))
            confidence = 0.82 if player_mode == "bright_avatar" else 0.55
            target = {
                "type": "quest_marker",
                "direction": float(bearing),
                "distance": float(proximity),
                "confidence": confidence,
            }
            quest_xy = (float(tx), float(ty))
            target_found = True
        else:
            target = {
                "type": "none", "direction": None,
                "distance": None, "confidence": 0.0,
            }

        red = (
            cv2.inRange(hsv, np.array([0, 120, 100], dtype=np.uint8),
                        np.array([10, 255, 255], dtype=np.uint8))
            | cv2.inRange(hsv, np.array([170, 120, 100], dtype=np.uint8),
                          np.array([179, 255, 255], dtype=np.uint8))
        )
        cyan = cv2.inRange(
            hsv, np.array([95, 110, 90], dtype=np.uint8),
            np.array([110, 255, 255], dtype=np.uint8))
        health = self._bar_fraction(red, w, h)
        stamina = self._bar_fraction(cyan, w, h)

        player = {
            "position": [float(px / w), float(py / h)],
            "heading": 0.0,
            "health": health,
            "stamina": stamina,
            "health_units": "fraction" if health is not None else "unknown",
        }

        proposals = self._humanoid_proposals(
            img, player_xy=(px, py), quest_xy=quest_xy)
        entity_tracks = self._tracker.update(proposals)
        stable_tracks = [
            t for t in entity_tracks
            if t["hits"] >= 3 and t["misses"] == 0
        ]

        ui = {
            "loading": False,
            "dialogue": False,
            "menu": False,
            "combat": False,
        }
        return {
            "player": player,
            "target": target,
            "enemies": [],
            "ui": ui,
            "abilities": [],
            "_notes": {
                "player_mode": player_mode,
                "quest_marker_detected": target_found,
                "health_bar_detected": health is not None,
                "stamina_bar_detected": stamina is not None,
                "entity_tracks": stable_tracks,
                "humanoid_track_count": sum(
                    1 for t in stable_tracks
                    if t["kind"] == "humanoid_unknown"),
                "quest_npc_track_count": sum(
                    1 for t in stable_tracks
                    if t["kind"] == "quest_npc"),
                "hostile_candidate_count": sum(
                    1 for t in stable_tracks
                    if t["kind"] == "hostile_candidate"),
                "enemy_detector": "role_evidence_v1_not_yet_promoted",
            },
        }


# ---------------------------------------------------------------------------

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
