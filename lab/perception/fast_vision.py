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
                 max_detect_width: int = 640,
                 role_detection: bool = True):
        super().__init__(bus, target_hz=target_hz)
        self.frames: StateChannel = bus.state(frames_channel + ".latest")
        self._last_frame_id = None
        self.obs_state: StateChannel = bus.state(self.TOPIC_OBS)
        self.events: StreamChannel = bus.stream(self.TOPIC_EVT, maxsize=64)
        self.detector = HeuristicFastVision()
        self.gpo_detector = GPOHeuristicFastVision(
            role_detection=role_detection)
        self.max_detect_width = int(max_detect_width)
        self.role_detection = bool(role_detection)

    def on_start(self) -> None:
        # The target laptop has only two physical CPU cores. OpenCV's
        # default internal thread pool can otherwise compete aggressively
        # with the canonical Brian2 subprocess and the game itself.
        try:
            import cv2
            cv2.setNumThreads(1)
            cv2.setUseOptimized(True)
        except Exception:
            pass
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

    def __init__(self, role_detection: bool = True):
        # Conservative association: false positives are more dangerous than
        # missed detections for the first autonomous movement gate.
        self.role_detection = bool(role_detection)
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

        # Direct hostile-role anchors from the observed starter-town
        # overhead red diamonds. This is stronger evidence than generic edge
        # texture and avoids depending on a hedge-prone humanoid proposal.
        hostile_anchors = []
        marker_roi = np.zeros_like(red)
        my0, my1 = int(0.24 * h), int(0.45 * h)
        mx0, mx1 = int(0.10 * w), int(0.93 * w)
        marker_roi[my0:my1, mx0:mx1] = red[my0:my1, mx0:mx1]
        nred, _rlab, rstats, rcents = cv2.connectedComponentsWithStats(
            marker_roi)
        area_scale = max(1.0, (w / 640.0) ** 2)
        for ri in range(1, nred):
            rx, ry, rw, rh, rarea = rstats[ri]
            # At the live detector width (640 px), observed Bandit diamonds
            # are tiny 2x2-ish components. Larger red components are usually
            # HUD bars, damage text, roofs, clothing, or shop scenery.
            if not (2 * area_scale <= rarea <= 5 * area_scale):
                continue
            if rw > max(3, int(0.005 * w)) or rh > max(3, int(0.009 * h)):
                continue
            rcx, rcy = map(float, rcents[ri])
            # The red diamond/name marker sits above the body. Build a
            # conservative body box below it. It remains a candidate until
            # the next passive evidence run confirms the cue.
            bw = 0.050 * w
            bh = 0.145 * h
            bx1 = max(0.0, rcx - bw * 0.5)
            bx2 = min(float(w), rcx + bw * 0.5)
            by1 = max(0.0, rcy + 0.018 * h)
            by2 = min(float(h), by1 + bh)
            hostile_anchors.append({
                "bbox": [bx1 / w, by1 / h, bx2 / w, by2 / h],
                "kind": "hostile_candidate",
                "confidence": 0.78,
            })

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

        # Merge direct hostile anchors with generic proposals. If an anchor
        # overlaps a generic candidate, upgrade that proposal instead of
        # duplicating it.
        for anchor in hostile_anchors:
            acx = (anchor["bbox"][0] + anchor["bbox"][2]) * 0.5
            acy = (anchor["bbox"][1] + anchor["bbox"][3]) * 0.5
            merged = False
            for det in out:
                dcx = (det["bbox"][0] + det["bbox"][2]) * 0.5
                dcy = (det["bbox"][1] + det["bbox"][3]) * 0.5
                if (dcx - acx) ** 2 + (dcy - acy) ** 2 < 0.055 ** 2:
                    if det.get("kind") != "quest_npc":
                        det["kind"] = "hostile_candidate"
                        det["confidence"] = max(
                            float(det.get("confidence", 0.0)), 0.78)
                    merged = True
                    break
            # Unmatched red markers are intentionally discarded.
            # A red pixel/diamond by itself is not enough to invent an enemy:
            # it must coincide with an independently detected non-player
            # humanoid proposal. This removes observed self/HUD false hostiles.
            if not merged:
                continue

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
        quest_direction = None
        quest_proximity = None
        quest_confidence = 0.0
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
            quest_direction = float(bearing)
            quest_proximity = float(proximity)
            quest_confidence = float(confidence)
            target_found = True
        else:
            target = {
                "type": "none", "direction": None,
                "distance": None, "confidence": 0.0,
            }

        # Recommended-quest waypoint: GPO shows a bright green circular
        # destination marker (with distance text) and red path arrows.
        # The green circle is a strong navigation cue and is preferred over
        # the local yellow QUEST marker when present. It is still only a
        # sensory target: it does NOT choose a key or bypass the fly brain.
        green = cv2.inRange(
            hsv, np.array([45, 160, 150], dtype=np.uint8),
            np.array([85, 255, 255], dtype=np.uint8))
        waypoint_roi = np.zeros_like(green)
        # Observed GPO marker: bright green circle with green distance text
        # directly below it. Keep the HUD/ability edges out of this search.
        wy0, wy1 = int(0.10 * h), int(0.68 * h)
        # The real GPO waypoint can legitimately sit almost on the
        # right edge of the gameplay view (confirmed from live screenshots).
        # Keep the text-support requirement below as the false-positive gate
        # instead of cropping away valid edge waypoints.
        wx0, wx1 = int(0.08 * w), int(0.995 * w)
        waypoint_roi[wy0:wy1, wx0:wx1] = green[wy0:wy1, wx0:wx1]
        ng, _glab, gstats, gcents = cv2.connectedComponentsWithStats(
            waypoint_roi)
        waypoint_candidates = []
        min_green_area = max(5, int(w * h * 0.000035))
        max_green_area = max(90, int(w * h * 0.0012))
        for gi in range(1, ng):
            gx, gy, gw, gh, garea = gstats[gi]
            if not (min_green_area <= garea <= max_green_area):
                continue
            aspect = gw / max(float(gh), 1.0)
            if not (0.70 <= aspect <= 1.35):
                continue
            if gw < 3 or gh < 3:
                continue

            # The real waypoint has green distance text (e.g. "179m") a
            # short distance below the circular marker. Require some green
            # support there so isolated world/UI green squares are rejected.
            sx0 = max(0, gx - 2 * gw)
            sx1 = min(w, gx + 3 * gw)
            sy0 = min(h, gy + gh)
            sy1 = min(h, gy + gh + max(3 * gh, int(0.065 * h)))
            support = int(np.count_nonzero(green[sy0:sy1, sx0:sx1]))
            min_support = max(3, int(garea * 0.10))
            if support < min_support:
                continue

            gcx, gcy = map(float, gcents[gi])
            compact = 1.0 - min(1.0, abs(aspect - 1.0))
            # Prefer the circle+text pair and slightly prefer central cues.
            centrality = max(0.25, 1.0 - abs(gcx / max(w, 1) - 0.5))
            score = (float(garea) * (0.7 + 0.3 * compact)
                     * (1.0 + min(1.5, support / max(float(garea), 1.0)))
                     * centrality)
            waypoint_candidates.append(
                (score, gcx, gcy, int(garea), int(gw), int(gh)))

        waypoint_found = False
        waypoint_xy = None
        if waypoint_candidates:
            _score, tx, ty, _ga, _gw, _gh = max(waypoint_candidates)
            bearing = self._bearing(px, py, tx, ty, w, h)
            screen_d = math.hypot((tx - px) / w, (ty - py) / h)
            proximity = max(0.0, min(1.0, 1.0 - screen_d / 0.70))
            target = {
                "type": "recommended_quest_waypoint",
                "direction": float(bearing),
                "distance": float(proximity),
                "confidence": 0.92,
            }
            waypoint_found = True
            waypoint_xy = (float(tx), float(ty))

        red = (
            cv2.inRange(hsv, np.array([0, 120, 100], dtype=np.uint8),
                        np.array([10, 255, 255], dtype=np.uint8))
            | cv2.inRange(hsv, np.array([170, 120, 100], dtype=np.uint8),
                          np.array([179, 255, 255], dtype=np.uint8))
        )

        # Quest-enemy objective marker. Direct live observation from the user
        # establishes the current tracker convention: after accepting a kill
        # quest the recommended target becomes a large red circular marker.
        # Detect only compact red circle + nearby red text support in the
        # gameplay area so the health bar, roofs and damage text cannot win.
        red_obj_roi = np.zeros_like(red)
        ry0, ry1 = int(0.10 * h), int(0.70 * h)
        # Keep objective-dot detection out of the persistent HUD sidebands.
        # The 2026-10-02 live run repeatedly selected x~34/502 from the
        # left-side Recommended Quest notification as a fake enemy marker.
        # Real actionable enemy markers need to enter the gameplay viewport.
        rx0, rx1 = int(0.12 * w), int(0.90 * w)
        red_obj_roi[ry0:ry1, rx0:rx1] = red[ry0:ry1, rx0:rx1]
        nr, _rrlab, rrstats, rrcents = cv2.connectedComponentsWithStats(
            red_obj_roi)
        red_objective_candidates = []
        min_red_obj_area = max(7, int(w * h * 0.000045))
        max_red_obj_area = max(120, int(w * h * 0.0025))
        for ri in range(1, nr):
            rx, ry, rw, rh, rarea = rrstats[ri]
            if not (min_red_obj_area <= rarea <= max_red_obj_area):
                continue
            aspect = rw / max(float(rh), 1.0)
            if not (0.72 <= aspect <= 1.38):
                continue
            if rw < 5 or rh < 5:
                continue
            fill_ratio = float(rarea) / max(float(rw * rh), 1.0)
            # Red Recommended-Quest arrows are thin/triangular and were
            # visible in the user's live screenshots even before a kill
            # objective. A real objective dot is much more compact/filled.
            if fill_ratio < 0.58:
                continue
            sx0 = max(0, rx - 2 * rw)
            sx1 = min(w, rx + 3 * rw)
            sy0 = min(h, ry + rh)
            sy1 = min(h, ry + rh + max(3 * rh, int(0.070 * h)))
            support = int(np.count_nonzero(red[sy0:sy1, sx0:sx1]))
            # Large circular target markers may have sparse distance text;
            # require only weak support, but reject isolated red scenery.
            if support < max(2, int(rarea * 0.05)):
                continue
            rcx, rcy = map(float, rrcents[ri])
            compact = 1.0 - min(1.0, abs(aspect - 1.0))
            centrality = max(0.30, 1.0 - abs(rcx / max(w, 1) - 0.5))
            score = (float(rarea) * (0.75 + 0.25 * compact)
                     * (1.0 + min(1.0, support / max(float(rarea), 1.0)))
                     * centrality)
            red_objective_candidates.append(
                (score, rcx, rcy, int(rarea), int(rw), int(rh)))

        enemy_marker_found = False
        enemy_marker_xy = None
        enemies = []
        if red_objective_candidates:
            _rscore, etx, ety, _ra, _rw, _rh = max(
                red_objective_candidates)
            ebearing = self._bearing(px, py, etx, ety, w, h)
            escreen_d = math.hypot((etx - px) / w, (ety - py) / h)
            eproximity = max(0.0, min(1.0, 1.0 - escreen_d / 0.70))
            target = {
                "type": "quest_enemy_marker",
                "direction": float(ebearing),
                "distance": float(eproximity),
                "confidence": 0.94,
            }
            enemies = [{
                "direction": float(ebearing),
                "distance": float(eproximity),
                "attacking": False,
                "threat": float(max(0.0, (eproximity - 0.55) / 0.45)),
                "confidence": 0.94,
                "type": "quest_enemy_marker",
            }]
            enemy_marker_found = True
            enemy_marker_xy = (float(etx), float(ety))

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

        if self.role_detection:
            proposals = self._humanoid_proposals(
                img, player_xy=(px, py), quest_xy=quest_xy)
            entity_tracks = self._tracker.update(proposals)
            stable_tracks = [
                t for t in entity_tracks
                if t["hits"] >= 3 and t["misses"] == 0
            ]
            role_mode = "role_evidence_v1_not_yet_promoted"
        else:
            # Navigation-light profile: skip Canny/Sobel/box-filter humanoid
            # proposals entirely. Waypoint + HUD sensing remain active.
            # Combat/quest-role recognition is re-enabled by the Gate-7
            # profile, not silently approximated here.
            stable_tracks = []
            role_mode = "disabled_navigation_light"

        # Resolve an actual quest-enemy BODY separately from the red
        # objective marker. The marker tells us where the quest wants us to
        # go; it is not proof that M1 is in range. Only a persistent tracked
        # hostile humanoid spatially associated with that marker is promoted
        # to quest_enemy_actor.
        quest_enemy_actor = None
        actor_candidates = [
            t for t in stable_tracks
            if (
                (t.get("kind") == "hostile_candidate"
                 and float(t.get("confidence", 0.0)) >= 0.58)
                or
                (t.get("kind") == "humanoid_unknown"
                 and float(t.get("confidence", 0.0)) >= 0.42)
            )
        ]
        if enemy_marker_found and enemy_marker_xy and actor_candidates:
            emx = float(enemy_marker_xy[0]) / max(float(w), 1.0)
            emy = float(enemy_marker_xy[1]) / max(float(h), 1.0)
            ranked = []
            for tr in actor_candidates:
                box = list(tr.get("bbox") or [])
                if len(box) != 4:
                    continue
                cx = (float(box[0]) + float(box[2])) * 0.5
                cy = (float(box[1]) + float(box[3])) * 0.5
                association = math.hypot(cx - emx, cy - emy)
                ranked.append((association, tr, cx, cy, box))
            if ranked:
                association, tr, cx, cy, box = min(
                    ranked, key=lambda x: x[0])
                # A track with its own red hostile cue gets a wider gate.
                # A generic humanoid can still be promoted when the quest
                # objective sits almost directly on that persistent body.
                association_limit = (
                    0.24 if tr.get("kind") == "hostile_candidate"
                    else 0.13)
                if association <= association_limit:
                    qx, qy = cx * w, cy * h
                    actor_bearing = self._bearing(px, py, qx, qy, w, h)
                    center_d = math.hypot(
                        cx - float(px / w), cy - float(py / h))
                    center_prox = max(
                        0.0, min(1.0, 1.0 - center_d / 0.52))
                    body_h = max(0.0, float(box[3]) - float(box[1]))
                    scale_prox = max(
                        0.0, min(1.0, body_h / 0.24))
                    actor_proximity = max(
                        0.0, min(
                            1.0,
                            0.62 * center_prox + 0.38 * scale_prox))
                    actor_conf = min(
                        0.95,
                        max(
                            0.70 if tr.get("kind") == "hostile_candidate"
                            else 0.66,
                            float(tr.get("confidence", 0.0)))
                        + 0.08)
                    quest_enemy_actor = {
                        "track_id": tr.get("track_id"),
                        "bbox": box,
                        "direction": float(actor_bearing),
                        "distance": float(actor_proximity),
                        "confidence": float(actor_conf),
                        "association_to_objective": float(association),
                        "role_evidence": str(tr.get("kind") or "unknown"),
                    }
                    target = {
                        "type": "quest_enemy_actor",
                        "direction": float(actor_bearing),
                        "distance": float(actor_proximity),
                        "confidence": float(actor_conf),
                    }
                    enemies = [{
                        "direction": float(actor_bearing),
                        "distance": float(actor_proximity),
                        "attacking": False,
                        "threat": float(max(
                            0.0, (actor_proximity - 0.40) / 0.60)),
                        "confidence": float(actor_conf),
                        "type": "quest_enemy_actor",
                    }]

        ui = {
            "loading": False,
            "dialogue": False,
            "menu": False,
            "combat": bool(quest_enemy_actor is not None),
        }
        return {
            "player": player,
            "target": target,
            "enemies": enemies,
            "ui": ui,
            "abilities": [],
            "_notes": {
                "player_mode": player_mode,
                "quest_marker_detected": target_found,
                "quest_marker_xy": quest_xy,
                "quest_marker_direction": quest_direction,
                "quest_marker_proximity": quest_proximity,
                "quest_marker_confidence": quest_confidence,
                "recommended_waypoint_detected": waypoint_found,
                "recommended_waypoint_xy": waypoint_xy,
                "quest_enemy_marker_detected": enemy_marker_found,
                "quest_enemy_marker_xy": enemy_marker_xy,
                "quest_enemy_actor_visible": quest_enemy_actor is not None,
                "quest_enemy_actor": quest_enemy_actor,
                "quest_phase_hint": (
                    "hunt_enemy" if enemy_marker_found
                    else "quest_giver" if target_found
                    else "travel" if waypoint_found
                    else "unknown"),
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
                "enemy_detector": role_mode,
                "role_detection_enabled": self.role_detection,
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
