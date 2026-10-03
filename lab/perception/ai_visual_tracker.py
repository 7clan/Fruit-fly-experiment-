"""Fast local tracker for the cloud AI's selected visual target.

The cloud model still decides WHAT to pursue. This worker only tracks the
AI-provided bounding box between cloud replies so a 1-3 second network delay
does not make camera/movement control stale.

No input is emitted here.
"""

from __future__ import annotations

import math
import time

from ..worker import Worker


class AIVisualTracker(Worker):
    name = "ai_visual_tracker"

    TRACKABLE = {
        "quest_giver",
        "quest_enemy_actor",
        "quest_objective",
        "waypoint",
        "ui",
    }

    def __init__(self, bus, target_hz: float = 8.0, max_width: int = 480):
        super().__init__(bus, target_hz=target_hz)
        self.frames = bus.state("capture.frames.latest")
        self.plan = bus.state("coach.plan")
        self.out = bus.state("ai.visual.track")
        self.max_width = max(240, int(max_width))

        self._plan_id = -1
        self._kind = "none"
        self._base_template = None
        self._base_w = 0
        self._base_h = 0
        self._scale = 1.0
        self._bbox = None
        self._misses = 0
        self._last_frame_seq = -1
        self._initialized_ns = 0
        self._semantic_key = None

        self.stats.update({
            "initializations": 0,
            "updates": 0,
            "misses": 0,
            "expires": 0,
            "last_confidence": 0.0,
            "last_kind": "none",
            "last_plan_id": -1,
            "plan_carries": 0,
        })

    @staticmethod
    def _valid_bbox(raw):
        if not isinstance(raw, (list, tuple)) or len(raw) != 4:
            return None
        try:
            x1, y1, x2, y2 = [float(v) for v in raw]
        except (TypeError, ValueError):
            return None
        x1 = max(0.0, min(1.0, x1))
        y1 = max(0.0, min(1.0, y1))
        x2 = max(0.0, min(1.0, x2))
        y2 = max(0.0, min(1.0, y2))
        if x2 - x1 < 0.025 or y2 - y1 < 0.035:
            return None
        return [x1, y1, x2, y2]

    def _frame(self):
        env = self.frames.read()
        if env is None:
            return None, -1
        payload = env.payload or {}
        # StateChannel timestamps can collide in very fast tests or on coarse
        # timer resolutions. Sequence is strictly monotonic per channel and is
        # the correct freshness key.
        return payload.get("data_ref"), int(env.envelope.seq)

    def _prepare(self, frame):
        import cv2
        img = frame
        h, w = img.shape[:2]
        scale = min(1.0, self.max_width / max(float(w), 1.0))
        if scale < 1.0:
            img = cv2.resize(
                img,
                (max(1, int(w * scale)), max(1, int(h * scale))),
                interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return gray

    @staticmethod
    def _semantic_plan_key(plan: dict) -> tuple:
        skill = str(plan.get("skill") or "").strip().upper()
        raw_target = str(plan.get("target") or "").strip().lower()
        # Cloud wording is not identity. "Corrupt Marine", "Corrupt Marine
        # NPC" and "the quest enemy Corrupt Marine" are the same semantic
        # target. Canonicalize generic role words so a harmless wording change
        # cannot drop a good local track.
        cleaned = "".join(
            ch if (ch.isalnum() or ch.isspace()) else " "
            for ch in raw_target)
        stop = {
            "the", "a", "an", "npc", "quest", "enemy", "actor",
            "target", "objective", "marker",
        }
        tokens = [t for t in cleaned.split() if t not in stop]
        target = " ".join(tokens)
        vt = plan.get("visual_target") or {}
        kind = str(vt.get("kind") or "none").strip().lower()
        return skill, target, kind

    def _initialize(self, frame, plan: dict, plan_id: int, now_ns: int):
        vt = plan.get("visual_target") or {}
        kind = str(vt.get("kind") or "none").strip().lower()
        bbox = self._valid_bbox(vt.get("bbox_norm"))
        try:
            conf = float(vt.get("confidence", 0.0))
        except (TypeError, ValueError):
            conf = 0.0
        semantic_key = self._semantic_plan_key(plan)
        valid_seed = bool(
            kind in self.TRACKABLE and bbox is not None and conf >= 0.72)
        if not valid_seed:
            # Gemini may repeat the SAME persistent fight/navigation goal but
            # omit the bbox on a later compact response. Throwing away a good
            # local track at that moment recreates cloud latency. Carry the
            # existing AI-selected target only when the semantic goal is still
            # the same; never jump to a new object on our own.
            old_skill, old_target, old_kind = (
                self._semantic_key
                if self._semantic_key is not None
                else ("", "", "none"))
            new_skill, new_target, new_kind = semantic_key
            same_goal = bool(
                self._base_template is not None
                and self._bbox is not None
                and new_skill == old_skill
                and new_target == old_target
                and (new_kind in {"none", old_kind}
                     or old_kind in {"none", new_kind}))
            if same_goal:
                self._plan_id = plan_id
                self._semantic_key = (
                    new_skill, new_target,
                    old_kind if new_kind == "none" else new_kind)
                self.stats["plan_carries"] += 1
                self.stats["last_plan_id"] = plan_id
                self._publish(
                    now_ns,
                    confidence=max(
                        0.45, float(self.stats.get(
                            "last_confidence", 0.55)) * 0.94),
                    source="plan_carry")
                return
            self._reset(plan_id=plan_id)
            self._semantic_key = semantic_key
            return

        gray = self._prepare(frame)
        h, w = gray.shape[:2]
        x1 = int(round(bbox[0] * (w - 1)))
        y1 = int(round(bbox[1] * (h - 1)))
        x2 = int(round(bbox[2] * (w - 1)))
        y2 = int(round(bbox[3] * (h - 1)))
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, max(x1 + 2, x2)), min(h, max(y1 + 2, y2))
        patch = gray[y1:y2, x1:x2]
        if patch.size < 64 or patch.shape[0] < 6 or patch.shape[1] < 6:
            self._reset(plan_id=plan_id)
            return

        self._plan_id = plan_id
        self._kind = kind
        self._base_template = patch.copy()
        self._base_h, self._base_w = patch.shape[:2]
        self._scale = 1.0
        self._bbox = list(bbox)
        self._misses = 0
        self._initialized_ns = now_ns
        self._semantic_key = semantic_key
        self.stats["initializations"] += 1
        self.stats["last_kind"] = kind
        self.stats["last_plan_id"] = plan_id
        self._publish(now_ns, confidence=min(1.0, conf), source="ai_seed")

    def _reset(self, plan_id: int | None = None):
        self._plan_id = -1 if plan_id is None else int(plan_id)
        self._kind = "none"
        self._base_template = None
        self._base_w = 0
        self._base_h = 0
        self._scale = 1.0
        self._bbox = None
        self._misses = 0
        if plan_id is None:
            self._semantic_key = None

    def _publish(self, now_ns: int, confidence: float, source: str):
        if not self._bbox:
            return
        x1, y1, x2, y2 = self._bbox
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        bh = max(0.0, y2 - y1)
        # Weak visual scale hint only; identity/goal still comes from the AI.
        proximity = max(0.0, min(1.0, (bh - 0.055) / 0.34))
        self.out.write({
            "ts_ns": int(now_ns),
            "plan_id": int(self._plan_id),
            "kind": self._kind,
            "x_norm": round(cx, 5),
            "y_norm": round(cy, 5),
            "bbox_norm": [round(v, 5) for v in self._bbox],
            "direction": round((cx - 0.5) * 2.8, 5),
            "proximity_hint": round(proximity, 5),
            "confidence": round(float(confidence), 5),
            "source": source,
            "misses": int(self._misses),
            "age_s": round(
                max(0.0, (now_ns - self._initialized_ns) / 1e9), 3),
            "ENGINEERED_TRACKING_ONLY": True,
        }, ts_ns=now_ns)
        self.stats["last_confidence"] = round(float(confidence), 5)

    def step(self) -> None:
        import cv2

        penv = self.plan.read()
        frame, frame_seq = self._frame()
        if penv is None or frame is None or frame_seq == self._last_frame_seq:
            return
        self._last_frame_seq = frame_seq
        plan = penv.payload or {}
        try:
            pid = int(plan.get("plan_id", -1))
        except (TypeError, ValueError):
            return
        now_ns = self.clock.now_ns()

        # Any new AI plan may point at a different semantic target.
        if pid != self._plan_id:
            self._initialize(frame, plan, pid, now_ns)
            return
        if self._base_template is None or self._bbox is None:
            return

        # Do not let a stale target persist indefinitely without cloud refresh.
        if now_ns - self._initialized_ns > int(12.0e9):
            self.stats["expires"] += 1
            self._reset()
            return

        gray = self._prepare(frame)
        h, w = gray.shape[:2]
        bx1, by1, bx2, by2 = self._bbox
        cx = ((bx1 + bx2) * 0.5) * w
        cy = ((by1 + by2) * 0.5) * h
        bw = max(8.0, (bx2 - bx1) * w)
        bh = max(8.0, (by2 - by1) * h)

        # Search locally around the last known position. Large enough for NPC
        # walking/camera motion, small enough to avoid jumping to a different
        # humanoid with similar clothes.
        margin_x = max(55.0, bw * 1.8)
        margin_y = max(50.0, bh * 1.5)
        sx1 = max(0, int(cx - bw * 0.5 - margin_x))
        sy1 = max(0, int(cy - bh * 0.5 - margin_y))
        sx2 = min(w, int(cx + bw * 0.5 + margin_x))
        sy2 = min(h, int(cy + bh * 0.5 + margin_y))
        search = gray[sy1:sy2, sx1:sx2]
        if search.size < 64:
            return

        best = None
        # Track modest scale changes as the player approaches/retreats.
        for rel in (0.84, 1.0, 1.18):
            scale = max(0.55, min(2.6, self._scale * rel))
            tw = max(6, int(round(self._base_w * scale)))
            th = max(6, int(round(self._base_h * scale)))
            if tw >= search.shape[1] or th >= search.shape[0]:
                continue
            templ = cv2.resize(
                self._base_template, (tw, th),
                interpolation=(
                    cv2.INTER_AREA if scale < 1.0
                    else cv2.INTER_LINEAR))
            res = cv2.matchTemplate(search, templ, cv2.TM_CCOEFF_NORMED)
            _, score, _, loc = cv2.minMaxLoc(res)
            if best is None or score > best[0]:
                best = (float(score), loc, tw, th, scale)

        if best is None or best[0] < 0.42:
            self._misses += 1
            self.stats["misses"] += 1
            if self._misses >= 5:
                self.stats["expires"] += 1
                self._reset()
            else:
                self._publish(
                    now_ns,
                    confidence=max(0.0, 0.60 - 0.08 * self._misses),
                    source="tracker_hold")
            return

        score, loc, tw, th, scale = best
        x1 = sx1 + int(loc[0])
        y1 = sy1 + int(loc[1])
        x2 = min(w, x1 + tw)
        y2 = min(h, y1 + th)
        self._bbox = [
            x1 / max(float(w), 1.0),
            y1 / max(float(h), 1.0),
            x2 / max(float(w), 1.0),
            y2 / max(float(h), 1.0),
        ]
        self._scale = scale
        self._misses = 0
        self.stats["updates"] += 1
        self._publish(
            now_ns,
            confidence=max(0.0, min(0.98, score)),
            source="template_track")
