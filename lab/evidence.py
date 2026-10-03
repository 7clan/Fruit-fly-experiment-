"""Low-rate visual evidence recorder for headless autonomous sessions.

Saves one raw and one annotated game frame whenever a new canonical brain
chunk appears. It reads latest-state snapshots only and never blocks control.
"""

from __future__ import annotations

from pathlib import Path

from .bus import Bus
from .worker import Worker


class EvidenceRecorder(Worker):
    name = "evidence_recorder"

    def __init__(self, bus: Bus, session_dir: Path, target_hz: float = 5.0,
                 save_raw: bool = True):
        super().__init__(bus, target_hz=target_hz)
        self.capture = bus.state("capture.frames.latest")
        self.brain = bus.state("brain.output")
        self.coach = bus.state("coach.plan")
        self.obs = bus.state("world.observation")
        self.out_dir = Path(session_dir) / "evidence"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._last_chunk = None
        self._last_plan = None
        self.save_raw = bool(save_raw)
        self.stats.update({
            "saved": 0, "raw_saved": 0,
            "brain_frames": 0, "ai_plan_frames": 0,
        })

    def step(self) -> None:
        b = self.brain.read()
        p = self.coach.read()
        c = self.capture.read()
        if c is None:
            return

        chunk = (b.payload or {}).get("chunk_id") if b is not None else None
        try:
            plan_id = int((p.payload or {}).get("plan_id", -1)) if p else -1
        except (TypeError, ValueError):
            plan_id = -1

        source = None
        source_id = None
        if plan_id >= 0 and plan_id != self._last_plan:
            source, source_id = "ai_plan", plan_id
        elif chunk is not None and chunk != self._last_chunk:
            source, source_id = "brain_chunk", int(chunk)
        else:
            return

        img = c.payload.get("data_ref")
        if img is None:
            return

        try:
            import cv2
            frame = img
            if frame.ndim == 3 and frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            raw = frame.copy() if self.save_raw else None
            annotated = frame.copy()
            h, w = annotated.shape[:2]

            o = self.obs.read()
            obs = o.payload if o is not None else {}
            notes = (obs or {}).get("notes") or {}
            for tr in notes.get("entity_tracks") or []:
                box = tr.get("bbox") or []
                if len(box) != 4:
                    continue
                x1, y1, x2, y2 = box
                p1 = (int(x1 * w), int(y1 * h))
                p2 = (int(x2 * w), int(y2 * h))
                kind = str(tr.get("kind", "unknown"))
                color = ((0, 220, 255) if kind == "quest_npc"
                         else (80, 80, 255) if kind == "hostile_candidate"
                         else (255, 180, 70))
                cv2.rectangle(annotated, p1, p2, color, 2)
                cv2.putText(
                    annotated, f"T{tr.get('track_id')} {kind}",
                    (p1[0], max(16, p1[1] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

            wp = notes.get("recommended_waypoint_xy")
            dshape = notes.get("detect_shape")
            if (wp and dshape and len(dshape) >= 2
                    and float(dshape[1]) > 0 and float(dshape[0]) > 0):
                sx = w / float(dshape[1])
                sy = h / float(dshape[0])
                px = int(float(wp[0]) * sx)
                py = int(float(wp[1]) * sy)
                cv2.circle(annotated, (px, py), 18, (0, 255, 0), 3)
                cv2.putText(
                    annotated, "recommended_waypoint",
                    (max(0, px - 80), max(18, py - 22)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                    cv2.LINE_AA)

            # Overlay the exact visual target the cloud AI grounded.
            if p is not None:
                plan = p.payload or {}
                vt = plan.get("visual_target") or {}
                box = vt.get("bbox_norm")
                if (isinstance(box, (list, tuple)) and len(box) == 4):
                    try:
                        x1, y1, x2, y2 = [float(v) for v in box]
                        p1 = (int(x1 * w), int(y1 * h))
                        p2 = (int(x2 * w), int(y2 * h))
                        cv2.rectangle(
                            annotated, p1, p2, (255, 0, 255), 2)
                        cv2.putText(
                            annotated,
                            "AI " + str(vt.get("kind") or "target"),
                            (p1[0], max(16, p1[1] - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                            (255, 0, 255), 1, cv2.LINE_AA)
                    except (TypeError, ValueError):
                        pass

            stem = f"{source}_{int(source_id):06d}"
            if self.save_raw:
                cv2.imwrite(
                    str(self.out_dir / f"{stem}_raw.jpg"), raw,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 78])
                self.stats["raw_saved"] += 1
            cv2.imwrite(
                str(self.out_dir / f"{stem}_annotated.jpg"), annotated,
                [int(cv2.IMWRITE_JPEG_QUALITY), 78])
            if source == "ai_plan":
                self._last_plan = int(source_id)
                self.stats["ai_plan_frames"] += 1
            else:
                self._last_chunk = chunk
                self.stats["brain_frames"] += 1
            self.stats["saved"] += 1
        except Exception as exc:
            self.stats["errors"] += 1
            self.stats["last_error"] = repr(exc)
