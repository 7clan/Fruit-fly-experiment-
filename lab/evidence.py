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

    def __init__(self, bus: Bus, session_dir: Path, target_hz: float = 5.0):
        super().__init__(bus, target_hz=target_hz)
        self.capture = bus.state("capture.frames.latest")
        self.brain = bus.state("brain.output")
        self.obs = bus.state("world.observation")
        self.out_dir = Path(session_dir) / "evidence"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._last_chunk = None
        self.stats.update({"saved": 0})

    def step(self) -> None:
        b = self.brain.read()
        c = self.capture.read()
        if b is None or c is None:
            return
        chunk = b.payload.get("chunk_id")
        if chunk is None or chunk == self._last_chunk:
            return
        img = c.payload.get("data_ref")
        if img is None:
            return

        try:
            import cv2
            frame = img
            if frame.ndim == 3 and frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            raw = frame.copy()
            annotated = raw.copy()
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

            stem = f"brain_chunk_{int(chunk):06d}"
            cv2.imwrite(
                str(self.out_dir / f"{stem}_raw.jpg"), raw,
                [int(cv2.IMWRITE_JPEG_QUALITY), 82])
            cv2.imwrite(
                str(self.out_dir / f"{stem}_annotated.jpg"), annotated,
                [int(cv2.IMWRITE_JPEG_QUALITY), 82])
            self._last_chunk = chunk
            self.stats["saved"] += 1
        except Exception as exc:
            self.stats["errors"] += 1
            self.stats["last_error"] = repr(exc)
