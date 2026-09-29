"""Lightweight tracklets for low-rate GPO perception.

This is intentionally detector-agnostic. It associates normalized boxes by
center distance/IoU and keeps short histories so semantic code can reason
about persistence and motion without a heavyweight tracking dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field


def _center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ba = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    den = aa + ba - inter
    return inter / den if den > 0 else 0.0


@dataclass
class Tracklet:
    track_id: int
    bbox: list
    kind: str = "humanoid_unknown"
    confidence: float = 0.0
    hits: int = 1
    misses: int = 0
    history: list = field(default_factory=list)

    def snapshot(self) -> dict:
        cx, cy = _center(self.bbox)
        return {
            "track_id": self.track_id,
            "bbox": list(self.bbox),
            "center": [cx, cy],
            "kind": self.kind,
            "confidence": round(float(self.confidence), 3),
            "hits": self.hits,
            "misses": self.misses,
            "history": [list(p) for p in self.history[-6:]],
        }


class SimpleTrackletTracker:
    """Greedy latest-frame association suitable for <=~30 proposals/frame."""

    def __init__(self, max_misses: int = 4, center_gate: float = 0.09):
        self.max_misses = int(max_misses)
        self.center_gate = float(center_gate)
        self._next_id = 1
        self._tracks: dict[int, Tracklet] = {}

    def update(self, detections: list[dict]) -> list[dict]:
        unmatched_tracks = set(self._tracks)
        unmatched_dets = set(range(len(detections)))
        pairs = []

        for tid, tr in self._tracks.items():
            tcx, tcy = _center(tr.bbox)
            for di, det in enumerate(detections):
                dcx, dcy = _center(det["bbox"])
                d2 = (tcx - dcx) ** 2 + (tcy - dcy) ** 2
                overlap = _iou(tr.bbox, det["bbox"])
                # Low cost is good. IoU helps preserve identity when boxes
                # overlap; center distance handles low-FPS displacement.
                cost = d2 - 0.02 * overlap
                if d2 <= self.center_gate ** 2 or overlap >= 0.05:
                    pairs.append((cost, tid, di))

        for _cost, tid, di in sorted(pairs):
            if tid not in unmatched_tracks or di not in unmatched_dets:
                continue
            tr = self._tracks[tid]
            det = detections[di]
            tr.bbox = list(det["bbox"])
            tr.kind = str(det.get("kind", tr.kind))
            tr.confidence = 0.65 * tr.confidence + 0.35 * float(
                det.get("confidence", tr.confidence))
            tr.hits += 1
            tr.misses = 0
            tr.history.append(_center(tr.bbox))
            tr.history = tr.history[-8:]
            unmatched_tracks.remove(tid)
            unmatched_dets.remove(di)

        for tid in unmatched_tracks:
            self._tracks[tid].misses += 1

        for di in unmatched_dets:
            det = detections[di]
            box = list(det["bbox"])
            tr = Tracklet(
                track_id=self._next_id,
                bbox=box,
                kind=str(det.get("kind", "humanoid_unknown")),
                confidence=float(det.get("confidence", 0.0)),
                history=[_center(box)],
            )
            self._tracks[self._next_id] = tr
            self._next_id += 1

        stale = [tid for tid, tr in self._tracks.items()
                 if tr.misses > self.max_misses]
        for tid in stale:
            del self._tracks[tid]

        return [
            tr.snapshot()
            for tr in self._tracks.values()
            if tr.misses <= 1
        ]
