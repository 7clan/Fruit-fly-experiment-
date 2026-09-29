"""Heavy perception tier: OCR + slow semantic work (TARGET 5–20 Hz).

FINAL_ARCHITECTURE §5/§11/§13: heavy vision handles OCR (dialogue,
quests, inventory, equipment, item/ability names, level/EXP, menus) and
detailed NPC recognition. **Combat must not wait for OCR** — this worker
runs on its own low-rate loop, consumes newest frames only, and writes
to a separate state channel ("world.semantics"). The fast path never
reads this channel synchronously.

Backends:
  * MockOCRBackend — deterministic stub for sandbox/tests (no external
    dependency); labels output source="mock_ocr".
  * TesseractOCRBackend — guarded pytesseract import (Windows install
    documented in setup_windows.ps1); caches results per region hash so
    unchanged text regions are not re-OCR'd every tick.

Output payload (state channel world.semantics):
  {"ts_ns", "source", "texts": [{"region": str, "text": str,
                                 "conf": float}], "level": int|None,
   "exp_frac": float|None, "ui_texts_changed": bool}
"""

from __future__ import annotations

import hashlib

from ..bus import Bus, StateChannel, StreamChannel
from ..worker import Worker


class MockOCRBackend:
    name = "mock_ocr"

    def ocr(self, img) -> list:
        # deterministic pseudo-OCR: empty on plain synthetic frames
        return []


class TesseractOCRBackend:
    name = "tesseract"

    def __init__(self):
        try:
            import pytesseract  # noqa: F401
            import pytesseract as pt
            self._pt = pt
        except Exception as e:
            raise RuntimeError(
                f"pytesseract unavailable: {e} — install per "
                "setup_windows.ps1 or use MockOCRBackend") from e

    def ocr(self, img) -> list:
        import numpy as np
        if img is None or not isinstance(img, np.ndarray):
            return []
        data = self._pt.image_to_data(img, output_type=self._pt.Output.DICT)
        out = []
        for i, txt in enumerate(data.get("text", [])):
            t = (txt or "").strip()
            conf = float(data["conf"][i]) if i < len(data.get("conf", [])) else -1
            if t and conf > 40:
                out.append({"region": "full", "text": t, "conf": conf / 100.0})
        return out


class HeavyVisionWorker(Worker):
    """Low-rate semantic worker. NEVER on the combat-critical path."""
    name = "heavy_vision"
    TOPIC_SEM = "world.semantics"
    TOPIC_EVT = "perception.heavy.events"

    def __init__(self, bus: Bus, target_hz: float = 8.0,
                 frames_channel: str = "capture.frames",
                 backend=None):
        super().__init__(bus, target_hz=target_hz)
        self.frames: StreamChannel = bus.stream(frames_channel, maxsize=4)
        self.sem_state: StateChannel = bus.state(self.TOPIC_SEM)
        self.events: StreamChannel = bus.stream(self.TOPIC_EVT, maxsize=64)
        self.backend = backend if backend is not None else MockOCRBackend()
        self._last_hash = None
        self._cache_hits = 0

    def step(self) -> None:
        env = self.frames.newest()
        if env is None:
            return
        payload = env.payload
        img = payload.get("data_ref")
        if img is None:
            return
        # region-hash cache: skip OCR when the frame region is unchanged
        fh = hashlib.blake2b(bytes(bytearray(img.tobytes()[:4096])),
                             digest_size=8).hexdigest()
        if fh == self._last_hash:
            self._cache_hits += 1
            return
        self._last_hash = fh
        t0 = self.clock.now_ns()
        texts = self.backend.ocr(img)
        ocr_ms = self.clock.elapsed_ms(t0)
        sem = {"ts_ns": self.clock.now_ns(), "source": self.backend.name,
               "frame_id": payload.get("frame_id"),
               "texts": texts, "level": None, "exp_frac": None,
               "ocr_ms": round(ocr_ms, 3)}
        self.sem_state.write(sem, ts_ns=sem["ts_ns"])
        self.events.publish({"kind": "heavy_vision", **sem}, ts_ns=sem["ts_ns"])
