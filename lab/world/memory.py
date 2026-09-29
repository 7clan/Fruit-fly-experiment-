"""Memory: semantic + episodic, with ASYNCHRONOUS writes.

FINAL_ARCHITECTURE §23: semantic memory (facts/relationships) and
episodic memory (events/outcomes). Memory writes are ASYNCHRONOUS —
never a large synchronous database write during combat. In-memory cache
with periodic/event-driven persistence.

Design: a bounded async write queue drained by a background thread
(non-real-time); persistence to JSONL files under runs/<session>/memory/.
If the queue is full, LOW-priority writes are DROPPED (memory never
blocks the combat loop) and the drop is counted honestly.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from pathlib import Path

from ..clock import SHARED_CLOCK


class MemoryStore:
    """Async-write semantic + episodic memory."""

    def __init__(self, persist_dir: Path | None = None,
                 max_queue: int = 512, clock=None):
        self.persist_dir = Path(persist_dir) if persist_dir else None
        self.clock = clock or SHARED_CLOCK
        self._q: queue.Queue = queue.Queue(maxsize=max_queue)
        self._semantic: dict = {}          # fact-key -> {"value", "tier", "ts"}
        self._episodic: list = []          # event dicts (bounded ring)
        self._episodic_cap = 4096
        self.dropped = 0
        self.written = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._drain, name="memory",
                                        daemon=True)

    # -- producer side (async, never blocks) ------------------------------
    def put_semantic(self, key: str, value, tier: str = "UNVERIFIED",
                     high_priority: bool = False) -> bool:
        return self._put({"kind": "semantic", "key": key, "value": value,
                          "tier": tier, "ts_ns": self.clock.now_ns()},
                         high_priority)

    def put_episode(self, event: dict, high_priority: bool = False) -> bool:
        return self._put({"kind": "episode", "event": event,
                          "ts_ns": self.clock.now_ns()}, high_priority)

    def _put(self, item: dict, high_priority: bool) -> bool:
        try:
            self._q.put_nowait(item)
            return True
        except queue.Full:
            if high_priority:
                # try to make room by dropping one LOW-priority item
                try:
                    self._q.get_nowait()
                    self.dropped += 1
                    self._q.put_nowait(item)
                    return True
                except Exception:
                    pass
            self.dropped += 1
            return False

    # -- consumer side -------------------------------------------------------
    def _drain(self) -> None:
        while not self._stop.is_set() or not self._q.empty():
            try:
                item = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            self._apply(item)
            self.written += 1

    def _apply(self, item: dict) -> None:
        if item["kind"] == "semantic":
            self._semantic[item["key"]] = {"value": item["value"],
                                           "tier": item["tier"],
                                           "ts_ns": item["ts_ns"]}
        else:
            self._episodic.append(item)
            if len(self._episodic) > self._episodic_cap:
                self._episodic = self._episodic[-self._episodic_cap:]
        if self.persist_dir and self.written % 64 == 0:
            self.persist()

    # -- reads (cache; last writer wins) -------------------------------------
    def get_semantic(self, key: str):
        e = self._semantic.get(key)
        return e["value"] if e else None

    def recent_episodes(self, n: int = 50) -> list:
        return [e["event"] for e in self._episodic[-n:]]

    # -- persistence (periodic/event-driven, NEVER on the combat path) -------
    def persist(self) -> Path | None:
        if not self.persist_dir:
            return None
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        (self.persist_dir / "semantic.json").write_text(
            json.dumps(self._semantic, indent=1))
        with open(self.persist_dir / "episodic.jsonl", "a") as fh:
            for e in self._episodic:
                fh.write(json.dumps(e) + "\n")
        return self.persist_dir

    # -- lifecycle ------------------------------------------------------------
    def start(self) -> None:
        if not self._thread.is_alive():
            self._thread = threading.Thread(target=self._drain, name="memory",
                                            daemon=True)
            self._stop.clear()
            self._thread.start()

    def stop(self, persist: bool = True) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        if persist:
            self.persist()

    def stats(self) -> dict:
        return {"written": self.written, "dropped": self.dropped,
                "qsize": self._q.qsize(), "semantic_keys": len(self._semantic),
                "episodes": len(self._episodic), "ASYNC": True}
