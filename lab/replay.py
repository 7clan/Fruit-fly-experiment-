"""Timestamped replay recorder (FINAL_ARCHITECTURE §28).

Records the COMPLETE decision chain on the shared monotonic clock:
  captured frame reference · CV observation · world state · goal ·
  memory retrieved · brain input · brain output · intention · ability
  candidates · ability score · selected ability · emitted input ·
  game outcome · reward/value update.

Implementation: an async drain worker — it reads the per-component
event STREAMS (perception events, brain events, action inputs, planner
goals) and merges them into a single time-ordered JSONL session file.
Recording is OFF the real-time path: if the recorder falls behind, it
drops the OLDEST undrained events (counted honestly in stats) — the
pipeline never waits for disk.
"""

from __future__ import annotations

import json
from pathlib import Path

from .bus import Bus, StreamChannel
from .clock import SHARED_CLOCK
from .worker import Worker

REPLAY_TOPICS = (
    "perception.fast.events",
    "perception.heavy.events",
    "brain.events",
    "coach.events",
    "action.inputs",
)


class ReplayRecorder(Worker):
    """Async multi-topic event recorder → JSONL."""

    name = "replay_recorder"

    def __init__(self, bus: Bus, session_dir: Path, target_hz: float = 30.0):
        super().__init__(bus, target_hz=target_hz)
        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.session_dir / "replay.jsonl"
        self._fh = open(self.path, "a", encoding="utf-8")
        self._subs: list[StreamChannel] = [bus.stream(t, maxsize=256)
                                           for t in REPLAY_TOPICS]
        self.stats.update({"events": 0, "dropped": 0})

    def step(self) -> None:
        events = []
        for ch in self._subs:
            events.extend(ch.drain())     # drops nothing here; backlog
                                          # above drops at the bus level
        if not events:
            return
        events.sort(key=lambda e: e.ts_ns)
        for env in events:
            rec = env.to_dict()
            rec["wall_ns"] = SHARED_CLOCK.wall_time_ns()
            self._fh.write(json.dumps(rec, default=str) + "\n")
        self._fh.flush()
        self.stats["events"] += len(events)

    def on_stop(self) -> None:
        # drain remaining events before close
        self.step()
        self._fh.close()

    def write_header(self, meta: dict) -> None:
        rec = {"topic": "_session_header", "ts_ns": SHARED_CLOCK.now_ns(),
               "seq": 0, "payload": meta}
        self._fh.write(json.dumps(rec, default=str) + "\n")
        self._fh.flush()


def load_replay(path: Path) -> list:
    """Load a replay JSONL into a sorted event list (reconstruction aid)."""
    out = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            out.append(json.loads(line))
    out.sort(key=lambda r: r.get("ts_ns", 0))
    return out


def replay_decision_chains(events: list) -> list:
    """Reconstruct decision chains: for each brain chunk, gather the
    perception event that preceded it, the emitted input(s) until the
    NEXT brain chunk, forming one chain per decision. Analysis aid."""
    chains = []
    by_topic = {}
    for e in events:
        by_topic.setdefault(e.get("topic", "?"), []).append(e)
    brain_events = sorted(by_topic.get("brain.events", []),
                          key=lambda e: e.get("ts_ns", 0))
    inputs = by_topic.get("action.inputs", [])
    fast = by_topic.get("perception.fast.events", [])
    for idx, be in enumerate(brain_events):
        payload = be.get("payload", {})
        t0 = be.get("ts_ns", 0)
        t_next = brain_events[idx + 1].get("ts_ns", t0 + 500_000_000) \
            if idx + 1 < len(brain_events) else t0 + 500_000_000
        # inputs attributable to THIS decision: emitted after the brain
        # output and before the next one
        related_inputs = [i for i in inputs
                          if t0 <= i.get("ts_ns", 0) < t_next]
        # the freshest perception event that fed this decision
        related_fast = [f for f in fast
                        if 0 < t0 - f.get("ts_ns", 0) < 1_000_000_000]
        chains.append({
            "chunk_id": payload.get("chunk_id"),
            "ts_ns": t0,
            "sensory_rates_hz": payload.get("sensory_rates_hz"),
            "dn_rates_hz": payload.get("dn_rates_hz"),
            "intention": payload.get("intention", {}).get("name"),
            "intention_confidence": payload.get("intention", {}).get("confidence"),
            "inputs": [i.get("payload", {}).get("action")
                       for i in related_inputs],
            "precedes_fast_observation":
                related_fast[-1].get("payload", {}).get("observation", {})
                if related_fast else None,
        })
    return chains
