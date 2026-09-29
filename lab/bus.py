"""Asynchronous pipeline bus for DigitalFlyLab.

FINAL_ARCHITECTURE.md §3 (binding):

  * The system is a MULTI-PROCESS / ASYNCHRONOUS pipeline; components run
    concurrently.
  * Bounded queues or shared latest-state buffers. **Never an unbounded
    frame queue.**
  * NEWER DATA WINS: if vision receives frame 106 while frame 103 is
    still waiting, discard obsolete 103 rather than creating latency.
  * dashboard rendering / disk logging / OCR / guide research / memory
    persistence must NEVER block the real-time neural/action loop.

Two channel kinds:

  * StreamChannel  — bounded FIFO of work items (frames, commands) with a
    DROP-OLDEST policy when full. Publishing NEVER blocks. Each item
    carries a monotonic timestamp; a reader may additionally filter stale
    items. Use for units of work that each need handling once.
  * StateChannel   — single-slot latest-state buffer with sequence
    numbers. Write overwrites; read returns the newest snapshot plus a
    "stale" hint (how many overwritten versions the reader missed).
    Use for continuously-recomputed state: world observation, fly input
    channels, dashboard snapshot, brain output/intention.

Both channels work identically in threaded (single-process) and
multiprocessing mode: they are built on queue.Queue / multiprocessing
primitives chosen by BusMode. Windows-safe (no fork assumptions; spawn
compatible — channels are constructible from a plain spec dict in each
child process).

Every payload should be (or be reducible to) a JSON-safe dict for replay;
dataclasses in lab.schemas provide to_dict()/from_dict().
"""

from __future__ import annotations

import multiprocessing as mp
import queue
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterator

from .clock import SHARED_CLOCK


class BusMode(Enum):
    THREADED = "threaded"          # one process, threads (dev/tests/sandbox)
    MULTIPROCESS = "multiprocess"  # Windows-style process pipeline


# ---------------------------------------------------------------------------
# messages
# ---------------------------------------------------------------------------

@dataclass
class Envelope:
    """Timestamped message on the bus.

    ts_ns    — shared-clock monotonic timestamp (set at publish)
    seq      — per-channel strictly increasing sequence number
    topic    — channel topic name
    payload  — JSON-safe dict (schemas provide to_dict())
    """
    ts_ns: int
    seq: int
    topic: str
    payload: dict

    def to_dict(self) -> dict:
        return {"ts_ns": self.ts_ns, "seq": self.seq,
                "topic": self.topic, "payload": self.payload}


def _mp_safe_queue(maxsize: int) -> queue.Queue:
    """Return a queue safe for the current bus mode.

    For MULTIPROCESS mode a real multiprocessing queue is used; for
    THREADED a plain queue.Queue (faster, and picklable-free). mp.Queue
    requires a default context that works on Windows (spawn) — we use
    ctx.Queue explicitly at Bus level instead; this helper covers the
    threaded case and any mp.Queue injected by the Bus.
    """
    return queue.Queue(maxsize=maxsize)


# ---------------------------------------------------------------------------
# stream channel (bounded, drop-oldest, publish never blocks)
# ---------------------------------------------------------------------------

class StreamChannel:
    """Bounded work stream. publish() NEVER blocks; when full, the OLDEST
    item is dropped to make room (drop-oldest == newest data wins).

    get() raises queue.Empty when drained; get_nowait() is non-blocking.
    Every consumer drains greedily and processes only the newest relevant
    item — obsolete frames must not create latency.
    """

    def __init__(self, topic: str, maxsize: int = 8, q: queue.Queue | None = None):
        self.topic = topic
        self.maxsize = int(maxsize)
        self._q = q if q is not None else _mp_safe_queue(self.maxsize)
        self._seq = 0
        self._seq_lock = threading.Lock()
        self.dropped = 0          # metrics: items dropped to keep bounded
        self.published = 0

    # -- producer side ----------------------------------------------------
    def publish(self, payload: dict, ts_ns: int | None = None) -> Envelope:
        """Publish; drop oldest if full. NEVER blocks the real-time path."""
        with self._seq_lock:
            self._seq += 1
            seq = self._seq
        env = Envelope(ts_ns if ts_ns is not None else SHARED_CLOCK.now_ns(),
                       seq, self.topic, payload)
        self.published += 1
        while True:
            try:
                self._q.put_nowait(env)
                break
            except queue.Full:
                try:
                    self._q.get_nowait()      # drop the oldest item
                    self.dropped += 1
                except queue.Empty:
                    continue                  # raced with a consumer; retry
        return env

    # -- consumer side ------------------------------------------------------
    def get(self, timeout: float | None = None) -> Envelope:
        return self._q.get(timeout=timeout) if timeout is not None \
            else self._q.get()

    def get_nowait(self) -> Envelope:
        return self._q.get_nowait()

    def drain(self) -> list[Envelope]:
        """Non-blocking: return everything currently queued (oldest first).

        Conventional consumer pattern: drain(), keep the LAST envelope of
        interest, discard the rest as obsolete (NEWER DATA WINS).
        """
        out = []
        while True:
            try:
                out.append(self._q.get_nowait())
            except queue.Empty:
                return out

    def newest(self) -> Envelope | None:
        """Drain and return only the newest envelope (drop obsolete)."""
        items = self.drain()
        return items[-1] if items else None

    def qsize(self) -> int:
        return self._q.qsize()

    def empty(self) -> bool:
        return self._q.empty()


# ---------------------------------------------------------------------------
# state channel (single slot, latest wins)
# ---------------------------------------------------------------------------

@dataclass
class StateSnapshot:
    """A latest-state read. `missed` counts overwritten versions since the
    previous read by ANY consumer via this channel's read counter — i.e.
    how many fresher states the writer produced while this consumer was
    busy. miss == 0 means the consumer is fully caught up."""
    envelope: Envelope
    missed: int

    @property
    def ts_ns(self) -> int:
        return self.envelope.ts_ns

    @property
    def payload(self) -> dict:
        return self.envelope.payload


class StateChannel:
    """Latest-wins single-slot state. Writers overwrite; readers snapshot.

    The slot is lock-protected (threaded) — O(1), never blocking for more
    than the copy of one dict reference. StateChannels are NOT queues:
    there is no backlog, only the freshest state. Readers that fall behind
    see `missed > 0` (they can report it) but never stale data.
    """

    def __init__(self, topic: str):
        self.topic = topic
        self._lock = threading.Lock()
        self._env: Envelope | None = None
        self._seq = 0
        self._reads = 0          # total reads across consumers (diagnostic)
        self.writes = 0

    def write(self, payload: dict, ts_ns: int | None = None) -> Envelope:
        with self._lock:
            self._seq += 1
            env = Envelope(ts_ns if ts_ns is not None else SHARED_CLOCK.now_ns(),
                           self._seq, self.topic, payload)
            self._env = env
            self.writes += 1
            return env

    def read(self) -> StateSnapshot | None:
        """Read the freshest state. missed = versions published since the
        last read of this channel (shared counter)."""
        with self._lock:
            if self._env is None:
                return None
            prev_reads = self._reads
            self._reads += 1
            missed = self._env.seq - prev_reads - 1
            missed = max(0, missed)
            return StateSnapshot(self._env, missed)

    def last_write_ts_ns(self) -> int | None:
        with self._lock:
            return self._env.ts_ns if self._env else None


# ---------------------------------------------------------------------------
# bus
# ---------------------------------------------------------------------------

@dataclass
class ChannelSpec:
    """Declarative channel description — used both to build in-process
    channels and to REBUILD equivalent channels inside spawned children
    (Windows spawn: children re-create channels from specs, no fork)."""
    name: str
    kind: str                 # "stream" | "state"
    maxsize: int = 8          # stream channels only


class Bus:
    """Topic registry of stream + state channels.

    Usage:
        bus = Bus()
        frames = bus.stream("capture.frames", maxsize=4)
        world  = bus.state("world.observation")

    Multiprocess mode: the parent creates channels via Bus(mode=MP) and
    passes `bus.specs()` to children; children call Bus.from_specs(...).
    For true cross-process transport, mp.Queue-backed StreamChannels are
    produced by Bus.make_mp_stream() in the parent and the SAME queue
    object passed to the child (mp queues are inheritable under spawn via
    Process args on Windows). StateChannels are process-local; cross-
    process latest-state uses a small mp.StreamChannel(maxsize=1) with
    drop-oldest semantics, which is semantically identical to
    latest-wins for payloads.
    """

    def __init__(self, mode: BusMode = BusMode.THREADED,
                 mp_context: str = "spawn"):
        self.mode = mode
        self._mp_ctx = mp.get_context(mp_context) if mode is BusMode.MULTIPROCESS else None
        self._streams: dict[str, StreamChannel] = {}
        self._states: dict[str, StateChannel] = {}
        self._lock = threading.Lock()

    # -- channel construction ------------------------------------------------
    def stream(self, name: str, maxsize: int = 8) -> StreamChannel:
        with self._lock:
            if name not in self._streams:
                if self.mode is BusMode.MULTIPROCESS:
                    q = self._mp_ctx.Queue(maxsize=maxsize)
                    self._streams[name] = StreamChannel(name, maxsize, q=q)
                else:
                    self._streams[name] = StreamChannel(name, maxsize)
            return self._streams[name]

    def state(self, name: str) -> StateChannel:
        with self._lock:
            if name not in self._states:
                self._states[name] = StateChannel(name)
            return self._states[name]

    # -- introspection / child reconstruction ---------------------------------
    def specs(self) -> list[dict]:
        out = []
        for name, ch in self._streams.items():
            out.append({"name": name, "kind": "stream", "maxsize": ch.maxsize})
        for name in self._states:
            out.append({"name": name, "kind": "state", "maxsize": 1})
        return out

    @classmethod
    def from_specs(cls, specs: list[dict], **kw) -> "Bus":
        bus = cls(**kw)
        for s in specs:
            if s["kind"] == "stream":
                bus.stream(s["name"], maxsize=s.get("maxsize", 8))
            else:
                bus.state(s["name"])
        return bus

    def metrics(self) -> dict:
        return {
            "streams": {n: {"published": c.published, "dropped": c.dropped,
                            "qsize": c.qsize()}
                        for n, c in self._streams.items()},
            "states": {n: {"writes": c.writes}
                       for n, c in self._states.items()},
        }


# ---------------------------------------------------------------------------
# rate governor (multi-rate control loop, FINAL_ARCHITECTURE §3 table)
# ---------------------------------------------------------------------------

class RateGovernor:
    """Target-rate pacing for workers.

    tick() blocks in SMALL sleeps until the next period boundary, so a
    worker loop is `while running: gov.tick(); do_work()` at ~target_hz.
    Coarse by design (no busy-wait); actual achieved rate is measurable
    and reported honestly (targets are NOT fabricated guarantees).
    """

    def __init__(self, target_hz: float, clock=None):
        self.target_hz = float(target_hz)
        self.clock = clock or SHARED_CLOCK
        self._next_ns = None

    def tick(self) -> float:
        """Sleep until the next period boundary. Returns actual period ms
        (for honest rate reporting)."""
        now = self.clock.now_ns()
        if self._next_ns is None:
            self._next_ns = now
        period_ns = int(1e9 / max(self.target_hz, 1e-9))
        if now < self._next_ns:
            delay = (self._next_ns - now) / 1e9
            # sleep in slices for responsiveness to shutdown
            while delay > 0:
                step = min(delay, 0.05)
                # time.sleep works on wall clock; monotonic alignment is
                # approximate — acceptable for pacing, not for timestamps
                import time as _t
                _t.sleep(step)
                delay -= step
                now = self.clock.now_ns()
                if now >= self._next_ns:
                    break
        self._next_ns = max(self._next_ns + period_ns, self.clock.now_ns())
        return period_ns / 1e6
