"""Shared monotonic clock for the whole DigitalFlyLab pipeline.

FINAL_ARCHITECTURE.md §10 (replay): every recorded event — captured frame
reference, CV observation, world state, goal, brain input/output,
intention, ability candidates/scores, selected ability, emitted input,
game outcome, value updates — must be timestamped on ONE clock so the
complete decision chain can be reconstructed and latencies measured
honestly (never fabricated).

Design:
  * The shared timeline is simply the PARENT process's monotonic clock.
  * Each Clock is `now_ns = local_monotonic_ns + offset` with offset 0 in
    the parent. Children adopt an exported parent reading and solve their
    own offset, so child timestamps continue the parent timeline.
  * In-process (threaded) mode: one shared instance, exact by construction.
  * Cross-process alignment is best-effort (spawn latency ~ sub-ms) and is
    only used for multi-process replay stitching; per-process latency
    deltas are always computed on the local monotonic clock and are exact.
  * wall_time_ns() is for human-readable logs only — NEVER latency math.
"""

from __future__ import annotations

import time


class Clock:
    """Monotonic pipeline clock with parent-timeline adoption."""

    def __init__(self, offset_ns: int = 0):
        # now_ns() = time.monotonic_ns() + offset_ns  (offset 0 in parent)
        self.offset_ns = int(offset_ns)

    # -- reads ----------------------------------------------------------
    def now_ns(self) -> int:
        return time.monotonic_ns() + self.offset_ns

    def now_ms(self) -> float:
        return self.now_ns() / 1e6

    def now_s(self) -> float:
        return self.now_ns() / 1e9

    @staticmethod
    def wall_time_ns() -> int:
        """Wall time for human-readable logs only (never latency math)."""
        return time.time_ns()

    # -- shared clock propagation ----------------------------------------
    def export(self) -> dict:
        """Serialize a timeline reading so a child process can adopt it."""
        return {"timeline_ns": self.now_ns()}

    @classmethod
    def adopt(cls, spec: dict) -> "Clock":
        """Create a child clock continuing the exported parent timeline.

        offset_child = parent_timeline_at_export - local_monotonic_now
        so child.now_ns() at adoption equals the parent's exported value
        and advances at the same rate from there.
        """
        parent_timeline = int(spec["timeline_ns"])
        return cls(offset_ns=parent_timeline - time.monotonic_ns())

    # -- math helpers ------------------------------------------------------
    @staticmethod
    def elapsed_ms(t0_ns: int, t1_ns: int | None = None) -> float:
        t1 = time.monotonic_ns() if t1_ns is None else t1_ns
        return (t1 - t0_ns) / 1e6


SHARED_CLOCK = Clock()
