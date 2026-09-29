"""DigitalFlyLab — the applied Windows application package.

Governing architecture: FINAL_ARCHITECTURE.md (authoritative, 2026-09-29).
Platform policy: docs/WINDOWS_RUNTIME_SPEC.md W3 — Windows-specific code
lives behind guarded imports; biology/vision/memory/planning stay
platform-independent.

Layering labels (TIER discipline, docs/MASTER_SPEC.md):
  * lab.brain      — wraps the canonical Shiu/FlyWire v783 brain
                     (TIER: BIOLOGICAL) plus an explicitly-labeled mock
                     runtime for pipeline/hardware-limited situations.
  * lab.perception — ENGINEERED computer vision.
  * lab.world      — ENGINEERED semantic world model / memory / value /
                     planner. These are NOT fly-brain functions.
  * lab.action     — ENGINEERED ability resolver + registry; motor
                     executor is gated (no autonomous input before the
                     corresponding gate passes).

Speed contract (FINAL_ARCHITECTURE.md §3): every real-time component
communicates through lab.bus with bounded drop-oldest queues and
latest-wins state slots. Nothing in the real-time path blocks on
dashboard rendering, disk logging, OCR, guide research or memory
persistence.
"""

__version__ = "0.1.0"
