# Gate-5 dry-run evidence: canonical brain in the live lab pipeline (synthetic source)

Session: 2026-09-29 10:19 UTC+8 sandbox (2-core, 4.1 GB) — dry-run stage
per GATE5_PREREG.md §4 (source=synthetic; does NOT count toward game-
window sessions).

- runtime: canonical_brian (138,639 neurons, Shiu v783, pinned 91bdd1e)
  through the validated D7 interactive machinery, wrapped init-once by
  lab/brain/runtime.py; chunk 50 ms; decoder intent-decoder-v1
- 40 s wall drive: 650 frames captured, 73 canonical chunks, 0 errors
- throughput 0.104 bio-s per wall-s (reproduces the D7 measured 0.099
  on the same class of box — wrapper adds no overhead)
- intentions: TURN_RIGHT x30, STOP x18, ESCAPE x13, TURN_LEFT x10,
  RETREAT x2 — from DN readouts only; looming drive 0..54 Hz tracks the
  synthetic enemy wind-up cycles (perception-driven only after the
  looming-honesty fix: helper context is NEVER wired into sensory rates)
- executor: SHADOW mode (0 inputs emitted; 1118 hold-refreshes; PASSIVE)
- files: replay.jsonl (full decision chain), session_report.json
