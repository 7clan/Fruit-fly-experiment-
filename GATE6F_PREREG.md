# GATE 6F — PRE-REGISTRATION (keyboard steering + low-load UI)

**Status:** FROZEN before first 6F live run.
Date: 2026-09-30

The preceding live run established that Windows input now works:
the user observed autonomous forward displacement and the terminal reported
non-zero emitted inputs, SendInput successes, zero SendInput failures, and
zero worker errors.

Gate 6F addresses two remaining engineering problems observed live:

1. movement translated mostly forward instead of visibly steering;
2. the OpenCV dashboard/mirror consumed noticeable CPU and could appear
   unresponsive on the two-core target laptop.

The canonical biological brain and D9 intention thresholds are unchanged.

## Movement translation

Movement-only outputs are hard-limited to keyboard navigation:

- TURN_LEFT  -> W + A, held <= 2.5 s per new canonical brain decision
- TURN_RIGHT -> W + D, held <= 2.5 s per new canonical brain decision
- APPROACH   -> W, held <= 2.5 s
- STOP / RETREAT / ESCAPE -> no movement in this gate

No mouse input, attack, block, interact, ability, dash, menu, or combat input
is permitted by the hard backend.

## Dashboard isolation

- autonomous dashboard uses a cached low-resolution preview;
- dashboard snapshots run at 0.5 Hz;
- capture/fast vision run at 3 Hz, heavy vision at 0.1 Hz;
- autonomous evidence JPEG writing is handled by EvidenceRecorder, not the
  OpenCV renderer;
- dashboard rendering failures disable only the dashboard and MUST NOT stop
  capture, brain, replay, or safety controls.

## Independent safety controls

F8 enable movement
F9 disable movement
F10 refocus the authorized game window
F11 release held movement keys
F12 emergency stop and end run

A >=6 percentage-point observed health drop while movement is enabled is a
movement-only safety exception: held movement is released and autonomy pauses.
It never selects a replacement movement action.

## Functional evidence

A useful 6F run should show:

- at least one non-STOP canonical intention;
- accepted W+A or W+D input when a turn intention occurs;
- visible lateral/diagonal correction rather than forward-only motion;
- SendInput failures = 0;
- dashboard lag/failure cannot terminate the control pipeline;
- any damage event pauses movement instead of causing continued movement into
  combat.

Combat remains unauthorized.
