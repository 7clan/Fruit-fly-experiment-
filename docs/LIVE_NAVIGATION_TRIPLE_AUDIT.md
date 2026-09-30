# Live navigation triple audit — 2026-09-30

This audit was triggered by a live run where the dashboard showed
MOVEMENT ENABLED but the canonical brain remained at STOP and the executor
reported zero emitted inputs.

## 1. Neural interface / calibration audit

The frozen D8 reference is explicit:

- 40 LC9 neurons per side;
- target sensory drive calibrated at **150 Hz**;
- triangular visual tuning with theta0=15 deg and theta_w=30 deg;
- D10 closed-loop rate schedules repeatedly use values up to 150 Hz;
- at 150 Hz, the frozen D8 calibration produced P9_left≈20 Hz for a left
  target and P9_right≈72 Hz for a right target.

The live adapter had drifted from that reference:

- it capped target sensory drive at **72 Hz**;
- 72 Hz was copied from an observed **P9 response**, not the calibrated LC9
  sensory input rate;
- its bearing tuning used a broad +/-90 deg side ramp instead of the frozen
  D8 15/30-deg tuning.

That under-drove the exact pathway the D9/D10 motor decoder depends on and
explains the observed STOP outputs despite visible targets.

Correction: live fly-channels-v3-d8-exact now mirrors the frozen D8 target
rate and angular tuning. The canonical brain, synapses, and decoder thresholds
are unchanged.

## 2. Vision / waypoint audit

The uploaded screenshots clearly show the real green recommended-quest marker
near the extreme right edge of the gameplay view, with green distance text
below it. The detector was excluding x > 0.92 of the view after an earlier
false-positive fix, so it fell back to the local yellow QUEST marker.

Correction: the real edge marker is allowed again. False-positive rejection
now relies on the observed circle + supporting green distance-text pattern
instead of cropping away valid screen positions.

## 3. Windows input / full-stack audit

The previous Windows SendInput backend did not check the return value from
SendInput, so a future input-injection failure could be silent. Microsoft
documents that SendInput returns the number of inserted events and can fail
or be blocked by UIPI.

Corrections:

- INPUT structures use pointer-sized ULONG_PTR-compatible fields;
- every SendInput call is checked;
- attempts, successes, failures, and last error are reported;
- dashboard shows SendInput success/failure counts;
- a blocked/failed input becomes a visible worker error instead of looking
  like neural inactivity.

## Guard against regression

tests/test_live_d8_interface.py locks the live target-rate ceiling and angular
tuning to brain/data/d7_d10/preregistration.json.

The next active run should therefore distinguish the remaining possibilities:

1. neural path now produces P9/turn/approach -> real W/mouse commands;
2. neural path still does not recruit walk populations -> neural/interface
   issue remains;
3. commands are produced but SendInput fails -> Windows/input problem is
   explicitly reported.
