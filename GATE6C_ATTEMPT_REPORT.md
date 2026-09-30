# GATE 6C — uploaded autonomous navigation attempt

Date: 2026-09-30
Session: `lab_20260930_153541`

Verdict: **INVALID AS A CONTROL TEST / ARMING-TIMING BUG**

The uploaded replay is highly informative but cannot be accepted as a valid
post-arm autonomous navigation test.

## What happened

- 132 canonical brain events were recorded.
- Intentions: 103 TURN_RIGHT, 29 STOP.
- 103 non-empty navigation action records were generated.
- The engineered recommended-waypoint detector fired in 684 fast-vision
  observations.
- Worker errors: 0.

However, the application created the MotorExecutor in active mode before the
main thread had declared brain READY, focused Roblox, and printed
`navigation armed`.

Of the 103 non-empty navigation commands, **101 occurred before the recorded
post-READY/armed assessment start**. Only one non-empty command occurred
inside the reported 14.6 s assessment window (one more appeared during
shutdown).

Therefore most SendInput events may have gone to whatever window held focus,
not necessarily Roblox. This is an application gating defect, not evidence
that autonomous GPO movement succeeded.

The run also exposed a waypoint false positive: the green waypoint detector
repeatedly selected a component at x≈0.972 of the detector image, whereas
the real GPO green quest waypoint observed in screenshots is a circle with
green distance text below it and is typically in the gameplay field.

## Corrections after this attempt

1. movement executor starts DISARMED even when autonomous navigation is
   requested;
2. canonical BrainWorker exposes an explicit ready event;
3. Roblox focus must succeed before autonomy is enabled;
4. F12 watcher is armed before movement input;
5. run deadline and reported assessment duration now use the same shared
   monotonic clock;
6. repeated TURN intentions re-press W after the prior pulse has expired;
7. waypoint detection now requires the observed green-circle + green-distance
   text pattern and rejects the right-edge HUD region;
8. headless autonomous sessions now save raw + annotated evidence frames.

This historical attempt remains invalid rather than being counted as a pass.
