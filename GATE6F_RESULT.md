# GATE 6F — live result

Date: 2026-09-30
Session: `lab_20260930_232120`

Verdict: **PASS — movement steering + damage safety**

Evidence from the uploaded session bundle:

- 45.141 s post-READY run
- canonical brain: 9 worker steps / 8 recorded brain events
- all 8 recorded intentions: TURN_RIGHT
- navigation action mapping: W + D
- 3 real repeated navigation input events after movement was enabled
- Win32 SendInput: 14 attempts / 14 successes / 0 failures
- motor worker errors: 0
- total worker errors: 0
- player health fell from ~0.82 to ~0.75 while movement was active
- damage-safety rule disabled autonomy and released held movement
- final held-key set: empty
- replay/evidence completed with no recorder drops

The saved evidence shows the avatar translated from the central path toward
the right-side building area before contact with a hostile NPC. The run then
recorded the health drop and movement pause rather than continuing deeper
into combat.

This pass authorizes continued navigation engineering. It does NOT authorize
combat yet: the validated neural decoder still has no DEFEND/ATTACK intention
mapping. Combat remains a separate Gate-7 validation problem.
