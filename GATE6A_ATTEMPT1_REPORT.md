# GATE 6A — active movement attempt 1

Date: 2026-09-30
Session: `lab_20260930_150026`

Verdict: **FAIL — no observable or actual movement command**

The user reported no autonomous movement. Replay confirms that result.

- 45.0 s post-READY active window
- 7 brain-worker steps / 6 recorded brain events
- all 6 recorded canonical intentions were `STOP`
- all six action records had empty bindings and zero mouse motion
- therefore **zero actual movement commands** were requested
- worker errors: 0

The previous `inputs_emitted=6` statistic was misleading: the executor was
incrementing the counter for non-shadow STOP/no-op events. That accounting bug
has been corrected so only non-empty keyboard/mouse output increments the
counter.

Root cause: the live sensory encoder discarded `TARGET_CENTER` because the
verified D8 biological entry interface exposes left/right visual populations
only. In this session the target produced weak, nearly symmetric left/right
drive (roughly 2–9 Hz) and no DN walk/turn activity, so the decoder correctly
failed safe to STOP.

Engineering corrections after this failed attempt:

1. GPO's bright-green recommended-quest waypoint is now recognized as a
   navigation target.
2. `fly-channels-v2-center-bilateral` represents a centered target as equal
   bilateral drive into the existing verified D8 left/right sensory entry
   populations. The biological network is unchanged.
3. TURN intentions use bounded horizontal camera movement rather than A/D
   strafing.
4. APPROACH produces a visible but bounded W pulse.
5. Output accounting only counts non-empty emitted commands.

Gate 6A remains a failed historical attempt.
