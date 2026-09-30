# GATE 5 v2 — assessment attempt 2

Date: 2026-09-30
Session: `lab_20260930_122519`

Verdict: **FAIL — V2-4 role-evidence conservatism**

This is an engineering verdict only. The canonical biological model and
interfaces are unchanged.

## Automatic measurements

- capture frames: **750** (V2-1 PASS; threshold 450)
- fast-perception events: **711** (V2-2 PASS; threshold 400)
- fast-vision errors: **0**
- health-or-stamina observed: **694 / 711 = 97.6%** (V2-3 PASS)
- canonical brain chunks: **19** (V2-6 count PASS; threshold 8)
- max inter-chunk wall gap: **10.75 s** (V2-6 PASS; threshold 15 s)
- canonical bio-time increased from 0.15 s to 1.05 s monotonically
- non-zero sensory drive + non-zero whole-brain activity occurred
  (V2-7 PASS)
- replay-recorder errors/drops: **0 / 0** (V2-8 PASS)
- MotorExecutor real inputs emitted: **0**; all 19 input events shadow-only
  (V2-9 PASS)
- total worker errors: **0**; report and ZIP written (V2-10 PASS)
- quest-marker observations: 164; quest-NPC track present in 113 fast
  observations (V2-5 supported)

## V2-4 failure

The saved evidence/replay spot-check exposed a persistent false-hostile mode:
unmatched small red markers could be promoted directly to
`hostile_candidate` even without an independently detected humanoid. In
later evidence frames this included a box over/near the player's own avatar,
and left-side HUD red markers could also seed candidate tracks.

That violates the conservative-role-evidence intent of V2-4, so this attempt
is recorded as FAIL rather than treating the other passing metrics as a
global PASS.

## Engineering correction after this failed attempt

- hostile red markers now **only upgrade an independently detected,
  non-player humanoid proposal**;
- unmatched red markers are discarded;
- future evidence saves both raw and annotated per-brain-chunk JPEGs for
  direct role-label review.

The frozen Gate-5 v2 criteria are unchanged. A new assessment under the same
preregistration is required for a PASS.
