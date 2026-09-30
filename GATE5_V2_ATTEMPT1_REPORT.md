# GATE 5 v2 — assessment attempt 1

Date: 2026-09-30
Session: `lab_20260930_030637`

Verdict: **INCOMPLETE / FAIL TO SATISFY v2 PASS CRITERIA**

The session was clean but did not run the required 120 seconds after
canonical brain READY. The replay timeline contains only about 12 seconds
of post-READY activity, so this attempt cannot count as a passing v2
assessment.

## Observed

- 436 captured frames
- 374 fast-perception events
- 76 heavy-perception events
- 1 completed canonical brain chunk
- canonical runtime: `canonical_brian`
- transport: `subprocess_pipe`
- 0 worker errors
- 0 real input emissions
- replay recorder drops/errors: 0
- health detected in 357 / 374 fast observations
- stamina detected in 356 / 374 fast observations
- quest-marker target in 281 / 374 fast observations
- quest-NPC track present in 254 / 374 fast observations
- hostile candidate present in 116 / 374 fast observations
- one completed brain chunk had non-zero target sensory drive and non-zero
  whole-brain activity, but DN readout was all zero and intention was STOP
- no evidence JPEGs were persisted because the single brain output arrived
  essentially at shutdown and the dashboard did not render another
  post-chunk snapshot

## v2 criteria

- V2-1 capture >=450: FAIL (436)
- V2-2 fast perception >=400: FAIL (374)
- V2-3 HUD observability >=80%: PASS
- V2-4 conservative role evidence: NOT ASSESSABLE from saved JPEGs
- V2-5 quest evidence: PASS in replay
- V2-6 >=8 canonical chunks: FAIL (1)
- V2-7 non-zero sensory + neural activity: PASS
- V2-8 replay integrity: PASS
- V2-9 zero real input: PASS
- V2-10 zero worker errors + report/ZIP: PASS

The frozen v2 criteria remain unchanged. A new qualifying 120-second
post-READY session may be run under the same preregistration; this failed
attempt remains in the record.
