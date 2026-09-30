# GATE 5 v2 — assessment attempt 3

Date: 2026-09-30
Session: `lab_20260930_142333`

Verdict: **FAIL — V2-6 canonical-brain continuity**

## Automatic measurements

- capture frames: **683**
- fast-perception events: **589**
- fast-vision errors: **0**
- health observed: **504 / 589**
- stamina observed: **482 / 589**
- quest-marker observations: **337 / 589**
- quest-NPC track present: **291 / 589**
- hostile-candidate observations: **24 / 589**, max one candidate at a time
- canonical brain worker steps: **6**
- recorded brain events: **5**
- recorded live chunk wall times: **6.53, 11.19, 12.35, 17.53, 16.32 s**
- max recorded inter-brain-event gap: **17.61 s**
- canonical runtime: `canonical_brian`
- transport: `subprocess_pipe`
- worker errors: **0**
- replay-recorder errors/drops: **0 / 0**
- MotorExecutor real inputs emitted: **0**

## Criteria

- V2-1 capture >=450: PASS
- V2-2 fast perception >=400: PASS
- V2-3 HUD observability >=80%: PASS
- V2-4 conservative role evidence: improved; saved annotated brain-chunk
  evidence contains no obvious persistent false-hostile box, but this does
  not rescue the throughput failure
- V2-5 quest evidence: PASS
- V2-6 >=8 completed chunks and no gap >15 s: **FAIL** (6 worker steps,
  5 recorded events; max recorded gap 17.61 s)
- V2-7 neural responsiveness: PASS (non-zero target drive produced
  whole-brain activity and TURN_RIGHT)
- V2-8 replay integrity: PASS (zero recorder drops/errors)
- V2-9 zero autonomous input: PASS
- V2-10 session integrity: PASS (zero worker errors; report + ZIP written)

## Interpretation

The remaining blocker in this attempt is hardware/live-workload throughput,
not a transport crash. The canonical brain is functional and responsive but
the full live workload can stretch a 50 ms biological chunk beyond the v2
15-second continuity limit.

The next engineering step is to reduce CPU contention around the unchanged
canonical brain before another assessment. No biological parameter or
decoder threshold is changed.
