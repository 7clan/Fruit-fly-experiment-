# GATE 6B — autonomous navigation attempt (uploaded short run)

Date: 2026-09-30
Session: `lab_20260930_152203`

Verdict: **INCOMPLETE / NO FORWARD LOCOMOTION**

The run ended after 23.781 s post-READY, so it did not complete the 180 s
6B window. The run was otherwise technically clean.

## Replay evidence

- canonical brain events: 10
- worker errors: 0
- replay drops/errors: 0
- real non-empty outputs: 3
- all three real outputs were `TURN_RIGHT` horizontal mouse commands
- no W key event was emitted
- no APPROACH intention occurred
- the recommended green quest waypoint was detected in 14 fast-vision
  observations; the yellow quest marker was also frequently visible
- centered/bilateral target stimulation reached the canonical brain and
  produced non-zero P9/GF activity plus TURN_RIGHT intentions

## Root cause

The Gate-6B action mapping interpreted P9 TURN_LEFT/TURN_RIGHT as camera
rotation only. In the validated motor interpretation P9 is associated with
forward walking plus ipsiversive turning, so the game avatar could rotate its
view without translating.

## Correction after this attempt

For the next navigation version, TURN_LEFT/TURN_RIGHT now mean:

- bounded horizontal camera turn; **and**
- a bounded W forward pulse.

STOP remains no input. No attacks/interactions/abilities are enabled. Repeated
turn intentions refresh W and re-issue only the bounded steering component.
