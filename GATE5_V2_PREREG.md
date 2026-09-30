# GATE 5 v2 — PRE-REGISTRATION (hardware-realistic passive Windows gate)

**Status:** FROZEN before the v2 assessment run.

Date: 2026-09-30

This is a new engineering preregistration. It does **not** modify or erase
the frozen Gate-5 v1 record. Gate-5 v1 remains FAIL / incomplete as recorded
in `GATE5_V1_ASSESSMENT.md`.

The biological brain, sensory interface, and intention decoder remain frozen.
No biological parameter is changed to make this gate pass.

## Question

Can the current Windows application run a useful passive
capture → perception → canonical brain → intention → shadow-action loop on
the target laptop, with replay/evidence sufficient to justify a first
movement-only experiment?

## Assessment

One 120-second game session **after canonical brain READY**.

Development sessions before this preregistration are diagnostic evidence only
and do not count as the v2 assessment.

The user manually exposes a mixture of:

- quest NPC / yellow QUEST marker;
- starter-town Bandits where practical;
- ordinary scenery with no nearby Bandit marker;
- normal camera movement.

No autonomous input is allowed during this session.

## Automatic FAIL conditions

- any non-shadow game input;
- canonical runtime other than `canonical_brian`;
- biological model/interface parameter changes;
- worker crash or non-shutdown worker error;
- missing replay/report/evidence bundle;
- fabricated or post-hoc timestamps.

## Pass criteria — all required

| ID | Criterion | Threshold |
|---|---|---|
| V2-1 | Capture alive | >= 450 published frames during the 120 s post-READY window |
| V2-2 | Fast perception alive | >= 400 fast-perception events; no fast-vision errors |
| V2-3 | Player/HUD observability | health or stamina available in >= 80% of fast observations |
| V2-4 | Role evidence is conservative | saved evidence spot-check shows no persistent HUD/scenery object labeled hostile; visible starter Bandit red-diamond examples produce `hostile_candidate` when present |
| V2-5 | Quest evidence | yellow QUEST examples produce `quest_marker` and/or `quest_npc`; absence is allowed to be explicit unknown/none |
| V2-6 | Canonical brain continuity | >= 8 completed 50 ms biological chunks; bio-time strictly increases; no inter-chunk wall gap > 15 s |
| V2-7 | Brain responsiveness | at least one assessment chunk has non-zero sensory drive and at least one completed chunk has non-zero whole-brain activity or DN activity |
| V2-8 | Replay integrity | brain/perception/action streams recorded with zero replay-recorder drops/errors |
| V2-9 | Input safety | MotorExecutor `inputs_emitted == 0`; all action events are shadow-only |
| V2-10 | Session integrity | total worker errors == 0 on normal shutdown; report + ZIP written |

## Interpretation

Passing v2 authorizes only preparation of Gate 6: low-risk **movement-only**
input. It does not authorize combat, attacks, quest interaction keys, or
long-term unattended play.

Gate 6 receives its own preregistration before input is armed.
