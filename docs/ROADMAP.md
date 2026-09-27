# ROADMAP — from "move toward target" to autonomous long-term game progression

This is the master phase plan. Each phase has a DEFINED GRADUATION GATE: a
measurable result that must be demonstrated before the next phase begins.
No phase is skipped; the game is touched only from Phase 6 onward.

Research question (the whole project exists to measure this):

> "How much useful autonomous game-playing behavior can be produced by
> combining a biological behavioral controller with computer vision, memory,
> planning, reinforcement learning, and virtual action translation?"

The answer is not assumed to be positive. The control conditions exist to
make the answer measurable either way.

---

## Milestone map

| Phase | Name | Environment | Uses real fly? | Uses real game? |
|-------|------|-------------|----------------|-----------------|
| 0 | Apparatus + synthetic validation | 2D artificial | no | no |
| 1 | Real-fly observation | arena camera | yes (video/live) | no |
| 2 | Stimulus preference assay | arena + cues | yes | no |
| 3 | Closed loop, 2D open field | 2D artificial | yes | no |
| 4 | Obstacles, shaped reward, first RL | 2D artificial | yes | no |
| 5 | Multi-step tasks + memory | 2D artificial | yes | no |
| 6 | Screen perception (GPO Phase A–B) | real game, observe only | no | observe only |
| 7 | Game movement (GPO Phase C–D) | real game | yes | input |
| 8 | Interaction (GPO Phase E–F) | real game | yes | input |
| 9 | Simple objectives (GPO Phase G) | real game | yes | input |
| 10 | Multi-step objectives (GPO Phase H) | real game | yes | input |
| 11 | Equipment/ability discovery | real game | yes | input |
| 12 | Exploration system (GPO Phase I) | real game | yes | input |
| 13 | Strategy library + selection | real game | yes | input |
| 14 | Long-term progression optimization (GPO Phase J–K) | real game | yes | input |

Autonomy ladder (measured at every phase from 7 on):
0 = human controls everything; 1 = human gives exact actions;
2 = human gives a sequence; 3 = human gives a task; 4 = human gives a goal;
5 = human gives only "maximize long-term progression".

---

## Phase 0 — Apparatus + synthetic validation (DELIVERED)

WHAT: the complete pipeline with a SIMULATED fly (ground truth known).
Components: synthetic/webcam/video sources, MOG2 + frame-difference tracker,
4-point calibration, behavior classifier (zones, MOVING/PAUSED), action
decoder (motion-direction + zone-dwell schemes), 2D game, sparse+shaped
reward, SQLite logging, learning curves, bootstrap statistics, five control
conditions.

WHY: validate that the MEASUREMENT SYSTEM works before any animal is
involved. If the apparatus cannot detect a known-injected "fly preference"
(the synthetic cue taxis), it could never detect a real one.

GATE (passed 2026-09-27, 30 trials/condition): success-rate bootstrap CIs
exclude 0 for `fly_cue − random` and `fly_cue − fly_cue_shuffled`.
Measured: random 3.3%, fixed 100%, fly_random 6.7%, fly_cue 76.7%,
fly_cue_shuffled 3.3%; all key CIs p<0.0001.

## Phase 1 — Real-fly observation (no game)

WHAT: record 5–10 videos (5 min each) of a real fly in the arena; run the
tracker offline (`--source video`); manually annotate >= 200 sampled frames
across videos; compute tracker error vs annotation.

GATE: tracking present on >= 95% of frames; mean position error < 5% of
arena width; behavior statistics (walk/pause bout structure) extracted and
documented per fly.

## Phase 2 — Stimulus preference assay (no game)

WHAT: present each candidate benign cue (dark stripe at edge, LED panel
pattern, brightness gradient) at randomized edges; measure approach latency,
edge-occupancy time, and first-choice frequency per cue type.

WHY: the closed loop NEEDS a stimulus the fly reliably approaches. This is
a property of the ANIMAL, so it must be measured, not assumed.

GATE: at least one cue with approach probability > chance (binomial or
permutation test, CI excluding chance). FALLBACK if none: pure
movement-direction decoding (no cue) — the fly's own walking direction
drives the player; performance will be near floor in goal-directed tasks
but can still drive free-exploration behaviors (Phase 12). The fallback
path is architecturally supported (`cue_mode: none`).

## Phase 3 — Closed loop with a real fly (2D open field)

WHAT: the Phase-1 demo conditions, but with a REAL fly: fly_cue vs
fly_random-equivalent (no-preference flies / no-cue sessions) vs random
controller vs fixed controller. Multiple sessions per fly across days.
Decoder parameters FROZEN across sessions (attribution requirement).

GATE: fly-controlled system success rate > random controller (bootstrap CI),
stable across >= 3 sessions. If not met: document, analyze (tracking
quality? cue preference variance? emission cadence?), iterate Phase 1–2
assays. The system must NOT silently "improve" via decoder retuning during
this phase — that would be the computer learning, not the fly.

## Phase 4 — Obstacles + shaped reward + first learned component

WHAT: obstacles scenario; reward shaping on; a LEARNED element enters:
the computer learns (tabular Q or tiny MLP) a target-selection policy on
top of fly movement. WHAT IS LEARNED vs WHAT IS BIOLOGICAL becomes
explicit in the analysis.

GATE: obstacle scenario success above no-obstacle-transfer baseline;
learning curve of the RL component documented separately from fly metrics.

## Phase 5 — Multi-step tasks + memory

WHAT: composite objectives (reach A, then B, then return), sequence memory,
task state machine in the 2D world; episodic/semantic memory modules
(world_model/ package introduced).

GATE: multi-step success rate significantly above chance chaining; transfer
test (new layout, same rule) above zero-transfer baseline.

## Phase 6 — Screen perception ONLY (GPO Phase A–B; no input)

WHAT: capture the game screen; build the screen-state machine
(UNKNOWN_SCREEN / MENU / LOADING / GAMEPLAY / OTHER) via CALIBRATION, not
hard-coded coordinates; detect player, HUD numbers (level/health via OCR),
obvious objects. All game mechanics remain UNKNOWN — see docs/GPO_PLAN.md.

GATE: screen-state classifier accuracy > 95% on a labeled session
(>= 30 min of varied states); HUD readback latency < 500 ms; perception
runs at >= 10 FPS on the target machine.

## Phase 7 — Game movement (GPO Phase C–D)

WHAT: fly controls a tiny subset of movement inside the real game via the
virtual controller; Autonomy 0–1 (human supervises; human provides exact
action targets). Requires: input-automation method chosen + ToS/compliance
review completed BY THE USER (see GPO_PLAN risk register), failure
detection (stuck screen, death, menu), and recovery to safe state.

GATE: fly-driven movement produces measurable, sustained character
displacement over 10-minute sessions; failure-recovery loop handles >= 90%
of stuck states without human help.

## Phase 8 — Interaction (GPO Phase E–F)

WHAT: approach + interact sequences on objects discovered on screen; the
ACTION->OBSERVED CONSEQUENCE->REWARD CHANGE exploration protocol for
UNKNOWN objects.

GATE: at least one interaction loop reliably produces an observable state
change; unknown-object probe protocol executes without deadlock.

## Phase 9 — Simple objectives (GPO Phase G)

WHAT: single objectives observable on screen (e.g., reach a place, interact
repeatedly, survive N seconds); progression deltas (level/experience as
visible on screen) feed the reward.

GATE: >= 1 hour of Autonomy-2 operation with nonzero measured progression
and no unrecovered failures.

## Phase 10 — Multi-step objectives (GPO Phase H)

WHAT: NPC -> objective -> destination -> return chains, with each step's
mechanics DISCOVERED EXPERIMENTALLY from screen observation (nothing
hard-coded beyond the generic chain template).

GATE: a full chain completed autonomously at least once per session across
3 sessions; per-step success rates logged.

## Phase 11 — Equipment / ability discovery

WHAT: before/after state differencing on equip changes; item effect memory
(health delta, visible stat changes); ability acquisition and testing
protocol; knowledge graph of discovered relations.

GATE: system correctly predicts the observable effect of >= 1 item/ability
it has tested, on a retest.

## Phase 12 — Exploration system (GPO Phase I)

WHAT: dedicated exploration mode; discovery of new locations/entities/
interactions; spatial memory; exploration/exploitation scheduler with
configurable rates.

GATE: exploration sessions produce measurable new-knowledge entries
(knowledge-graph growth) without human guidance.

## Phase 13 — Strategy library + selection

WHAT: abstract strategy templates (repeat-nearby-task, travel, fight,
explore, obtain-equipment) filled with discovered facts; per-strategy
statistics (progression/time, risk, failure rate); predicted-vs-observed
outcome updates.

GATE: strategy selector outperforms (progression/time) a random-strategy
baseline over >= 3 sessions.

## Phase 14 — Long-term progression optimization (GPO Phase J–K)

WHAT: Autonomy 5 — the system receives only "maximize long-term
progression" and chooses what to investigate, where to go, which strategies
to repeat or abandon. Human intervention only for safety/compliance.

GATE: sustained autonomous progression across >= 3 multi-hour sessions;
strategy-mix shifts correlate with measured progression rates; full
attribution analysis published (fly contribution vs computer contribution,
with control conditions).

---

## What is deliberately NOT in this roadmap

- No neural implants, optogenetics, or any invasive technique. Ever.
- No hidden-game-data access, no game APIs, no memory inspection: screen
  observation only, per the source policy (docs/GPO_PLAN.md).
- No hard-coded game mechanics. Anything not visible on the permitted page
  is UNKNOWN and must be discovered experimentally.
- No claiming the fly "understands" the game. The fly contributes movement
  behavior; the computer contributes everything else; every phase report
  separates the two explicitly.
