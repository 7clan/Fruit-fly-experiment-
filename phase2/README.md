# Phase 2 — Real-Fly Behavioral Validation (non-invasive observation)

**Status: apparatus BUILT, pipeline VALIDATED on synthetic ground truth
(SA1–SA11 all PASS). Real-fly data: NOT YET COLLECTED — every real-fly
outcome is UNKNOWN until the hardware protocol is executed.**

Phase 2 asks exactly one question: *does a real fruit fly produce
sufficiently reliable, repeatable observable behavior that can be converted
into a small action vocabulary?* It is answered by NON-INVASIVE observation
only — no surgery, implants, genetic modification, toxic substances,
deliberate injury, extreme temperatures, or any harmful manipulation. The
real fly is NEVER connected to game input in this phase.

| Document | What it is |
|---|---|
| `PHASE2_GATE_PREREGISTRATION.md` | **Read first.** Quantitative gate PHASE2-GATE-1.0.0, fixed BEFORE any Phase-2 code/data existed (thresholds, vocabulary ladder, diagnosis map, software acceptance SA1–SA11) |
| `PHASE2_PROTOCOL.md` | Real-fly experimental protocol: apparatus, welfare rules, session structure, annotation protocol |
| `PHASE2_REPORT.md` | Honest status report (what was built / what the real fly demonstrated / what is UNKNOWN) |
| `config/gate.yaml` | Machine-readable pre-registered gate (single source of truth) |
| `config/tracking.yaml` | Detector/recording parameters (apparatus level) |

## Quick start

```bash
# 1. validate the MEASUREMENT CODE on synthetic ground-truth videos (~2 min)
#    (software test only — NOT evidence about real flies)
python phase2/run_phase2.py validate

# 2. record a live session on the camera machine (after reading the protocol)
python phase2/run_phase2.py record --camera 0 --minutes 10 --label baseline \
    --fly-id F01 --px-per-mm 6.7 \
    --arena '{"type":"circle","center_px":[640,360],"radius_px":450}'
python phase2/run_phase2.py record --minutes 10 --label stimulus \
    --stimulus --session-index 0 --fly-id F01 --px-per-mm 6.7 \
    --arena '{"type":"circle","center_px":[640,360],"radius_px":450}'

# 3. track + annotate + analyze + gate
python phase2/run_phase2.py track   data/sessions/<dir>
python phase2/run_phase2.py annotate data/sessions/<dir> --n 50
python phase2/run_phase2.py analyze data/sessions/<a> data/sessions/<b> ... \
    --out results/first_analysis
python phase2/run_phase2.py gate --analysis results/first_analysis \
    --annotations data/sessions/<a>/annotations.csv ...
```

## What the software measures (all pre-registered constants in gate.yaml)

- continuous recording into immutable session dirs (video + metadata)
- per-frame position detection with confidence; px→mm calibration; arena
  geometry (circular/rectangular); tracking coverage
- velocity, movement direction (circular statistics, moving frames only)
- pauses and movement bouts (hysteresis speed thresholds)
- wall/edge interactions (distance, wall-following, contact rate)
- stimulus timing read back from in-view margin markers + response
  analysis (permutation test, seeded)
- session-to-session variability (state occupancy, Jensen-Shannon divergence)
- action-vocabulary determination: 1-s windows → PAUSE / cardinal sectors →
  pre-registered reduction ladder L_A(5)…L_E(1) with leave-one-session-out
  decoder reliability
- the gate: G1 coverage ≥ 0.90, G2 annotation error ≤ 2.0 mm (≥ 150
  frames), G3 occupancy ≥ 0.05/session + [0.5, 2.0] stability band, G4 LOSO
  accuracy ≥ 0.80; G5 stimulus response secondary/informative

## Layer separation (unchanged from the project architecture)

```
BIOLOGICAL LOOP:   REAL FLY → camera → behavior detection → action decoder
                              → VIRTUAL INPUT CONTROLLER → game
                              (game = 2D virtual env first; real game only
                               after every prior gate passes)

COMPUTER LOOP:     GAME SCREEN → CV → world model → memory
                              → goal/task/strategy → high-level command
                              → cue/stimulus channel → biological loop
```

The fly is never expected to understand game concepts (level, quest, sword,
devil fruit, accessory, NPC, map, inventory, progression). Those live in
the computer-side stack.

## Source policy

No game access occurs in Phase 2. No game input is sent. For Grand Piece
Online the only permitted source remains the single Roblox page
(`docs/GPO_PLAN.md`); every other game fact stays UNKNOWN and must be
discovered experimentally in later phases.
