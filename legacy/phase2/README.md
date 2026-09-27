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
| `config/apparatus_checks.yaml` | Pre-session apparatus checks (preflight/blank/intake thresholds; Amendment 2) |

## Quick start

```bash
# 1. validate the MEASUREMENT CODE on synthetic ground-truth videos
#    (software test only — NOT evidence about real flies)
python phase2/run_phase2.py validate --duration 120

# 2. ON THE RIG MACHINE, every recording day, in this order (no fly yet):
python phase2/run_phase2.py calibrate --camera 0 --arena-mm 95   # once per build
python phase2/run_phase2.py calibrate --recheck-scale            # daily drift check
python phase2/run_phase2.py preflight --camera 0                 # ~20 s
python phase2/run_phase2.py blank --camera 0                     # >= 3 min + LED blinks

# 3. fly sessions (record REFUSES unless 2 passed today; labels unique):
python phase2/run_phase2.py record --minutes 10 --label baseline_01 \
    --fly-id F01 --age 5 --sex M --temperature 23 --humidity 45
python phase2/run_phase2.py record --minutes 10 --label stimulus_01 \
    --stimulus --session-index 0 --fly-id F01 --age 5 --sex M \
    --temperature 23 --humidity 45

# 4. after every recording day: immutable, verified backup
python phase2/run_phase2.py backup --to /path/to/usb-drive

# 5. when recordings come back: INTAKE FIRST (integrity + compliance,
#    no analysis, no tuning), then the fixed pre-registered sequence:
python phase2/run_phase2.py intake data/sessions
python phase2/run_phase2.py track   data/sessions/<dir>
python phase2/run_phase2.py annotate data/sessions/<dir> --n 50
python phase2/run_phase2.py analyze data/sessions/<a> data/sessions/<b> ... \
    --out results/first_analysis
python phase2/run_phase2.py gate --analysis results/first_analysis \
    --annotations data/sessions/<a>/annotations.csv ...
```

See `docs/PHASE2_START_GUIDE.md` for the full physical-experiment
procedure (shopping list, arena construction, camera/lighting setup,
calibration, blank test, recording, backup, pass/fail criteria).

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
