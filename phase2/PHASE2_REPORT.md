# PHASE-2 REPORT — Real-Fly Behavioral Validation

**Date:** 2026-09-27 | **Gate:** PHASE2-GATE-1.0.0 (pre-registered, commit
`520eb2f`, before any Phase-2 code or data) | **Pipeline validation:**
SA1–SA11 ALL PASS

---

## Executive summary

Phase 2 set out to determine whether a REAL fruit fly produces sufficiently
reliable, repeatable observable behavior to serve as a biological
controller. The complete **measurement apparatus** (software) is now built
and validated against synthetic ground truth. **No real-fly data has been
collected in this environment — no camera rig exists here — so every
question about real-fly behavior remains UNKNOWN.** The pre-registered gate
is defined, implemented, and exercised end-to-end (including its
failure-diagnosis paths); it will evaluate automatically the moment real
sessions are recorded on hardware. Nothing in Phase 2 touches the game.

## 1. What was built

**Apparatus software** (`phase2/flyrec/`, ~2,600 lines, all runnable):

| Capability | Where |
|---|---|
| Continuous recording into immutable session dirs (video + metadata + seeded open-loop stimulus schedule) | `recording/recorder.py` |
| Fly detection (static-median or MOG2 background, dark-blob, arena ROI mask) with per-frame confidence | `tracking/detector.py` |
| px→mm calibration; circular/rectangular arena geometry; wall distance & tangent | `tracking/calibration.py` |
| Video→tracks pipeline with stimulus-timing readback from in-view margin markers | `tracking/tracker.py` |
| Gap interpolation, smoothing, velocity/heading (moving-frame convention) | `analysis/trajectory.py` |
| Pause/bout detection (hysteresis), wall-following & contact metrics | `analysis/behaviors.py` |
| Circular statistics: heading distribution, mean resultant length, autocorrelation | `analysis/circular.py` |
| Stimulus-response analysis with seeded permutation test | `analysis/stimulus.py` |
| Session-to-session variability (occupancy, Jensen-Shannon divergence) | `analysis/variability.py` |
| Master behavioral statistics (every required metric) | `analysis/statistics.py` |
| Action-vocabulary determination: 1-s windows, cardinal sectors + PAUSE, pre-registered ladder L_A…L_E, leave-one-session-out decoder reliability | `vocabulary/` |
| Gate evaluator G1–G4 (+G5 secondary) with pre-registered failure→diagnosis map; fails closed without annotations | `gate/evaluate.py` |
| Manual annotation tool (interactive; G2 accuracy) | `tracking/annotate.py` |
| Synthetic ground-truth generator + end-to-end software validation SA1–SA11 | `validation/` |
| CLI: `validate / record / track / annotate / analyze / gate` | `run_phase2.py` |

**Documents:** pre-registered gate (`PHASE2_GATE_PREREGISTRATION.md`,
`config/gate.yaml` — committed BEFORE the code), real-fly protocol
(`PHASE2_PROTOCOL.md`: welfare rules, apparatus, session structure,
annotation protocol), this report.

**Pre-registration provenance:** the gate thresholds were committed at
`520eb2f` on 2026-09-27, before any Phase-2 analysis code or data existed.
One amendment is recorded (heading statistics restricted to moving frames —
a correctness fix caught by the software validation itself; no threshold
values changed; made before any real-fly data).

## 2. What the real fly actually demonstrated

**Nothing yet — and this report refuses to pretend otherwise.** No real
fruit fly has been recorded in this environment (it is a code workspace;
the hardware rig does not exist here). Zero real-fly data exists in this
repository. Therefore:

- The real fly has demonstrated NO behavioral capability, NO decodable
  states, NO stimulus response, and NO game-related capability.
- The Phase-1 result (76.7% goal-directed success) was SYNTHETIC-ONLY
  (`PHASE1_REPORT.md`) and must never be cited as real-animal performance.
- The Phase-2 synthetic validation videos below are SOFTWARE TESTS of the
  measurement code with known ground truth — they are **not** a model of
  Drosophila behavior and **not** evidence about real flies.

The honest statement is: *the apparatus is ready; the animal question is
open.* Answering it requires executing `PHASE2_PROTOCOL.md` on hardware.

## 3. What remains UNKNOWN

- **Every real-fly behavioral statistic**: tracking coverage on real
  footage, real speed/pause/bout/wall statistics, real heading
  distributions, real trial-to-trial variability. All UNKNOWN.
- **The usable action vocabulary** — whether the real fly produces 5, 4, 3,
  2 or 1 reliably distinguishable state(s). The ladder is pre-registered so
  the DATA will choose; the design does not force the 5-action vocabulary.
- **Real-fly stimulus response** (the future steering channel). UNKNOWN.
- **The Phase-2 gate outcome.** NOT EVALUABLE until real sessions exist;
  the gate deliberately FAILS CLOSED without real annotations.
- **All Grand Piece Online facts beyond the single permitted page.** No
  Roblox resource was accessed during Phase 2; no game input was sent; no
  game automation exists in this repository. Controls, mechanics, UI,
  quests, progression — all UNKNOWN, to be discovered experimentally only
  in later phases.

## 4. Measured behavioral statistics

**(a) Software validation on synthetic ground truth (CODE, not flies).**
5 sessions (3 baseline + 2 stimulus, 120 s each, 30 fps, 480×360, 60 mm
circular arena @ 5 px/mm) + 1 blank. The test pattern is a deliberately
5-state cardinal-concentrated bout/pause walker with stimulus taxis —
designed so the code's correctness can be checked against known truth
(`phase2/results/phase2_pipeline_validation/`):

| Criterion | Result | Threshold |
|---|---|---|
| SA1 tracking coverage | 100.0% every session | ≥ 98% |
| SA2 position error | median 0.08 mm, p95 0.12 mm | ≤ 1.0 / 2.5 mm |
| SA3 mean speed recovery | within 2% of GT | ± 15% |
| SA4 pause frequency | within 8% | ± 20% |
| SA5 median bout duration | within 5% | ± 20% |
| SA6 wall-following fraction | within 0.03 absolute | ± 0.10 |
| SA7 mean resultant length | within 0.06 | ± 0.15 |
| SA8 stimulus detection | p = 0.014 / 0.005 in both stimulus sessions; 0 false positives in 3 baselines | detect ≥ 2; FP ≤ 1 |
| SA9 vocabulary selected | L_A (full 5-state) | L_A |
| SA10 gate machinery | clean PASS; 20%-dropout negative control correctly FAILS G1 with tracking diagnosis | both required |
| SA11 blank video | 0 detections | 0 |

Leave-one-session-out decoder accuracy on the synthetic 5-state data:
0.813 (chance 0.236). Cross-session state-occupancy stability passed the
[0.5, 2.0] band at every level. Figures: `*_overview.png`,
`vocabulary_ladder.png` in the results directory.

**(b) Real-fly statistics: NONE EXIST.** The exact metrics that will be
produced per session (coverage, confidence, speed distribution, direction
rose, pause/bout rates and durations, persistence, wall metrics, state
occupancy, variability, stimulus response) are implemented and
demonstrated in (a); the values themselves await hardware data.

## 5. Did the predefined Phase-2 gate pass?

**The gate is defined, pre-registered (commit `520eb2f`), implemented, and
verified — but its real-fly evaluation has NOT HAPPENED, so the honest
verdict is: PENDING, not passed.** What was verified in this session:

- the gate machinery runs end-to-end and reports PASS on clean synthetic
  tracks (G1 1.000 ≥ 0.90; G2 median 0.08 mm ≤ 2.0 mm over 150 GT-derived
  annotations; G3 all five states ≥ 0.05 in every session within the
  stability band; G4 LOSO 0.813 ≥ 0.80);
- it correctly FAILS (G1, with the pre-registered apparatus/tracking
  diagnosis) when 20% of detections are dropped — the negative control;
- it fails CLOSED when annotations are missing (G2 cannot be waved
  through).

**The real-fly gate outcome is UNKNOWN until ≥ 3 baseline + ≥ 3 stimulus
sessions (≥ 10 min, ≥ 30 fps, ≥ 2 days) are recorded and 150 frames are
manually annotated, per the pre-registered data requirements.** If it
fails, the pre-registered diagnosis map (G1→apparatus/tracking,
G2→calibration, G3→repertoire/variability, G4→decoder design) is followed —
no post-hoc threshold changes.

## 6. What must be built next (in order)

1. **Hardware rig** per `PHASE2_PROTOCOL.md` / `docs/HARDWARE.md` (camera
   ≥ 30 fps with locked exposure, 60–100 mm circular arena on uniform
   backlight, 4 stimulus LED segments visible in the camera margin,
   light-tight enclosure; blank-video false-positive check before any
   fly).
2. **Execute the protocol**: 3 baseline + 3 stimulus sessions across ≥ 2
   days, single fly, habituation ≥ 30 min, welfare rules enforced;
   `record → track → annotate (150 frames) → analyze → gate`.
3. **Gate decision**: PASS → Phase 3 (closed loop with the REAL fly in the
   2D VIRTUAL environment — still no game). FAIL → work the diagnosis map,
   improve the apparatus, register an amendment, re-run.
4. Then, only after Phase 3 succeeds: screen-perception tooling (observe
   only), virtual input controller, and the staged GPO integration per
   `docs/ROADMAP.md` — each phase behind its own measurable gate.

## 7. Exactly how the real fly will eventually connect to the virtual controller

The wiring is already defined and its interfaces exist in code:

```
            ┌───────────────────── COMPUTER SIDE (planning) ─────────────────────┐
            │ game screen → CV → world model → memory → goal/task/strategy      │
            │      │                                                               │
            │      └── high-level command (e.g. "move toward X") ──────────────┐ │
            └─────────────────────────────────────────────────────────────────│─┘
                                                                                ▼
REAL FLY → overhead camera (30–60 fps) → flyrec tracker (position,        CUE CHANNEL
  speed, heading, confidence) → 1-s window behavioral-state classifier    (4 benign LED
  (PAUSE / cardinal sector, arena frame) → action decoder                 segments in the
  (same 5-action vocabulary as Phase 1, arena→game rotation in           arena margin)
  config, exactly like `config/actions.yaml`) → VIRTUAL INPUT            ◄── presents the
  CONTROLLER → game input (Phase 3: the 2D virtual arena; later,          direction the
  only after every prior gate, the real game)                             computer wants
                                                                          the agent to go)
```

Key points, unchanged from the project architecture:

- **The fly contributes only low-level behavioral variety** (which
  direction it walks, when it pauses). **The computer contributes
  everything else**: perception of the game, memory, planning, strategy,
  reward, statistics — and the cue that biases which direction is
  available/rewarded. Attribution stays measurable through the control
  conditions (random / fixed / no-cue / shuffled-cue), exactly as in the
  frozen Phase-1 design.
- **The fly never sees or understands game concepts** (level, quest,
  sword, devil fruit, accessory, NPC, map, inventory, progression). Those
  belong to the computer-side perception/planning stack unless experiments
  ever demonstrate otherwise.
- **Decoder parameters will be frozen across sessions** in Phase 3
  (attribution requirement): if performance improves, it must be the
  closed loop, not silent retuning of the computer's decoder.
- **Nothing connects to Grand Piece Online** until real-fly validation
  (this phase), virtual-environment closed-loop control (Phase 3), and
  screen perception are all demonstrated behind their gates. All game
  facts not observable on the single permitted page remain UNKNOWN and
  must be discovered experimentally.

## Claim discipline

This report claims: the Phase-2 measurement apparatus is built; its code
is validated against synthetic ground truth (SA1–SA11 PASS); the gate is
pre-registered and its machinery verified including failure paths. It does
NOT claim: that a real fly can be tracked (untested on real footage), that
a real fly produces decodable states (UNKNOWN), that a real fly responds to
the stimulus (UNKNOWN), or that this system can play any game (never yet
demonstrated — no game input has ever been sent).
