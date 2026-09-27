# PHASE-1 REPORT — Synthetic Closed-Loop Validation

**Baseline:** `PHASE1-BASELINE-1.0.0` (tag `phase1-baseline-v1.0.0`)
**Source revision:** `7296f6b` | **Reproduction:** IDENTICAL (commit `4be769a`)

---

## READ THIS FIRST — what Phase 1 is and is not

> **The Phase-1 "fly" is a SYNTHETIC BEHAVIORAL MODEL** (`flyagent/biological_agent/synthetic_fly.py`):
> a scripted burst-pause walker with a hard-coded visual-cue preference
> (`cue_bias: 0.85`). **No real animal was used, observed, or involved at any
> point in Phase 1.**
>
> Therefore **Phase 1 does NOT demonstrate that a real fruit fly can control a
> game.** It also does not demonstrate anything about real Drosophila
> behavior, real-fly tracking performance outside synthetic renders, or any
> game capability.
>
> What Phase 1 DOES demonstrate: (1) the proposed
> **biological-behavior → behavior-detection → action-decoder →
> virtual-input → game → reward** closed-loop architecture can be implemented
> end-to-end and tested experimentally with control groups and statistics; and
> (2) a behavior source with a cue-modulated preference — the *synthetic
> stand-in* for a cue-responsive animal — **contributes useful goal-directed
> behavior through that loop** (76.7% vs 3.3% success), an effect that is
> destroyed when the cue is randomized (3.3%).
>
> **Phase-1 results must never be described as real-animal performance.**
> Whether a real fly can play any role in this loop is precisely what Phase 2
> tests, beginning with non-invasive observation only.

## 1. What was built (frozen at rev `7296f6b`)

The complete Phase-1 apparatus, validated without any animal:

1. **Fly tracking pipeline** — frame sources (synthetic render / webcam /
   recorded video behind one interface), MOG2 background subtraction with
   frame-difference fallback, arena-ROI masking, blob detection, 4-point
   homography calibration (px → normalized → mm), trajectory buffer with
   kinematics (speed, heading).
2. **Biological-agent layer** — behavior classifier (WALK/PAUSE + heading)
   and action decoder (two configurable schemes: `motion_direction` default,
   `zone_dwell` alternative; STOP after sustained pause; cooldowns). The
   5-action vocabulary and all mappings live in `config/actions.yaml`.
3. **2D artificial environment** — player, target, optional obstacles,
   30 s trials, timeout; the game the architecture is validated on BEFORE any
   real game.
4. **Controllers behind one interface** — fly (real CV pipeline), random
   (floor), fixed greedy (ceiling), keyboard (human reference).
5. **Learning-agnostic reward machinery** — sparse + shaped reward computed
   and logged per trial (nothing is learned in Phase 1; learning enters
   Phase 4+).
6. **Experiment machinery** — identical trial loop for every controller;
   5-condition control-group demo; bootstrap statistics; immutable run
   directories; SQLite schema v1 (experiments / trials / events / frames).
7. **Test suite** — `tests/test_pipeline.py` (tracker, decoder, arena+reward,
   database, calibration) — ALL PASS.

## 2. The Phase-1 experiment (5 conditions × 30 trials, seed 20260927)

All conditions run the SAME trial loop, metrics, and database; only the
controller differs. The fly conditions pull frames through the REAL computer-
vision pipeline (tracker → calibration → trajectory → classifier → decoder).

| condition | controller | cue | result |
|---|---|---|---|
| `random` | uniform random actions | — | **3.3%** (floor) |
| `fixed` | hand-coded greedy | — | **100.0%** (ceiling) |
| `fly_random` | synthetic fly, `cue_bias=0` | presented, ignored | **6.7%** |
| `fly_cue` | synthetic fly, `cue_bias=0.85` | goal-directed | **76.7%** |
| `fly_cue_shuffled` | synthetic fly, `cue_bias=0.85` | random side | **3.3%** |

Bootstrap 95% CI of success-rate differences (10,000 iterations):

| contrast | diff | 95% CI | p(diff≤0) |
|---|---|---|---|
| `fly_cue` − `random` | +0.733 | [+0.567, +0.900] | 0.0000 |
| `fly_cue` − `fly_random` | +0.700 | [+0.533, +0.867] | 0.0000 |
| `fly_cue` − `fly_cue_shuffled` | +0.733 | [+0.567, +0.900] | 0.0000 |
| `fixed` − `random` | +0.967 | [+0.900, +1.000] | 0.0000 |

**Interpretation.** The cue-following behavior source converts a computer-
selected goal signal into goal-directed game behavior: 76.7% success, far
above every no-goal-information control (3.3–6.7%) and below the algorithmic
ceiling (100%). The shuffled-cue control shows the performance is due to the
closed loop between computer goal selection and behavior-source taxis, not to
fly behavior per se, decoder machinery, or environmental regularity. The
Phase-1 graduation gate (both `fly_cue − random` and
`fly_cue − fly_cue_shuffled` CIs exclude 0) **PASSED**.

## 3. Attribution — who contributes what (Phase 1)

| Contribution | Source | Status |
|---|---|---|
| Goal selection (which cue to present) | **Computer** (axis-dominant direction to target + hysteresis) | hard-coded |
| Action translation, reward, metrics, logging, statistics | **Computer** | hard-coded |
| Movement behavior that selects among cued options | **Behavior source** (synthetic fly) | hard-coded model parameter |
| Cue preference (`cue_bias 0.85`) | synthetic fly parameter | **NOT learned** — a modeling assumption |
| Anything learned | — | **nothing**; no learning algorithm is active in Phase 1 |

The control groups exist precisely so that no improvement can later be
mis-attributed: any real-fly phase must beat `random` (no controller),
`fixed` (pure computer control), `fly_random`-analogues (behavior without
goal information), and shuffled-cue analogues (goal information destroyed).

## 4. Reproduction record

`python main.py demo` re-run from the frozen baseline on 2026-09-27 produced
**bitwise-identical results**: all 150 per-trial metric sets across all 5
conditions, all aggregates, all 4 bootstrap contrasts, and the gate outcome
(`scripts/verify_reproduction.py`, exit 0; record in
`runs/demo_20260927-131700/reproduction_verification.txt`). No parameters
were tuned and no code was modified. Determinism rests on fully seeded numpy
PCG64 streams (seed table in `PHASE1_BASELINE.md` §3); timestamps occur only
in run-directory names and metadata.

## 5. What this enables (and does not enable)

ENABLED: proceeding to Phase 2 — real-fly behavioral validation by
NON-INVASIVE observation only (no surgery, implants, genetic modification,
toxins, injury, extreme temperatures, or any harmful manipulation), with a
pre-registered quantitative gate, before any connection between a real fly
and any game input is attempted.

NOT ENABLED (still UNKNOWN until demonstrated experimentally):
- whether a real fruit fly produces decodable, reliable behavioral states;
- whether a real fly responds to the benign cue channel;
- any capability involving the actual game (no game input has ever been sent;
  no game automation exists in this codebase; per source policy, the only
  permitted game reference is the single Roblox page for Grand Piece Online,
  and every game fact not observable there remains UNKNOWN and must be
  discovered experimentally before use).
