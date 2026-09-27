# PHASE-2 GATE PRE-REGISTRATION — PHASE2-GATE-1.0.0

**Pre-registered:** 2026-09-27 (UTC), committed to git BEFORE any Phase-2
analysis code was written and BEFORE any real-fly data was collected.
The enclosing git commit is the tamper-evident timestamp. Machine-readable
copy: `phase2/config/gate.yaml` (the single source of truth consumed by the
gate evaluator).

**Phase-2 objective (fixed):** determine whether a REAL fruit fly produces
sufficiently reliable, repeatable observable behavior that can be converted
into a small action vocabulary — by NON-INVASIVE observation only. The real
fly is NOT connected to any game input in this phase.

**Mapping note:** user-facing "Phase 1" = repo ROADMAP Phase 0 (synthetic
validation, DELIVERED, frozen as PHASE1-BASELINE-1.0.0). This phase ("Phase
2 — real-fly behavioral validation") covers repo ROADMAP Phases 1–2
(real-fly observation + stimulus response measurement) plus the new
action-vocabulary determination.

---

## 1. Non-negotiable ethical constraints (pre-registered)

Observation and benign visual stimuli ONLY. PROHIBITED in this project, at
any phase: surgery, neural implants, genetic modification, toxic substances,
deliberate injury, extreme temperatures, electric shock, aversive stimuli of
any kind, food/water deprivation as a manipulation, or any other harmful
procedure. Sessions ≤ 30 min, ≥ 24 h rest between sessions per fly, stop the
session if the fly is immobile > 10 min. If the user's institution requires
oversight for animal observation, the protocol stops until that oversight is
obtained. Full welfare text: `PHASE2_PROTOCOL.md`.

## 2. Data requirement for gate evaluation (fixed)

| Requirement | Value |
|---|---|
| Baseline (free-behavior) sessions | ≥ 3, each ≥ 10 min, ≥ 30 fps |
| Stimulus sessions | ≥ 3, each ≥ 10 min, ≥ 30 fps |
| Recording days | ≥ 2 (sessions spread across days) |
| Subject | 1 fly preferred (cohort allowed if documented; per-fly analysis reported) |
| Manual annotation (for G2) | ≥ 150 frames total, seeded random sample (50/session × 3 sessions) |

## 3. THE GATE — quantitative criteria (all thresholds FIXED in advance)

Evaluated by `python phase2/run_phase2.py gate --analysis <dir>`.
PASS requires G1–G4 **all** to pass. G5 is secondary/informative (see §5).

### G1 — Tracking coverage (apparatus works)
Per session: fraction of frames with a fly detection at confidence ≥ 0.5
must be **≥ 0.90**.

### G2 — Tracking accuracy (positions are true)
Median manual-annotation position error, pooled over ≥ 150 annotated frames,
must be **≤ 2.0 mm** (via px→mm calibration; in px if calibration absent:
≤ 2.0 mm × px-per-mm).

### G3 — State occupancy and stability (behavior is rich and repeatable)
Using 1.0 s non-overlapping windows and the SELECTED vocabulary level (§4):
- every included state has window occupancy **≥ 0.05 in EVERY session**, and
- every per-session occupancy lies within **[0.5×, 2.0×] the across-session
  mean** occupancy of that state.

### G4 — Decoder reliability (states are distinguishable)
Leave-one-session-out nearest-centroid classifier (features: window mean
speed, displacement x, displacement y, displacement magnitude; standardized;
deterministic), accuracy on window labels **≥ 0.80**. Evaluated whenever the
level has ≥ 2 states; waived for the 1-state level (L_E).

## 4. Action-vocabulary ladder (pre-registered; the data choose, not the design)

Candidate 5-action vocabulary per project spec: LEFT, RIGHT, FORWARD,
BACKWARD, STOP. Window labeling: a 1.0 s window is PAUSE (candidate STOP) if
mean speed < 2.0 mm/s; otherwise its displacement angle is binned into four
90° sectors centered on the arena-frame cardinals (FORWARD = −y / camera-up,
BACKWARD = +y, LEFT = −x, RIGHT = +x). Arena→game rotation is a config
mapping (as in Phase-1 `actions.yaml`), not a behavior question.

Reduction ladder (largest first; the LARGEST level whose states ALL satisfy
G3+G4 is selected — deterministic):

| Level | States | Construction |
|---|---|---|
| L_A (5) | FORWARD, BACKWARD, LEFT, RIGHT, PAUSE | full vocabulary |
| L_B (4) | 3 cardinals + PAUSE | drop the cardinal with the lowest across-session mean occupancy |
| L_C (3) | LEFT, RIGHT, PAUSE *or* FORWARD, BACKWARD, PAUSE | the axis (horizontal/vertical) with higher total occupancy + PAUSE; windows assigned by the sign of dominant-axis displacement |
| L_D (2) | MOVE, PAUSE | any movement vs pause |
| L_E (1) | MOVE | fly presence = continuous action |

**Minimum for directional game control: L_C.** If the data support only
L_D/L_E, the fly cannot steer directionally; the project then documents the
pivot (computer-side planning with the fly as GO/NO-GO signal, or apparatus/
stimulus improvement) instead of forcing the 5-action design.

## 5. G5 — Stimulus response (SECONDARY; informative for the future cue channel; NOT required for gate PASS)

Per stimulus event (20 s ON), response statistic = mean cos(heading −
stimulus angle) over [0, 3] s after onset MINUS mean over [−8, −1] s before
onset. Two-sided permutation test, 1000 circular shifts of event times
within the session, seed 20260927, α = 0.05. A session "responds" if
p < 0.05. Informative outcome: response in ≥ 2 of ≥ 3 stimulus sessions
indicates a usable steering channel for Phase 3; failure triggers the
pre-registered stimulus-design diagnosis (§6), not silent abandonment.

## 6. Failure→diagnosis map (pre-registered; no post-hoc reinterpretation)

| Failed criterion | Pre-registered diagnosis to work through |
|---|---|
| G1 | Apparatus/tracking: lighting stability, focus, locked exposure, background thresholds, blob-area range, ROI masking (see docs/HARDWARE.md failure table) |
| G2 | Calibration (scale, geometry) or detector center bias; annotation protocol quality (re-annotate a subsample, check self-consistency) |
| G3 occupancy | Repertoire too narrow or elicitation failure: session time-of-day (circadian activity), arena size, stimulus design; ladder already reduces vocabulary automatically |
| G3 stability | Behavioral variability across sessions: time-of-day alignment, habituation, age/sex, temperature/humidity logging; add sessions/days — decoder tuning is NOT the fix |
| G4 | Decoder design: window length (re-registration required to change from 1.0 s), feature set; if persistent, ladder reduces vocabulary |
| G5 | Stimulus modality/salience: test benign alternatives (position, luminance, pattern, spectrum) before concluding the fly cannot be steered; fallback `cue_mode: none` exists in the architecture |

## 7. Analysis constants (fixed; consumed from `phase2/config/gate.yaml`)

| Constant | Value |
|---|---|
| Window length | 1.0 s, non-overlapping |
| PAUSE speed threshold | 2.0 mm/s (window mean) |
| Sector definition | four 90° bins centered on cardinals, arena frame |
| Position smoothing | 5-frame median filter |
| Velocity | central difference on smoothed positions (±1 frame) |
| Bout onset speed | 3.0 mm/s (hysteresis off-threshold 2.0 mm/s) |
| Min bout / pause duration | 0.25 s / 0.30 s (gap merge ≤ 0.2 s) |
| Wall band | 10.0 mm; wall-following = in band AND heading within 45° of wall tangent |
| Stimulus windows | response [0, 3] s; baseline [−8, −1] s; permutation 1000 shifts, two-sided, seed 20260927 |
| Occupancy minimum | 0.05 per session |
| Stability band | [0.5, 2.0] × across-session mean |
| LOSO accuracy minimum | 0.80 |
| Annotation minimum | 150 frames; G2 median error ≤ 2.0 mm |

## 8. Software acceptance criteria (pipeline validation — CODE, not flies)

Before any real-fly data, the measurement software itself must pass
`python phase2/run_phase2.py validate` on SYNTHETIC ground-truth videos:

| ID | Criterion |
|---|---|
| SA1 | tracking coverage ≥ 0.98 per synthetic session |
| SA2 | median position error ≤ 1.0 mm; p95 ≤ 2.5 mm |
| SA3 | mean speed within ±15% of ground truth |
| SA4 | pause frequency (pauses/min) within ±20% |
| SA5 | median bout duration within ±20% |
| SA6 | wall-following fraction within ±0.10 absolute |
| SA7 | mean resultant length of headings within ±0.15 of ground truth |
| SA8 | stimulus response DETECTED (p<0.05) in ≥ 2 stimulus sessions; NOT detected in ≥ 2 of 3 baseline sessions (false-positive check) |
| SA9 | vocabulary selection on synthetic data = L_A |
| SA10 | gate machinery: PASS on clean synthetic tracks (GT-derived annotations); a 20%-frame-drop negative control makes G1 FAIL with the tracking diagnosis |
| SA11 | blank (no-fly) video: zero confident detections |

**These synthetic videos are software tests with known ground truth. They
are NOT a model of real Drosophila behavior and provide NO evidence about
real flies.** The synthetic-fly model from Phase 1 must likewise never be
cited as evidence about real-fly behavior.

## 9. Consequences (fixed)

- **G1–G4 PASS** → proceed to Phase 3: closed loop with the real fly in the
  2D VIRTUAL environment (repo ROADMAP Phase 3; still no game).
- **Any of G1–G4 FAIL** → work the pre-registered diagnosis map (§6),
  improve the apparatus/protocol, register an amendment (new gate ID
  PHASE2-GATE-1.x.0 with documented rationale), and re-run. Thresholds are
  never changed after seeing real-fly data.
- **In no case** does Phase 2 connect the real fly to game input, implement
  game automation, or assume any undocumented game mechanic. For Grand Piece
  Online the only permitted source remains the single game page; everything
  else about the game is UNKNOWN until experimentally discovered in later
  phases.

## 10. Amendments

(none)
