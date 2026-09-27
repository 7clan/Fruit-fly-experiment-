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

### Amendment 1 (2026-09-27, during software validation, BEFORE any real-fly data)

Two correctness fixes identified by the SA1–SA11 pipeline validation itself
(the validation suite doing its job); neither changes any registered
threshold value:

1. **Heading statistics use MOVING frames only.** A heading (direction of
   velocity) is mathematically undefined while the animal stands still; on
   exact ground-truth input, paused frames produce a degenerate
   `atan2(0,0)=0` heading that inflated the mean resultant length (R̄).
   All direction/persistence metrics (mean angle, R̄, circular std, heading
   histogram, heading autocorrelation) now use only frames with speed
   ≥ 2.0 mm/s — the same convention already used by the stimulus-response
   module (§5). Thresholds unchanged; SA7 now compares like-for-like.
2. **Analysis trims to the subject's first appearance.** Recordings start
   before the fly is introduced (the static background needs fly-free
   lead-in frames). Frames before the first detection are reported as
   `lead_in_s` and excluded from coverage/pause statistics — they are not
   "lost tracking" and must not create a phantom leading pause. Thresholds
   unchanged.

### Amendment 2 (2026-09-28, while preparing the PHYSICAL experiment, BEFORE any real-fly data)

Registered when moving from software validation to the physical apparatus.
**No G1–G5 threshold, no analysis constant, no session-structure rule, and
no welfare rule is changed.** Everything below is apparatus-level. The
machine-readable additions live in `phase2/config/apparatus_checks.yaml`
(id `PHASE2-APPARATUS-1.0.0`), deliberately separate from this gate file.

**Motivating problem found during physical-experiment preparation:** the
pre-registered session flow (PHASE2_PROTOCOL §5.1: the fly habituates
≥ 30 min INSIDE the closed enclosure, then recording starts) means real
sessions have **no fly-free lead-in** — but the `static` detector builds
its background from the first 40 recorded frames. The fly-free-lead-in
assumption holds for the synthetic validation videos only; the protocol
and the tracker were incompatible as written for the real rig.

1. **Pre-session apparatus checks (new, pre-registered):** `preflight`
   (~20 s live capture: fps, exposure/lighting stability, focus, resolution
   vs calibration, background stability, sensor noise, arena/marker
   geometry) and a **blank-arena test** (≥ 3 min no-fly recording: zero
   confident false detections, no raw foreground motion, lighting
   stability, no static "fly-like" dark features, sensor noise, and an
   N/E/S/W stimulus-LED blink check read back from the video margin). The
   recorder REFUSES animal sessions unless both passed within 24 h under
   the same config/code hash; `--force` overrides are recorded as protocol
   deviations in `session.json` and flagged by `intake`.
2. **Detector `reference` background mode (new):** the background is the
   per-pixel median of the day's PASSED blank recording (identical
   lighting, fly-free by construction). Real fly sessions are tracked in
   this mode, which keeps the pre-registered session flow (habituation →
   record) intact. The `static` mode and the synthetic validation suite
   are unchanged — re-run after this amendment: **bit-identical tracks and
   identical SA1–SA11 outcomes** (SA8 p = 0.014 / 0.005).
3. **New software acceptance SA12:** reference-mode tracking on synthetic
   sessions rendered with the fly present FROM FRAME 0, background built
   from a separate fly-free clip (exactly the real-rig flow), must meet the
   SA1/SA2 thresholds (coverage ≥ 0.98, median error ≤ 1.0 mm, p95
   ≤ 2.5 mm). Observed: coverage 1.000, median error 0.08 mm, p95 0.12 mm
   — PASS.
4. **Data provenance (new):** every `session.json` records `session_id`,
   config/code hashes (tracking.yaml, gate.yaml, apparatus_checks.yaml,
   flyrec source tree, git revision), the rig calibration reference +
   hash, preflight/blank PASS references, the background reference + hash,
   and the camera's reported vs measured frame rate (the container fps is
   set to the measured rate when the camera report is implausible, so
   timestamps stay truthful). Raw video is never overwritten; session
   directories are immutable and labels unique; backups are
   content-addressed (SHA-256 manifests) and never overwrite.
5. **Per-frame derived export (new):** `analyze` additionally writes
   `per_frame.csv` (session id, frame, t, calibrated x/y, speed, heading,
   pause flag, wall-in-band, wall-following, stimulus state, confidence,
   interpolated flag) — derived and regenerable like `tracks.csv`; nothing
   feeds back from it into any metric or gate criterion.
6. **Intake procedure (new, fixed order):** when recordings come back,
   `intake` verifies file integrity (video opens, frame count, metadata,
   SHA-256 manifest) and protocol compliance (the §2 data requirements,
   seeded stimulus schedule regeneration, provenance consistency, welfare
   duration cap) BEFORE any tracking/analysis/gate evaluation. No tuning
   precedes a PASS/FAIL report.
7. **Camera/geometry clarification (derived from the pre-registered
   software geometry, not a new degree of freedom):** the working
   resolution must place the arena (≥ 600 px across, §3) with ≥ 60 px
   clearance between the arena rim and every frame edge (recommended
   ≥ 150 px) so the four stimulus marker boxes (tracking.yaml
   `stimulus_markers`: 30×14 px at an 18 px frame margin) lie outside the
   arena ROI. In practice this requires ≥ 1280×960 or 1920×1080 at
   ≥ 30 fps; **1280×720 cannot satisfy the pre-registered software
   geometry and is not acceptable for the real rig.**
8. **Validation duration note (reproduction):** the registered software
   validation run uses `--duration 120` (8 stimulus events per synthetic
   session). A 60 s run halves the number of stimulus events and reduces
   the permutation test's power (observed: stimulus_01 p = 0.142 at 60 s
   vs p = 0.014 at 120 s). Reproduce with
   `python phase2/run_phase2.py validate --duration 120`.
