# PHASE-2 PROTOCOL — Real-Fly Behavioral Validation (non-invasive observation)

**Status:** apparatus software BUILT and pipeline-validated on synthetic
ground-truth videos (software acceptance SA1–SA11). **Real-fly data: NOT YET
COLLECTED** — this protocol is executed by the user on hardware. All
real-fly outcomes are UNKNOWN until then. Gate:
`PHASE2_GATE_PREREGISTRATION.md` / `config/gate.yaml` (pre-registered).

**Phase 2 never connects the fly to game input.** No game automation is
implemented or attempted. The stimuli are open-loop (presentation schedule
fixed in advance; the fly's behavior cannot change what is shown).

---

## 1. Objective

Determine whether a real fruit fly (D. melanogaster) produces sufficiently
reliable, repeatable observable behavior that can be converted into a small
action vocabulary (candidate: LEFT / RIGHT / FORWARD / BACKWARD / STOP; the
pre-registered reduction ladder may select fewer). Secondary question: does
the fly respond to a benign directional visual stimulus (informative for the
future steering channel)?

## 2. Welfare rules (binding)

1. OBSERVATION + BENIGN VISUAL STIMULI ONLY.
2. Prohibited always: surgery, neural implants, genetic modification, toxic
   substances, deliberate injury, extreme temperatures, electric or
   mechanical shock, aversive stimuli, food/water deprivation as a
   manipulation, or any procedure with foreseeable harm.
3. Session length ≤ 30 min; ≥ 24 h rest per fly between sessions; maximum
   6 sessions per fly per week.
4. End a session early if: the fly is immobile > 10 min, escapes, or shows
   any sign of impaired locomotion. Record the reason in `session.json`.
5. Transfer by gentle aspiration only. No CO₂, cold, or chemical
   immobilization within 24 h before a session (all alter behavior).
6. Flies from a registered stock center / education supplier (species
   certainty; not wild-caught). Standard medium, 22–25 °C, 12:12 light:dark.
7. If the user's institution requires approval for animal observation, stop
   until obtained. This project's procedures are designed to be entirely
   non-invasive observation, but institutional rules prevail.

## 3. Apparatus (see docs/HARDWARE.md for the full BOM)

```
        light-tight enclosure
  ┌──────────────────────────────────────────┐
  │   diffused constant LED panel (no flicker)│
  │    ┌── circular arena, matte floor ──┐     │
  │    │        single fly               │     │
  │    └─────────────────────────────────┘     │
  │  4 stimulus LED segments (N/E/S/W),        │
  │  visible in camera margin, OUTSIDE the     │
  │  arena ROI (timing read back from video)   │
  │            [camera, top-down, fixed]       │
  └─────────────────┬──────────────────────────┘
                    USB → computer (this software)
```

Key requirements:
- Camera ≥ 30 fps (60 preferred), arena spanning ≥ 600 px, manual/locked
  exposure, rigid mount, fixed focus.
- Circular arena 90–100 mm inner diameter (matches `analysis.arena`
  geometry; rectangular also supported by config), matte floor, closed lid
  with air holes.
- Uniform, flicker-free backlighting; light-tight enclosure (record a blank
  video first — SA11 requires zero false detections).
- Stimulus: four dim, luminance-matched LED segments at N/E/S/W, placed so
  they are INSIDE the camera field of view but OUTSIDE the arena ROI (the
  tracker reads exact stimulus timing back from the video margin — no manual
  log trust). Benign visual stimuli only.
- Log room temperature (22–25 °C) and humidity each session.

## 4. Software apparatus (BUILT — this is what Phase 2 adds)

| Capability | Command / module |
|---|---|
| Rig calibration (ruler + arena circle, geometry checks) | `python phase2/run_phase2.py calibrate` |
| Camera/environment PREFLIGHT (fail-closed) | `python phase2/run_phase2.py preflight` |
| Blank-arena test + reference background | `python phase2/run_phase2.py blank` |
| Continuous recording + immutable session dirs | `python phase2/run_phase2.py record` |
| Session intake: integrity + compliance (no analysis) | `python phase2/run_phase2.py intake` |
| Immutable content-addressed backup | `python phase2/run_phase2.py backup --to <dir>` |
| Fly detection, position, tracking confidence | `flyrec/tracking/detector.py`, `tracker.py` |
| px→mm calibration + arena geometry (circle/rect) | `flyrec/tracking/calibration.py` |
| Velocity / heading estimation | `flyrec/analysis/trajectory.py` |
| Pause / bout / wall-edge interaction detection | `flyrec/analysis/behaviors.py` |
| Direction distribution / circular statistics | `flyrec/analysis/circular.py` |
| Stimulus-timing readback + response analysis | `flyrec/analysis/stimulus.py` |
| Trial-to-trial (session) variability | `flyrec/analysis/variability.py` |
| Full behavioral statistics report | `flyrec/analysis/statistics.py` |
| Action-vocabulary determination (ladder + LOSO) | `flyrec/vocabulary/` |
| Pre-registered gate evaluation + diagnosis | `flyrec/gate/evaluate.py` |
| Manual annotation tool (for G2) | `python phase2/run_phase2.py annotate` |
| Pipeline self-validation (synthetic GT, no fly) | `python phase2/run_phase2.py validate` |

## 5. Session structure (pre-registered)

### 5.0 Pre-session apparatus checks (Amendment 2 — REQUIRED, enforced by software)

Every recording day, BEFORE any fly is introduced, in this exact order:

1. **`calibrate`** (once per rig build; re-verify daily): with the printed
   reference card mounted beside the arena, measure px/mm on the ruler dot
   pairs (X and Y — the camera-tilt/anisotropy check is ≤ 2 %) and fit the
   arena circle from ≥ 3 clicks on the INNER wall edge. Built-in checks:
   arena ≥ 600 px across, ≥ 60 px rim-to-frame-edge clearance (marker boxes
   must lie outside the ROI), fitted radius vs measured physical inner
   diameter within ± 3 mm. Daily quick check: `calibrate --recheck-scale`
   (≤ 1 % drift) or full `--verify`.
2. **`preflight`** (~20 s, no fly, enclosure closed, LEDs off): frame rate
   ≥ 30 fps, exposure/lighting stability (frame-mean gray std ≤ 2.0,
   drift ≤ 2.0), focus (Laplacian variance on the reference card ≥ 60),
   background stability (no fly-like blobs in the arena ROI), sensor noise,
   camera resolution vs the stored calibration. FAIL closes the day.
3. **`blank`** (≥ 3 min, no fly): the software records and tracks an empty
   arena and requires: **zero** confident false detections, no raw
   foreground motion, lighting stability, **no static "fly-like" dark
   features** (smudges/shadows/rim arcs), sensor noise within limits, and —
   with the operator switching each stimulus LED on for 10 s in turn
   following the printed timetable — all four N/E/S/W markers readable
   from the video margin. A PASS stores the day's reference background
   (`data/rig/background.png`) used to track all fly sessions that day.

`record` refuses baseline/stimulus sessions unless steps 2–3 passed within
24 h under the current config/code hash (checked automatically). A `--force`
override is legal only for apparatus debugging and is recorded as a
protocol deviation (flagged by `intake`). No animal session may begin on a
failed or skipped blank test.

For each fly:
1. **Habituation**: fly in arena ≥ 30 min before recording starts (enclosure
   closed, stimulus LEDs off, camera already running).
2. **Baseline sessions** ×3 (≥10 min each): free behavior, no stimulus.
3. **Stimulus sessions** ×3 (≥10 min each): 10 events per session; each
   event = stimulus ON 20 s → OFF 40 s; side sequence from a SEEDED RNG
   (seed = 20260927 + session index; recorded in `session.json`).
4. Sessions spread over ≥ 2 days, same time-of-day (± 1 h) to control
   circadian state; ≥ 24 h between a given fly's sessions.
5. Record metadata every session: fly id, age (days post-eclosion), sex,
   temperature, humidity, start time, stimulus seed, any deviations.

Session directory (immutable — files are only ever ADDED, raw video never
modified):
```
data/sessions/<UTC-timestamp>_<label>/
    video.(mp4|avi)        # raw recording, never edited, never overwritten
    session.json           # metadata + provenance + stimulus schedule
    tracks.csv             # derived (regenerable): per-frame tracking
    per_frame.csv          # derived (regenerable): t, x/y mm, speed, heading,
                           #   pause, wall-in-band, wall-following, stimulus
                           #   state, confidence, interpolated flag
    annotations.csv        # manual G2 annotations (added by `annotate`)
    ground_truth.csv       # synthetic validation sessions only
```
`session.json` records (Amendment 2): unique `session_id`, fly metadata,
reported vs measured frame rate, `config_hashes` (tracking.yaml, gate.yaml,
apparatus_checks.yaml, flyrec source tree, git revision), the rig
`calibration` reference + hash, the `preflight`/`blank` PASS references, the
`tracking` recipe (reference mode + background hash) and any `--force`
override. Session labels must be unique (enforced).

Data management (Amendment 2):
- **Backup after every recording day**: `python phase2/run_phase2.py backup
  --to <drive>` — content-addressed copies (SHA-256 verified), never
  overwrites; a changed raw file after backup raises an immutability alarm.
- **When recordings come back for analysis**: run `intake` FIRST (file
  integrity + protocol compliance, SHA-256 manifest). Only after intake
  reports no FAIL: `track` → `annotate` → `analyze` → `gate`. Never tune
  anything before the gate verdict is reported.

## 6. Manual annotation protocol (for gate criterion G2)

1. `python phase2/run_phase2.py annotate <session_dir> --n 50 --seed 20260927`
   on 3 sessions (150 frames total, seeded random frame sample).
2. Human clicks the fly's center in each displayed frame; results saved to
   `annotations.csv` (frame, x, y) in the session dir.
3. Self-consistency check: re-annotate 20 frames once; the report includes
   the human repeatability median distance (context for the 2.0 mm G2
   threshold).
4. `python phase2/run_phase2.py gate --analysis <dir>` consumes the
   annotations and computes median error in mm.

## 7. Analysis plan (pre-registered — see gate.yaml for every constant)

Per session: tracking coverage, speed distribution, heading distribution
(circular), pause frequency, bout-duration distribution, directional
persistence (mean resultant length, heading autocorrelation, bout
straightness), wall-following fraction and wall-contact rate, occupancy and
stability of vocabulary states, LOSO decoder accuracy, stimulus-response
permutation test. Cross-session: JSD between state-occupancy distributions,
per-state occupancy ranges. The gate (G1–G4, + secondary G5) is then
evaluated automatically and its verdict recorded.

## 8. What Phase 2 explicitly does NOT do

- No game input of any kind; no game automation; no GPO access. All game
  facts remain UNKNOWN (single-source policy, `docs/GPO_PLAN.md`).
- No closed-loop stimulation (the fly cannot change what the stimuli do).
- No learning claims: nothing in Phase 2 trains the fly, and the analysis
  vocabulary/ladder is fixed in advance.
- No inference from synthetic validation videos to real-fly behavior — they
  validate the measurement CODE only.
- The fly is not expected to understand any game concept (level, quest,
  sword, devil fruit, accessory, NPC, map, inventory, progression) — those
  belong to the computer-side perception/planning stack in later phases.

## 9. Layer separation (target architecture, unchanged)

```
BIOLOGICAL CONTROL LOOP (low level)          COMPUTER SIDE (high level)
REAL FLY                                    GAME SCREEN
  ↓ camera                                    ↓ computer vision
behavior detection                           WORLD MODEL
  ↓ action decoder                           ↓ memory
VIRTUAL INPUT CONTROLLER                     ↓ goal / task / strategy
  ↓ game                                     ↓ high-level command
(game = 2D virtual env first, real game      → into the biological loop
 only after all prior gates pass)              (cue/stimulus channel)
```
