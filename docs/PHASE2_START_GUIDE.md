# PHASE-2 PHYSICAL EXPERIMENT START GUIDE

**Scope:** everything needed to run the real-fly Phase-2 experiment safely
and reproducibly on the bench. No real-fly result is claimed anywhere in
this document — every real-fly outcome remains UNKNOWN until you provide
actual recordings.

**Authority chain:** `phase2/PHASE2_GATE_PREREGISTRATION.md` (gate
PHASE2-GATE-1.0.0 + Amendments 1–2) → `phase2/PHASE2_PROTOCOL.md`
(pre-registered procedure) → this guide (bench-level execution detail).
Where texts differ on a threshold, the pre-registration wins.

**Hard rules for the whole phase**
- The fly is NEVER connected to the game in Phase 2. No game input, no
  game automation, no game access. All game facts stay UNKNOWN.
- Non-invasive observation + benign visual stimuli only (see §J).
- No threshold is ever changed after seeing animal data. Fixes go through
  the apparatus (lighting, focus, cleaning) or a registered amendment.
- The software FAILS CLOSED: `record` refuses fly sessions unless
  `preflight` and `blank` passed within 24 h under the same code/config
  hash. `--force` exists only for rig debugging and is recorded as a
  protocol deviation.

**First-time bring-up order** (new rig): A shopping → B arena build →
C/D/E camera+lighting+LEDs → F calibrate → G blank test →
`python phase2/run_phase2.py validate --duration 120` on the analysis
machine → H fly sessions.
**Every recording day** (rig already built): F-daily recheck → preflight →
blank → fly sessions → backup.

---

## A. Shopping list

Every item: REQUIRED spec, MINIMUM acceptable, why, and the failure it
prevents. Nothing expensive is needed — total ≈ $150–300 excluding the
computer. Do not buy laboratory-grade gear beyond this list.

| # | Item | REQUIRED spec | MINIMUM acceptable | Why required | Failure it prevents |
|---|---|---|---|---|---|
| 1 | Camera | ≥ 30 fps at 1280×960 or 1920×1080; manual/lockable exposure AND gain; lockable or fixed focus; rigid mount thread | Used USB webcam, 1080p30, exposure lockable via vendor tool or `v4l2-ctl` (Logitech C922 class) | Pre-registered `min_fps: 30`; ≥ 960 px frame height is needed to fit arena ≥ 600 px + marker margin; auto-exposure pumps the background model | G1 coverage loss (exposure pumping), marker geometry failure, dropped-frame sessions refused by `record` |
| 2 | Lens (only for CS-mount board cams) | 8–25 mm CS-mount, manual iris + manual focus. 12 mm is the default pick | 16 mm CS (slightly taller rig) | Sets working distance ≈ 250–450 mm with the arena ≥ 600 px wide; focus stays put | Focus creep → G2 accuracy drift; nightly re-calibration |
| 3 | Arena ring | Clear acrylic ring, inner Ø 95 mm ± 1 (accept 90–100), wall 3 mm, height 15 mm, ground top edge | 100×15 mm glass Petri dish IF its measured inner floor Ø is 90–100 mm | Protocol arena geometry (ROI, 10 mm wall band); measured geometry feeds calibration | Unmeasured geometry → G2 error; warped/scratched wall → reflections |
| 4 | Arena lid | Clear acrylic 105×105×3 mm with one 15×15 mm vent window covered by 0.3 mm nylon mesh (glued) | Any clear, fly-proof lid that seats fully | Flies escape through surprisingly small gaps; escapes are lost subjects | Lost subject mid-experiment; lid reflections (use AR-clean acrylic, keep it dusted) |
| 5 | Floor mat | Matte light-gray card/felt 110×110 mm, uniform, zero printed texture | Matte gray craft card | Dark-blob detector needs a bright, texture-free field | Texture = permanent false foreground → blank test FAIL |
| 6 | Lighting | One white LED panel ≥ 150×150 mm, 4000–6500 K, CONSTANT-current (non-PWM), + frosted diffuser sheet | Any panel that passes `preflight` (frame-mean gray std ≤ 2.0) | Uniform bright field; PWM flicker is invisible to the eye and fatal to background subtraction | Blank/preflight FAIL; flicker-induced false detections |
| 7 | Stimulus LEDs | 4 × diffused white/warm-white 5 mm LEDs, 4 × 220–470 Ω resistors, 4 toggle switches, hookup wire, 4×AA battery holder | Same, with alligator clips instead of switches | N/E/S/W benign visual stimuli + in-view timing markers (read-back needs ≥ 40 gray-level on-delta); batteries = ripple-free constant current | Untrusted stimulus timing (timing is read back from video, so manual switching is fine); unmatched sides (use 4 identical parts) |
| 8 | Enclosure | Light-tight box ≈ 350×350 mm footprint, height per camera build (webcam ≈ 250 mm; CS 12 mm ≈ 400–450 mm), matte-black lining, light-trapped vent at the bottom | Sturdy cardboard box + matte-black paper + taped seams | Blocks window light, monitor glow, people walking past | Lighting-change false positives; session-to-session variability |
| 9 | Mounting | Rigid base plate ≥ 300×300 mm (MDF/acrylic); camera arm or retort stand; printed reference card (see §F) fixed beside the arena; position marks for arena + 4 LEDs | Anything rigid that moves as ONE piece | Repeatability between sessions; calibration stability | Camera nudge → calibration drift (G2); geometry change → marker misalignment |
| 10 | Cables + power | USB cable for the camera (length ≥ enclosure height + 0.5 m), powered USB hub if the bus sags; LED panel supply (usually included); 4×AA for the stimulus LEDs | — | Stable power = stable frames and lighting | Brownout frame drops (refused at intake); mains ripple modulating markers |
| 11 | Computer | Any machine with USB 2.0+, the repo Python env (`requirements.txt`), ≥ 10 GB free disk per recording day, mains power (laptop ON CHARGER) | — | 10-min 1080p30 ≈ 0.6–1.5 GB/session; CPU throttling drops frames | Dropped-frame sessions (intake FAIL); lost sessions to disk-full |
| 12 | Consumables | D. melanogaster from a registered stock center / education supplier (NOT wild-caught); vials + standard medium; aspirator (mouth pooter or pump); spares of mat + lid | — | Species certainty, parasite-free, legal cleanliness; gentle transfer | Misidentified species; stressed/injured flies (welfare + data quality) |

Optional (not required): Arduino Nano clone (~$8) to switch the stimulus
LEDs from a script. Manual switching is fully supported because exact
timing is recovered from the video margin markers.

---

## B. Arena construction

All dimensions millimetres. N/E/S/W are defined **in the camera image**:
camera upright (never rotated), N = image top. The arena sits centered.

### Parts
1. Acrylic ring: ID 95.0 ± 1.0, OD 101, H 15, clear, polished edges.
2. Floor mat: 110 × 110 matte light-gray card (≈ 60 % gray), uniform.
3. Lid: 105 × 105 × 3 clear acrylic; 15 × 15 center window, 0.3 mm nylon
   mesh glued over it (fly-proof ventilation).
4. Base plate ≥ 300 × 300 with marked positions (below).

### Assembly
1. Mark the arena center on the base plate; draw the N–S and E–W axes
   through it (these are the compass lines).
2. Mark the four LED positions on the compass lines at **75–80 mm from the
   arena center** (≈ 25–30 mm outside the ring wall).
3. Fix the printed reference card flat on the base, on the SW diagonal,
   ~70 mm from the arena center (outside the future camera ROI, clear of
   all four marker boxes at the frame edges).
4. Lay the floor mat centered under the ring; place the ring on it (no
   glue — removable for cleaning).
5. Place the four LED modules on their marks, domes facing the camera.
6. Mount the camera arm; camera looks straight down at the arena center.
7. Stand the LED panel + diffuser above (enclosure lid or a frame at
   150–250 mm above the floor).
8. Close the enclosure; route cables through a light-trapped slot.

### Geometry contract (software-enforced at calibrate/preflight)
- Arena spans **≥ 600 px** across in the image (raise the camera if not).
- **≥ 60 px clearance** (recommended ≥ 150 px) between arena rim and every
  frame edge — the four 30×14 px stimulus marker boxes sit at an 18 px
  margin from the frame edges and MUST be outside the arena ROI.
  ⇒ 1280×720 cannot satisfy this; use 1280×960 or 1920×1080.
- Camera perpendicular: X vs Y scale anisotropy ≤ 2 % (checked in §F).

---

## C. Camera setup

1. **Position:** top-down, centered over the arena, image axis vertical.
   Never rotate the camera (image-up = arena N = behavioral FORWARD).
2. **Height (lens-dependent).** Set it empirically: open a live preview
   and adjust until the arena (with the lid on) spans 600–800 px across
   and the rim keeps ≥ 150 px from every frame edge. Typical starting
   points: webcam ≈ 150–220 mm; CS 8 mm ≈ 230 mm; 12 mm ≈ 340 mm;
   16 mm ≈ 450 mm above the floor.
3. **Lock everything:** exposure manual + fixed, gain manual + minimum,
   focus manual, white balance off/fixed. On Linux:
   `v4l2-ctl -d /dev/video0 --set-ctrl=exposure_auto=1` (menu value 1 =
   manual) then set `exposure_absolute`; on Windows use the vendor tool.
   Auto-exposure WILL fail the preflight (frame-mean gray std > 2).
4. **Rigid:** no dangling cables pulling on the camera; the camera, arena,
   LEDs, and card all live on the ONE base plate inside the enclosure.
5. The probe at `record` start re-measures fps; below 25 fps measured the
   session is refused (the pre-registered requirement is ≥ 30).

## D. Lighting setup

1. One diffused white LED panel, constant ON, ABOVE the arena (150–250 mm
   above the floor or mounted at the enclosure top); frosted diffuser
   between panel and arena so no LED hot-spot images on the floor.
2. No dimmers, no PWM drivers, no desk lamps, no monitors facing the box.
   Room lights OFF during every check and session; enclosure closed.
3. The stimulus LEDs are the ONLY things that change brightness, and only
   during stimulus sessions / blank blink checks.
4. Log room temperature (22–25 °C required) and humidity each session
   (`record --temperature --humidity`).
5. Reflection control: matte floor + matte-black enclosure lining; keep
   the lid dusted; if the blank test flags static dark features, clean the
   mat/lid and re-run — do not proceed with a dirty rig.
6. Flicker control is verified, not assumed: preflight FAILs if frame-mean
   gray std > 2.0 (aliasing PWM shows up exactly there).

## E. Stimulus LED setup

1. Four IDENTICAL modules (same LED, same resistor, same battery pack) at
   the marks from §B: N and S on the vertical axis, E and W on the
   horizontal, 25–30 mm outside the ring, sitting on the base plane.
2. Each LED's bright image must land INSIDE its marker box near the frame
   edge: N = top-center, S = bottom-center, W = left-center, E =
   right-center of the image. Verify with the `calibrate` preview; the
   blank test confirms read-back automatically (each side must be seen).
3. Benign by construction: diffused, dim, warm-white, battery-driven, no
   UV, no heat concentration near the fly.
4. Switching: manual toggles following the console prompts (`record`
   prints the pre-registered schedule; `blank` prints the blink
   timetable). You do NOT need frame-precise timing — the tracker reads
   the exact on/off moments back from the video margin markers, and the
   stimulus analysis uses those recovered times, never your watch.

---

## F. Calibration

**Reference card** (print once, keep mounted): 100 × 100 mm print
containing (a) two crosshair dot pairs exactly 50.00 mm apart along X and
along Y, (b) a sharp high-contrast grid/fan for the focus check. Print at
exactly 100 % scale, then VERIFY the dot spacing with steel calipers —
if your printer scaled it, enter the measured value at the prompt.

**Full calibration** (once per rig build, and after any physical change):
```
python phase2/run_phase2.py calibrate --camera 0 --arena-mm 95
```
1. Click the two X ruler dots, confirm the real distance (mm).
2. Click the two Y ruler dots, confirm the distance — the X/Y mismatch
   (anisotropy) must be ≤ 2 % or the camera is tilted/card not flat.
3. Click ≥ 5 points spread around the INNER edge of the arena wall.
4. The tool checks: circle-fit residual ≤ 3 px, arena ≥ 600 px, rim
   clearance, marker boxes outside the ROI, fitted radius vs the measured
   95 mm inner Ø within ± 3 mm. On success it saves
   `phase2/data/rig/calibration.json` and prints the values — copy them
   into your lab notebook.
5. Re-run `calibrate --verify` any time for a non-interactive re-check.

**Daily re-verification** (before preflight, no full re-calibration if the
rig was untouched):
```
python phase2/run_phase2.py calibrate --recheck-scale
```
Click the X dot pair again; ≤ 1 % drift vs the stored scale = PASS. If it
drifts, do a full `calibrate` again (and then preflight + blank).

---

## G. Blank-arena test (REQUIRED before any animal session)

Runs after `preflight`, with NO fly anywhere in the enclosure:
```
python phase2/run_phase2.py preflight --camera 0     # ~20 s, must PASS
python phase2/run_phase2.py blank --camera 0          # >= 3 min
```
During the blank, follow the printed LED timetable (N ON 10 s at t≈30 s,
E at t≈60 s, S at t≈90 s, W at t≈120 s — switch each off after 10 s).

The software determines, automatically and fail-closed:
| Check | Requirement |
|---|---|
| B1 tracker false positives | **0** confident detections in the whole blank |
| B2 any detections | ≤ 0.1 % of frames |
| B3 background motion | ≤ 1 % of frames with foreground ≥ min blob area |
| B4 lighting | frame-mean gray std ≤ 3.0, drift ≤ 3.0 |
| B5 reflections / static dark features | **0** fly-like dark spots inside the arena ROI |
| B6 sensor noise | median temporal noise ≤ 8 gray levels |
| B7 stimulus marker channel | all four N/E/S/W markers read back from the video |

A PASS also stores the day's reference background
(`phase2/data/rig/background.png`) — every fly session that day is tracked
against it (Amendment 2). **If the blank FAILS: no animal session.** Fix
the indicated cause (clean mat/lid, seal the enclosure, kill stray light,
re-seat cables, replace flickering panel) and re-run. Do not proceed, and
never `--force` an animal session past a failed blank.

---

## H. Real-fly recording procedure

Pre-registered structure (PHASE2_PROTOCOL §5) — do not improvise:
1. **Per recording day, before the fly:** `calibrate --recheck-scale` →
   `preflight` → `blank` (all with no fly present, enclosure closed).
2. **Introduce the fly** by gentle aspiration (no CO₂, no cold, no
   chemicals within 24 h of a session). Close the enclosure.
3. **Habituate ≥ 30 min** inside the closed enclosure (LEDs off, camera
   may run — no session recording needed yet).
4. **Baseline sessions ×3** (≥ 10 min each, no stimulus):
```
python phase2/run_phase2.py record --minutes 10 --label baseline_01 \
    --fly-id F01 --age 5 --sex M --temperature 23 --humidity 45
```
5. **Stimulus sessions ×3** (≥ 10 min each; session-index 0/1/2 fixes the
   seeded N/E/S/W schedule; follow the console ON/OFF prompts):
```
python phase2/run_phase2.py record --minutes 10 --label stimulus_01 \
    --stimulus --session-index 0 --fly-id F01 --age 5 --sex M \
    --temperature 23 --humidity 45
```
6. Sessions spread over **≥ 2 days**, same time-of-day ± 1 h, **≥ 24 h
   between this fly's sessions**, ≤ 30 min per session, ≤ 6 sessions per
   fly per week. Labels are unique (the software enforces it).
7. **Stop early** if the fly is immobile > 10 min, escapes, or shows
   impaired locomotion — say why in `--operator-notes`.
8. **After the recordings:** annotate 3 sessions (50 frames each,
   seeded): `python phase2/run_phase2.py annotate data/sessions/<dir>
   --n 50` (needs a display; click the fly's center). Re-annotate 20
   frames once on one session for the human self-consistency check.
9. **Back up the day** (§I) before touching anything else.
10. Protocol changes: if something genuinely must change, it is recorded
    as an amendment BEFORE the modified procedure is used for
    confirmatory analysis. Thresholds are never moved after the fact.

---

## I. Data backup procedure

After every recording day:
```
python phase2/run_phase2.py backup --to /path/to/usb-drive
```
- Copies every new session into `<usb>/flyagent-backup/sessions/<name>/`,
  re-hashes every copied byte, and writes a SHA-256 manifest.
- Never overwrites: already-backed-up-and-unchanged sessions are skipped;
  a changed raw file raises an immutability ALARM (investigate — raw
  recordings must never change).
- A repo snapshot (config hashes + git revision) travels with each
  backup, so it is self-describing.
- Keep at least one additional off-site/cloud copy of the backup folder.
- Never edit anything inside `data/sessions/<...>/` — derived files
  (tracks.csv, per_frame.csv, annotations.csv) are regenerated by the
  software, never hand-edited.

---

## J. Safety / welfare checklist (all boxes required, every session)

- [ ] Observation + benign visual stimuli ONLY.
- [ ] NO surgery, neural implants, genetic modification, toxic substances,
      deliberate injury, extreme heat/cold, electric/mechanical shock,
      aversive stimuli, or starvation/dehydration as a manipulation —
      not in this or any phase of the project.
- [ ] Session ≤ 30 min; ≥ 24 h rest for this fly; ≤ 6 sessions/fly/week.
- [ ] Stop early (and note it) if immobile > 10 min, escape, or impaired
      locomotion.
- [ ] Transfer by gentle aspiration only; no CO₂/cold/chemical
      immobilization within 24 h before a session.
- [ ] Flies from a registered stock/education supplier; standard medium;
      22–25 °C; 12:12 light:dark.
- [ ] **Institutional/legal:** if your institution or jurisdiction
      requires approval, registration, or oversight for animal
      observation experiments, obtain it BEFORE the first session.
      Institutional and legal requirements prevail over this protocol.
- [ ] No game connection of any kind during Phase 2.

---

## K. Phase-2 analysis commands (fixed order when recordings come back)

When you return with recordings, the sequence is fixed — **no tuning
before the PASS/FAIL verdict is reported**:
```
# 1. integrity + protocol compliance FIRST (no analysis):
python phase2/run_phase2.py intake data/sessions
# 2. track each session (uses each session's recorded recipe):
python phase2/run_phase2.py track data/sessions/<dir>          # x6
# 3. annotate 3 sessions (50 frames each) if not already done:
python phase2/run_phase2.py annotate data/sessions/<dir> --n 50
# 4. full behavioral analysis of all six sessions:
python phase2/run_phase2.py analyze data/sessions/<a> ... --out results/phase2
# 5. evaluate the pre-registered gate:
python phase2/run_phase2.py gate --analysis results/phase2 \
    --annotations data/sessions/<a>/annotations.csv \
                  data/sessions/<b>/annotations.csv \
                  data/sessions/<c>/annotations.csv
```
Steps 1–2 verify and prepare; 3–4 measure; 5 reports PASS or FAIL with
the pre-registered diagnosis attached to any failed criterion. The
integrity manifest (SHA-256) is written during step 1.

## L. Phase-2 pass/fail criteria (pre-registered PHASE2-GATE-1.0.0)

**Data requirements** (all must hold, else the gate is not evaluable):
≥ 3 baseline + ≥ 3 stimulus sessions, each ≥ 10 min at ≥ 30 fps, spread
over ≥ 2 recording days, ≥ 150 manually annotated frames (50 × 3
sessions, seeded sample).

**PASS requires ALL of G1–G4; G5 is secondary/informative:**
| ID | Criterion | Threshold |
|---|---|---|
| G1 | Tracking coverage per session (detections at confidence ≥ 0.5) | ≥ 0.90 |
| G2 | Median manual-annotation position error (≥ 150 frames) | ≤ 2.0 mm |
| G3 | State occupancy + stability at the SELECTED vocabulary level | every state ≥ 0.05 in EVERY session and within [0.5, 2.0]× its across-session mean |
| G4 | Leave-one-session-out nearest-centroid decoder accuracy | ≥ 0.80 |
| G5 | Stimulus response (permutation test per stimulus session) | informative only: response in ≥ 2 of ≥ 3 stimulus sessions suggests a usable steering channel |

**Vocabulary ladder** (the data choose, largest level that satisfies
G3+G4 wins): L_A 5 actions → L_B 4 → L_C 3 → L_D 2 → L_E 1.
**Minimum for directional game control: L_C.** If only L_D/L_E hold, the
project documents the pivot instead of forcing 5 actions.

**Consequences (fixed):**
- **G1–G4 PASS** → proceed to Phase 3: closed loop with the real fly in
  the 2D VIRTUAL environment (still no game).
- **Any FAIL** → work the pre-registered diagnosis map
  (tracking/calibration → apparatus; occupancy → elicitation/circadian;
  stability → variability, add sessions; decoder → window/features) —
  improve the apparatus, register an amendment if the procedure changes,
  and re-run. Thresholds are never changed after seeing real-fly data.
- **In no case** does Phase 2 connect the fly to game input or make any
  claim about game performance. Every game fact remains UNKNOWN until
  experimentally discovered in later phases.
