# HARDWARE — the exact physical setup

> **PHASE-2 BUILD?** This file is the original Phase-1 hardware sketch.
> For the real-fly Phase-2 experiment, build instead to
> **`docs/PHASE2_START_GUIDE.md`** (the authoritative shopping list, arena
> construction dimensions, camera/lighting/LED setup, calibration and
> blank-test procedure). Phase-2 requirements supersede this sketch where
> they differ: circular arena 90–100 mm inner diameter, camera ≥ 30 fps at
> ≥ 1280×960 or 1920×1080 (720p cannot fit the marker margin geometry), and
> the pre-session preflight + blank-arena checks are mandatory.

## 1. System overview

```
            [ light-tight enclosure ]
  ┌─────────────────────────────────────────┐
  │      diffused LED panel (constant)      │   ← no flicker (see 2.3)
  │                                          │
  │   ┌─── transparent arena (fly) ───┐      │
  │   │  matte gray floor mat         │      │
  │   └───────────────────────────────┘      │
  │    cue LED strips / e-ink side markers   │   ← benign visual stimuli
  │                                          │
  │            [ camera, top-down ]          │   ← 60+ fps, macro lens
  └──────────────────────┬───────────────────┘
                         │ USB
                    [ computer ]  ← Python, this repo
```

## 2. Bill of materials (with selection criteria)

### 2.1 Camera (~$60–150)
Criteria, in priority order:
1. >= 60 fps at the working resolution (fly walking is fast; velocity
   estimates need >= 30 effective fps; 60 gives margin).
2. Manual exposure/gain control via UVC/V4L2 (auto-exposure will pump the
   background model and destroy MOG2 stability).
3. Resolution such that the arena spans >= 800 px across (a 3 mm fly at
   90 mm arena width = ~27 px; comfortably above the tracker's area filter).
4. Global shutter preferred (rolling shutter distorts fast flies);
   acceptable at 60+ fps.
5. Mount thread (CS/M12 or tripod) for rigid fixing — ANY camera wobble
   breaks the calibration.

Concrete options: used industrial USB cam (IMX477/IMX291 module, ~$40–90)
+ 16–25 mm CS lens; or a manual-focus webcam (Logitech C920 class) as the
minimum viable option (60 fps at 720p, exposure controllable via v4l2-ctl
on Linux; on Windows use the vendor tool + lock settings).

### 2.2 Arena (~$20–40)
- Rectangular clear acrylic/glass chamber, ~90 x 60 x 15 mm interior
  (matches `tracking.yaml: arena_mm`). Rectangular beats round: zone/edge
  logic is grid-based.
- Removable lid with 5–10 tiny air holes; lid must CLOSE (flies escape
  otherwise; escapes are lost subjects, not just lost data).
- Matte gray floor insert (uniform felt/paper): high contrast vs the fly,
  zero texture (texture = false foreground).
- Anti-reflective top surface or slight camera tilt to kill LED glare on
  the lid.

### 2.3 Lighting (~$15–30)
- One large diffused white LED panel ABOVE the arena (constant ON).
- CRITICAL: no PWM flicker at exposure times — test by recording a blank
  wall and checking per-frame variance; flickering LEDs look fine to the
  eye and destroy background subtraction. High-quality panel or
  current-driven LEDs; avoid cheap dimmable strips.
- The whole rig goes inside a light-tight box (cardboard + matte black
  lining): blocks window light, monitor glow, and people walking past.

### 2.4 Cue stimuli (~$15–35)
Phase 2 requirement: presentable at each of the four arena edges.
- Option A (simplest): four small LED assemblies (one per edge), diffused,
  driven by an Arduino/ESP32 over USB serial from `source.set_cue()`
  (the hook already exists in the code).
- Option B (richest): a small display/tablet under a TRANSPARENT arena
  floor — arbitrary patterns; requires the tracker to tolerate the
  display content (ROI masking + stable patterns; test carefully).
- Brightness: dim, comparable to arena illumination; NO UV; no heat
  concentration near the flies.

### 2.5 Rig + misc (~$25)
- Camera stand (retort stand or 3D-printed mount), 150–250 mm above the
  arena; everything on one rigid base that can be covered by the enclosure.
- Aspirator (mouth pooter or pump) for gentle fly transfer; spare vials.

### 2.6 Flies + husbandry (~$30–60)
- D. melanogaster from a registered stock center or education supplier
  (Carolina Biological, university stock centers). NOT wild-caught:
  species certainty, parasite-free, legal cleanliness.
- Instant Drosophila medium, vials, plugs; room at 22–25 C with a
  12:12 light cycle (a cheap outlet timer suffices).

Total budget: roughly $150–300 excluding the computer.

## 3. Setup procedure (when hardware arrives — Phase 1)

1. Assemble the rig; camera fixed rigidly; LED panel on; enclosure closed.
2. `python main.py calibrate` — click the four arena corners; the tool
   writes `config/calibration.npz`.
3. Set `tracking.yaml: source.type: webcam`, `source.arena_mm: [90, 60]`,
   `calibration.file: config/calibration.npz`.
4. Record 60 s with no fly; run a blank session; the tracker should find
   NOTHING (zero false positives) — if it does, fix lighting before
   proceeding.
5. Record 5 min with a fly; `python main.py run --controller fly
   --source video --video recording.mp4 --trials 5` (offline mode);
   inspect `frames` table: fly position must track continuously.
6. Only then: Phase 1 gate measurements (docs/ROADMAP.md).

## 4. Failure modes and their fixes

| symptom | cause | fix |
|---|---|---|
| tracker finds phantom blobs | lighting drift / glare | enclosure, kill auto-exposure |
| fly fades while paused | learning rate too high | keep lr=0.0005; enclosure for stability |
| position jumps to arena edge | cue/stripe bleed into ROI | stripes stay OUTSIDE the calibrated arena rect |
| velocity estimates noisy | low fps or EMA too weak | 60 fps; tune `vel_span`/`vel_alpha` |
| calibration drifts over days | camera nudged | re-run calibrate; compare corners in manifest |
