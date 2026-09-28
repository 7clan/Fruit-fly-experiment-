# ROADMAP — Digital Drosophila plays a video game (D1–D24)

Supersedes the real-fly roadmap (archived at
`legacy/docs/ROADMAP_REAL_FLY.md`). Governing specification:
`docs/MASTER_SPEC.md`. Development order is D1–D24; the previous 15-step
Windows priority is preserved in intent and renumbered into this table
(`docs/WINDOWS_RUNTIME_SPEC.md` W12).

Research question:

> "How much useful autonomous game-playing behavior can be produced by a
> connectome-derived Drosophila brain model (Shiu et al. LIF model on
> FlyWire v783) acting through engineered sensory and motor interfaces,
> with computer vision, memory, and planning — measured against control
> conditions that remove each contribution?"

The answer is not assumed to be positive. Control conditions A–E
(MASTER_SPEC §8) exist to measure it either way.

Autonomy ladder (measured from D18 on): 0 = human controls everything;
1 = human gives exact actions; 2 = human gives a sequence; 3 = human gives
a task; 4 = human gives a goal; 5 = human gives only "maximize long-term
progression".

---

## Stage I — the brain (D1–D6)

| Step | WHAT | GATE | Status |
|---|---|---|---|
| D1 | Install + reproduce the Shiu et al. whole-brain LIF model from a pinned commit | Model runs; its example/tutorial executes | **DONE 2026-09-28** |
| D2 | Configure/test FlyWire v783 connectome data | Data loads; SHA-256 manifest verifies; structural counts recorded and cross-checked against the model's published values | **DONE 2026-09-28** (138,639 / 15,091,983) |
| D3 | Benchmark the whole-brain simulation | Metrics table: RAM, CPU, initialization time, simulation speed (bio-s per wall-s), neuron count, synapse/connection count | **DONE 2026-09-28** (0.148 bio-s/wall-s here) |
| D4 | Identify sensory/visual neural populations (optic lobe targets, other sensory neuropils) in the model | Documented population inventory with model node IDs and stimulation entry points | **DONE (2026-09-28)** - see D4_D6_NEURAL_IO_MAP.md + brain/data/io_map/ |
| D5 | Identify mushroom-body/dopamine learning circuitry (Kenyon cells, MBONs, PAM, PPL1) | Documented inventory + literature notes (MB research base for Stage C plasticity) | **DONE (2026-09-28)** - incl. three-tier reward interface proposal |
| D6 | Identify descending/motor-related populations | Documented inventory; candidate motor readout sites | **DONE (2026-09-28)** - 1,305 DNs; MDN/DNp09/BPN/RRN/GF/FG/BB/BRK resolved to v783 IDs; GATE 2 PASS |

**Gate 1 (blocks everything):** the connectome-derived digital Drosophila
brain runs locally and produces reproducible neural activity — **PASSED
2026-09-28** (`DIGITAL_BRAIN_REPORT.md`).

## Stage II — artificial-environment closed loop (D7–D13)

> D7–D10 were re-scoped by the user's Gate-3 task brief (D7 = engineering
> runtime, D8 = sensory encoder + artificial environment, D9 = motor
> decoder, D10 = closed loop + control matrix). Delivered accordingly.

| Step | WHAT | GATE | Status |
|---|---|---|---|
| D7 | Dual-mode engineering runtime: REFERENCE (canonical full brain, single-run) vs INTERACTIVE (chunked stepping, dynamic rates, checkpoints, profiling); chunk-size study 25/50/100/200 ms; C++ standalone replay; optional sparse variant | Chunked interactive stepping BIT-IDENTICAL to reference at all four chunk sizes; checkpoints deterministic; interactive 0.099 bio-s/wall-s @ 506 ms/chunk (this box); cpp replay bit-identical & 3.0x; sparse variant honestly workload-bound | **DONE (2026-09-28)** — `brain/results/d7_runtime/` |
| D8 | Artificial 2D environment + sensory encoder onto Gate-2 populations (LC9-L/R target channels, LPLC2/LC4 looming); open-loop probes with full logging; pre-registered calibration ladder | Lateralized P9 recruitment (left 20 Hz / right 72 Hz, contra 0), looming->GF 90–118 Hz, silent baseline, sensory→DN latency ≤ 50 ms chunk; rung 1 frozen BEFORE closed loop | **DONE (2026-09-28)** — `brain/results/d8_encoder/`, sign-inversion regression caught by unit tests pre-run |
| D9 | Motor decoder: explicit rules (thresholds, P9 differential + hysteresis, min duration, STOP failsafe, conflict logging) over side-split DN readouts | Decoder reads ONLY DN spike counts (API-contract test); demo + unit tests pass; conflicts logged not hidden | **DONE (2026-09-28)** — `brain/scripts/d9_decoder.py` |
| D10 | Closed-loop target approach + looming escape; 5 conditions x 5 scenarios x 3 reps pre-registered; no hidden shortcut; bit-exact replays; basic dashboard | Intact 9/9 target successes + 3/3 escapes; shuffled-sensory 0/9; shuffled-motor 0/9; random 0/9 (all bootstrap CI [1.00, 1.00]); replays bit-identical incl. C++ standalone | **DONE (2026-09-28) — GATE 3: PASS** — `D7_D10_FIRST_CONTROL_REPORT.md`, `brain/results/d10_closed_loop/` |
| D11 | Reward-modulated learning (MB/dopamine plasticity on the Gate-3 runtime; pairing+test paradigm) | Learning improves task metric vs control C (fixed connectome); engineered additions labeled | **DONE - behavioural criterion FAIL (v1), honest report** |
| D11v2 | GATE 4 v2: source-audited associative learning (corrected transmitter-anchored MBON valence; compartment-matched PAM01-11 US; balanced-boundary open-loop assay P(left)=0.500; two independent verdicts G4A neural / G4B behavioural) | G4A and G4B criteria in GATE4_V2_PREREG.md section 8, fail-closed | **IN PROGRESS (matrix running; v1 FAIL permanently recorded)** |
| D12 | Internal-state visualization (real variables only, MASTER_SPEC §9) | Dashboard shows live actual variables; no invented decorative activity | DONE |
| D13 | Memory + multi-step artificial tasks (delayed reinforcement + cue-memory conditions A/B) | Multi-step success above chance-chaining baseline | DONE (results in report; behavioural change still Gate-4-blocked) |

## Stage III — Windows application + passive game perception (D14–D17)

| Step | WHAT | GATE | Status |
|---|---|---|---|
| D14 | Windows dashboard (DigitalFlyLab UI shell; OBSERVER/GAME/BRAIN/SCIENTIST/REPLAY layouts; real values only) | Runs on Windows with live real values; Linux never claimed as Windows validation | pending |
| D15 | Windows window capture (WGC preferred; single pipeline serves CV + mirror; only the target window) | Live mirror + CV fed from ONE stream on Windows; emergency-stop hotkey demonstrated | pending |
| D16 | Passive game computer vision (screen states, entities, OCR; confidence on every output) | Screen-state classifier + entity/OCR readback at target FPS on labeled session | pending |
| D17 | Non-Roblox GPO knowledge database (UNVERIFIED GUIDE KNOWLEDGE -> VERIFIED IN GAME tiers; direct observation wins) | Schema + ingestion + verification workflow; source policy enforced | pending |

## Stage IV — game interaction (D18–D23)

| Step | WHAT | GATE | Status |
|---|---|---|---|
| D18 | Game movement (brain -> action decoder -> Windows input adapter) | Measurable sustained displacement; emergency stop verified; user's ToS/compliance review recorded FIRST | pending |
| D19 | Interactions/combat | >= 1 interaction loop produces observable state change; unknown-object probe protocol runs | pending |
| D20 | Quest/level progression | Measurable progression over sessions (visible EXP/level readback) | pending |
| D21 | Equipment/fruits/accessories | Before/after effect memory for >= 1 item, verified on retest | pending |
| D22 | Exploration + strategy selection | Strategy selector beats random-strategy baseline over >= 3 sessions | pending |
| D23 | Long-term autonomous progression | Autonomy 4–5 sustained across multi-hour sessions; full A–E control-condition attribution analysis | pending |

## Stage V — optional automation (D24)

| Step | WHAT | GATE | Status |
|---|---|---|---|
| D24 | Optional launcher/menu automation | User-approved, ToS-reviewed, account policy intact (W6) | pending |

---

## What is deliberately NOT in this roadmap

- No real animal, no physical apparatus, no animal-care workflow
  (corrected 2026-09-28; old direction archived in `legacy/`).
- No renderer embedding, no DLL injection, no hidden-game-data access, no
  game APIs, no memory inspection: window capture + screen observation
  only, per the source policy (MASTER_SPEC §6, WINDOWS_RUNTIME_SPEC W2).
- No hard-coded game mechanics. Non-Roblox guide knowledge is stored as
  UNVERIFIED until verified by direct observation.
- No claiming the system "understands" the game or "experiences"
  anything. Every report separates connectome data / modeled dynamics /
  engineered interfaces / added learning / external systems.
- No bypassing the Drosophila brain (no CV -> direct action).
