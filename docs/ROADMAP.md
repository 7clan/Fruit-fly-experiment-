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
| D1 | Install + reproduce the Shiu et al. whole-brain LIF model from a pinned commit | Model runs; its example/tutorial executes | **IN PROGRESS** |
| D2 | Configure/test FlyWire v783 connectome data | Data loads; SHA-256 manifest verifies; structural counts recorded and cross-checked against the model's published values | **IN PROGRESS** |
| D3 | Benchmark the whole-brain simulation | Metrics table: RAM, CPU, initialization time, simulation speed (bio-s per wall-s), neuron count, synapse/connection count | **IN PROGRESS** |
| D4 | Identify sensory/visual neural populations (optic lobe targets, other sensory neuropils) in the model | Documented population inventory with model node IDs and stimulation entry points | pending |
| D5 | Identify mushroom-body/dopamine learning circuitry (Kenyon cells, MBONs, PAM, PPL1) | Documented inventory + literature notes (MB research base for Stage C plasticity) | pending |
| D6 | Identify descending/motor-related populations | Documented inventory; candidate motor readout sites | pending |

**Gate 1 (blocks everything):** the connectome-derived digital Drosophila
brain runs locally and produces reproducible neural activity
(MASTER_SPEC §13).

## Stage II — artificial-environment closed loop (D7–D13)

| Step | WHAT | GATE | Status |
|---|---|---|---|
| D7 | Simple artificial visual environment (reuse the Phase-1 2D arena where practical) | Env steps + renders headless and reproducibly | pending |
| D8 | Sensory encoder: env/game state -> stimulation of D4 sensory populations (defensible mapping documented; confidence carried) | Encoder output drives measurable brain activity changes | pending |
| D9 | Motor decoder: neural activity -> action vocabulary (all emissions logged) | Decoder reads D6 populations; actions reproducible from identical activity | pending |
| D10 | Closed-loop target navigation in the artificial env | Brain-in-the-loop navigation beats control A (random); no CV->action bypass anywhere in the path | pending |
| D11 | Reward-modulated learning (Stage B readout first; Stage C MB plasticity where defensible) | Learning improves task metric vs control C (fixed connectome); engineered additions labeled | pending |
| D12 | Internal-state visualization (real variables only, MASTER_SPEC §9) | Dashboard shows live actual variables; no invented decorative activity | pending |
| D13 | Memory + multi-step artificial tasks | Multi-step success above chance-chaining baseline | pending |

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
