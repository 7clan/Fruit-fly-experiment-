# D7–D10 — First Autonomous Digital-Fly Control (Gate 3)

**Stage:** closed-loop sensorimotor control with the FIXED connectome-derived
brain — perception → action → environment change, reproducibly.
**No learning** (fixed-brain baseline, per the D10 mandate). **No Roblox, no
computer vision** — everything runs on an artificial 2D environment.
**Canonical brain:** Shiu et al. LIF model on FlyWire v783 (138,639 neurons /
15,091,983 connections), third-party code unmodified, never replaced.

Deliverables in this stage:

| Piece | Artifact |
|---|---|
| D7 dual-mode runtime | `brain/scripts/d7_runtime.py`, studies in `brain/results/d7_runtime/` |
| D8 sensory encoder + arena | `brain/scripts/d8_encoder.py`, `brain/data/d7_d10/sensory_interface.json`, `motor_readout.json`, probes in `brain/results/d8_encoder/` |
| D9 motor decoder | `brain/scripts/d9_decoder.py` (explicit rules, frozen config) |
| D10 closed loop | `brain/scripts/d10_closed_loop.py`, `d10_dashboard.py`, matrix in `brain/results/d10_closed_loop/` |
| Pre-registration | `brain/data/d7_d10/preregistration.json` (frozen before the matrix) |
| D8 calibration freeze | `brain/results/d8_encoder/d8_calibration_frozen.json` |
| Tests | `tests/test_d7_d10.py` (29 brain-free unit tests) |

---

## 1. D7 — Engineering runtime (REFERENCE / INTERACTIVE)

### 1.1 Design

Two modes over the SAME canonical network:

* **REFERENCE** — full canonical brain, numpy runtime, ONE `net.run()` for a
  whole pre-known schedule; the schedule is delivered through a
  `TimedArray`-driven `PoissonGroup` (rates fixed at build time). This is the
  scientific reference execution.
* **INTERACTIVE** — the same full canonical brain, advanced in **discrete
  chunks** (25/50/100/200 ms biological time) with sensory rates re-set
  between chunks from the environment, `store()/restore()` checkpointing,
  per-chunk wall-time + RSS instrumentation. This is the closed-loop engine.

Both use one engineered **dynamic sensory interface** (`SensoryInterface`):
a `PoissonGroup` with one slot per driven neuron, 1:1 `Synapses` adding
`w_syn * f_poi` to `v` per event, driven cells made a-refractory — **exactly
the semantics of the model's own `dbm.poi()`** (which bakes a fixed rate into
compiled code and therefore cannot be used for closed-loop drive). The LIF
equations, parameters, and connectome synapses are untouched.

Why not C++ standalone for the interactive engine? It was measured and
rejected for this role **on architecture, not speed**: Brian2's
`cpp_standalone` compiles the ENTIRE run schedule at build time, so runtime
decisions (rates depending on decoded spikes) cannot influence it. It remains
the fastest FULL-brain route for **offline replay/verification** (§1.4).

### 1.2 Chunked stepping is bit-identical (chunk study)

Workload: pre-registered 2 s schedule (silence → left target → silence →
looming → silence → right target), seed 11, full brain, 160 interface slots.
A single-run REFERENCE was compared with chunked INTERACTIVE runs at
25/50/100/200 ms:

| chunk | identical to reference | wall/chunk (mean) | bio-s / wall-s |
|---|---|---|---|
| 25 ms | **YES (bit-identical)** | 332 ms | 0.075 |
| 50 ms | **YES (bit-identical)** | 493 ms | 0.101 |
| 100 ms | **YES (bit-identical)** | 761 ms | 0.131 |
| 200 ms | **YES (bit-identical)** | 1296 ms | 0.154 |

Why identical: Brian2's `PoissonGroup` draws exactly ONE uniform per neuron
per timestep (`rand() < rates*dt`) regardless of the rate value, and `run()`
boundaries do not touch the RNG — so chunking changes only *when the harness
may change rates*, never the numerical trajectory. Evidence:
`brain/results/d7_runtime/chunk_study.json` (+ per-chunk spike files).

**Honest boundary-quantization finding** (kept as
`chunk_study_misaligned.json`): with the first schedule version (a 500 ms
segment), the 200 ms chunking legitimately shifted a mid-chunk rate change
(1500 → 1600 ms) and therefore diverged — a real property of any discrete
control loop, not a defect. The pre-registered schedule was corrected to be
chunk-aligned and the identity claim applies to aligned schedules.

### 1.3 Checkpointing (checkpoint study)

* fresh process vs `restore('ep0') + seed(s)`: **bit-identical** (sha256
  `415d4737…`)
* mid-episode checkpoint + **recorded reseed** resume: **bit-identical**
* disk `store(filename)` round-trip (238.5 MB checkpoint): **bit-identical**
* 10 s continuous-drive soak: RSS flat (3115.5 → 3115.6 MB; **−0.9 MB**
  “growth”), 96 s wall → no memory leak across chunks.

Documented limitation (Brian 2.10): `store()/restore()` does **not** restore
the Poisson RNG stream. Deterministic episodes therefore use the protocol
`restore() → brian2.seed(episode_seed)` — which the study proves is
bit-reproducible.

### 1.4 Benchmarks (this 2-core box; the user's laptop will re-run)

Bilateral LC9 target drive (the D10 workload), 1 s, 50 ms chunks, seed 11:

| mode | bio-s/wall-s | decision latency | RAM |
|---|---|---|---|
| INTERACTIVE (chunked, full brain) | 0.098 | 512 ms / 50 ms-bio chunk (p95 …) | 2.94 GB |
| REFERENCE (single run, same schedule) | 0.094 | n/a | 2.84 GB |
| spike fidelity interactive vs reference | **bit-identical** | | |

Route comparison on realistic closed-loop workloads (episode replay):

| route | role | speed | RAM | fidelity |
|---|---|---|---|---|
| numpy full brain (INTERACTIVE) | live closed loop (default) | 0.099 bio-s/wall-s (506 ms / 50 ms chunk) | 2.9 GB | bit-identical to reference (verified) |
| C++ standalone full brain | offline replay/verify only (schedule fixed at build time) | 0.294 bio-s/wall-s (3.0×) | 0.68 GB binary | **bit-identical on a real 4.35 s episode** (§4.5) |
| sparse subcircuit (87,960 neurons; OPTIONAL) | low-RAM interactive variant | 0.131 bio-s/wall-s (1.3×) | 2.05 GB | bit-identical on the D7 study workload (1349/1349 trains) but **NOT exact on the real 4.35 s episode** (1089 vs 1099 actives, 99.65 % spikes) — data-driven ROIs are workload-bound; per-episode verification mandatory, never a silent replacement |

### 1.5 What D7 does NOT claim

* No approximation was silently adopted: the INTERACTIVE engine IS the full
  canonical brain; the sparse variant is optional, measured, and
  exactness-verified, never a replacement.
* No speed magic: interactive control runs at ~0.1 bio-s per wall-second on
  this box — 10× slower than real time, fine for offline experiments and
  watchable demos, to be re-measured on the target laptop.

---

## 2. D8 — Sensory encoder on an artificial 2D environment

### 2.1 Environment (Arena2D, pre-registered)

2D arena [−1, 1]², agent (position + heading), bright target disc, dark
neutral background, optional looming threat (expanding disc from t = 0.5 s),
optional obstacle. Idealized geometry (no pixels — the CV/pixel stage is a
later, game-facing step). Kinematics: FORWARD 0.25 u/s, TURN arcs 0.15 u/s @
150°/s, BACKWARD 0.15 u/s, JUMP = 0.30 u escape hop (once), reach radius
0.15 u.

### 2.2 Channels (biological entry points, Gate-2 verified)

| channel | entry population | biological basis |
|---|---|---|
| `target_left` | 40 left-hemisphere LC9 (seeded sample) | LC9 = visual target for directed walking; v783 wiring LC9_left → **932/940** synapses onto LEFT DNs, top partner = left P9 (385 syn) |
| `target_right` | 40 right-hemisphere LC9 | mirror: LC9_right → 1582/1599 onto RIGHT DNs, top = right P9 (866 syn) |
| `looming` | Gate-2 sample verbatim (60 LPLC2 + 20 LC4) | direct LPLC2/LC4 → giant-fiber escape route |

Excluded (recorded, not forced): T4/T5 optic-flow channel — Gate 2 showed
lamina/T4/T5 entries propagate only 1–2 hops below DN threshold under default
parameters, so no optic-flow channel is wired into the closed loop yet.

Encoder tuning (pre-registered): triangular hemifield drive
`rate_L = R·clamp((θ0+b)/θw)`, `rate_R = R·clamp((θ0−b)/θw)` with θ0 = 15°,
θw = 30°, R = 150 Hz; b = target bearing (positive = left of heading).
Looming drive saturates linearly between 15° and 60° of angular size.

**Sign-convention regression caught before any closed-loop run:** the arena
math convention (positive bearing = LEFT) was initially inverted in the
encoder formula and scenarios — the fly would have turned *away* from
targets. Caught by `tests/test_d7_d10.py`, fixed everywhere, regression test
added (`test_arena_to_encoder_end_to_end_sign`).

### 2.3 Open-loop calibration (frozen BEFORE the closed loop)

Pre-registered ladder (rung 1: 40/side @150 Hz; rung 2: full side @150;
rung 3: full side @250) with acceptance criterion: steady ipsilateral P9
≥ 20 Hz AND (ipsi − contra) ≥ 15 Hz on both sides. **Rung 1 passed on both
sides** — no escalation needed:

| probe (1 s, last 500 ms steady) | P9_left | P9_right | GF | notes |
|---|---|---|---|---|
| baseline (no drive) | 0 | 0 | 0 | 0 spikes total — brain silent |
| target_left | **20.0 Hz** | 0.0 | 0 | fully lateralized |
| target_right | 0.0 | **72.0 Hz** | 0 | fully lateralized |
| target_center | 20.0 | 2.0 | 0 | bilateral drive |
| looming | 0 | 0 | **90–118 Hz** | escape channel |
| latency probe (onset at 0.5 s) | — | first spikes in 1st chunk | first spikes in 1st chunk | sensory→DN latency ≤ 50 ms (chunk resolution) |

The left/right asymmetry (20 vs 72 Hz) is the biological wiring asymmetry
(385 vs 866 LC9→P9 synapses), not a tuning choice. Full logs:
`brain/results/d8_encoder/probe_*_rung1.json`; freeze:
`d8_calibration_frozen.json`.

---

## 3. D9 — Motor decoder (explicit rules, frozen)

Readout populations (side-split via FlyWire v783 `side` annotation):
P9_left/P9_right (DNp09), BPN, RRN, MDN, giant fiber (DNp01), Foxglove,
Bluebell, Brake, DN_ALL_left/right, DN_LEG.

Decoder (300 ms sliding window over 50 ms chunks):

* **JUMP/ESCAPE** if giant-fiber mean rate > 25 Hz
* **BACKWARD** if MDN > 4 Hz (conflict with walk → dominance + LOGGED)
* **TURN_LEFT/RIGHT** if P9 differential (left − right) beyond ±8 Hz with
  ±4 Hz hysteresis
* **FORWARD** if walk-population mean (P9 + BPN + RRN) > 5 Hz
* **STOP** if walk-OFF populations (Foxglove + Bluebell) > 4 Hz, or the
  failsafe fires (no walk population above 1.5 Hz and no MDN/GF)
* minimum action duration 2 chunks; all simultaneous intents evaluated and
  conflicts logged, never hidden

The decoder sees ONLY population spike counts — it has no access to the
target, bearing, or any arena state (API-contract unit test included).

---

## 4. D10 — The first closed loop

### 4.1 The loop (real, no shortcuts)

```
ARENA (target / looming)  →  SENSORY ENCODER (LC9-L/R, LPLC2/LC4 rates)
  →  DIGITAL DROSOPHILA (full v783 brain, 50 ms chunks)
  →  DN ACTIVITY (P9, GF, MDN, walk-OFF, …)  →  MOTOR DECODER
  →  AGENT MOVEMENT  →  NEW VISUAL STATE  →  repeat
```

The environment knows the target location; only the encoder converts it to
neural stimulation; only the decoder converts neural activity to actions.
The scripted-optimal controller exists **only** as the explicit, labelled
control B. Episode determinism: `restore() → seed(episode_seed)` (§1.3),
fresh subprocess per brain episode, full per-chunk JSONL logs + canonical
spike files + rate schedules (exact replay possible).

### 4.2 Experiment matrix (pre-registered)

5 conditions × 5 scenarios × 3 reps = 75 episodes:

* conditions: A random (no brain) · B scripted-optimal (no brain) ·
  **C intact** · D shuffled-sensory (each channel drives a wrong same-size
  random subset of LC9∪LPLC2∪LC4, seeded per episode) · E shuffled-motor
  (decoder readout slots permuted, seeded per episode, brain intact)
* scenarios: target_left (+60°) · target_right (−60°) · target_center ·
  no_target · looming (no target; expanding threat from t = 0.5 s)

### 4.3 Results

Full table in `brain/results/d10_closed_loop/aggregates.json`; per-episode
records in `experiments.csv` + `episodes/*/summary.json`.

**Target approach (success rate over 3 reps per scenario):**

| condition | target_left | target_right | target_center | mean time-to-target | path efficiency | mean \|bearing\| |
|---|---|---|---|---|---|---|
| B scripted-optimal (control) | 3/3 | 3/3 | 3/3 | 2.73 s | 1.20 | 8.1° |
| **C intact digital fly** | **3/3** | **3/3** | **3/3** | **4.20 s** | **1.15** | **16.4°** |
| D shuffled-sensory | 0/3 | 0/3 | 0/3 | — | — | 57.7° |
| E shuffled-motor | 0/3 | 0/3 | 0/3 | — | — | 59.5° |
| A random | 0/3 | 0/3 | 0/3 | — | 1.09 | 61.8° |

**Other scenarios:**

| condition | no_target behavior | looming escape | escape latency |
|---|---|---|---|
| C intact | **STOP 100 %**, never leaves origin, 0 spikes (silent brain) | **3/3 JUMP** (GF 53–76 Hz) | **583 ms** (≈ 260–310 ms after the loom channel crosses drive threshold) |
| D shuffled-sensory | STOP 100 % | 3/3 (slower) | 733 ms |
| E shuffled-motor | STOP 100 % | 3/3 | 700 ms |
| A random | wanders | 0/3 | — |

Closed-loop neural signature (intact episodes): target_left drives
P9_left 16–18 Hz vs P9_right 13–17 Hz; target_right drives P9_right 22–23 Hz
vs P9_left 14 Hz — the ipsilateral P9 dominance predicted by the v783
wiring. MDN/BPN/RRN stay silent (no backward/stop contamination); GF stays
below jump threshold except under looming. **Zero decoder conflicts** were
logged across all 15 intact episodes (the walk populations never fired
incompatible combinations in these scenarios).

Honest detail: shuffled-sensory episodes often emit maladaptive JUMPs during
target scenarios (escape rate 1.0 in the table's target rows) — the shuffled
interface draws its wrong target sets from LC9 ∪ LPLC2 ∪ LC4, so wrong wiring
sometimes hits the looming pathway and triggers escape instead of approach.
Approach behavior collapses exactly as the control requires.

### 4.4 Controls analysis (Gate 3 criteria, pre-registered)

| criterion | result |
|---|---|
| G3.1 intact closed loop produces reproducible neural→motor actions that change the environment; success clearly above random | **PASS** (9/9 target successes; random 0/9) |
| G3.2 shuffled-sensory control degrades behavior | **PASS** (0/9 vs 9/9) |
| G3.3 shuffled-motor control degrades behavior | **PASS** (0/9 vs 9/9) |
| G3.4 random controller substantially worse than intact | **PASS** (0/9 vs 9/9) |

Bootstrap contrasts on per-episode success (target scenarios pooled,
5000 resamples): intact vs random **+1.00 [1.00, 1.00]**, intact vs
shuffled-sensory **+1.00 [1.00, 1.00]**, intact vs shuffled-motor
**+1.00 [1.00, 1.00]** (all significant); intact vs scripted **0.00** — the
fly matches the optimal controller's SUCCESS RATE while being ~1.5 s slower
(4.20 vs 2.73 s) with more heading churn (mean |bearing| 16.4° vs 8.1°).
The intact fly also does something the scripted controller never does:
escaping the looming threat (scripted has no escape rule; 0/3).

Neural activity was preserved under shuffling (brain still runs: ~44–69k
spikes/episode vs ~83k intact on target scenarios), i.e. the degradation is
attributable to the BROKEN INTERFACE/READOUT MAPPING, not to a silenced
brain — exactly what the controls were designed to prove.

### 4.5 Replay verification (bit-exact)

| episode | replay | result |
|---|---|---|
| intact_target_left_r0 (4.35 s) | interactive re-run from exact schedule + seed | **bit-identical** (sha `bba1053a…`) |
| intact_looming_r0 | interactive re-run | **bit-identical** |
| shuffled_motor_target_right_r1 | interactive re-run (permuted decoder replayed through the same log) | **bit-identical** |
| intact_target_left_r0 | **C++ standalone replay** (fastest full-brain route, offline) | **bit-identical**, binary-only 14.8 s for 4.35 bio-s = 0.294 bio-s/wall-s (3.0× the runtime engine), 24.6 s incl. codegen+compile |

Two replay-harness bugs were found and fixed during this verification
(both documented): per-chunk rates were logged rounded to 0.1 Hz (exact
rates are now reconstructed deterministically from seed + recorded actions,
with an assertion against the rounded log), and the first schedule segment
was compressed to zero duration (chunk-start vs duration semantics). No
brain or experiment data was re-run after the fixes except the replay
verifications themselves — the matrix results stand as originally produced.

**Runtime performance across the 15 intact episodes:** decision latency
506 ± 30 ms per 50 ms biological chunk (p95 534 ms), 0.0988 bio-s/wall-s
average, ~38 s wall per episode, ~58k spikes/episode, peak RSS ~2.9 GB,
serial subprocess per episode.

---

## 5. Machine-readable results

* `brain/results/d10_closed_loop/experiments.csv` — per-episode metrics
* `brain/results/d10_closed_loop/aggregates.json` — per condition × scenario
* `brain/results/d10_closed_loop/verdict.json` — Gate 3 criteria (fails closed)
* `brain/results/d10_closed_loop/episodes/<cond>_<scen>_r<k>/` — chunks.jsonl,
  summary.json, spikes.txt, rate_schedule.json (+ dashboard.mp4 for
  representative episodes)
* `brain/results/d7_runtime/*.json` — chunk/checkpoint/bench studies

## 6. Gate 3 verdict

**GATE 3: PASS**

> THE CONNECTOME-DERIVED DIGITAL DROSOPHILA CAN PERCEIVE A SIMPLE VISUAL
> TARGET THROUGH THE VERIFIED SENSORY INTERFACE AND PRODUCE REPRODUCIBLE
> MOTOR ACTION THAT CHANGES THE ENVIRONMENT IN CLOSED LOOP.

* reproducible motor action changing the environment: 9/9 target-approach
  successes + 3/3 looming escapes, all episodes bit-exactly replayable
  (interactive AND C++ standalone), per-episode determinism via
  restore+reseed;
* shuffled sensory control degrades behavior: 0/9 (bootstrap CI [1.00,1.00]);
* shuffled motor control degrades behavior: 0/9 (bootstrap CI [1.00,1.00]);
* random controller substantially worse: 0/9 (bootstrap CI [1.00,1.00]).

No parameter was tuned after seeing results: the encoder, decoder, arena,
kinematics, scenarios, metrics and criteria were all frozen in
`preregistration.json` + `d8_calibration_frozen.json` BEFORE the matrix ran.
Every bug found after the matrix (replay-harness only) is documented in §4.5
and did not alter any recorded experiment.

## 7. What this stage does NOT claim

* No learning, no plasticity, no reward-modulated change (fixed connectome;
  learning is D11).
* No biological equivalence beyond the wiring evidence cited; the encoder,
  decoder, arena and kinematics are ENGINEERED interfaces, labelled as such.
* No claim that the fly "understands" the target — only that the wired
  sensorimotor path produces approach behavior that depends on the intended
  biological pathway (which is exactly what the controls test).
* No Roblox, no computer vision, no game input of any kind (next gate).
