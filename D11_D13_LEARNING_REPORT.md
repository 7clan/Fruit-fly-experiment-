# D11–D13 — Reward-Modulated Learning, Internal States, and Multi-Step Memory

**Stage:** reward-modulated learning through the mushroom-body / dopamine
pathway of the canonical digital Drosophila brain, with the mandated
honest separation of BIOLOGICAL / MODELED / ENGINEERED tiers
(brain/closedloop/__init__.py).
**Canonical brain:** Shiu et al. LIF model on FlyWire v783, unmodified,
driven by the Gate-3 chunked interactive runtime (d7_runtime.py) — the
learning runtime is COMPOSED ON TOP of the Gate-3 closed loop (not a
replacement).

| Piece | Artifact |
|---|---|
| Anatomy verification (P0) | `brain/results/d11_anatomy/verify.json` |
| Dynamic probes (P1–P5) | `brain/scripts/d11_probes.py`, `brain/results/d11_probes/probes.json` |
| Bilateral calibration (P6) | `brain/results/d11_probes/bilateral_calibration.json` |
| Learning runtime | `brain/scripts/d11_learning.py` (LearningBrain + MBPlasticity) |
| Sessions + matrix | `brain/scripts/d11_sessions.py`, `brain/results/d11_learning/{B,A,C,D,E_rl_baseline}` |
| Pre-registration (frozen) | `brain/data/d11_d13/preregistration.json` (v2 + erratum) |
| Internal states (D12) | `brain/scripts/d12_states.py` + per-session `dashboard_d12.html` |
| Multi-step + memory (D13) | `brain/scripts/d13_multistep.py`, `brain/results/d13_multistep/` |
| Tests | `tests/test_d11_d13.py` (24 brain-free unit tests) |

---

## 1. What the connectome gives and what it does not (probes, P0–P5)

The D4–D6 map established the mushroom-body substrate anatomically:
**KC→MBON = 62,261 connections / 256,719 synapses (77% ipsilateral —
re-verified P0, exactly matching D5)**, PAM (307, reward/appetitive DANs),
PPL1 (16, aversive), APL/DPM feedback, and MBON valence classes anchored
to Aso 2014 / Owald 2015 (37 approach / 24 avoidance / 35 unclassified
after fixing the interrupted session's `hemibrain_type` bug).

The DYNAMIC probes then produced this stage's most important honest
findings about the pinned model — all recorded in
`brain/results/d11_probes/probes.json`:

| probe | question | result |
|---|---|---|
| P1 | does the biological sugar entry (GRN_SUGAR @150 Hz) recruit the reward DANs? | **NO — PAM/PPL1/KC all 0 Hz.** The sugar→PAM route is anatomically present (h2–h3) but dynamically silent — the same class of finding as Gate-2's L2/T4/T5 propagation limit |
| P1b | direct PAM stimulation (optogenetic-style) | **WORKS** — PAM 293 Hz; MBONs driven (approach 54 / avoidance 93 Hz) |
| P2 | does visual (LC9) drive engage KCs side-specifically? | **NO — 0 spikes, 0.0001 mV mean depolarization.** LC9→KC (h2–h3) is dynamically silent |
| P2b | direct KC ensemble stimulation | **WORKS** — MBONs respond (approach 69 / avoidance 85 Hz) |
| P3 | does silencing ALL 62,261 KC→MBON synapses change network activity under visual drive? | **BIT-IDENTICAL trajectory** — the KC→MBON pathway carries zero spikes under sensory drive |
| P4 | do MBONs reach the walk command populations? | MBON activation touches DN_ALL (~3.9 Hz) but **NOT P9/BPN/RRN/MDN (all 0 Hz)** — behavioural expression cannot flow in-network through the Gate-3 decoder's populations |
| P5 | is the learning protocol deterministic? | **YES — bit-identical** (reset + reseed + weight re-application) |

Consequence (all labelled): the CS is delivered to per-side **KC
ensembles** (seeded 200/side) and the US as **direct PAM stimulation**
— exactly the optogenetic-style entries pre-registered in D4–D6 §3.3
tier-3a; behavioural expression uses the pre-registered **MBON
valence-bias readout** (tier-3c, probe-P4 evidence). Reward NEVER touches
action directly: it enters only as DAN stimulation; behaviour is read
only from DN/MBON spike counts.

## 2. The learning runtime (D11)

`LearningBrain` (d11_learning.py) composes on the Gate-3 `DualModeBrain`:
the KC→MBON synapse rows are mapped from the connectivity parquet
(fail-closed check that live `syn.w` equals the canonical weights), and
per-trial the protocol is `restore('ep0') → seed → re-apply MODELED
weight scales → closed-loop chunks`. The plasticity rule (MODELED tier,
the Owald-2015/Aso-2014 sign rule):

* reward (PAM rate ≥ 10 Hz in the US window, measured from the network)
  depresses **KC→avoidance-MBON** synapses of eligible KCs;
* punishment (PPL1 ≥ 10 Hz) depresses **KC→approach-MBON**;
* eligibility = per-KC spike trace, τ = 2 s, induction nonlinearity
  (e/max)²; floor 0.05; per-trial relaxation toward canonical (τ = 50
  trials) provides forgetting/extinction.

**Task redesign forced by honest pilots.** The instrumental two-target
design could not work: the bilateral regime is a **decision boundary**
(calibrated balance → 75–92% STOP-timeout trials; any imbalance → a
one-sided basin; no explorable 50/50 regime exists — closed-loop
bisection evidence in `d11_learning/calib_A_*`). The frozen brain's
decoder has no spontaneous exploration. The pre-registered v2 design is
therefore the classic **forced-pairing + free-choice-test** paradigm
(Aso-2014 style): the US is paired with one side in forced single-target
exposure (innate approach carries the fly; no choice needed), and the
preference is read in unrewarded two-target test trials.

## 3. The matrix and the honest Gate-4 verdict (v1)

Pre-registered (`preregistration.json` v2, frozen before the matrix):
B (plastic) full battery — acquisition 6×(4 pair RIGHT + 3 test),
extinction 12 test, reversal 6×(4 pair LEFT + 3 test); controls A (no
plasticity, US still delivered), C (dopamine gates lagged one trial —
broken CS–US contingency), D (plasticity through a permuted
eligibility→KC mapping), E (conventional ε-greedy Q-learner, explicitly
non-biological).  Gate-4 criteria fail closed.

**Results (all sessions complete, `analysis.json`):**

| measure | B (plastic) | A (no plasticity) | C (shuffled DA) | D (shuffled KC/MBON) | E (RL) |
|---|---|---|---|---|---|
| acquisition tests, P(paired side \| decided) | 1.000 | 1.000 | 0.889 | 0.889 | 1.000 |
| B − A bootstrap 95% CI | **0.000 [0.000, 0.000]** — not significant | | | | |
| pairing approach rate | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| US delivered | 24/24 | 24/24 | 24/24 | 24/24 | 24/24 |
| extinction P(right) | 1.000 (no decline) | — | — | — | 0.917 |
| reversal P(left) | **0.000 (no flip)** | — | — | — | 0.944 |

**GATE 4: FAIL (v1) — reported honestly, no post-hoc tuning.**

The fly's test choices never changed in a reward-dependent way: the A
control shows the operative regime has an **innate 100% RIGHT test
preference** (18/18 decided), so B's acquisition sat at the innate
ceiling (B − A = 0.000 [0, 0]); extinction did not decline; reversal
(24 LEFT-pairings with US, approach rate 1.0, gates firing) produced
**zero** left test choices.

**Mechanistic diagnosis (all measured, `B/trials.jsonl`):**
the learning machinery DID operate end-to-end — the US drove PAM
(114–117 Hz in the US window), eligibility was correctly lateralized
(elig_R ≈ 11–15 vs elig_L ≈ 0.003–5 on right-pairings), 46% of KC→MBON
synapses ended below canonical (avoidance-class mean 0.949), and the
learned MBON-rate shift exists (after LEFT pairing, avoid_L fell ~15 Hz
below avoid_R). But (i) the net decoder valence bias averaged only
**−1.0 Hz (std 5.0)** in the reversal phase — an approach-class
asymmetry cancels most of the avoidance-class shift, and (ii) the innate
wiring lean (P9-based, ≫ 8 Hz threshold) dominates the turn decision.
The learned signal is real but behaviourally silent at these parameters.
Controls C and D behaved like A (0.889), i.e. the shuffled-pathway
controls are uninformative here because the intact pathway itself did
not shift behaviour — reported as such.

**What WOULD demonstrate Gate 4 (pre-registered follow-up v2):** a
behavioural contest that does not fight the innate lean — test trials
in the balanced-boundary regime (innate lean ≈ 0 among decided trials;
timeouts tolerated and reported) and/or a stronger CS (full-hemisphere
KC ensembles so the depression moves MBON rates by 10s of Hz), frozen in
a fresh pre-registration that references this FAIL. Not run within this
stage's claims; the v1 verdict stands.

## 4. D12 — internal states (real variables only)

`d12_states.py` computes every mandated state as a **documented
transform of measured model variables** — no invented emotion, no
natural-language thought; friendly labels sit beside the numeric
variables (see the dashboard's state→variable table):

| state | underlying measured variable |
|---|---|
| positive / negative valence | mean approach- / avoidance-MBON rates (L+R) |
| valence bias | (appr_L−avoid_L) − (appr_R−avoid_R), Hz |
| reward expectation (per side) | appr/(appr+avoid+1 Hz) of that side's MBONs |
| reward-prediction error | US(1/0) − expectation(chosen side) |
| novelty | 1 − cosine(KC-ensemble eligibility, EMA of past patterns) |
| exploration drive | Shannon entropy of the last 8 choices |
| threat drive | loom-channel drive + giant-fiber rate (0 in D11; live in D13/GPO) |
| uncertainty | exponentially-weighted std of recent RPEs |
| arousal-like | z-score(PAM+PPL1+DN_ALL choice-window rates) |

`dashboard_d12.html` (self-contained, per session) renders the state
time series, choice/US raster, DAN and MBON rates, and the synaptic
scales — with the honesty note and tier labels in the header.

## 5. D13 — delayed reinforcement and multi-step memory

**E1 (delayed reinforcement, 1.5 s US delay).** The D11 pairing+test
design with the US delayed 1.5 s after commit (the MODELED eligibility
trace, τ = 2 s, must bridge it), followed by the retention battery:
immediate tests, delayed recall (after 8 no-stimulus filler trials),
context change (goals at ±40°), distractor (0.5 s centre flash before
the choice), extinction tests. Results: `brain/results/d13_multistep/`
and §5 numbers below.

**E2 (multi-step cue-memory; conditions A vs B).** Per trial: CUE
(single target at the cue side, random L/R) → CHOICE (cue removed; the
rewarded goal is the cue's side) → APPROACH → INTERACT (0.5 s dwell) →
DELAY (1.5 s) → US. Condition **A (brain-only)**: the cue must be
carried across the cue→choice gap by the brain's own state — the
canonical model has NO persistent activity (silence is its resting
state, Gate-1), so the honest expectation is cue-following near chance.
Condition **B (brain + EXTERNAL episodic memory)**: the harness logs the
cue and injects a labelled ±12 Hz bias during CHOICE — **engineering,
never credited to the fly brain** (mandate).

E1/E2 results (see `brain/results/d13_multistep/analysis.json`):

<!-- D13_RESULTS -->

## 6. Machine-readable evidence

* `brain/results/d11_learning/{B,A,C,D,E_rl_baseline}/trials.jsonl` —
  per-trial records (choice, US, PAM/PPL1 rates, MBON valence, weight
  stats, KC eligibility incl. the 400-cell ensemble pattern, telemetry)
* `brain/results/d11_learning/{B}/chunks.jsonl` — per-chunk closed-loop
  log of the full B battery (exact replay possible)
* `brain/results/d11_learning/analysis.json` + `learning_curves.png`
* `brain/results/d11_probes/` — probes.json, bilateral_calibration.json
* `brain/results/d13_multistep/` — E1 battery + E2 conditions A/B
* performance: B session 96 trials, 1,678 s wall, peak RSS 3,019 MB,
  ~0.1 bio-s/wall-s (Gate-3 profile preserved; one brain subprocess at
  a time; one network build per session)

## 7. What this stage claims and does NOT claim

* CLAIMED: the reward→PAM→KC→MBON plasticity pathway is implemented,
  deterministic, and measurably changes the connectome's own KC→MBON
  synapses and MBON population rates in a reward-contingent,
  eligibility-specific way; internal states are real computed variables;
  the multi-step task and both memory conditions run with honest
  attribution.
* NOT CLAIMED: **Gate 4 is NOT passed** — the digital Drosophila did
  NOT change its future BEHAVIOUR as a consequence of reward history
  under the pre-registered v1 parameters; the innate wiring lean
  dominated every test choice. No Roblox connection was made (mandate:
  none before Gate 4 passes).
* Every engineered entry (CS ensembles, direct-PAM US, valence-bias
  readout, arena, pairing schedule) is labelled and evidence-backed;
  nothing was silently substituted for the brain.
