# GATE 4 v2 — Source-Audited Associative Learning Experiment — FINAL REPORT

**Verdicts (pre-registered, fail-closed, GATE4_V2_PREREG.md §8):**

- **G4A — NEURAL ASSOCIATIVE LEARNING: FAIL** (A2 PASSES decisively;
  A1 and A4 FAIL)
- **G4B — BEHAVIOURAL EXPRESSION: FAIL** (B1, B2 FAIL; B3 vacuous —
  see §3.4)
- **OVERALL GATE 4 (v2): FAIL**

**Per the pre-registration (§8) and the v2 brief (Section P): biological
tuning STOPS here. The result is preserved as-is.** The v1 record remains
PERMANENTLY FAIL (tag `gate4-v1-FAIL`). This report does not alter either
verdict, and no post-hoc parameter search was performed at any point.

**Commit ordering (mandated):** prereg `e152676` → calibration freeze
`0b35496` → this final outcome record. Every session was auto-pushed as
it completed.

---

## 1. What v2 changed and why (Sections A–D of the brief)

The v2 brief asked one question: **can the already-demonstrated
mushroom-body synaptic plasticity produce behavioural learning under a
scientifically valid assay that is not saturated by the innate motor
bias?** The source audit (primary texts fetched, archived in
`brain/data/gate4_v2/sources/`, and itemized in
`brain/data/gate4_v2/SOURCE_TABLE.md` — 17 claims with source/evidence
type/confidence/use) forced three corrections before any v2 outcome
data existed:

1. **MBON activation-valence corrected.** Aso 2014 (eLife 04580, direct
   optogenetic screen): *"all MBONs eliciting aversion were glutamatergic
   and all the MBONs eliciting attraction were either GABAergic or
   cholinergic."* Owald 2015: M4/M6 activation **induces avoidance**;
   their inhibition converts avoidance into attraction; reward
   **depresses** their CS+ drive. The v1 dictionary had the M4/M6 family
   labelled `approach` — backwards. v2 classifies all 96 MBONs by their
   **measured v783 neurotransmitter predictions**: 22 avoidance-class
   (GLUT) / 74 approach-class (GABA+ACh); 23 of 35 type-level v1
   discrepancies documented in `mbon_valence_corrected.json`.
2. **The US became compartment-specific.** v1 stimulated all 307 PAM
   cells as "reward". Aso 2014: reward is conveyed by *"a distributed
   set of PAM cluster DANs that innervate the compartments of
   glutamatergic MBONs."* v2 stimulates **PAM01–PAM11** (252 cells; the
   compartment-matched set; the Aso↔v783 DAN numbering was itself
   verified against the connectome — DAN→MBON co-innervation + KC-class
   + transmitter cross-checks, 17/21 anchors match, all of PAM01–11
   match; full table in `dan_compartment_table.md`). Labelled everywhere
   as an **ENGINEERED OPTOGENETIC-STYLE US ENTRY** — the anatomical
   sugar→PAM route is dynamically silent in the pinned model (probe P1),
   so this is not a naturally simulated sugar pathway.
3. **The assay moved to a calibrated balanced boundary.** v1's
   free-choice test was read through the closed-loop P9 turn decision,
   which the innate wiring lean saturated (A-control 100% RIGHT). v2
   reads preference in **open-loop preference tests** (both CS ensembles
   at frozen rates; the fly stationary; no motor loop to saturate),
   calibrated BEFORE any reward-side assignment to **P(left) = 0.500**
   (12 trials per rate value; all tested values in
   `calibration_report.md`: dose-response S1 0.67 / S2 0.33 / S3 0.17 /
   S4 1.00-saturated; boundary sweep 0.50/0.83/0.67/0.33/0.33 →
   kc_cs_left = 35 Hz selected at exactly 0.500). The native closed-loop
   route (A) is still recorded every block.

The plasticity rule itself (dopamine-gated depression of KC→MBON,
eligibility trace τ=2 s with (e/max)² induction nonlinearity, per-trial
relaxation τ=50, η=0.30, floor 0.05) is **unchanged from the
v1-demonstrated mechanism** — v2 changed the assay and the source-audited
target set, not the learning rule. Reward NEVER touches action: it
enters only as compartment-matched DAN stimulation; behaviour is read
only from MBON/DN spike counts; the D13 external episodic memory is not
linked anywhere (decoder `external_bias_hz` hard-zero).

Dataset provenance (Section B) is documented in
`brain/data/V783_MODEL_COVERAGE_REPORT.md`: official v783 = 139,255
neurons vs model 138,639; the 616 excluded = 537 photoreceptors + 4
cervical DNge cells outside every used readout population + 75
peripheral cells. **Zero overlap with any sensory, MB, dopamine, or
motor population used here — not blocking.**

## 2. The matrix (Sections J–L; prereg §7, frozen)

11 brain sessions (one network build each, ~35 min, 94 trials, ~3 GB,
~0.1 bio-s/wall-s), each auto-pushed on completion: B_plastic × 3 seeds
× both reward sides (R/L/R/L/R/L, pre-registered seeds 20261101–06); A
no-plasticity (US still delivered) × both sides; C lagged-dopamine
gates; D permuted eligibility→KC attribution; F untrained; E =
conventional ε-greedy Q-learner (explicitly NON-biological). Battery per
session: 6 baseline prefs → 4×(4 pairings + 2 prefs + 1 route-A trial)
acquisition → 8 fillers + 3 prefs retention → 12 prefs extinction →
4×(4 pairings OTHER side + 2 prefs + 1 route-A) reversal → 3 prefs at
×0.8 and 3 at ×1.2 CS rates (context change) → 3 prefs with a 0.5 s
loom pre-pulse (distractor).

## 3. Results

### 3.1 G4A — the synapse-level associative trace is REAL and
cue-specific; it does not become a directional population signal

- **A2 (cue-specific synaptic change): PASS, decisively.** In every B
  session, KC→GLUT-MBON synapses of the **paired-side CS ensemble** end
  acquisition at mean scale **0.198–0.219** (≈80% depression) while the
  unpaired-side ensemble's stay at **0.77–1.00**; per-session contrast
  +0.55 to +0.79, pooled **+0.686, bootstrap CI95 [0.608, 0.756]**.
  The compartment-corrected target set produces exactly the
  Owald/Aso-predicted substrate trace: reward-paired context → strongly
  depressed KC→avoidance-class synapses, unpaired context → untouched.
- **A1 (eligibility lateralization): FAIL at 0.807** (gate ≥ 0.90).
  In ~19% of pairing trials the US itself (PAM01-11 at 150 Hz, 48k
  DAN→KC synapses) ignites a **bilateral KC cascade**, so the paired
  side's eligibility advantage (typically ≫100:1, e.g. 0.02 vs 2.6)
  collapses to near-parity in those trials. The US entry that carries
  the reward also contaminates the credit it is supposed to assign.
- **A3 (descriptive): the population response moves OPPOSITE to the
  local trace.** The avoidance-class MBON response to the PAIRED CS
  **rises** in all six B sessions (+2.7 to +51.4 Hz, baseline→acquisition)
  even though the paired context's KC→GLUT synapses are the ones being
  depressed. The depression of 100 ensemble KCs' synapses is swamped by
  a network-gain effect: KC→MBON depression reduces MBON drive to the
  feedback loop (APL), which disinhibits the remaining ~5,000
  non-ensemble KCs, and the GLUT-MBON population — which pools inputs
  from all KCs — fires MORE. The local synaptic trace and the
  population-level readout have opposite signs.
- **A4 (learned valence signal toward the rewarded side): FAIL**
  (pooled −4.1 Hz, CI95 [−16.1, +7.9]). Per-session values split by
  reward side (see §3.3): right-rewarded sessions move −9 to −26 Hz,
  left-rewarded +3 to +14 Hz — a lateralized drift, not a
  reward-directed signal.

### 3.2 The balanced boundary sits on a bistable network gain state

The calibrated regime (kc_cs_left = 35 Hz, P(left) = 0.500) turns out to
sit at a bifurcation: across the session, the left CS ensemble's
measurement-window firing rate is **bimodal** — a stimulus-locked state
(~27–40 Hz) and an **amplified state** (~50–76 Hz), and route-B choice
follows the state almost deterministically (low state → V≈−10 → right;
amplified state → V≈+15..+40 → left). The context-change probes make
this explicit: at ×0.8 CS intensity every B_R session reads right
(amplification not engaged), at ×1.2 every one reads left. The "0.500
neutral baseline" is therefore the 50/50 tipping point of a bistable MB
gain loop, and any process that biases the tipping — history, US
sensitization, plasticity — appears as a "preference shift".

### 3.3 What actually moves the readout: a lateralized,
plasticity-dependent but credit-INdependent drift

Acquisition shifts (P(rewarded side), acquisition − own baseline):

| condition | sessions | shifts | reading |
|---|---|---|---|
| B, RIGHT-rewarded | 3 | −0.417, −0.250, −0.125 | **away** from reward |
| B, LEFT-rewarded | 3 | +0.125, 0.000, +0.375 | toward reward |
| A (US, no plasticity) | R/L | −0.083, 0.000 | no consistent shift |
| C (lagged gates) | R | −0.375 | reproduces B_R |
| D (permuted eligibility) | R | −0.417 | reproduces B_R |
| F (untrained) | R | −0.167 | mild drift |
| E (Q-learner, non-biological) | R/L | +0.583 / 0.000 | learns on the same schedule |

Everything with the plasticity pathway active drifts LEFT regardless of
which side is rewarded (right-rewarded sessions move AWAY from reward;
the left-rewarded sessions' "correct-direction" shifts are the same
leftward drift wearing the right sign). The lagged-gate (C) and
permuted-eligibility (D) controls reproduce the B_R shift almost
exactly: the drift needs the US-driven dopamine gate to be open and
depression to land **somewhere** in the KC→GLUT population, but not on
the right synapses — it is not credit-specific. The US alone (A) or no
US (F) does not produce it reliably. Mechanistically this is consistent
with §3.1's gain story: bilateral eligibility contamination (A1) plus
indiscriminate depression tips the left MB loop into its amplified
state.

### 3.4 Route comparison (Section I) and honest control notes

- **Route A (native DN / closed loop):** mixed, no reward-directed
  change (P(left) 0.125–0.714 across B sessions; expected from v1's P4
  finding that MBON→DN wiring does not reach walk-command populations).
- **Route B (aggregate MBON valence, ENGINEERED tier-3c):** the only
  route with dynamic range — and it reads the bistable drift (§3.2/3.3).
- **Route C (MBON→reachable-DN aggregate, ENGINEERED):** saturated
  LEFT in 100% of pref trials in ALL 11 sessions (the approach-routed DN
  aggregate inherits the leftward dominance) — uninformative as a
  differential readout.
- **B3 disclosure:** the registered B3 gate fails only when a control
  shows a significant POSITIVE (toward-rewarded) shift; because the
  entire system moved away-from-rewarded or not at all, the gate
  registers "pass" **vacuously**. Read honestly (as intended): C and D
  DO reproduce the right-rewarded B sessions' away-shift (−0.375,
  −0.417 vs B_R −0.125..−0.417), i.e. the observed shift is NOT
  causally credit-specific. G4B fails on B1 and B2 regardless of how B3
  is read. Similarly, the C control is weak in this regime by
  construction (US on every pairing trial ⇒ the lag-1 gate is also open
  on every pairing trial) — documented as a protocol limitation, not
  re-run.
- **E (conventional RL)** learns the same schedule (B_R1 0.167→0.75) —
  the task is learnable by an ordinary learner; the failure is specific
  to expressing the fly-brain trace, not to the assay's structure.

### 3.5 Battery details (descriptive)

Extinction did not produce a clean decline in the B_R sessions (the
amplified state persists; first3→last3 stays 0.0→0.0-0.33); B_L1/B_L2/
B_L3 show partial declines (0.33→0.33/0.67/0.67). Reversal: the three
B_R sessions read P(new=left)=0.875–1.0 (the drift direction again, not
a flip), B_L sessions 0.125–0.375. Retention after 8 fillers: present
in some sessions, not others. Full per-trial data in each session's
`trials.jsonl`.

## 4. What this stage claims and does NOT claim

**CLAIMED:**
- The full v2 protocol ran exactly as pre-registered: calibration frozen
  and pushed before reward-side assignment; both reward sides
  demonstrated; all controls run; deterministic replay preserved
  (seeded, per-trial reset+re-apply protocol identical to the v1/P5
  verified scheme); every session's data pushed at completion.
- The source audit corrected a real classification error (M4/M6
  valence) against the primary literature, and the compartment-matched
  PAM01-11 US works dynamically (gate 148 Hz measured).
- **The mushroom-body plasticity produces a strong, cue-specific
  associative trace at its own synapses** (A2: ~80% depression of
  paired-context KC→GLUT synapses, CI far from 0) — the cleanest
  synapse-level demonstration in the project so far.
- That trace does NOT propagate to a directional population signal
  (A4), a preference (B1/B2), or a causally credit-specific behavioural
  change (C/D reproduce the drift). The readout is dominated by (i) a
  bistable left-lateralized MB network-gain state that the calibrated
  boundary sits on, and (ii) US-cascade eligibility contamination.

**NOT CLAIMED:**
- Gate 4 is NOT passed. The digital Drosophila still has not
  demonstrated reward-dependent behavioural change. No post-hoc tuning
  was performed; per the pre-registration, biological tuning STOPS with
  this result.
- The HYBRID MBON readout option (Section P) is NOT validated by these
  data either: route B is the only readout with dynamic range and it
  reads the drift, not the associative trace (route C saturated; route
  A inert). The only reliable associative signal found is the A2
  synapse-level trace itself — a weight inspection, not a neural
  activity readout.
- No Roblox/game connection was made at any point (mandate).

## 5. Performance (Section N)

| session | wall s | bio s | peak RSS MB | mean pref wall s | trials |
|---|---|---|---|---|---|
| B_R1–B_L3 (6) | 1958–2179 | 196–202 | 3007–3025 | 23.3–25.9 | 94 |
| A_R / A_L | 2020 / 2001 | ~197 | ~3024 | 24.4 / 24.1 | 94 |
| C_R / D_R / F_R | 2050 / 2008 / ~2010 | ~197 | ~3020 | ~24 | 94 |

Full canonical numpy runtime (Gate-3 profile preserved; no reduced
network, no silent substitution); decision latency ≈ 24 s per 2.3-s
preference trial (~0.1 bio-s/wall-s); checkpoints (weights.npz) 121 KB.

## 6. Machine-readable evidence

- `brain/data/gate4_v2/` — SOURCE_TABLE.md, dan_compartment_table
  .{json,md}, mbon_valence_corrected.json, mbon_to_dn_routes.json,
  FROZEN_PARAMS.json, calibration_report.{json,md}, interface_v2.json,
  readout_v2.json, v783_missing_ids.json, sources/ (fetched texts)
- `brain/data/V783_MODEL_COVERAGE_REPORT.md`
- `brain/results/gate4_v2/{B_R1..F_R,E_rl}/` — trials.jsonl, session.json,
  weights*.npz per session; `analysis/verdicts.json` +
  `analysis/report_sections.md`; `calibration/` (ALL tested values)
- scripts: `g4v2_audit.py`, `g4v2_coverage_report.py`, `g4v2_specs.py`,
  `g4v2_core.py`, `g4v2_calibration.py`, `g4v2_sessions.py`,
  `g4v2_analyze.py`; tests: `tests/test_g4v2.py` (11; full suite 92 PASS)

## 7. Where this leaves the project

Two pre-registered assays (v1 closed-loop, v2 balanced-boundary
open-loop) have now both failed at the behavioural expression step,
each with a fully diagnosed mechanism (v1: innate motor wiring lean
dominates the turn decision; v2: the readout sits on a bistable
lateralized network-gain state and the US cascades contaminate credit
assignment). What survives — and is now better characterized than
before — is the substrate: the connectome's own KC→MBON synapses carry
a strong, cue-specific, dopamine-gated associative trace (A2). What
does not yet exist in this pinned model is a path from that trace to
behaviour that does not pass through a dominating non-associative
network dynamic.

Per the v2 brief Section P: **STOP BIOLOGICAL TUNING. Preserve this
result.** Any future game-agent work that needs a value signal must
either (a) use a clearly labelled non-biological value module, or (b)
await a principled model-side change (e.g., a plasticity site/gain
architecture that survives the APL feedback loop) — which would be a NEW
pre-registered iteration, not a retune of this one. No autonomous Roblox
control is initiated.
