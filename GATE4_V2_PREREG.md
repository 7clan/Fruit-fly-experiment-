# GATE 4 v2 — PRE-REGISTRATION (source-audited associative-learning experiment)

**Status:** FROZEN before any v2 outcome data. This document is committed
and pushed BEFORE the calibration outcome data and BEFORE the final v2
matrix. Nothing in it may change after the matrix begins; any deviation
discovered mid-run is recorded as a deviation, not silently applied.

**Relationship to v1:** Gate 4 v1 is PERMANENTLY RECORDED AS FAIL
(git tag `gate4-v1-FAIL`, commit `38a65a9`). v2 is not a retune of v1. It
is a new, source-audited assay designed to answer ONE question: **can the
already-demonstrated mushroom-body synaptic plasticity produce behavioral
learning under an assay that is not saturated by the innate motor bias?**
The v1 mechanistic findings (dopamine gating operates; eligibility
lateralizes; KC→MBON weights change; MBON responses change; behavior
nevertheless unchanged) all stand.

---

## 0. Source hierarchy and the v1→v2 corrections (Section A)

Primary sources consulted (full text fetched and archived in
`brain/data/gate4_v2/sources/`; every assumption itemized in
`brain/data/gate4_v2/SOURCE_TABLE.md`): Shiu 2024 (model), Dorkenwald
2024 (v783), Schlegel 2024 (cell typing), Aso 2014 eLife 04577 + 04580,
Owald 2015, plus the project's own measured probes (P1–P6).

The audit produced three corrections to the v1 design, all
source-anchored, all frozen here:

1. **MBON activation-valence corrected.** Aso 2014 (eLife 04580) direct
   screen: *"all MBONs eliciting aversion were glutamatergic and all the
   MBONs eliciting attraction were either GABAergic or cholinergic"*;
   Owald 2015: M4/M6 activation **induces avoidance**, their inhibition
   converts avoidance into attraction, and reward **depresses** their CS+
   drive. The v1 dict had the M4/M6 family labelled `approach` —
   backwards. v2 classifies every MBON by its **measured v783
   neurotransmitter prediction** (`nt_type`, official annotations):
   GLUT → `avoidance` class (22 cells: MBON01/03/04/05/06/07/25/30),
   GABA+ACh → `approach` class (74 cells). Two tie-breaks documented
   (MBON03, MBON05: nt split 1/2, Aso-anchored glutamatergic). Generated
   by `brain/scripts/g4v2_audit.py` → `mbon_valence_corrected.json`.
2. **US population made compartment-specific.** v1 stimulated all 307
   PAM cells. Aso 2014: reward is conveyed by *"a distributed set of PAM
   cluster DANs that innervate the compartments of glutamatergic MBONs."*
   v2 US = **PAM01–PAM11** (252 cells; the Aso-2014 compartment-matched
   types; numbering verified against the connectome 17/21, PAM01–11 all
   matching). This is an **ENGINEERED OPTOGENETIC-STYLE US ENTRY**
   (probe P1: the anatomical sugar→PAM route is dynamically silent in the
   pinned model — this is NOT a naturally simulated sugar pathway).
3. **Assay moved to a balanced boundary.** v1's free-choice test was
   read through the closed-loop P9 turn decision, which the innate wiring
   lean saturates (A-control 100% RIGHT decided). v2 reads preference in
   **open-loop preference tests** (both CS ensembles at fixed calibrated
   rates; the fly stationary; no motor loop to saturate), with the
   closed-loop native route (A) still recorded. Calibration brings the
   baseline to P(left) ∈ [0.35, 0.65] BEFORE any reward-side assignment.

**Unchanged from v1 (the demonstrated mechanism):** the plasticity rule
form and all its constants (Section M below), the canonical brain, the
Gate-3 runtime, the chunked session protocol, determinism (bit-identical
replay), and the fail-closed verdict discipline.

## 1. Dataset provenance (Section B)

`brain/data/V783_MODEL_COVERAGE_REPORT.md` (computed, committed):
official v783 = 139,255 neurons; model roster = 138,639; the 616
excluded = 537 photoreceptors (no brain synapses) + 4 cervical DNge
cells (outside every used readout population) + 75 peripheral/ascending/
optic/endocrine cells. **Zero overlap with any sensory, MB, dopamine, or
motor population used in v2.** Not blocking.

## 2. Experimental apparatus (tiers labelled)

* **[BIOLOGICAL]** canonical Shiu LIF brain on v783, unmodified, driven
  by the frozen Gate-3 chunked runtime (`d7_runtime.py`); the v2 learning
  runtime composes on it via `d11_learning.LearningBrain` (KC→MBON
  synapse-row map, per-chunk KC spike counts, weight re-imposition) —
  NOT a replacement runtime.
* **[MODELED]** dopamine-gated depression rule (Section M).
* **[ENGINEERED]** the choice task, CS delivery (per-side KC ensemble
  stimulation; probes P2/P3: LC9→KC dynamically silent), US delivery
  (compartment-matched PAM01–11 stimulation; probe P1: sugar→PAM
  dynamically silent), the arena (route A only), the readouts (B and C),
  the session scheduler, seeding, and metrics.

**Learning path (the only route from reward to choice):**
CUE (CS ensemble) → KC representation (eligibility) → dopamine-gated
KC→MBON depression at avoidance-class (GLUT) MBONs → changed MBON
output (measured) → readout activity (B: aggregate MBON valence;
C: MBON→reachable-DN aggregate; A: native DN→decoder) → behavioral
choice. **Forbidden shortcuts (absent by construction and by test):**
reward→action, reward side→decoder, target side→correct answer, and any
use of the D13 external episodic-memory system (not linked; the decoder's
`external_bias_hz` is hard-zero in v2).

## 3. Populations (frozen; `brain/data/gate4_v2/`)

* CS ensembles: per-side seeded samples of left-/right-hemisphere KCs,
  sampling seed **20260929**, nested prefixes (400/side drawn; smaller
  sizes are prefixes). Size and rate frozen by the calibration (§4).
* US set: PAM01–11, 252 cells (`interface_v2.json:pam_us`).
* MBON classes: corrected (§0.1) — avoidance 22 / approach 74
  (`readout_v2.json`).
* Route-C DN sets: DNs reachable from approach-class MBONs (pure: 107)
  and from avoidance-class MBONs (pure: 3; mixed: 3), per side
  (`readout_v2.json`, from `mbon_to_dn_routes.json`).

## 4. Calibration (Sections F & G — runs BEFORE any reward-side assignment)

**Phase G — CS dose-response (pre-registered settings; ALL tested values
will be reported).** Symmetric per-side settings:

| id | n KC/side | CS rate/side (Hz) |
|----|-----------|-------------------|
| S1 | 100 | 75 |
| S2 | 200 | 75 |
| S3 | 200 | 100 |
| S4 | 400 | 100 |

For each setting, in ONE subprocess (seed **c1 = 20260930**), 6 open-loop
preference trials (protocol §5.1), no plasticity, no US. Measured per
setting: (i) CS-ensemble KC spike rate; (ii) approach-/avoidance-class
MBON rates per side; (iii) DN/walk-command population rates and the P9
left−right differential; (iv) route-B choice distribution (6 trials);
(v) route-C; (vi) stability (std of the valence differential across the
6 trials); (vii) wall-time/trial. **Selection rule (ordered):** among
settings with (i) KC ensemble rate ≥ 5 Hz AND (ii) both MBON class rates
≥ 2 Hz AND (iii) |P9 differential| < 8 Hz (the CS itself must not
dictate a turn) AND (iv) P(left|B) ∈ [0.15, 0.85] (not saturated),
choose the setting with the largest avoidance-class MBON response
(largest dynamic range for plasticity); tie-break: smaller ensemble
(lower cost). If NO setting passes, v2 is reported
FAIL-AT-CALIBRATION with all values — no parameter invention.

**Phase F — balanced-boundary calibration.** With the selected setting:
sweep kc_cs_left over {R−40, R−20, R, R+20, R+40} Hz (R = selected
symmetric rate; kc_cs_right fixed = R), 6 preference trials per value,
seeds **c1 = 20260930** and **c2 = 20261001** (two subprocesses; 12
trials/value pooled). Allowed knobs are ONLY these symmetric interface
parameters (per-side CS input strength); no reward-coupled or
action-side correction exists in the code path. **Neutral target:**
P(left | route B, decided) ∈ [0.35, 0.65]. **Selection rule:** the
left-rate minimizing |P(left) − 0.5|; if several qualify, the closest to
0.5; if NONE qualifies, the closest value is used AND the miss is
disclosed in the report (no further sweeping, no other knob touched).

**US feasibility check (interface verification, same class as v1 probes
P1b/P2b, not outcome data):** one pairing-protocol trial (§5.2) at the
frozen settings must show US-set PAM rate ≥ 10 Hz in the US window
(measured) and a measurable MBON response change; else v2 is reported
FAIL-AT-CALIBRATION.

**Freeze:** selected n, R, kc_cs_left, all constants below, the ensemble
ids, and the readout/interface specs are written to
`brain/data/gate4_v2/FROZEN_PARAMS.json` and committed + pushed. Only
then is the reward side assigned per session (§7). No parameter changes
after the freeze; deviations are recorded as deviations.

## 5. Trial protocols (frozen)

**5.1 Preference test (pref; open-loop; the G4B primary readout).**
reset_episode(seed) → re-apply plastic weights → 0.5 s silence →
**1.5 s both-CS window** (kc_cs_left and kc_cs_right at their frozen
rates; no US, no arena, no visual targets) → 0.3 s tail. Per trial we
record, over the last 1.2 s of the CS window: per-class per-side MBON
rates; CS-ensemble KC rates; DN population rates (incl. walk-command
pops and the P9 differential); route-B value
`V = (appr_L − avoid_L) − (appr_R − avoid_R)` (Hz) and choice_B =
left if V > 0, right if V < 0, undecided if exactly 0; route-C value
`C = (pure_appr_L − pure_avoid_L) − (pure_appr_R − pure_avoid_R)` and
choice_C likewise; and eligibility (for diagnostics).

**5.2 Pairing trial (classical conditioning; open-loop).**
reset → re-apply weights → 0.5 s silence → 1.5 s **paired-side CS**
(only the rewarded side's ensemble, at its frozen rate) with the **US
during the last 1.0 s** (pam_us channel at 150 Hz) → 0.5 s tail.
Eligibility trace runs over the whole trial (τ = 2 s, spike mode).
After the trial: relaxation step, then the dopamine-gated update with
the **measured** US-set PAM rate (gate ≥ 10 Hz) and measured PPL1 rate
(gate ≥ 10 Hz; expected 0 — no punishment in v2).

**5.3 Filler trial:** 1.5 s silence (retention delay).

**5.4 Context-change preference test:** §5.1 with BOTH CS rates scaled
by ×0.8 (three trials) and ×1.2 (three trials).

**5.5 Distractor preference test:** §5.1 preceded by a 0.5 s looming
channel pulse (the frozen Gate-3 loom rate) immediately before the CS
window.

**5.6 Route-A closed-loop test (native expression; unchanged from v1):**
the v1 ChoiceArena2D "choice" trial, the UNCHANGED Gate-3 decoder
(P9 turn source, thresholds, β = 1.5 valence-bias term, visual regime
target_left = 134 Hz), both CS ensembles fixation-gated as in v1.
Commit/timeout recorded. This route is expected to stay
innate-lean-dominated (v1 evidence) and is reported as the honest native
baseline.

## 6. Plasticity model (Section M — every equation labelled)

Substrate **[BIOLOGICAL]**: the connectome's own KC→MBON synapses
(62,261 rows; canonical weights fail-closed-verified against the live
network).

Rule **[MODELED]** — unchanged in form and constants from the v1
demonstration; only the target-set membership is corrected (§0):

* effective weight: `w = w_canonical × s`, s ∈ [0.05, 1], s₀ = 1.
* eligibility **[MODELED]**: per-KC exponential trace, τ_e = 2.0 s,
  spike mode; induction nonlinearity `(e/max(e))²` (frozen from v1
  pilot evidence).
* reward gate **[MODELED threshold on MEASURED activity]**: mean US-set
  PAM rate ≥ 10 Hz in the US window → depress KC→**avoidance-class**
  (GLUT) synapses of eligible KCs: `s ← max(s × (1 − 0.30 × ê), 0.05)`.
* punishment gate: mean PPL1 ≥ 10 Hz → depress KC→approach-class
  synapses (same form). NOT exercised in v2 (no aversive US).
* forgetting **[MODELED]**: per-trial relaxation toward 1,
  `s ← 1 − (1−s)·exp(−1/50)` (τ = 50 trials).

These rules are an **extension** to the published Shiu model — the
static model contained NO learning rules. Biology anchoring the FORM:
Aso 2014 (dopamine-gated depression at compartment-specific KC→MBON),
Owald 2015 (reward depresses M4/M6 CS+ drive), Hige 2015 / Cohn 2015 /
Davis 2023 (direction and compartmentalization). The specific constants
are modeling choices frozen since v1.

## 7. Session matrix (frozen)

Every session is its own subprocess (one network build), deterministic
(seeds below), full battery per §5. Reward side per session is assigned
from this pre-registered list — fixed for the whole session:

| session | condition | master seed | rewarded side |
|---|---|---|---|
| B_R1 | B_plastic | 20261101 | RIGHT |
| B_L1 | B_plastic | 20261102 | LEFT |
| B_R2 | B_plastic | 20261103 | RIGHT |
| B_L2 | B_plastic | 20261104 | LEFT |
| B_R3 | B_plastic | 20261105 | RIGHT |
| B_L3 | B_plastic | 20261106 | LEFT |
| A_R | A_no_plastic (US delivered, no plasticity) | 20261107 | RIGHT |
| A_L | A_no_plastic | 20261108 | LEFT |
| C_R | C_shuffled_da (dopamine gates lagged one trial) | 20261109 | RIGHT |
| D_R | D_shuffled_kcmbon (permuted eligibility→KC mapping) | 20261110 | RIGHT |
| F_R | F_untrained (no US, no plasticity; drift control) | 20261111 | RIGHT |
| E_rl | conventional ε-greedy Q-learner (NON-biological) | 20261112 | both (schedule-matched) |

**Battery (B/A/C/D/F identical structure; F's pairing trials are CS-only
no-US; A/C/D receive US as scheduled):**
1. baseline: 6 pref
2. acquisition: 4 blocks × (4 pairing → 2 pref → 1 route-A)
3. retention: 8 filler → 3 pref
4. extinction: 12 pref
5. reversal: 4 blocks × (4 pairing OTHER side → 2 pref → 1 route-A)
6. context change: 3 pref ×0.8 + 3 pref ×1.2
7. distractor: 3 pref with loom pre-pulse

Trial-level seeds: `master_seed + trial_index` (deterministic replay;
protocol proven bit-identical in v1/P5).

**Controls:** A = plasticity off (frozen weights, US still delivered);
C = dopamine gates lagged one trial (breaks CS–US contingency);
D = seeded permutation of the eligibility→KC attribution (same
depression magnitude, wrong synapses); E = Q-learner (learning-rate 0.3,
ε = 0.1, same trial schedule — explicitly non-biological comparison);
F = never trained. The D13 external episodic memory is NOT used
anywhere.

## 8. Verdict criteria (fail-closed; computed by `g4v2_analyze.py`)

**G4A — NEURAL ASSOCIATIVE LEARNING (independent verdict).** PASS iff
all of:
* **A1 (eligibility lateralization):** in ≥ 90% of B pairing trials,
  mean ensemble eligibility of the paired side > 2× the unpaired side.
* **A2 (cue-specific synaptic change):** end-of-acquisition, mean weight
  scale s over synapses (pre-KC ∈ paired-side CS ensemble) × (post-MBON
  ∈ avoidance class) is lower than over (pre-KC ∈ unpaired-side
  ensemble) × (post-MBON ∈ avoidance class); bootstrap 95% CI of the
  difference excludes 0 (pooled B sessions, n = 6).
* **A4 (learned valence signal):** pooled over B sessions, the
  acquisition-phase V (§5.1) shifts toward the rewarded side relative to
  that session's baseline mean (sign-normalized for LEFT-rewarded
  sessions); bootstrap 95% CI of the mean shift excludes 0.
* A3 (avoidance-class MBON response decrease to the paired CS) is
  reported but is not a gate.

**G4B — BEHAVIORAL EXPRESSION (independent verdict).** PASS iff all of:
* **B1:** pooled over the 6 B sessions, the route-B preference shifts
  toward the rewarded side: mean over sessions of
  (P(rewarded side | acquisition prefs) − P(rewarded side | baseline
  prefs)) > 0 with bootstrap 95% CI excluding 0.
* **B2 (both directions, Section J):** the RIGHT-rewarded subgroup mean
  shift > 0 AND the LEFT-rewarded subgroup mean shift > 0 (n = 3 each;
  subgroup point estimates reported with their own CIs; the gate is on
  the sign of each subgroup mean, not its CI).
* **B3 (causality through the learning pathway):** controls C and D do
  not reproduce the effect (each control's pooled shift CI overlaps 0
  OR is significantly below B's shift), and A and F show no shift
  (CI overlaps 0). If any control reproduces B's effect, G4B FAILS.
* Route A and route C results are reported alongside (labels:
  route B/C are ENGINEERED readouts; route A is the native DN route).
* Extinction decline and reversal flip are REPORTED (supportive
  evidence), not gates.

**OVERALL GATE 4: PASS only if G4B passes** (behavioral learning
causally attributable to the implemented MB/dopamine learning pathway).
**If G4A passes and G4B fails: STOP BIOLOGICAL TUNING.** The result is
preserved as-is; the future game agent may use a clearly labelled
HYBRID MBON readout, and the biological model is not modified further
to force a pass. No autonomous Roblox control is initiated during or
after Gate 4 v2 within this stage.

**Multiple testing discipline:** the gates above are the ONLY pass/fail
tests; all other quantities are labelled descriptive.

## 9. Performance recording (Section N)

Per session: wall time, biological time, peak RSS, CPU time, mean
wall-time per pref trial (decision latency), weights-checkpoint size,
bio-s/wall-s ratio. The full canonical model remains the reference
(NOT substituted); v2 runs the same numpy reference runtime as v1
(Gate-3 profile ≈ 3 GB, ~0.1 bio-s/wall-s). Any use of C++
acceleration would be validated bit-identical first; not planned for
the matrix.

## 10. Commit ordering (mandated)

1. THIS prereg + audit outputs + specs + coverage report (before
   outcome data).
2. Calibration outcomes + `FROZEN_PARAMS.json` (post-freeze, pre-matrix).
3. Final matrix results + analysis + `GATE4_V2_REPORT.md` + tests.

All three commits are pushed to `github.com/7clan/Fruit-fly-experiment-`
(main) as they happen, so a sandbox reset cannot lose the record.
