# D4–D6: Neural I/O Map of the Digital Drosophila Brain (v783)

**Status: COMPLETE — GATE 2 (sensory → digital Drosophila → motor) PASSES.**
**Game control / Roblox / computer-vision integration: NOT STARTED (as instructed).**

Canonical brain: the pinned Shiu et al. leaky-integrate-and-fire whole-brain model
(commit `91bdd1e`, MIT) on the public **FlyWire v783** connectome
(138,639 neurons / 15,091,983 connections). Nothing in this document replaces
that reference; acceleration modes in §7 are measured alternatives, never silent
substitutions.

Terminology labels used throughout (per `docs/MASTER_SPEC.md`):
**[DATA]** biological connectome/annotation data · **[LIT]** published biological
evidence · **[MODEL]** modelled dynamics of the Shiu LIF implementation ·
**[ENGINEERED]** our interface and learning additions.

---

## 1. New data acquired for this stage (provenance)

The pinned model bundles connectivity only — no cell-type information. For
D4–D6 we therefore retrieved the **official FlyWire/Codex annotation tables for
exactly the v783 materialization** from the public Codex data release
(`storage.googleapis.com/flywire-data/codex/data/fafb/783/`), now vendored under
`brain/data/flywire_annotations_v783/`:

| file | content |
|---|---|
| `classification.csv.gz` | 139,255 neurons: `super_class`, `class`, `sub_class`, `side`, `nerve`, `hemilineage` |
| `consolidated_cell_types.csv.gz` | 138,327 neurons: consolidated `primary_type` (e.g. `T4b`, `KCg-m`, `PAM05`, `MDN`) |
| `neurons.csv.gz` | neurotransmitter prediction (`nt_type` + scores) and neuropil group |
| `column_assignment.csv.gz` | retinotopic column coordinates (x, y, p, q) for 45,528 columnar visual neurons |
| `labels.csv.gz` / `processed_labels.csv.gz` | 4 M+ author-deposited labels — **including the locomotor-circuit labels deposited by the Bidaye lab itself** (MDN, BPN, Foxglove, Bluebell, BRK, DNp09/P9 …) |
| `neuropil_synapse_table.csv.gz`, `cell_stats.csv.gz`, `visual_neuron_types.csv.gz`, `fw_and_hemibrain_types.csv.gz` | region occupancy, morphology stats, optic-lobe catalog, hemibrain cross-match |

Verification: **100.0 % of the 138,639 connectome neurons carry an annotation**
(138,639/138,639); annotation root IDs cover 99.56 % of the classification
table from the connectome side (the surplus rows are non-connectome neurons).
All population ID lists in this map are derived from these tables by
`brain/scripts/io_map_build.py`; nothing is hand-copied.

---

## 2. D4 — Sensory / visual input populations

### 2.1 Inventory (all counts are v783 [DATA]; function text is [LIT])

Sign = mean of the model's `Excitatory` column over outgoing connections
(+1 excitatory / −1 inhibitory, derived from FlyWire neurotransmitter
predictions; [MODEL] interprets dopamine/serotonin/octopamine as +1).

| population | region | n (L/R) | out-synapses | partners | sign | biological function | key reference |
|---|---|---|---|---|---|---|---|
| LMC **L1** | lamina | 1,775 (982/793) | 429,076 | 16,572 | −0.97 | ON / brightness channel | Borst 2018 |
| LMC **L2** | lamina | 1,728 (937/791) | 764,485 | 23,390 | +1.00 | OFF / contrast channel | Fu et al. 2020 |
| LMC **L3** | lamina | 1,477 (739/738) | 317,878 | 23,748 | +1.00 | sustained luminance | Kenner et al. 2022 |
| **Tm1/2/3/4/9** | medulla | 7,871 (3,981/3,890) | 3,394,777 | 71,949 | +1.00 | OFF-motion relay to lobula | Takemura et al. 2017 |
| **Mi1/4/9,T2,T3** | medulla | 7,806 (3,926/3,880) | 3,136,868 | 67,357 | +0.19 | ON-motion relay to T4 | Borst 2018 |
| **T4a–d** | medulla→LP | 6,243 (3,139/3,104) | 877,677 | 31,055 | +0.92 | ON direction-selective motion | Maisak et al. 2013 |
| **T5a–d** | lobula→LP | 6,002 (3,006/2,996) | 893,524 | 27,767 | +0.93 | OFF direction-selective motion | Maisak et al. 2013 |
| **HSN/HSE/HSS** | lobula plate | 6 (3/3) | 5,888 | 2,247 | +1.00 | horizontal optic flow | Hausen 1982 |
| **VS1–VS8** | lobula plate | 16 (8/8) | 8,725 | 2,814 | +0.89 | vertical optic flow | Hengstenberg 1982 |
| **LPLC2** | lobula+LP | 210 (108/102) | 84,760 | 7,665 | +1.00 | looming (expansion) detector | von Reyn et al. 2017 |
| **LPLC1/LPLC4** | lobula+LP | 250 (124/126) | 155,745 | 12,856 | +1.00 | looming family | Wu et al. 2016 |
| **LC4** | lobula | 104 (54/50) | 33,644 | 4,678 | +1.00 | looming escape timing | Williamson et al. 2018 |
| **LC16** | lobula | 152 (78/74) | 58,112 | 4,153 | +0.89 | small-object approach → backward walking | Sen et al. 2017 |
| **LC17** | lobula | 276 (134/142) | 148,105 | 3,680 | +1.00 | edge motion / approach | Klapoetke et al. 2022 |
| **LC9** | lobula | 179 (87/92) | 96,223 | 5,714 | +1.00 | visual target → directed walking | Bidaye et al. 2020 |
| **MeTu** | medulla→AOTU | 904 (457/447) | 198,608 | 13,600 | +0.99 | anterior visual pathway (navigation) | Omoto et al. 2017 |
| GRN sugar set | SEZ | 20 (tutorial) | 6,878 | 285 | +1.00 | sugar taste (tutorial precedent) | Shiu et al. 2025 |
| ORN (register) | antennal lobe | 2,281 | 358,178 | 3,184 | +1.00 | olfaction — reserved, unused | [DATA] |
| mechanosensory (register) | AMMC/SEZ | 2,674 | 319,736 | 5,602 | +0.99 | vibration/sound — reserved | [DATA] |

Non-visual entries (ORN, mechanosensory, gustatory) are **registered but not
part of the initial game interface**: the game display provides only visual
information. Johnston's-organ mechanosensory neurons remain a documented
option should game audio ever be mapped.

### 2.2 Proposed game-sensory channels [ENGINEERED interface]

| channel | entry populations | encoding (first version) | confidence |
|---|---|---|---|
| `vis_target_left` / `vis_target_right` | LMC L1/L2 + Tm (per **eye**) | hemisphere = visual hemifield; Poisson drive scaled by target salience | medium |
| `looming_threat` | **LPLC2 + LC4** | rate ∝ simulated angular-expansion velocity | **high (direct motor path)** |
| `optic_flow` | HS + VS | signed differential drive | medium |
| `motion_directional` | T4a–d / T5a–d | subtype-selective drive matching CV motion vector | medium (needs subtype→direction calibration) |
| `brightness_contrast` | L1 (ON), L2 (OFF), L3 (ambient) | drive ratio encodes luminance change | high |
| `object_approach` | LC16/LC17 (+LPLC2 for fast expansion) | drive scaled by object size/approach speed | medium |
| `reward_pulse` | **PAM** (D5) | brief burst on reward events (game EXP/gain/level-up) | medium, see §3.3 |
| `punish_pulse` | **PPL1** (D5) | brief burst on punishment events (damage/death) | medium, see §3.3 |

Refinement path (not needed for first closed loop): `column_assignment.csv.gz`
gives (x, y, p, q) retinotopic coordinates for 45,528 columnar neurons, so the
coarse hemifield channel can later become a true screen-position channel.

**Entry-class finding (Gate-2 empirical, §6):** convergent visual-projection
entries (LPLC2, LC4, LC9 — each tens-to-hundreds of cells feeding a specific
behavioural channel) reach motor populations directly; distributed retinotopic
entries (L2, T4/T5) propagate 1–2 hops and then fall below spike threshold at
default parameters. This mirrors the biology — LC/LPLC cells are exactly the
bottleneck where distributed retinotopic information converges onto behavioural
programmes — and it dictates the interface design above.

---

## 3. D5 — Mushroom body / dopamine / learning circuitry

### 3.1 Inventory

| population | n (L/R) | types | out-synapses | sign | function [LIT] |
|---|---|---|---|---|---|
| Kenyon cells (KC) | 5,177 (2,580/2,597) | KCg-m 2,189 · KCab 1,643 · KCapbp-* 916 · KCg-d 295 · other 34 | 984,448 | +1.00 | sparse coders; sites of dopaminergic plasticity |
| MBON | 96 (48/48) | 35 types (MBON01–35 + variants) | 204,864 | mixed (+18,728 / −18,719 conns) | compartment readout; single-type activation drives approach/avoidance (Aso et al. 2014) |
| PAM DANs | 307 (154/153) | PAM01–15 | 59,285 | +1.00 | **reward/appetitive** reinforcement (sugar, water) |
| PPL1 DANs | 16 (8/8) | PPL1-01…08 | 30,975 | +1.00 | **punishment/aversive** reinforcement (shock, bitter) |
| PPL2 DANs | 8 (4/4) | PPL2-01…04 | 11,557 | +1.00 | less-characterised cluster; registered, unused initially |
| APL | 2 (1/1) | — | 110,784 | −1.00 | GABAergic wide-field feedback; gain control/normalisation |
| DPM | 2 (1/1) | — | 15,716 | +1.00 | neuromodulatory feedback; memory consolidation |
| DAN-DNs (DopaMeander candidates) | 7 (4/3) | DNc01/02, DNp32×2, DNg86, DNge047 | 5,033 | +1.00 | dopaminergic **descending** modulators; DopaMeander drives forward walking + turning (Liessem et al. 2026) |

Direct pathway checks [DATA]: **KC → MBON = 62,261 connections / 256,719
synapses** (the associative-plasticity site); **PAM+PPL1 → MBON/KC = 48,273
connections / 74,496 synapses** (the neuromodulation site). Both sites are
therefore present and wired in the model exactly as the MB literature requires.

### 3.2 What the evidence does and does not support

Biologically observed [LIT]: compartmental organisation (KC axons → 15
compartments, each with a characteristic DAN type and MBON dendrites);
dopamine-gated plasticity at KC→MBON synapses (reward via PAM, punishment via
PPL1); MBON valence (activation of individual types elicits approach or
avoidance); DAN heterogeneity (absolute vs relative punishment signalling);
APL gain control; DPM consolidation.

Not yet resolved at this stage: the per-type mapping MBONxx/PAMxx ↔ the 15 fine
compartments (v783 neuropil segmentation is coarse: MB calyx/ML/PED/VL only).
Refinement requires the hemibrain MB tables or the v783 synapse-coordinate
release; documented as follow-up work, no mapping was guessed.

### 3.3 Proposed reward–learning interface (three explicit tiers)

**Tier 1 — biologically observed mechanisms [LIT/DATA]** (wired in v783, to be
used as-is): PAM/PPL1 compartmental innervation; KC→MBON convergence; MBON
output valence; APL feedback. Nothing here is invented.

**Tier 2 — modelled assumptions [MODEL]** (already implicit in the Shiu LIF
model): synapses are static in the released model; dopamine is not a distinct
signal type — DAN activity is just spiking with sign +1; stimulation is
Poisson-input onto chosen neurons (the model's own optogenetics abstraction).

**Tier 3 — engineered additions [ENGINEERED]** (ours; must stay labelled, and
are *not* yet implemented at D4–D6): (a) mapping game events to PAM/PPL1 bursts
(the game's "reward" is not sugar — the analogy is explicit); (b) a Stage-C
plasticity rule at KC→MBON synapses gated by DAN activity (dopamine-gated
depression/potentiation in the established MB tradition), implemented as Brian2
synapse dynamics on top of the pinned model; (c) reading MBON population rates
as a valence bias for action selection. **No arbitrary RL will be labelled as
biological** — any Q-learning/value-network machinery, if ever added at Stage D,
lives outside the brain, talks to it only through PAM/PPL1 stimulation and
MBON readout, and is reported as an external planner-memory module.

---

## 4. D6 — Descending / motor output populations

### 4.1 Inventory

v783 contains **1,305 descending neurons of 473 types** (647 L / 650 R / 8
centre; 1,303 leave via the cervical connective) [DATA], consistent with the
~1,000–2,000 range of published DN counts (Namiki et al. 2018). 106 brain
motor neurons (proboscis/pharynx; incl. MN9 = CB0701, the tutorial readout)
close the loop inside the brain for feeding-related actors.

Key command populations (IDs from author-deposited FlyWire labels where
available — the original experimenters picked these very neurons):

| population | n | v783 type | nt | known behaviour [LIT] | evidence |
|---|---|---|---|---|---|
| **MDN** | 4 | `MDN` | ACH | backward walking (activation sufficient & necessary) | Bidaye et al. 2014 (Science); Sen et al. 2017 |
| **DNp09 (P9)** | 2 | `DNp09` | ACH | forward walking **with ipsilateral turning**; visual/courtship driven | Bidaye et al. 2020 (Neuron); Sapkal et al. 2024 |
| **BPN** | 33 (types 1–4) | `SMP461`, `SMP459`, `CL209/210`, … | ACH | straight fast forward walking | Bidaye et al. 2020 |
| **RRN (roadrunner)** | 2 | `CB0257` | ACH | initiates forward walking, sets speed | Dallmann et al. 2026 (bioRxiv) |
| leg-DN cluster | 24 | `DNg101/102`, `DNp64`, `DNge050`, … | mixed | leg-level forward programme downstream of RRN | Dallmann et al. 2026 |
| **DNp01 (giant fibers)** | 2 | `DNp01` | ACH | escape jump / startle; driven by LPLC2+LC4 | King & Wyman 1980; von Reyn 2017; Ache 2019 |
| **FG (Foxglove)** | 2 | `CB0890` | **GABA** | walk-OFF: halts **forward** walking | Sapkal et al. 2024 (Nature) |
| **BB (Bluebell)** | 2 | `DNg60` | **GABA** | walk-OFF: halts **turning** | Sapkal et al. 2024 |
| **BRK (brake)** | 6 | `AN_GNG_53/54/76` (ascending) | ACH | active arrest of stepping (grooming context) | Sapkal et al. 2024 |
| MooSEZ | 2 | `CB0191` | ACH | moonwalker-related SEZ neuron | Israel et al. 2022 |
| iAN-MDN | 2 | `AN_multi_41` (ascending) | GABA | inhibitory feedback onto MDN | [labels, Bidaye lab] |

Direct v783 pathway checks [DATA]: looming LPLC2+LC4 → giant fiber =
**293 connections / 1,885 synapses (direct)**; LC9 → DNp09 =
**114 / 1,251 (direct)**; RRN → leg-DN cluster = 10 / 41 (weak direct, function
multi-hop per Dallmann et al.); LC16 → MDN = **0 direct connections** — the
Sen et al. 2017 hypothesis (LC16 induces backward walking *via* MDNs) is
mediated, not monosynaptic, exactly as their paper proposed; FG/BB halting is
also predominantly multi-hop (2 direct synapses FG→forward command neurons),
matching Sapkal et al.'s network-level account — their paper used this very
Shiu model to make the same point.

### 4.2 Are FORWARD / BACKWARD / LEFT / RIGHT / STOP biologically separable?

**Yes — with honest qualifications.**

- **BACKWARD (MDN)** — cleanly separable, activation-sufficient, bilateral pair.
- **FORWARD** — separable at two levels: command level (BPN for straight
  fast walking; RRN for initiation/speed) and DN level (DNp09 drives forward
  *with* turning; the leg-DN cluster executes). Straight-forward vs
  forward-and-turn can be dissociated (BPN vs DNp09).
- **LEFT / RIGHT** — separable **as turning, not as pure translation**: DNp09
  is a bilateral pair whose left/right members drive **ipsiversive** turns
  (left neuron → turn left); DopaMeander DAN-DNs additionally correlate with
  ipsiversive turning (candidate stage). A "strafe" style pure lateral
  translation has no known fly analogue — the game must map turning to its
  steering controls.
- **STOP** — separable, and unusually well characterised for a "stop": FG halts
  forward walking, BB halts turning (both GABAergic walk-OFF, brain-resident);
  BRK actively brakes (partly VNC-resident — its ascending axons are in v783,
  its leg-side execution is not, because the brain connectome ends at the neck).
  A locomoting fly stopping also occurs when walk-command neurons simply go
  quiet, so STOP additionally serves as the decoder's **default/failsafe**
  state.

**Bonus action: STARTLE/JUMP (giant fiber)** — the best-characterised command
neuron in the fly; maps to a game jump/dash if Grand Piece Online exposes one.

### 4.3 Proposed decoder rules [ENGINEERED]

- `BACKWARD` iff rate(MDN) > θ_bwd (either side; both sides = straight backward).
- `FORWARD` iff [rate(BPN-t1)+rate(RRN)+rate(DNp09)]/n > θ_fwd sustained T ms.
- `TURN-LEFT` iff rate(DNp09_L) − rate(DNp09_R) > δ (ipsiversive: left→left);
  `TURN-RIGHT` mirror.
- `STOP` iff rate(FG)+rate(BB) > θ_stop, **or** all walk populations idle for
  T_idle — STOP is the default state (failsafe).
- `JUMP` iff rate(DNp01) > θ_jump.
- Thresholds θ are to be calibrated from baseline/stimulation rate
  distributions (D8); per-side normalization by population size; hysteresis
  on all transitions. Decoder statistics per population (firing rates under
  stimulation) are already recorded by the Gate-2 harness.

---

## 5. Gate-2 verification — sensory → brain → motor

Pre-declared criteria and outcomes (all evidence in `brain/results/gate2_io/`;
protocol: 150 Hz Poisson drive onto seeded population samples of 40–80 neurons,
1 s trials, seed 11, fresh subprocess per trial):

| condition | active | spikes | DNs active | DN spikes | GF (DNp01) | notes |
|---|---|---|---|---|---|---|
| baseline (no stim) | 0 | 0 | 0 | 0 | 0 | brain silent without input |
| vis_L2_L (left eye) | 295 | 12,452 | 0 | 0 | 0 | 2-hop spread in optic lobe |
| vis_L2_R (right eye) | 376 | 14,828 | 0 | 0 | 0 | distinguishable from L (Jaccard 0.42) |
| **loom (LPLC2+LC4)** | 466 | 23,524 | **63** | **2,087** | **220 spikes (~110 Hz)** | escape path fires |
| motion (T4+T5) | 51 | 6,079 | 0 | 0 | 0 | 1-hop spread |
| **target (LC9)** | 580 | 18,121 | **92** | **1,381** | 2 | MDN 6 spikes |
| relay-silence note | 293 | 12,435 | 0 | 0 | 0 | silencing 120/7,871 Tm cells ≈ no effect |
| **entry-off control** | 40 | 5,793 | 0 | 0 | 0 | downstream activity collapses to **0** |
| vis_L2_L seed 12 | 293 | 12,737 | 0 | 0 | 0 | seed sensitivity |

| criterion | result | evidence |
|---|---|---|
| G2.1 stimulus effect | **PASS** | every sensory condition evokes activity above the silent baseline |
| G2.2 motor readout | **PASS** | looming → 63 DNs incl. giant fibers at ~110 Hz; visual-target LC9 → 92 DNs (incl. MDN); honest caveat: L2/T4/T5 samples attenuate below DN threshold (multi-hop) |
| G2.3 reproducibility | **PASS** | same seed, fresh processes → bit-identical spike files (SHA-256 equal: `a113fa41…`, `ee52eafd…`) |
| G2.4 sensitivity | **PASS** | seed 11 vs 12 → different files (`a113fa41…` ≠ `3320861…`) |
| G2.5 causal control | **PASS** | identical drive with entry synapses zeroed → downstream active = 0 of 255 |
| G2.6 lateralization | **PASS** | left vs right eye stimulation produce distinct activity fingerprints |

**GATE 2: PASS — a verified sensory → digital Drosophila → motor path exists
and can be stimulated/read reproducibly.** Two independent motor-reaching
sensory channels are established (loom→escape via giant fiber; visual
target→DN population recruitment), both reproducible bit-for-bit.

Honest limitations recorded with the pass: (1) lamina/motion single-population
samples do not reach DNs under default parameters — the interface therefore
enters at convergent VPN classes for motor-relevant channels (as §2.2 does);
(2) DNp09 itself did not spike under the 40-cell LC9 sample (its direct LC9
input exists but the neuron needs more convergence; MDN and 90 other DNs did
fire) — calibrating sample size/rate is D8 work; (3) the relay-silencing
observation shows population-level (not 1.5 %-sample) silencing is needed for
manipulation studies, consistent with the optic lobe's massive parallelism.

---

## 6. Performance study (D7 pre-work)

Machine: this 2-core, 3.9 GB box. Standard workload: the Gate-2 looming
stimulus (80 neurons), seed 11, 1 bio-second, 23,524 spikes. The canonical
full brain remains the reference; numbers below are measured alternatives.

| mode | neurons | connections | build s | run s | bio-s / wall-s | peak RSS MB | fidelity vs reference |
|---|---|---|---|---|---|---|---|
| 1. full_reference (numpy) | 138,639 | 15,091,983 | 2.1 | 10.7 | 0.094 | 2,842 | — (reference) |
| 2. subcircuit (1+1-hop ROI) | 49,817 | 2,881,511 | 0.5 | 7.3 | 0.137 | **793** | Jaccard 0.63 · 94.5 % spikes · 73/63 DNs (approximate) |
| 3. sparse_active (+1 hop) | 53,781 | 6,579,005 | 0.8 | 8.1 | 0.123 | 1,297 | **exact**: Jaccard 1.0 · 23,524/23,524 spikes · 63/63 DNs |
| 4a. cython (cold, incl. compile) | 138,639 | 15,091,983 | 4.5 | 178.4 | — | 3,138 | bit-identical |
| 4b. cython (warm) | 138,639 | 15,091,983 | 1.9 | 10.6 | 0.094 | 2,841 | **bit-identical** (Jaccard 1.0) |
| 5. cpp_standalone (codegen+compile+run) | 138,639 | 15,091,983 | 1.7 | 36.9 | — | 2,955 | **bit-identical** (Jaccard 1.0) |
| 5b. cpp compiled binary alone | 138,639 | 15,091,983 | — | **3.5** | **0.283** | **678** | same run, timed directly |

(Full machine-readable table incl. RAM details: `brain/results/perf_study/perf_study.json`.)

Findings, without any replacement of the reference model:

1. **The task-relevant subcircuit is intrinsically brain-wide.** A ≤3+3-hop
   entry→motor bridge already spans 135,009/138,639 neurons (97.4 %); 2+2 hops
   spans 85 %. Only the tight 1+1 radius yields an executable subcircuit
   (49,817 neurons, 36 %), which still needs 74 % of the runtime (the per-step
   cost is dominated by the active/synaptic set, not by neuron count) but cuts
   **RAM by 72 %** and reproduces 94.6 % of spikes.
2. **Sparse active-neuron simulation is exact, not approximate.** Simulating
   only the reference trial's active set + one downstream hop reproduces the
   reference output *exactly* (466/466 active, 23,524/23,524 spikes) — a
   provable consequence of never-spiking neurons carrying no synaptic events
   in this deterministic-LIF regime. This is the safest acceleration for
   repeated-trial learning protocols.
3. **Cython codegen gives no speedup here** (warm 10.6 s vs 10.7 s numpy;
cold 178.4 s is compile-dominated and peaks at 3.14 GB — the very reason
in-process cython attempts OOM'd this box). Output remains bit-identical.
4. **C++ standalone compilation is the real speed lever on a laptop-class
   budget: the compiled binary runs the identical full-brain trial in
   3.5 s at 678 MB — 3.1× faster and 4× lighter than the numpy runtime** —
   after a one-time ~37 s codegen+compile (g++ project ≈ 589 MB). Output is
   bit-identical to the reference (Jaccard 1.0, 23,524/23,524 spikes,
   63/63 active DNs). Engineering caveat: the seed is baked into the
   generated sources, so per-trial reseeding needs a cheap main-module
   recompile — a D7 engineering task, not a science change.
5. Implication for the Windows laptop target: **subcircuit (RAM) + C++
   standalone (speed) + sparse-active (exactness for trial batches)** together
   make real-time-ish closed-loop operation plausible on consumer hardware,
   while the full-brain numpy mode remains the scientific reference for
   validation runs.

---

## 7. Machine-readable artifacts

- `brain/data/io_map/neural_io_map.json` — the complete map (populations with
  per-population v783 stats, pathway checks, channels, actions, provenance)
- `brain/data/io_map/ids/<pop_id>.json` — full v783 root-ID lists per population
- `brain/data/io_map/*.csv` — sensory / learning / motor populations, input
  channels, output actions
- `brain/results/gate2_io/` — manifest (exact stimulation sets), spike files,
  trial records, `readouts.csv`, `verdict.json`
- `brain/results/perf_study/` — `perf_study.json`, `perf_study.csv`, ROI ladder,
  filtered connectomes, C++ project
- Scripts (all in `brain/scripts/`): `io_map_populations.py`,
  `io_map_build.py`, `io_trial_runner.py`, `verify_io_path.py`,
  `perf_trial.py`, `perf_study.py`

## 8. What this document does NOT claim

- No claim of biological equivalence, consciousness, or faithfulness beyond
  the cited connectome data and LIF dynamics.
- No computer-vision pipeline, no game input, no Roblox access at this stage
  (next gate work only).
- No plasticity implemented yet; the reward interface in §3.3 is a labelled
  proposal, not a running mechanism.
- DAN→compartment fine mapping and DopaMeander's exact v783 identity remain
  open items, recorded as such rather than guessed.

## 9. Key references

Aso et al. 2010 Science; Aso et al. 2012 Curr Biol; Aso et al. 2014 eLife;
Bidaye et al. 2014 Science; Bidaye et al. 2020 Neuron (P9/BPN);
Borst 2018; Dallmann et al. 2026 bioRxiv (RRN, 10.64898/2026.01.04.697356);
Kenner et al. 2022; Klapoetke et al. 2022; Li et al. 2020 eLife (MB connectome);
Liessem et al. 2026 Curr Biol 36:3808–3822 (MDN/DopaMeander);
Maisak et al. 2013 Nature; Namiki et al. 2018 eLife; Omoto et al. 2017;
Sapkal et al. 2024 Nature (FG/BB/BRK halting); Sen et al. 2017 Cell Rep (LC16/MDN);
Shiu et al. 2025 (the LIF brain model); Takemura et al. 2017;
Timaeus et al. 2020; von Reyn et al. 2017; Williamson et al. 2018;
Wu et al. 2016 eLife (LC catalogue); Yamagata et al. 2016; plus the FlyWire
v783 annotation release (Codex) and author-deposited FlyWire labels.
