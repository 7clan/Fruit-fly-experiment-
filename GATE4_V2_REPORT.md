# GATE 4 v2 — Source-Audited Associative Learning Experiment — FINAL REPORT

**Status: [TO BE FILLED — matrix in flight when this skeleton was
written; every number below is inserted from
`brain/results/gate4_v2/analysis/verdicts.json` after the matrix
completes. No number in this report is hand-entered.]**

**Pre-registration:** `GATE4_V2_PREREG.md` (commit `e152676`, pushed
BEFORE any outcome data). **Calibration frozen:** commit `0b35496`
(ALL tested values in `brain/data/gate4_v2/calibration_report.md`).
**v1 record:** PERMANENTLY FAIL (tag `gate4-v1-FAIL`, commit `38a65a9`);
this report does not alter it.

---

## 1. What v2 changed and why (Sections A–D of the brief)

The v2 brief asked one question: **can the already-demonstrated
mushroom-body synaptic plasticity produce behavioural learning under a
scientifically valid assay that is not saturated by the innate motor
bias?** The source audit (primary texts fetched, archived, and quoted in
`brain/data/gate4_v2/SOURCE_TABLE.md`) forced three corrections before
any v2 data was collected:

1. **MBON activation-valence was corrected.** Aso 2014 (eLife 04580,
   direct optogenetic screen): *"all MBONs eliciting aversion were
   glutamatergic and all the MBONs eliciting attraction were either
   GABAergic or cholinergic."* Owald 2015: M4/M6 activation **induces
   avoidance**; their inhibition converts avoidance into attraction;
   reward **depresses** their CS+ drive. The v1 dictionary had the M4/M6
   family labelled `approach` — backwards. v2 classifies all 96 MBONs by
   their **measured v783 neurotransmitter predictions**: 22
   avoidance-class (GLUT) / 74 approach-class (GABA+ACh).
2. **The US became compartment-specific.** v1 stimulated all 307 PAM
   cells. Aso 2014: reward is conveyed by *"a distributed set of PAM
   cluster DANs that innervate the compartments of glutamatergic
   MBONs."* v2 stimulates **PAM01–PAM11** (252 cells; numbering verified
   against the connectome, 17/21 Aso anchors match, all of PAM01–11
   match). Labelled everywhere as an **ENGINEERED OPTOGENETIC-STYLE US
   ENTRY** — the anatomical sugar→PAM route is dynamically silent in the
   pinned model (probe P1), so this is not a naturally simulated sugar
   pathway.
3. **The assay moved to a calibrated balanced boundary.** v1's
   free-choice test was read through the closed-loop P9 turn decision,
   which the innate wiring lean saturated. v2 reads preference in
   **open-loop preference tests** (both CS ensembles at frozen rates;
   no motor loop), calibrated to **P(left) = 0.500** before any
   reward-side assignment, with the native closed-loop route (A) still
   recorded.

The plasticity rule itself (dopamine-gated depression of KC→MBON,
eligibility trace, constants) is **unchanged from the v1-demonstrated
mechanism** — v2 changes the assay and the source-audited target set,
not the learning rule. Dataset provenance is documented in
`brain/data/V783_MODEL_COVERAGE_REPORT.md` (the 616 excluded neurons
touch no population used here).

## 2. The two independent verdicts (Section E)

The v2 preregistration separates **G4A (neural associative learning)**
from **G4B (behavioural expression)**. A G4A PASS with G4B FAIL is a
scientifically acceptable outcome that terminates biological tuning; it
does not trigger endless parameter search.

[G4A/G4B verdict tables inserted from the analyzer after the matrix]

## 3. Mechanistic findings (from the session data, labelled by tier)

[TO BE FILLED: the network-gain/bistability analysis, eligibility
contamination by US cascades, control contrasts A/C/D/F, route A/B/C
comparison, extinction/reversal/context/distractor behaviour]

## 4. What this stage claims and does NOT claim

[TO BE FILLED after verdicts]

## 5. Machine-readable evidence

- `brain/data/gate4_v2/` — SOURCE_TABLE.md, dan_compartment_table.{json,md},
  mbon_valence_corrected.json, mbon_to_dn_routes.json, FROZEN_PARAMS.json,
  calibration_report.{json,md}, interface_v2.json, readout_v2.json,
  sources/ (fetched primary texts)
- `brain/results/gate4_v2/{B_R1..F_R}/` — trials.jsonl, session.json,
  weights*.npz per session (auto-pushed as each completed)
- `brain/results/gate4_v2/analysis/verdicts.json` + `report_sections.md`
- `brain/results/gate4_v2/calibration/` — ALL dose/boundary/uscheck values
- tests: `tests/test_g4v2.py` (11 unit tests; full suite 92 PASS)
