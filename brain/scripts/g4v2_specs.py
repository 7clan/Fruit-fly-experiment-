"""GATE 4 v2 — frozen experiment specs (readout populations + interface).

Builds the v2 experiment's data files from the source-audited audit outputs
(brain/data/gate4_v2/dan_compartment_table.json + mbon_valence_corrected.json):

  readout_v2.json   per-side MBON valence populations with the CORRECTED
                    (transmitter-anchored) classes, the reward-US PAM
                    population (PAM01-11), route-C DN populations
                    (approach-/avoidance-routed reachable DNs per side),
                    plus the frozen Gate-3 motor readout populations.
  interface_v2.json CS ensemble channels (seeded per-side KC samples),
                    the US channel (PAM01-11 set), the frozen Gate-3
                    visual/loom channels for route-A trials.

Also emits brain/data/gate4_v2/SOURCE_TABLE.md (Section A of the brief):
every biological assumption with CLAIM / SOURCE / EVIDENCE TYPE /
CONFIDENCE / HOW USED.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import brainlib as bl  # noqa: E402
import d11_learning as d11  # noqa: E402

BRAIN = bl.ROOT
G4 = BRAIN / "data" / "gate4_v2"
ANN = BRAIN / "data" / "flywire_annotations_v783"

CS_SEED = 20260929          # v2 CS ensemble sampling seed (frozen)
CS_N_PER_SIDE_DEFAULT = 200  # candidate; dose-response selects, then frozen


def side_map():
    ann = pd.read_csv(ANN / "classification.csv.gz", usecols=["root_id", "side"])
    return dict(zip(ann.root_id.astype("int64"), ann.side))


def build_specs():
    G4.mkdir(parents=True, exist_ok=True)
    audit = json.loads((G4 / "dan_compartment_table.json").read_text())
    val = json.loads((G4 / "mbon_valence_corrected.json").read_text())
    routes = json.loads((G4 / "mbon_to_dn_routes.json").read_text())
    sides = side_map()
    classes = {int(k): v for k, v in val["corrected_classes"].items()}

    # ---- readout populations (v2, corrected classes) -------------------
    d9_spec = json.loads(
        (BRAIN / "data" / "d7_d10" / "motor_readout.json").read_text())
    # DN_ALL_left/right removed: they overlap the route-C DN pops and
    # DualModeBrain._build_readout assigns each neuron to exactly ONE
    # population ("later pops win overlaps").  DN_ALL rates were
    # descriptive only; the route-A decoder uses P9/MDN/WALK/STOP/GF.
    pops = {k: list(v) for k, v in d9_spec["populations"].items()
            if k not in ("DN_ALL_left", "DN_ALL_right")}

    def by_side(ids, cls):
        sel = [i for i in ids if classes.get(int(i)) == cls]
        return ([i for i in sel if sides.get(int(i)) == "left"],
                [i for i in sel if sides.get(int(i)) == "right"],
                [i for i in sel if sides.get(int(i)) not in ("left", "right")])

    mbon_ids = d11.load_ids("MBON")
    al, ar, au = by_side(mbon_ids, "approach")
    vl, vr, vu = by_side(mbon_ids, "avoidance")
    pops["MBON_approach_left"], pops["MBON_approach_right"], \
        pops["MBON_approach_unside"] = al, ar, au
    pops["MBON_avoidance_left"], pops["MBON_avoidance_right"], \
        pops["MBON_avoidance_unside"] = vl, vr, vu

    # reward-US DAN population: PAM01-11 (Aso compartment-matched set).
    # Overlap discipline: ONLY the full set is a readout population (no
    # left/right children), so its spike counts are complete.
    us_ids = sorted(i for t in audit["reward_us_set"]
                    for i in audit["dan_types"][t]["ids"])
    pops["PAM_US_SET"] = us_ids
    ppl1 = d11.load_ids("PPL1")
    pops["PPL1_left"] = [i for i in ppl1 if sides.get(int(i)) == "left"]
    pops["PPL1_right"] = [i for i in ppl1
                          if sides.get(int(i)) == "right"]

    # route-C DN populations: DNs reachable from approach- vs avoidance-
    # class MBONs (v783 MBON->DN direct wiring), per side.  Pure sets are
    # disjoint by construction; "both" cells go to their own pops.
    appr_dn = set(routes["reachable_dn_approach_valence"])
    avoid_dn = set(routes["reachable_dn_avoidance_valence"])
    pops["DN_apprRoute_left"] = [i for i in sorted(appr_dn - avoid_dn)
                                 if sides.get(int(i)) == "left"]
    pops["DN_apprRoute_right"] = [i for i in sorted(appr_dn - avoid_dn)
                                  if sides.get(int(i)) == "right"]
    pops["DN_avoidRoute_left"] = [i for i in sorted(avoid_dn - appr_dn)
                                  if sides.get(int(i)) == "left"]
    pops["DN_avoidRoute_right"] = [i for i in sorted(avoid_dn - appr_dn)
                                   if sides.get(int(i)) == "right"]
    pops["DN_bothRoute_left"] = [i for i in sorted(appr_dn & avoid_dn)
                                 if sides.get(int(i)) == "left"]
    pops["DN_bothRoute_right"] = [i for i in sorted(appr_dn & avoid_dn)
                                  if sides.get(int(i)) == "right"]

    readout_spec = dict(
        version=3,
        note="GATE-4 v2 readout. MBON valence classes CORRECTED "
             "(transmitter-anchored: GLUT->avoidance, GABA/ACh->approach; "
             "Aso 2014 eLife 04580 screen + Owald 2015; measured v783 "
             "nt_type). PAM_US_SET = PAM01-11 (Aso-2014 compartment-"
             "matched reward DANs). Route-C DNs = v783 MBON->DN direct "
             "wiring by valence class.",
        populations=pops)
    (G4 / "readout_v2.json").write_text(json.dumps(readout_spec, indent=1))

    # ---- interface (v2) -------------------------------------------------
    d7_spec = json.loads(
        (BRAIN / "data" / "d7_d10" / "sensory_interface.json").read_text())
    keep = {ch: dict(d) for ch, d in d7_spec["channels"].items()
            if ch in ("target_left", "target_right", "looming")}

    sides_kc = d11.side_map()
    kc_ids = d11.load_ids("KC")
    rng = np.random.default_rng(CS_SEED)
    ens = {}
    for side in ("left", "right"):
        cand = np.array([i for i in kc_ids
                         if sides_kc.get(int(i)) == side])
        # pre-draw the largest candidate ensemble size; smaller sizes are
        # prefixes (deterministic, nested across the dose-response)
        pick = rng.choice(cand, size=min(400, len(cand)), replace=False)
        ens[side] = [int(x) for x in pick]

    interface = dict(
        version=3, sampling_seed=CS_SEED,
        interface_weight="w_syn * f_poi (model defaults, mirrors dbm.poi)",
        note="GATE-4 v2 interface. CS = per-side KC ensembles (ENGINEERED "
             "entry; probes P2/P3: LC9->KC anatomically present, "
             "dynamically silent in the pinned model). US = compartment-"
             "matched PAM01-11 stimulation - ENGINEERED OPTOGENETIC-STYLE "
             "US ENTRY (probe P1: sugar->PAM anatomically present, "
             "dynamically silent; NOT a naturally simulated sugar "
             "pathway). Reward NEVER touches action directly.",
        channels={
            **keep,
            "kc_cs_left": dict(population="KC left ensemble (seeded, "
                               "nested prefixes)", ids=ens["left"],
                               biological_role="CS: left context "
                               "(ENGINEERED entry)"),
            "kc_cs_right": dict(population="KC right ensemble (seeded, "
                                "nested prefixes)", ids=ens["right"],
                                biological_role="CS: right context "
                                "(ENGINEERED entry)"),
            "pam_us": dict(population="PAM01-11 (Aso compartment-matched "
                           "reward DANs)", n=len(us_ids), ids=us_ids,
                           biological_role="US: dopamine reward entry "
                           "(ENGINEERED, optogenetic-style)"),
        })
    (G4 / "interface_v2.json").write_text(json.dumps(interface, indent=1))

    counts = {k: len(v) for k, v in pops.items()}
    print(f"[specs] readout_v2: {len(pops)} populations")
    print(f"  MBON appr L/R/U: {len(al)}/{len(ar)}/{len(au)}; "
          f"avoid L/R/U: {len(vl)}/{len(vr)}/{len(vu)}")
    print(f"  PAM_US_SET: {len(us_ids)} cells")
    print(f"  route-C DNs: appr {len(appr_dn)}, avoid {len(avoid_dn)}, "
          f"both {len(appr_dn & avoid_dn)}")
    print(f"[specs] interface_v2: channels "
          f"{sorted(interface['channels'])}")
    return readout_spec, interface


def write_source_table():
    """Section A: every biological assumption in the v2 design."""
    rows = [
        # (claim, source, evidence type, confidence, how used)
        ("The pinned Shiu et al. LIF whole-brain model on FlyWire v783 is "
         "the canonical digital Drosophila brain (138,639 neurons).",
         "Shiu et al. 2024 Nature; github.com/philshiu/"
         "Drosophila_brain_model @ 91bdd1e (SHA-256 verified)",
         "MODELED / ENGINEERED", "HIGH",
         "Unchanged substrate; the v2 learning runtime composes on the "
         "Gate-3 chunked interactive runtime of this model."),
        ("Official FlyWire FAFB v783 = 139,255 neurons; the 616-cell "
         "difference vs the model roster is 537 photoreceptors + 4 unused "
         "cervical DNs + 75 peripheral cells; no used population affected.",
         "Dorkenwald et al. 2024 Nature; Codex v783 annotations; "
         "V783_MODEL_COVERAGE_REPORT.md (computed)",
         "CONNECTOMIC / DATASET", "HIGH",
         "Section B deliverable; confirms v2 populations are complete."),
        ("KC->MBON synapses (62,261 connections / 256,719 synapses, 77% "
         "ipsilateral) are the plasticity substrate.",
         "v783 connectivity parquet (computed, D5 + P0 verified)",
         "CONNECTOMIC", "HIGH",
         "The MODELED weight-scale vector lives on exactly these rows."),
        ("PAM-cluster DANs convey reward (appetitive reinforcement); "
         "PPL1-cluster DANs convey punishment.",
         "Aso 2014 eLife 04577/04580; Burke 2012; Liu 2012; Schroll 2006",
         "EXPERIMENTAL", "HIGH",
         "Sign of the dopamine gate in the plasticity rule (reward branch "
         "exercised in v2)."),
        ("Reward is conveyed specifically by the PAM DANs innervating the "
         "compartments of GLUTAMATERGIC MBONs.",
         "Aso 2014 eLife 04580 ('reward signals are mediated by a "
         "distributed set of PAM cluster DANs that innervate the "
         "compartments of glutamatergic MBONs'); Yamagata 2015; "
         "Owald 2015",
         "EXPERIMENTAL", "HIGH",
         "v2 US = PAM01-11 (the Aso-2014 compartment-matched types), "
         "replacing v1's all-307-PAM stimulation."),
        ("v783 hemibrain DAN numbering PAM01-15/PPL101-108 matches the "
         "Aso-2014 PAM-01..14/PPL1-01..06 compartments.",
         "Aso 2014 eLife 04577 Table 1 + v783 connectome co-innervation "
         "(17/21 direct matches; PAM01-11 all match; "
         "dan_compartment_table.md)",
         "CONNECTOMIC + EXPERIMENTAL", "HIGH for PAM01-11",
         "Compartment assignment of the US set; PAM15/PPL107/108 excluded "
         "(no Aso anchor)."),
        ("Activation of GLUTAMATERGIC MBONs drives avoidance; activation "
         "of GABAergic or cholinergic MBONs drives approach.",
         "Aso 2014 eLife 04580 ('all MBONs eliciting aversion were "
         "glutamatergic and all the MBONs eliciting attraction were either "
         "GABAergic or cholinergic')",
         "EXPERIMENTAL", "HIGH",
         "MBON valence classes anchored to MEASURED v783 nt_type "
         "predictions (transmitter column of the official annotations)."),
        ("M4/M6 (MBON-beta2beta'2a, MBON-beta'2mp, MBON-gamma5beta'2a): "
         "reward learning DEPRESSES their CS+ drive; their optogenetic "
         "activation INDUCES avoidance; their inhibition converts "
         "avoidance into attraction.",
         "Owald et al. 2015 (Neuron, PMC4416108)",
         "EXPERIMENTAL", "HIGH",
         "The v1 dict labelled this family 'approach' - BACKWARDS. v2 "
         "corrects it; the M4/M6 cells are in the depression target set "
         "(avoidance class)."),
        ("Appetitive memory = dopamine-gated DEPRESSION of KC->MBON "
         "synapses at the reward-DAN compartments (reduced avoidance-"
         "channel output to the CS+ => net approach).",
         "Aso 2014 eLife 04580 model; Owald 2015 (CS+ response decrease)",
         "EXPERIMENTAL (site/direction) + MODELED (implementation)",
         "HIGH (site/direction) / MEDIUM (exact equation form)",
         "v2 plasticity: reward gate depresses KC->GLUT-MBON (avoidance-"
         "class) synapses of eligible KCs. Constants unchanged from the "
         "v1-demonstrated rule."),
        ("Each MB compartment couples one DAN set and one MBON set "
         "(compartmentalized neuromodulation).",
         "Aso 2014 eLife 04577; Cohn 2015; reviews (Davis 2023)",
         "EXPERIMENTAL + CONNECTOMIC", "HIGH",
         "Compartment framing of the US set and the DAN->MBON co-"
         "innervation audit."),
        ("MBON->DN direct wiring exists (361 connections from 48 MBON "
         "types to 107 DNs) but does not dynamically reach the walk-"
         "command populations (P9/BPN/RRN/MDN) in the pinned model.",
         "v783 connectivity (computed); probe P4 (measured)",
         "CONNECTOMIC + MODELED (dynamics)", "HIGH",
         "Route A (native DN expression) is reported but not expected to "
         "carry the learned signal; routes B/C are labelled ENGINEERED."),
        ("The sugar (GRN_SUGAR) -> PAM route and the LC9 -> KC route are "
         "anatomically present but DYNAMICALLY SILENT in the pinned model.",
         "probes P1/P2/P3 (measured, v1 record, frozen)",
         "MODELED (dynamics of the pinned model)", "HIGH",
         "Justifies the ENGINEERED CS (direct KC stimulation) and the "
         "ENGINEERED OPTOGENETIC-STYLE US ENTRY (direct compartment-"
         "matched DAN stimulation)."),
        ("PPL1-gamma1pedc can substitute for electric shock as an aversive "
         "US.", "Aso 2014 eLife 04580 (MB438B/dTrpA1)",
         "EXPERIMENTAL", "HIGH",
         "Documented; the punishment branch is NOT exercised in the v2 "
         "matrix (reward-only US, as in v1)."),
        ("Eligibility trace tau = 2 s with (e/max)^2 induction "
         "nonlinearity; per-trial relaxation toward canonical weights "
         "with tau = 50 trials (forgetting/extinction).",
         "MODELED (this project, v1-demonstrated); calcium-trace "
         "nonlinearity motivated by Hige 2015 / Cohn 2015",
         "MODELED", "MEDIUM",
         "Unchanged from the v1-demonstrated rule (dopamine gating, "
         "lateralized eligibility, measured 46% weight change, MBON "
         "shifts)."),
        ("MBON transmitter classes (GLUT/GABA/ACh) per v783 cell.",
         "Official FlyWire v783 annotations nt_type predictions "
         "(neurotransmitter prediction from EM, Eckstein et al. 2024)",
         "CONNECTOMIC (predicted)", "HIGH at type level",
         "Corrected valence classification: 22 avoidance (GLUT) / 74 "
         "approach (GABA+ACh) of 96 MBONs; two documented Aso-anchored "
         "tie-breaks (MBON03, MBON05)."),
        ("The Gate-3 motor decoder and its thresholds, and the arena "
         "kinematics, are ENGINEERED readouts of DN activity.",
         "this project D9 (Gate 3, frozen)",
         "ENGINEERED", "N/A (engineering)",
         "Route A trials use the UNCHANGED Gate-3 decoder; route B (the "
         "v2 primary readout) is the aggregate MBON-valence difference - "
         "ENGINEERED tier-3c, clearly labelled."),
        ("Balanced-boundary assay: baseline preference must be neutral "
         "(P(left) in 0.35-0.65) before reward-side assignment, achieved "
         "by adjusting only symmetric interface parameters (left/right CS "
         "rates), never reward/action shortcuts.",
         "v2 brief Section F; v1 finding (innate lean saturates the "
         "closed-loop assay)",
         "ENGINEERED (protocol)", "N/A",
         "Pre-registered calibration procedure; frozen before the matrix; "
         "reward side assigned after the freeze."),
    ]
    lines = ["# Gate-4 v2 — source table (Section A)", "",
             "Every biological assumption in the v2 design, with its "
             "source, evidence tier, confidence, and use. Generated by "
             "`brain/scripts/g4v2_specs.py` (write_source_table).", ""]
    lines += ["| # | CLAIM | SOURCE | EVIDENCE TYPE | CONFIDENCE | "
              "HOW USED IN OUR MODEL |", "|---|---|---|---|---|---|"]
    for i, (c, s, t, conf, u) in enumerate(rows, 1):
        c = c.replace("|", "/")
        s = s.replace("|", "/")
        u = u.replace("|", "/")
        lines.append(f"| {i} | {c} | {s} | {t} | {conf} | {u} |")
    lines += ["", "## Primary sources (fetched and read for this audit)",
              "",
              "- Aso Y, et al. 2014. The neuronal architecture of the "
              "mushroom body provides a logic for associative learning. "
              "eLife 3:e04577 (PMC4273437) - Table 1 (DAN/MBON types, "
              "compartments, transmitters, drivers).",
              "- Aso Y, et al. 2014. Mushroom body output neurons encode "
              "valence and guide memory-based action selection in "
              "Drosophila. eLife 3:e04580 (PMC4273436) - activation-valence "
              "screen; appetitive/aversive compartment model.",
              "- Owald D, et al. 2015. Activity of defined mushroom body "
              "output neurons underlies learned olfactory behavior "
              "(PMC4416108) - M4/M6 reward trace direction.",
              "- Shiu Y, et al. 2024. A Drosophila computational brain "
              "model reveals sensorimotor processing. Nature - the pinned "
              "model (github 91bdd1e).",
              "- Dorkenwald S, et al. 2024. Neuronal wiring diagram of an "
              "adult brain. Nature - FlyWire v783.",
              "- Schlegel P, et al. 2024. Whole-brain annotation and "
              "multi-connectome cell typing of Drosophila. Nature - "
              "hemibrain_type labels.",
              "- Eckstein N, et al. 2024. Neurotransmitter classification "
              "from electron microscopy - the official nt_type "
              "predictions.",
              "- Burke 2012; Liu 2012; Yamagata 2015; Perisse 2013; "
              "Cohn 2015; Hige 2015; Berry 2018; Davis 2023 - reinforcement "
              "DAN biology and plasticity direction (cited via the Aso/"
              "Owald papers' direct statements above).",
              "",
              "## Tier definitions (docs/MASTER_SPEC.md)",
              "",
              "- BIOLOGICAL: measured connectome anatomy / measured model "
              "dynamics (probes).",
              "- EXPERIMENTAL: published primary experimental results.",
              "- CONNECTOMIC: computed from the v783 connectome.",
              "- MODELED: our model rule (documented, not claimed as "
              "biology).",
              "- ENGINEERED: task/protocol machinery, labelled, never "
              "credited to the brain."]
    (G4 / "SOURCE_TABLE.md").write_text("\n".join(lines) + "\n")
    print(f"[specs] SOURCE_TABLE.md -> {G4/'SOURCE_TABLE.md'} ({len(rows)} "
          "claims)")


if __name__ == "__main__":
    build_specs()
    write_source_table()
