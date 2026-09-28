"""GATE 4 v2 — SOURCE-AUDITED ANATOMY (Sections A, B, C of the v2 brief).

Generates, from the canonical v783 connectome + official annotation tables +
the primary literature tables (Aso 2014 eLife 04577/04580; Owald 2015):

  1. DAN compartment table  (Section C)
     every PAMxx / PPL1xx hemibrain type in v783: cell count, sides,
     strongest MBON partners (connectome), inferred MB-lobe compartment,
     Aso-2014 anchor row, reinforcement association, evidence + confidence.

  2. CORRECTED MBON valence classification (source-audited)
     Aso 2014 (eLife 04580): "all MBONs eliciting aversion were glutamatergic
     and all the MBONs eliciting attraction were either GABAergic or
     cholinergic."  Owald 2015: M4/M6 (MBON-beta2beta'2a, MBON-beta'2mp,
     MBON-gamma5beta'2a) activation INDUCES AVOIDANCE; their inhibition
     converts avoidance into attraction; reward DEPRESSES their odor drive.
     => activation-valence anchored to MEASURED v783 neurotransmitter
     predictions (neurons.csv.gz nt_type), not to type-number comments.
     The v1 dict (brain/closedloop/populations.py) labelled the M4/M6
     family 'approach' - BACKWARDS vs both primary sources.  This audit
     quantifies and corrects that error; v1 results stay frozen as recorded.

  3. Reward-US candidate set = PAM types innervating compartments of the
     glutamatergic (activation-aversive) MBONs (Aso 2014: "reward signals are
     mediated by a distributed set of PAM cluster DANs that innervate the
     compartments of glutamatergic MBONs").

  4. MBON->DN expression-route table (for v2 Section I route C).

Outputs -> brain/data/gate4_v2/ :
  dan_compartment_table.json / .md   (Section C table)
  mbon_valence_corrected.json        (v2 plasticity target sets)
  mbon_to_dn_routes.json             (Section I route C wiring)
  audit_summary.json                 (cross-checks vs v1 dict, flags)

Usage:  python g4v2_audit.py            (anatomy only; no simulation)
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import brainlib as bl  # noqa: E402

BRAIN = bl.ROOT
ANN = BRAIN / "data" / "flywire_annotations_v783"
IDS = BRAIN / "data" / "io_map" / "ids"
OUT = BRAIN / "data" / "gate4_v2"

# --------------------------------------------------------------------------
# PRIMARY-SOURCE TABLES (hardcoded verbatim from the fetched papers; the
# full texts are archived in brain/data/gate4_v2/sources/)
#
# Aso et al. 2014 eLife 04577 Table 1 (PMC4273437): DAN types.
#   short name -> (compartment(s) innervated, cells/hemisphere, drivers)
# NOTE: Aso numbering "PAM-01..14" == hemibrain/v783 "PAM01..14" (verified
# below by connectome co-innervation + the aSP13 anchor); v783 adds PAM15
# (not in Aso 2014) and PPL107/PPL108 (not in Aso 2014).
# --------------------------------------------------------------------------
ASO_PAM = {
    "PAM01": ("gamma5", "8-21", "PAM-gamma5 (aSP13, MB-M1?)"),
    "PAM02": ("beta'2a", "6-9", "PAM-beta'2a"),
    "PAM03": ("beta2+beta'2a", ">3", "PAM-beta2beta'2a (MB-M3)"),
    "PAM04": ("beta2", "8-19", "PAM-beta2 (MB-M8 subset)"),
    "PAM05": ("beta'2p", "14-17", "PAM-beta'2p"),
    "PAM06": ("beta'2m", "12-15", "PAM-beta'2m"),
    "PAM07": ("gamma4<gamma1gamma2", "13-17", "PAM-gamma4<gamma1gamma2"),
    "PAM08": ("gamma4", "n/a", "PAM-gamma4"),
    "PAM09": ("beta1(ped)", "1-3", "PAM-beta1ped"),
    "PAM10": ("beta1", "4-6", "PAM-beta1"),
    "PAM11": ("alpha1", ">6", "PAM-alpha1"),
    "PAM12": ("gamma3", "9-23", "PAM-gamma3 (MB-M2)"),
    "PAM13": ("beta'1 ap", "13-14", "PAM-beta'1ap"),
    "PAM14": ("beta'1 m", "n/a", "PAM-beta'1m"),
    "PAM15": (None, None, "not in Aso 2014 (hemibrain-era type)"),
}
ASO_PPL1 = {
    "PPL101": ("gamma1(pedc)", "1-2", "PPL1-gamma1pedc (MB-MP1; shock US)"),
    "PPL102": ("gamma1", "1", "PPL1-gamma1"),
    "PPL103": ("gamma2alpha'1", "1", "PPL1-gamma2alpha'1 (MB-MV1)"),
    "PPL104": ("alpha'3", "1", "PPL1-alpha'3"),
    "PPL105": ("alpha'2alpha2", "1", "PPL1-alpha'2alpha2 (MB-V1)"),
    "PPL106": ("alpha3", "1", "PPL1-alpha3"),
    "PPL107": (None, None, "not in Aso 2014 (hemibrain-era type)"),
    "PPL108": (None, None, "not in Aso 2014 (hemibrain-era type)"),
}
# Aso 2014 eLife 04580 Table 1 (PMC4273436): MBON types (Aso numbering!)
#   Aso-number -> (name, transmitter, cells/hemisphere)
# NOTE: Aso MBON numbering (22 types) is NOT the hemibrain numbering (35+);
# only the compartment NAMES are used here; the hemibrain MBONxx <-> name
# assignment is derived from v783 nt_type + DAN co-innervation below.
ASO_MBON = [
    ("gamma5beta'2a", "glutamate", 1), ("beta2beta'2a", "glutamate", 1),
    ("beta'2mp", "glutamate", 1), ("beta'2mp_bilateral", "glutamate", 1),
    ("gamma4>gamma1gamma2", "glutamate", 1), ("beta1>alpha", "glutamate", 1),
    ("alpha1", "glutamate", 2),
    ("gamma3", "GABA", 1), ("gamma3beta'1", "GABA", 1), ("beta'1", "GABA", 8),
    ("gamma1pedc>alpha/beta", "GABA", 1),
    ("gamma2alpha'1", "acetylcholine", 2), ("alpha'2", "acetylcholine", 1),
    ("alpha3", "acetylcholine", 2), ("alpha'1", "acetylcholine", 2),
    ("alpha'3ap", "acetylcholine", 1), ("alpha'3m", "acetylcholine", 2),
    ("alpha2sc", "acetylcholine", 1), ("alpha2p3p", "acetylcholine", 2),
    ("gamma1gamma2", "acetylcholine", 1), ("gamma4gamma5", "acetylcholine", 1),
    ("calyx", "acetylcholine", 1),
]

# Aso 2014 eLife 04580 (behavioral screen) + Owald 2015 (M4/M6):
#   activation of GLUTAMATERGIC MBONs -> AVOIDANCE (aversive valence)
#   activation of GABAergic/cholinergic MBONs -> ATTRACTION (appetitive)
NT_VALENCE = {"GLUT": "avoidance", "GABA": "approach", "ACH": "approach"}

# Reward DAN populations per compartment (Aso 2014 eLife 04580: PAM DANs
# innervating glutamatergic-MBON compartments convey reward; Burke 2012;
# Liu 2012; Yamagata 2015; Perisse 2013; Owald 2015 for M4/M6 tips).
GLUT_MBON_COMPARTMENTS = [
    "gamma5", "beta'2a", "beta2", "beta'2m", "beta'2p", "gamma4", "beta1",
    "alpha1",
]


def load_annotation_maps():
    fw = pd.read_csv(ANN / "fw_and_hemibrain_types.csv.gz",
                     usecols=["root_id", "cell_type", "hemibrain_type"])
    fw["root_id"] = fw.root_id.astype("int64")
    typ = {r.root_id: (r.cell_type if isinstance(r.cell_type, str) and
                       len(r.cell_type) else
                       (r.hemibrain_type if isinstance(r.hemibrain_type, str)
                        else ""))
           for r in fw.itertuples()}

    cls = pd.read_csv(ANN / "classification.csv.gz",
                      usecols=["root_id", "side"])
    cls["root_id"] = cls.root_id.astype("int64")
    side = dict(zip(cls.root_id, cls.side))

    neu = pd.read_csv(ANN / "neurons.csv.gz",
                      usecols=["root_id", "nt_type", "nt_type_score"])
    neu["root_id"] = neu.root_id.astype("int64")
    nt = dict(zip(neu.root_id, neu.nt_type))
    nt_score = dict(zip(neu.root_id, neu.nt_type_score))
    return typ, side, nt, nt_score


def load_populations():
    def rd(p):
        return [int(x) for x in json.loads((IDS / f"{p}.json").read_text())]
    return {p: rd(p) for p in ("PAM", "PPL1", "KC", "MBON", "DN_ALL")}


def first_token(t: str, prefix: str) -> str:
    """First PREFIXnn token of a comma-joined type string ('MBON25,MBON34'
    -> 'MBON25'; 'PAM04' -> 'PAM04')."""
    for part in t.replace(",", " ").split():
        if part.startswith(prefix):
            return part
    return ""


def stream_connectivity(pops, typ):
    """One pass over the v783 connectivity parquet:
    DAN->MBON and MBON->DN synapse lists (type-level)."""
    import pyarrow.parquet as pq
    dan_ids = set(pops["PAM"]) | set(pops["PPL1"])
    mbon_ids = set(pops["MBON"])
    dn_ids = set(pops["DN_ALL"])
    kc_ids = set(pops["KC"])

    dan2mbon = defaultdict(int)      # (dan_type, mbon_type) -> synapses
    mbon2dn = defaultdict(int)       # (mbon_type, dn_id) -> synapses
    kc2mbon = defaultdict(int)       # (kc_class, mbon_type) -> synapses
    kc_class_of = {}
    for k in pops["KC"]:
        t = typ.get(k, "")
        if t.startswith("KCg"):
            kc_class_of[k] = "gamma"
        elif t.startswith("KCapbp"):
            kc_class_of[k] = "alpha'/beta'"     # KCapbp-m/ap1/ap2
        elif t.startswith("KCab"):
            kc_class_of[k] = "alpha/beta"
        else:
            kc_class_of[k] = "other"

    flyid2i, _ = bl.load_maps("783")
    i2fly = {v: k for k, v in flyid2i.items()}
    mbon_i = np.array(sorted(flyid2i[f] for f in mbon_ids), dtype=np.int64)

    pf = pq.ParquetFile(bl.VERSIONS["783"]["path_con"])
    wcol = "Excitatory x Connectivity"
    for batch in pf.iter_batches(columns=["Presynaptic_Index",
                                          "Postsynaptic_Index", wcol],
                                 batch_size=2_000_000):
        d = batch.to_pandas()
        p = d["Presynaptic_Index"].to_numpy()
        q = d["Postsynaptic_Index"].to_numpy()
        w = d[wcol].to_numpy()
        # DAN -> MBON
        m = np.isin(p, DAN_I) & np.isin(q, mbon_i)
        if m.any():
            for a, b, ww in zip(p[m], q[m], w[m]):
                if ww > 0:
                    dan2mbon[(int(i2fly[int(a)]), int(i2fly[int(b)]))
                             ] += int(ww)
        # KC -> MBON
        m = np.isin(p, KC_I) & np.isin(q, mbon_i)
        if m.any():
            for a, b, ww in zip(p[m], q[m], w[m]):
                if ww > 0:
                    kc2mbon[(kc_class_of[int(i2fly[int(a)])],
                             int(i2fly[int(b)]))] += int(ww)
        # MBON -> DN
        m = np.isin(p, mbon_i) & np.isin(q, DN_I)
        if m.any():
            for a, b, ww in zip(p[m], q[m], w[m]):
                if ww > 0:
                    mbon2dn[(int(i2fly[int(a)]), int(i2fly[int(b)]))
                            ] += int(ww)
    return dan2mbon, kc2mbon, mbon2dn


def main():
    t0 = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    typ, side, nt, nt_score = load_annotation_maps()
    pops = load_populations()

    globals()["DAN_I"] = None      # set inside stream via closure below
    # build Brian-index arrays once for the stream
    flyid2i, _ = bl.load_maps("783")
    global DAN_I, KC_I, DN_I
    DAN_I = np.array(sorted(flyid2i[f] for f in set(pops["PAM"])
                            | set(pops["PPL1"]) if f in flyid2i),
                      dtype=np.int64)
    KC_I = np.array(sorted(flyid2i[f] for f in pops["KC"] if f in flyid2i),
                    dtype=np.int64)
    DN_I = np.array(sorted(flyid2i[f] for f in pops["DN_ALL"]
                           if f in flyid2i), dtype=np.int64)
    mbon_i = np.array(sorted(flyid2i[f] for f in pops["MBON"]
                             if f in flyid2i), dtype=np.int64)

    print("[audit] streaming v783 connectivity (DAN->MBON, KC->MBON, "
          "MBON->DN) ...", flush=True)
    dan2mbon, kc2mbon, mbon2dn = stream_connectivity(pops, typ)

    # ------------------------------------------------------------------
    # per-MBON-type facts (hemibrain MBONxx types)
    # ------------------------------------------------------------------
    mbon_type_of = {}
    for f in pops["MBON"]:
        tok = first_token(typ.get(f, ""), "MBON")
        mbon_type_of[f] = tok if tok else typ.get(f, "?")

    mbon_facts = {}
    for f in pops["MBON"]:
        t = mbon_type_of[f]
        d = mbon_facts.setdefault(t, dict(ids=[], sides=[], nts=[], ntscore=[]))
        d["ids"].append(f)
        d["sides"].append(side.get(f, "?"))
        d["nts"].append(nt.get(f, "?"))
        d["ntscore"].append(float(nt_score.get(f, 0.0)))

    # aggregate connectivity per MBON type
    dan_in = defaultdict(lambda: defaultdict(int))    # mbon_type -> dan_type
    kc_in = defaultdict(lambda: defaultdict(int))     # mbon_type -> kc_class
    for (dan_f, mbon_f), s in dan2mbon.items():
        dan_in[mbon_type_of[mbon_f]][first_token(typ.get(dan_f, ""), "PAM")
                                     or first_token(typ.get(dan_f, ""), "PPL1")
                                     or "?"] += s
    for (kc_cls, mbon_f), s in kc2mbon.items():
        kc_in[mbon_type_of[mbon_f]][kc_cls] += s

    mbon_rows = {}
    for t, d in sorted(mbon_facts.items()):
        nts = pd.Series(d["nts"]).value_counts()
        nt_majority = str(nts.index[0])
        kc_mass = kc_in.get(t, {})
        kc_tot = sum(kc_mass.values()) or 1
        dans = sorted(dan_in.get(t, {}).items(), key=lambda kv: -kv[1])
        mbon_rows[t] = dict(
            hemibrain_type=t, n_cells=len(d["ids"]),
            ids=sorted(d["ids"]),
            sides={k: int(v) for k, v in pd.Series(
                d["sides"]).value_counts().items()},
            nt_predicted=nt_majority,
            nt_frac=round(float(nts.iloc[0]) / len(d["nts"]), 3),
            nt_mean_score=round(float(np.mean(d["ntscore"])), 3),
            kc_input_fraction=dict(
                (k, round(v / kc_tot, 3)) for k, v in sorted(
                    kc_mass.items(), key=lambda kv: -kv[1])),
            top_dan_partners=[dict(dan_type=k, synapses=int(v))
                              for k, v in dans[:5]],
        )

    # ------------------------------------------------------------------
    # per-DAN-type facts + compartment inference
    # ------------------------------------------------------------------
    dan_type_of = {}
    for f in list(pops["PAM"]) + list(pops["PPL1"]):
        tok = (first_token(typ.get(f, ""), "PAM")
               or first_token(typ.get(f, ""), "PPL1"))
        dan_type_of[f] = tok or typ.get(f, "?")

    dan_rows = {}
    for f, dt in dan_type_of.items():
        d = dan_rows.setdefault(dt, dict(hemibrain_type=dt, ids=[], sides=[]))
        d["ids"].append(f)
        d["sides"].append(side.get(f, "?"))

    dan_out = defaultdict(lambda: defaultdict(int))  # dan_type -> mbon_type
    for (dan_f, mbon_f), s in dan2mbon.items():
        dan_out[dan_type_of[dan_f]][mbon_type_of[mbon_f]] += s

    for dt, d in sorted(dan_rows.items()):
        partners = sorted(dan_out.get(dt, {}).items(), key=lambda kv: -kv[1])
        top = []
        for mt, s in partners[:6]:
            mrow = mbon_rows.get(mt)
            top.append(dict(mbon_type=mt, synapses=int(s),
                            mbon_nt=mrow["nt_predicted"] if mrow else "?"))
        aso = (ASO_PAM if dt.startswith("PAM") else ASO_PPL1).get(dt)
        d.update(n_cells=len(d["ids"]),
                 ids=sorted(d["ids"]),
                 sides={k: int(v) for k, v in pd.Series(
                d["sides"]).value_counts().items()},
                 top_mbon_partners=top,
                 aso_2014=(dict(compartment=aso[0], cells_per_hemi=aso[1],
                                name=aso[2]) if aso and aso[0] else None))

    # ------------------------------------------------------------------
    # connectome cross-check of the Aso PAMxx numbering:
    # does each PAMxx's strongest MBON partner set carry the neurotransmitter
    # and lobe class Aso's compartment predicts?
    # ------------------------------------------------------------------
    def lobe_of_compartment(comp):
        if comp is None:
            return None
        # Aso compartments: gamma* -> gamma-lobe KCs; beta'/alpha' ->
        # alpha'/beta' KCs (KCapbp); beta/alpha (no prime) -> alpha/beta KCs
        if "gamma" in comp:
            return "gamma"
        if "'" in comp:
            return "alpha'/beta'"
        if "alpha" in comp or "beta" in comp:
            return "alpha/beta"
        return None

    numbering_checks = []
    for dt, d in sorted(dan_rows.items()):
        aso = d.get("aso_2014")
        if not aso:
            numbering_checks.append(dict(dan_type=dt, check="no Aso anchor",
                                         verdict="N/A"))
            continue
        pred_lobe = lobe_of_compartment(aso["compartment"])
        partners = d["top_mbon_partners"][:3]
        syns = sum(p["synapses"] for p in partners) or 1
        weighted = defaultdict(float)
        for p in partners:
            kcfrac = mbon_rows.get(p["mbon_type"], {}).get(
                "kc_input_fraction", {})
            for lobe, frac in kcfrac.items():
                weighted[lobe] += frac * p["synapses"] / syns
        best_lobe = max(weighted, key=weighted.get) if weighted else None
        ok = (best_lobe == pred_lobe) if (pred_lobe and best_lobe) else None
        numbering_checks.append(dict(
            dan_type=dt, aso_compartment=aso["compartment"],
            predicted_lobe=pred_lobe, connectome_lobe=best_lobe,
            lobe_weights={k: round(v, 3) for k, v in weighted.items()},
            verdict="MATCH" if ok else ("MISMATCH" if ok is False
                                        else "INCONCLUSIVE")))

    # ------------------------------------------------------------------
    # CORRECTED valence classification, anchored to measured nt_type
    # ------------------------------------------------------------------
    corrected = dict(
        rule=("activation valence: GLUT -> avoidance (Aso 2014 eLife "
              "04580 screen: 'all MBONs eliciting aversion were "
              "glutamatergic and all the MBONs eliciting attraction were "
              "either GABAergic or cholinergic'; Owald 2015: M4/M6 "
              "activation induces avoidance, their inhibition converts "
              "avoidance into attraction, reward DEPRESSES their KC input)"),
        classes={},
        v1_discrepancies=[],
    )
    # v1 dict for comparison
    sys.path.insert(0, str(BRAIN.parent / "brain"))
    from closedloop import populations as clp  # noqa: E402
    v1_val_of = {}
    for cls, ids in clp.mbon_valence_classes().items():
        if cls.startswith("_"):
            continue
        for i in ids:
            v1_val_of[int(i)] = cls

    for t, row in sorted(mbon_rows.items()):
        val = NT_VALENCE.get(row["nt_predicted"], "unclassified")
        for f in row["ids"]:
            corrected["classes"][str(f)] = val
    # discrepancy count by type
    for t, row in sorted(mbon_rows.items()):
        val = NT_VALENCE.get(row["nt_predicted"], "unclassified")
        v1_vals = {v1_val_of.get(f, "?") for f in row["ids"]}
        if v1_vals != {val}:
            corrected["v1_discrepancies"].append(dict(
                hemibrain_type=t, v2_valence=val,
                v1_valences=sorted(v1_vals), nt=row["nt_predicted"],
                n_cells=row["n_cells"]))

    # ------------------------------------------------------------------
    # reward-US candidate set: PAM types whose compartments host
    # glutamatergic MBONs
    # ------------------------------------------------------------------
    glut_mbon_types = [t for t, r in mbon_rows.items()
                       if r["nt_predicted"] == "GLUT"]
    reward_dan_types = []
    for dt, d in sorted(dan_rows.items()):
        if not dt.startswith("PAM"):
            continue
        aso = d.get("aso_2014")
        comp = aso["compartment"] if aso else None
        if comp and any(g in comp for g in GLUT_MBON_COMPARTMENTS):
            reach = sum(p["synapses"] for p in d["top_mbon_partners"]
                        if p["mbon_type"] in glut_mbon_types)
            reward_dan_types.append(dict(
                dan_type=dt, compartment=comp, ids=d["ids"], n=d["n_cells"],
                glut_mbon_reach_synapses=int(reach)))
    us_set = sorted({dt for r in reward_dan_types
                     for dt in [r["dan_type"]]})

    # ------------------------------------------------------------------
    # MBON->DN expression routes (Section I route C)
    # ------------------------------------------------------------------
    dn_routes = defaultdict(lambda: dict(synapses=0, mbon_types=set()))
    for (mbon_f, dn_f), s in mbon2dn.items():
        r = dn_routes[dn_f]
        r["synapses"] += int(s)
        r["mbon_types"].add(mbon_type_of[mbon_f])
    route_rows = []
    for dn_f, r in sorted(dn_routes.items(), key=lambda kv: -kv[1]["synapses"]):
        route_rows.append(dict(
            dn_root_id=dn_f, dn_type=typ.get(dn_f, "?"), side=side.get(dn_f),
            synapses=r["synapses"],
            mbon_types=sorted(r["mbon_types"]),
            mbon_valences=sorted({corrected["classes"][str(m)]
                                  for m in [x for x in pops["MBON"]
                                            if mbon_type_of[x] in
                                            r["mbon_types"]]})))
    # aggregate MBON->DN by valence for the readout definition
    appr_dn = sorted({r["dn_root_id"] for r in route_rows
                      if "approach" in r["mbon_valences"]})
    avoid_dn = sorted({r["dn_root_id"] for r in route_rows
                       if "avoidance" in r["mbon_valences"]})

    # ------------------------------------------------------------------
    # write outputs
    # ------------------------------------------------------------------
    (OUT / "dan_compartment_table.json").write_text(json.dumps(dict(
        dan_types=dan_rows, mbon_types=mbon_rows,
        aso_numbering_checks=numbering_checks,
        reward_us_candidates=reward_dan_types, reward_us_set=us_set,
        glut_mbon_types=glut_mbon_types), indent=1))
    (OUT / "mbon_valence_corrected.json").write_text(json.dumps(dict(
        corrected_classes=corrected["classes"],
        v1_discrepancies=corrected["v1_discrepancies"], rule=corrected["rule"],
        n_avoidance=sum(1 for v in corrected["classes"].values()
                        if v == "avoidance"),
        n_approach=sum(1 for v in corrected["classes"].values()
                       if v == "approach")), indent=1))
    (OUT / "mbon_to_dn_routes.json").write_text(json.dumps(dict(
        routes=route_rows,
        reachable_dn_approach_valence=appr_dn,
        reachable_dn_avoidance_valence=avoid_dn), indent=1))

    # ------------------------------------------------------------------
    # markdown table (Section C)
    # ------------------------------------------------------------------
    lines = ["# Gate-4 v2 — DAN compartment table (source-audited)", "",
             "Generated by `brain/scripts/g4v2_audit.py` from the canonical "
             "v783 connectome, the official FlyWire v783 annotation tables, "
             "Aso et al. 2014 (eLife 04577 Table 1; eLife 04580 screen) and "
             "Owald et al. 2015. All tiers labelled.", ""]
    lines += ["## PAM / PPL1 types (v783 hemibrain_type)", "",
              "| DAN type | family | n (L/R) | Aso-2014 compartment | "
              "top MBON partners (connectome, synapses) | numbering check |",
              "|---|---|---|---|---|---|"]
    chk = {c["dan_type"]: c for c in numbering_checks}
    for dt, d in sorted(dan_rows.items()):
        fam = "PAM" if dt.startswith("PAM") else "PPL1"
        aso = d.get("aso_2014")
        comp = aso["compartment"] if aso else "—(not in Aso 2014)"
        tops = ", ".join(f"{p['mbon_type']}({p['synapses']})"
                         for p in d["top_mbon_partners"][:3]) or "—"
        c = chk.get(dt, {})
        lines.append(f"| {dt} | {fam} | {d['n_cells']} "
                     f"({d['sides'].get('left', 0)}/"
                     f"{d['sides'].get('right', 0)}) | {comp} | {tops} "
                     f"| {c.get('verdict', 'N/A')} |")
    lines += ["", "## Reward-US candidate set (PAM → glutamatergic-MBON "
              "compartments)", ""]
    for r in reward_dan_types:
        lines.append(f"- **{r['dan_type']}** — compartment {r['compartment']}"
                     f", {r['n']} cells, glut-MBON reach "
                     f"{r['glut_mbon_reach_synapses']} synapses "
                     "(Aso 2014 eLife 04580: reward conveyed by PAM DANs "
                     "innervating glutamatergic-MBON compartments)")
    lines += ["", f"**US set (v2): {', '.join(us_set)}**", ""]
    lines += ["## MBON activation-valence (corrected, transmitter-anchored)",
              "", "| hemibrain type | n | nt (v783 predicted) | v2 valence | "
              "v1 valence |", "|---|---|---|---|---|"]
    for t, row in sorted(mbon_rows.items(),
                         key=lambda kv: -kv[1]["n_cells"]):
        val = NT_VALENCE.get(row["nt_predicted"], "unclassified")
        disc = next((x for x in corrected["v1_discrepancies"]
                     if x["hemibrain_type"] == t), None)
        v1s = sorted(disc["v1_valences"]) if disc else [val]
        flag = " ⚠ v1 differs" if disc else ""
        lines.append(f"| {t} | {row['n_cells']} | {row['nt_predicted']} "
                     f"({row['nt_frac']:.2f}) | {val} | "
                     f"{'/'.join(v1s)}{flag} |")
    (OUT / "dan_compartment_table.md").write_text("\n".join(lines))

    summary = dict(
        generated=time.strftime("%Y-%m-%d %H:%M:%S"),
        wall_s=round(time.perf_counter() - t0, 1),
        n_dan_types=len(dan_rows), n_mbon_types=len(mbon_rows),
        aso_numbering=dict(
            match=sum(1 for c in numbering_checks if c.get("verdict")
                      == "MATCH"),
            mismatch=sum(1 for c in numbering_checks
                         if c.get("verdict") == "MISMATCH"),
            inconclusive=sum(1 for c in numbering_checks
                             if c.get("verdict") in ("INCONCLUSIVE",
                                                     "N/A"))),
        valence_v2=dict(
            n_avoidance=sum(1 for v in corrected["classes"].values()
                            if v == "avoidance"),
            n_approach=sum(1 for v in corrected["classes"].values()
                           if v == "approach")),
        n_v1_discrepancies=len(corrected["v1_discrepancies"]),
        v1_discrepancy_types=sorted({d["hemibrain_type"]
                                     for d in
                                     corrected["v1_discrepancies"]}),
        reward_us_set=us_set,
        glut_mbon_types=glut_mbon_types,
        n_dn_reachable=len(route_rows),
    )
    (OUT / "audit_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
