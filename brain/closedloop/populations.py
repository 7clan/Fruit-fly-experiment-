"""Population loading for the closed-loop brain.

Everything here is TIER: BIOLOGICAL (IDs verified in D4-D6, stored under
brain/data/io_map/) plus hemisphere splits from the vendored official
FlyWire v783 annotation tables (classification.csv.gz 'side').

MBON valence classes follow Aso et al. 2014 (eLife) / Owald & Waddell 2015:
activation of 'approach' MBON types drives approach behaviour, activation
of 'avoidance' types drives avoidance; appetitive learning (PAM) depresses
KC->avoidance-MBON, aversive learning (PPL1) depresses KC->approach-MBON.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

BRAIN = Path(__file__).resolve().parent.parent
IO = BRAIN / "data" / "io_map"
ANN = BRAIN / "data" / "flywire_annotations_v783"

# Aso et al. 2014 numbering (MBON01..MBON35). Valence = behavioural effect
# of optogenetic MBON activation, anchored to published results:
#   approach  : alpha'3 family MBON10/11/25/26/34 (+MBON04 alpha'3ap) -
#               Aso 2014 Fig 7, Placais 2013; alpha2sc MBON02/23 - Aso 2014;
#               M4/M6 = gamma5beta'2a MBON21, beta2beta'2a MBON17,
#               beta'2mp MBON05/14 - Owald et al. 2015 (drive conditioned
#               reward expression); MBON17-like follows MBON17.
#   avoidance : gamma2alpha'1 MBON07 - Berry 2018 (aversive memory readout);
#               gamma1pedc>alpha/beta MBON06/19 - Hige 2015 / Aso 2014
#               (PPL1-recipient, aversive); gamma4>gamma1gamma2 MBON09/24,
#               gamma3 family MBON08/29/30, beta'2a MBON13, beta'2p MBON18 -
#               Aso 2014 screen (avoidance-biased).
#   unclassified: types with ambiguous/no published activation valence
#               (MBON01, 03, 12, 15, 16, 20, 22, 27, 28, 31, 32, 33, 35,
#               15-like). EXCLUDED from plasticity targets (conservative).
# Sign rule (Davis 2023 review, after Cohn 2015 / Hige 2015 / Owald 2015):
#   aversive (PPL1)  -> decreases activity of approach-class MBONs
#   appetitive (PAM) -> decreases activity of avoidance-class MBONs
MBON_VALENCE = {
    "MBON01": "unclassified",  # alpha'1 bilateral
    "MBON02": "approach",      # alpha2sc
    "MBON03": "unclassified",  # alpha'2
    "MBON04": "approach",      # alpha'3ap (bilateral; alpha'3 family)
    "MBON05": "approach",      # beta'2mp bilateral (M4/6 class, Owald 2015)
    "MBON06": "avoidance",     # gamma1pedc>alpha/beta
    "MBON07": "avoidance",     # gamma2alpha'1 (Berry 2018)
    "MBON08": "avoidance",     # gamma3beta'1
    "MBON09": "avoidance",     # gamma4>gamma1gamma2
    "MBON10": "approach",      # alpha'3ap
    "MBON11": "approach",      # alpha'3m
    "MBON12": "unclassified",  # alpha2p3p
    "MBON13": "avoidance",     # beta'2a
    "MBON14": "approach",      # beta'2mp (unilateral; M4/6 class)
    "MBON15": "unclassified",  # beta'2m
    "MBON16": "unclassified",  # beta1>alpha
    "MBON17": "approach",      # beta2beta'2a (M6, Owald 2015)
    "MBON18": "avoidance",     # beta'2p
    "MBON19": "avoidance",     # gamma1pedc>alpha/beta (2nd)
    "MBON20": "unclassified",  # beta'1
    "MBON21": "approach",      # gamma5beta'2a (M4, Owald 2015)
    "MBON22": "unclassified",  # alpha'2 unilateral
    "MBON23": "approach",      # alpha2sc unilateral
    "MBON24": "avoidance",     # gamma4>gamma1gamma2 unilateral
    "MBON25": "approach",      # alpha'3 unilateral
    "MBON26": "approach",      # alpha'3m unilateral
    "MBON27": "unclassified",
    "MBON28": "unclassified",
    "MBON29": "avoidance",     # gamma3>gamma3d
    "MBON30": "avoidance",     # gamma3
    "MBON31": "unclassified",  # gamma4
    "MBON32": "unclassified",  # gamma1
    "MBON33": "unclassified",  # gamma2>gamma2
    "MBON34": "approach",      # alpha'3ap (2nd)
    "MBON35": "unclassified",  # gamma1pedc
}
MBON_VALENCE["MBON17-like"] = "approach"   # follows MBON17
MBON_VALENCE["MBON15-like"] = "unclassified"

_CACHE: dict = {}


def ids_of(pop: str) -> list[int]:
    return json.loads((IO / "ids" / f"{pop}.json").read_text())


def _side_map() -> dict[int, str]:
    if "_side" not in _CACHE:
        cls = pd.read_csv(ANN / "classification.csv.gz",
                          usecols=["root_id", "side"])
        _CACHE["_side"] = dict(zip(cls.root_id.astype("int64"), cls.side))
    return _CACHE["_side"]


def _type_map() -> dict[int, str]:
    if "_type" not in _CACHE:
        typ = pd.read_csv(ANN / "fw_and_hemibrain_types.csv.gz",
                          usecols=["root_id", "cell_type",
                                   "hemibrain_type"])
        # v783 facts (D11 verification): MBON identities live in
        # hemibrain_type (cell_type is empty for all 96 MBONs); most other
        # classes are labelled in cell_type.  Use cell_type when present,
        # else hemibrain_type.
        ct = typ.cell_type.fillna("")
        ht = typ.hemibrain_type.fillna("")
        combined = (ct.where(ct.str.len() > 0, ht)
                    if hasattr(ct, "where") else ct)
        _CACHE["_type"] = dict(
            zip(typ.root_id.astype("int64"), combined))
    return _CACHE["_type"]


def split_sides(pop: str) -> dict[str, list[int]]:
    sm = _side_map()
    out = {"left": [], "right": [], "unknown": []}
    for i in ids_of(pop):
        out[sm.get(int(i), "unknown")].append(int(i))
    return out


def mbon_valence_classes() -> dict[str, list[int]]:
    """MBON flywire IDs partitioned by modelled valence class."""
    tm = _type_map()
    sm = _side_map()
    out = {"approach": [], "avoidance": [], "unclassified": []}
    for i in ids_of("MBON"):
        t = tm.get(int(i), "")
        # take the first MBONxx token of the type string
        tok = ""
        for part in t.replace(",", " ").split():
            if part.startswith("MBON") and part[4:6].isdigit():
                tok = part[:6]
                break
        val = MBON_VALENCE.get(tok, "unclassified")
        out[val].append(int(i))
    out["_sides"] = {str(i): sm.get(int(i), "") for i in ids_of("MBON")}  # type: ignore[assignment]
    return out


def population_defs() -> dict:
    """All population -> (flywire ids, brian indices later) definitions
    used by the closed loop, with tier provenance."""
    return {
        # sensory entry points (D4-D6 verified)
        "LC9_L": ("LC9", "left", "visual target, left hemifield"),
        "LC9_R": ("LC9", "right", "visual target, right hemifield"),
        "LPLC2_L": ("LPLC2", "left", "looming, left"),
        "LPLC2_R": ("LPLC2", "right", "looming, right"),
        "LC4_L": ("LC4", "left", "looming/escape, left"),
        "LC4_R": ("LC4", "right", "looming/escape, right"),
        # reward entry (Shiu tutorial sugar set, verified in v783 D1-D3)
        "GRN_SUGAR": ("GRN_SUGAR", None, "sugar taste (engineered reward delivery)"),
        # reinforcement (D5 map; PAM/PPL1 also usable as direct entry)
        "PAM": ("PAM", None, "reward DANs"),
        "PPL1": ("PPL1", None, "aversive DANs"),
        # plasticity substrate
        "KC": ("KC", None, "Kenyon cells"),
        "MBON": ("MBON", None, "MB output neurons"),
        "APL": ("APL", None, "MB feedback GABA"),
        "DPM": ("DPM", None, "MB feedback neuromodulatory"),
        # motor readout (D6 map)
        "BPN": ("BPN", None, "forward"),
        "RRN": ("RRN", None, "forward (pair)"),
        "MDN": ("MDN", None, "backward"),
        "DNp09_L": ("DNp09_P9", "left", "forward + ipsiversive turn (left)"),
        "DNp09_R": ("DNp09_P9", "right", "forward + ipsiversive turn (right)"),
        "GF": ("DNp01_GF", None, "giant fiber escape"),
        "FG": ("FG_CB0890", None, "walk-OFF Foxglove"),
        "BB": ("BB_DNg60", None, "walk-OFF Bluebell"),
    }
