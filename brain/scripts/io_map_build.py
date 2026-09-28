"""Build the D4-D6 neural I/O map (machine-readable) from the pinned v783 stack.

Outputs (brain/data/io_map/):
  neural_io_map.json        - full map incl. per-population stats + sample IDs
  ids/<pop_id>.json         - complete v783 root-ID lists per population
  sensory_populations.csv   - D4 table
  learning_populations.csv  - D5 table
  descending_motor_populations.csv - D6 table
  input_channels.csv        - proposed sensory->brain channels
  output_actions.csv        - proposed brain->motor actions
  pathway_checks.json       - direct synaptic paths between key populations

Everything is derived from the pinned, hash-manifested inputs; no manual edits.
Provenance labels: [DATA] FlyWire/Codex v783 tables, [LIT] published work,
[ENGINEERED] our interface choices.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                                   # .../brain
ANN = ROOT / "data" / "flywire_annotations_v783"
OUT = ROOT / "data" / "io_map"
IDS_DIR = OUT / "ids"
MODEL_CON = ROOT / "third_party" / "Drosophila_brain_model" / "Connectivity_783.parquet"

sys.path.insert(0, str(HERE))
from io_map_populations import POPULATIONS, INPUT_CHANNELS, OUTPUT_ACTIONS  # noqa: E402


def load_annotations():
    cls = pd.read_csv(ANN / "classification.csv.gz")
    tp = pd.read_csv(ANN / "consolidated_cell_types.csv.gz")
    nr = pd.read_csv(ANN / "neurons.csv.gz")
    m = cls.merge(tp, on="root_id", how="left")
    m = m.merge(nr[["root_id", "nt_type", "nt_type_score", "group"]], on="root_id", how="left")
    return m


def load_connectome():
    con = pd.read_parquet(MODEL_CON,
                          columns=["Presynaptic_ID", "Postsynaptic_ID",
                                   "Connectivity", "Excitatory"])
    return con


def pop_stats(sel_ids, con, m):
    """Connectivity + annotation stats for one population (set of root IDs)."""
    ids = set(sel_ids)
    sub = m[m.root_id.isin(ids)]
    out_con = con[con.Presynaptic_ID.isin(ids)]
    in_con = con[con.Postsynaptic_ID.isin(ids)]
    post = out_con.groupby("Postsynaptic_ID")["Connectivity"].sum()
    pre = in_con.groupby("Presynaptic_ID")["Connectivity"].sum()

    # top downstream partners by summed synapses
    post_ann = m.set_index("root_id").reindex(post.index)
    top_post = (post.groupby(post_ann["super_class"].fillna("?").values)
                .sum().sort_values(ascending=False))
    top_post_types = (post.groupby(post_ann["primary_type"].fillna("?").values)
                      .sum().sort_values(ascending=False).head(8))

    stats = dict(
        n=len(sub),
        side_counts={str(k): int(v) for k, v in sub.side.value_counts().items()},
        nt_counts={str(k): int(v) for k, v in sub.nt_type.value_counts().items()},
        super_class_counts={str(k): int(v) for k, v in sub.super_class.value_counts().items()},
        n_out_connections=int(len(out_con)),
        n_out_synapses=int(out_con.Connectivity.sum()),
        n_unique_post_partners=int(post.shape[0]),
        n_in_connections=int(len(in_con)),
        n_in_synapses=int(in_con.Connectivity.sum()),
        n_unique_pre_partners=int(pre.shape[0]),
        frac_excitatory_out=float(out_con.Excitatory.mean()) if len(out_con) else None,
        downstream_by_superclass={str(k): int(v) for k, v in top_post.items()},
        downstream_top_types={str(k): int(v) for k, v in top_post_types.items()},
    )
    return stats


def check_pathways(con, m, pairs):
    """Direct (1-hop) synaptic strength between named ID sets."""
    res = {}
    for src_ids, dst_ids, name in pairs:
        s, d = set(src_ids), set(dst_ids)
        sel = con[con.Presynaptic_ID.isin(s) & con.Postsynaptic_ID.isin(d)]
        res[name] = dict(
            n_connections=int(len(sel)),
            total_synapses=int(sel.Connectivity.sum()),
            frac_excitatory=float(sel.Excitatory.mean()) if len(sel) else None,
            n_unique_src=int(sel.Presynaptic_ID.nunique()),
            n_unique_dst=int(sel.Postsynaptic_ID.nunique()),
        )
    return res


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    IDS_DIR.mkdir(parents=True, exist_ok=True)
    print("loading annotations ...")
    m = load_annotations()
    print("loading connectome ...")
    con = load_connectome()

    pops_out = []
    for p in POPULATIONS:
        sel = p.get("sel", "m.root_id.isin([])")
        if "__IDS__" in sel:
            sel = sel.replace("__IDS__", str([i for i in p.get("label_ids", []) if i is not None]))
        mask = eval(sel, {"m": m, "np": np, "pd": pd})
        if isinstance(mask, pd.Series):
            sel_ids = sorted(m.root_id[mask.eq(True)].tolist())
        else:
            sel_ids = sorted(set(int(i) for i in mask))
        stats = pop_stats(sel_ids, con, m) if sel_ids else {}

        entry = dict(
            pop_id=p["pop_id"], group=p["group"], name=p["name"],
            region=p["region"], biological_function=p["function"],
            references=p["refs"],
            provenance=["[DATA] FlyWire/Codex v783 tables"] + (
                ["[LIT] " + ", ".join(p["refs"])] if p.get("refs") else []),
            present_in_v783=len(sel_ids) > 0,
            v783=stats,
            proposed_mapping=p.get("game_channel") or p.get("game_action", ""),
            confidence=p["confidence"],
            v783_root_ids=sel_ids if len(sel_ids) <= 400 else sel_ids[:400],
            n_root_ids_total=len(sel_ids),
            root_ids_file=f"ids/{p['pop_id']}.json" if sel_ids else None,
            label_ids=p.get("label_ids", []),
        )
        if sel_ids:
            (IDS_DIR / f"{p['pop_id']}.json").write_text(json.dumps(sel_ids))
        pops_out.append(entry)
        print(f"  {p['pop_id']:16s} n={stats.get('n', 0):6d} out_syn={stats.get('n_out_synapses', 0):8d}")

    # key pathway checks (direct synapses between populations)
    by_id = {p["pop_id"]: p for p in POPULATIONS}
    id_of = {}
    for p in POPULATIONS:
        sel = p.get("sel", "m.root_id.isin([])")
        if "__IDS__" in sel:
            sel = sel.replace("__IDS__", str([i for i in p.get("label_ids", []) if i]))
        mask = eval(sel, {"m": m, "np": np, "pd": pd})
        if isinstance(mask, pd.Series):
            id_of[p["pop_id"]] = set(m.root_id[mask.eq(True)].tolist())
        else:
            id_of[p["pop_id"]] = set(int(i) for i in mask)

    pairs = [
        (id_of["LPLC2"] | id_of["LC4"], id_of["DNp01_GF"], "loom(LPLC2+LC4) -> giant fiber (DNp01)"),
        (id_of["LC16"], id_of["MDN"], "LC16 -> MDN (object -> backward)"),
        (id_of["LC9"], id_of["DNp09_P9"], "LC9 -> P9/DNp09 (visual -> directed walk)"),
        (id_of["BPN"] , id_of["MDN"], "BPN -> MDN (forward vs backward crosstalk)"),
        (id_of["RRN"], id_of["DN_LEG_CLUSTER"], "RRN -> leg-DN cluster"),
        (id_of["RRN"], id_of["MDN"], "RRN -> MDN"),
        (id_of["FG_CB0890"], id_of["DNp09_P9"] | id_of["BPN"] | id_of["RRN"], "FG(halt fwd) -> forward command neurons"),
        (id_of["BB_DNg60"], id_of["DNp09_P9"], "BB(halt turn) -> DNp09"),
        (id_of["KC"], id_of["MBON"], "Kenyon cells -> MBONs (learning synapse site)"),
        (id_of["PAM"] | id_of["PPL1"], id_of["MBON"] | id_of["KC"], "DANs (PAM+PPL1) -> MBON+KC (modulation site)"),
        (id_of["LMC_L2"] | id_of["LMC_L1"], id_of["TM_OFF"] | id_of["MI_ON"], "L1/L2 -> medulla relays (visual relay check)"),
        (id_of["T4"] | id_of["T5"], id_of["LPTC_HS"] | id_of["LPTC_VS"], "T4/T5 -> HS/VS (motion integration)"),
    ]
    pathways = check_pathways(con, m, pairs)

    result = dict(
        meta=dict(
            title="D4-D6 neural I/O map of the digital Drosophila brain",
            connectome="FlyWire v783 (pinned in third_party/Drosophila_brain_model)",
            model="Shiu et al. leaky-integrate-and-fire whole-brain model (MIT, pinned commit 91bdd1e)",
            annotations="FlyWire/Codex public annotation tables for v783 (classification, consolidated_cell_types, neurons, labels, column_assignment); see brain/data/flywire_annotations_v783/",
            terminology=[
                "[DATA] biological connectome data (FlyWire v783)",
                "[LIT] published biological evidence",
                "[MODEL] modelled dynamics (Shiu LIF parameters)",
                "[ENGINEERED] our interface/learning additions",
            ],
            disclaimer="This map never claims biological equivalence; it documents which v783 populations we will stimulate/read and the evidence behind each choice.",
            generated=str(Path(__file__).name),
        ),
        sensory_populations=[p for p in pops_out if p["group"] == "sensory"],
        learning_reward_populations=[p for p in pops_out if p["group"] == "learning"],
        descending_motor_populations=[p for p in pops_out if p["group"] == "motor"],
        pathway_checks=pathways,
        proposed_input_channels=INPUT_CHANNELS,
        proposed_output_actions=OUTPUT_ACTIONS,
    )
    (OUT / "neural_io_map.json").write_text(json.dumps(result, indent=1))

    # CSVs
    def flat(p):
        d = dict(pop_id=p["pop_id"], name=p["name"], region=p["region"],
                 function=p["biological_function"], references="; ".join(p["references"]),
                 present_in_v783=p["present_in_v783"], n=p["v783"].get("n"),
                 left=p["v783"].get("side_counts", {}).get("left"),
                 right=p["v783"].get("side_counts", {}).get("right"),
                 nt_distribution=json.dumps(p["v783"].get("nt_counts", {})),
                 out_synapses=p["v783"].get("n_out_synapses"),
                 unique_post_partners=p["v783"].get("n_unique_post_partners"),
                 in_synapses=p["v783"].get("n_in_synapses"),
                 frac_excitatory_out=p["v783"].get("frac_excitatory_out"),
                 proposed_mapping=p["proposed_mapping"],
                 confidence=p["confidence"],
                 root_ids_file=p["root_ids_file"], n_root_ids_total=p["n_root_ids_total"])
        return d

    for fname, group in [
        ("sensory_populations.csv", "sensory"),
        ("learning_populations.csv", "learning"),
        ("descending_motor_populations.csv", "motor"),
    ]:
        rows = [flat(p) for p in pops_out if p["group"] == group]
        pd.DataFrame(rows).to_csv(OUT / fname, index=False)
    pd.DataFrame(INPUT_CHANNELS).to_csv(OUT / "input_channels.csv", index=False)
    pd.DataFrame(OUTPUT_ACTIONS).to_csv(OUT / "output_actions.csv", index=False)

    print("\npathway checks:")
    for k, v in pathways.items():
        print(f"  {k:55s} conns={v['n_connections']:6d} syn={v['total_synapses']:7d}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
