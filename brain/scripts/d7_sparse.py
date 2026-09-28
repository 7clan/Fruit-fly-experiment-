"""D7 sparse-interactive variant: an EXACT-by-construction task ROI for
RAM-constrained interactive use, verified against the full canonical brain.

Route (from the Gate-2 performance study, mode m3):
  subcircuit = union of active sets under every stimulation pattern the
               encoder can produce (left / right / bilateral / looming at
               full drive) + 1-hop downstream + all interface + readout
               populations.  Induced subgraph, same model, same params.
  Exactness contract: dropped neurons never spike under these workloads
  (provable for a fixed workload - Gate 2 m3 measured Jaccard 1.0). Here
  we VERIFY it empirically on (a) the D7 study schedule and (b) real D10
  episode schedules, comparing bit-identity with the full brain.

The FULL brain remains the scientific reference and the default INTERACTIVE
engine; this variant is an OPTIONAL low-RAM accelerator, never a silent
replacement (docs/ROADMAP.md D7).

Usage:
  python d7_sparse.py build          # probes + ROI + filtered files + study
  python d7_sparse.py verify --episode-dir <ep>
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import brainlib as bl  # noqa: E402
import d7_runtime as rt  # noqa: E402
import d8_encoder as d8  # noqa: E402

DATA = d8.DATA
OUT = bl.ROOT / "results" / "d7_runtime"
SPARSE = OUT / "sparse"

PROBE_SCHEDULES = {
    "left": [(1000, {"target_left": 150.0})],
    "right": [(1000, {"target_right": 150.0})],
    "bilateral": [(1000, {"target_left": 150.0, "target_right": 150.0})],
    "loom": [(1000, {"looming": 150.0})],
}


def probe_active_sets(interface=DATA / "sensory_interface.json", seed=11):
    """Full-brain 1 s probe per pattern; collect active flywire IDs."""
    channel_ids = rt.load_interface_spec(interface)
    active = set()
    report = {}
    for name, sched in PROBE_SCHEDULES.items():
        brain = rt.DualModeBrain("interactive", seed=seed,
                                 channel_ids=channel_ids)
        brain.reset_episode(seed)
        for dur, rates in sched:
            for _ in range(int(round(dur / 50.0))):
                brain.step(rates, chunk_ms=50.0)
        act = set(brain.spikes_by_flyid().keys())
        active |= act
        report[name] = len(act)
        print(f"[sparse] probe {name}: {len(act)} active")
        del brain
        bl.trim_memory()
    return active, report


def build_roi(interface=DATA / "sensory_interface.json",
              readout=DATA / "motor_readout.json", seed=11):
    active, report = probe_active_sets(interface, seed)
    iface_spec = json.loads(Path(interface).read_text())
    readout_pops = json.loads(Path(readout).read_text())["populations"]

    flyid2i, comp = bl.load_maps("783")
    i2fly = {v: k for k, v in flyid2i.items()}
    act_idx = sorted({flyid2i[int(f)] for f in active if int(f) in flyid2i})

    adj, _ = bl.adjacency("783")
    one_hop = adj[np.asarray(act_idx, dtype=np.int64)].indices
    roi_idx = set(act_idx) | {int(i) for i in one_hop}

    forced = set()
    for ch in iface_spec["channels"].values():
        forced.update(int(i) for i in ch["ids"])
    for pop in readout_pops.values():
        forced.update(int(i) for i in pop)
    roi_idx |= {flyid2i[int(f)] for f in forced if int(f) in flyid2i}

    del adj, act_idx, one_hop
    gc.collect()
    bl.trim_memory()
    roi_fly = sorted(i2fly[i] for i in roi_idx)
    report.update(n_active_union=len(active), n_roi=len(roi_fly))
    return roi_fly, report


def build_filtered(roi_fly, tag="sparse"):
    SPARSE.mkdir(parents=True, exist_ok=True)
    roi = sorted(set(int(i) for i in roi_fly))
    new_index = {fid: k for k, fid in enumerate(roi)}
    con_pqt = SPARSE / f"{tag}_connectivity.parquet"
    comp_csv = SPARSE / f"{tag}_completeness.csv"

    pf = pq.ParquetFile(bl.VERSIONS["783"]["path_con"])
    n_kept = 0
    writer = None
    for batch in pf.iter_batches(batch_size=2_000_000):
        t = batch.to_pandas()
        mask = t.Presynaptic_ID.isin(new_index) & t.Postsynaptic_ID.isin(new_index)
        if mask.any():
            sub = t[mask].copy()
            sub["Presynaptic_Index"] = [new_index[f] for f in sub.Presynaptic_ID]
            sub["Postsynaptic_Index"] = [new_index[f] for f in sub.Postsynaptic_ID]
            tbl = pyarrow.Table.from_pandas(sub, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(con_pqt, tbl.schema)
            writer.write_table(tbl)
            n_kept += len(sub)
        del t, batch
    if writer is not None:
        writer.close()
    pd.DataFrame({"root_id": roi, "Completed": True}).to_csv(
        comp_csv, index=False)
    return dict(n_neurons=len(roi), n_connections=n_kept,
                comp=str(comp_csv), con=str(con_pqt))


def cmd_build():
    t0 = time.perf_counter()
    roi, report = build_roi()
    meta = build_filtered(roi)
    report.update(meta, build_wall_s=round(time.perf_counter() - t0, 1))
    bl.save_json(SPARSE / "sparse_build.json", report)
    print("[sparse]", json.dumps(report, indent=1))

    # chunk-study on the sparse network vs the FULL-brain reference
    full = json.loads((OUT / "chunk_study.json").read_text())
    res = rt.study_chunk(
        out_dir=SPARSE, comp=meta["comp"], con=meta["con"],
        tag="sparse_chunk_study")
    res["full_reference_sha"] = full["reference"]["sha256"]
    res["exact_vs_full"] = all(
        c["identical_to_reference"] for c in res["chunkings"].values())
    res["n_neurons"] = meta["n_neurons"]
    bl.save_json(SPARSE / "sparse_chunk_study.json", res)
    print(f"[sparse] EXACT vs full brain: {res['exact_vs_full']}")


def cmd_verify(episode_dir):
    """Replay a recorded episode schedule on the sparse network; compare
    bit-identity with the episode's full-brain spikes.txt."""
    ep = Path(episode_dir)
    summary = json.loads((ep / "summary.json").read_text())
    sched = json.loads((ep / "rate_schedule.json").read_text())
    iface = DATA / "sensory_interface.json"
    if summary["condition"] == "shuffled_sensory":
        iface = ep.parent / (f"shuffled_iface_{summary['condition']}_"
                             f"{summary['scenario']}_r{summary['rep']}.json")
    channel_ids = {ch: d["ids"] for ch, d in
                   json.loads(Path(iface).read_text())["channels"].items()}
    comp = str(SPARSE / "sparse_completeness.csv")
    con = str(SPARSE / "sparse_connectivity.parquet")

    brain = rt.DualModeBrain("interactive", seed=summary["seed"],
                             channel_ids=channel_ids, comp=comp, con=con)
    brain.reset_episode(summary["seed"])
    for _, rates in sched:
        brain.step(rates, chunk_ms=summary["chunk_ms"])

    # compare in FLYWIRE space (the sparse net has its own brian index space)
    def parse(path):
        d = {}
        for line in open(path):
            if line.strip():
                h, _, ts = line.partition(":")
                d[int(h)] = [float(x) for x in ts.split(",")]
        return d

    import pandas as pd
    full_comp = pd.read_csv(
        bl.VERSIONS["783"]["path_comp"], index_col=0)
    full_i2fly = {i: int(j) for i, j in enumerate(full_comp.index)}
    sp = brain.spikes_by_flyid()
    full = {full_i2fly[b]: ts for b, ts in
            ((int(k), v) for k, v in parse(ep / "spikes.txt").items())}
    common = set(sp) & set(full)
    identical = (common == set(sp) == set(full)
                 and all(sp[k] == full[k] for k in common))
    n_train_mismatch = sum(1 for k in common if sp[k] != full[k])
    out = dict(episode=str(ep), n_neurons=brain.n_neurons,
               identical_to_full_brain=identical,
               comparison="flywire-id spike trains (sparse net has its own "
                          "brian index space)",
               n_active_sparse=len(sp), n_active_full=len(full),
               n_active_common=len(common),
               n_train_mismatches=n_train_mismatch,
               spikes_sparse=sum(len(v) for v in sp.values()),
               spikes_full=sum(len(v) for v in full.values()))
    bl.save_json(ep / "sparse_verification.json", out)
    print("[sparse-verify]", json.dumps(out, indent=1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "verify"])
    ap.add_argument("--episode-dir", default="")
    args = ap.parse_args()
    if args.cmd == "build":
        cmd_build()
    else:
        cmd_verify(args.episode_dir)


if __name__ == "__main__":
    main()
