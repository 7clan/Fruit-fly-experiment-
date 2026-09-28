"""D7 performance study: five execution modes, one standard workload.

The canonical full brain (Shiu LIF on FlyWire v783) remains the scientific
reference and is NEVER replaced. The other modes are accelerations /
restrictions, each measured for speed, RAM and fidelity impact vs the
reference on the SAME stimulus (Gate-2 looming set, seed 11, 1 s, 150 Hz).

Modes:
  1. full_reference   numpy codegen, full brain (the Gate-1/2 reference)
  2. subcircuit       induced subgraph on the task-relevant ROI
                      (D4 entries + D5 learning core + D6 motor + <=1+1-hop
                      entry->motor bridge), same model code + parameters.
                      Hop-radius ladder (measured): 3+3 bridge = 135,009
                      neurons (97.4% of brain); 2+2 = 112,053 bridge / 118,156
                      total (85%); the sensory->motor interface is intrinsically
                      brain-wide at radius >= 2, so the EXECUTABLE subcircuit
                      uses the tightest meaningful radius, 1+1 hops.
  3. sparse_active    data-driven ROI: active set of the reference trial
                      + 1-hop downstream, same model code + parameters
  4. cython           brian2 cython codegen on the full brain (cold + warm)
  5. cpp_standalone   brian2 C++ standalone codegen on the full brain

Memory discipline (this box: 2 cores, 3.9 GB): the parent process stays
light - all ROI/connectivity-file preparation happens FIRST, its big
intermediates (CSR adjacency, streamed parquet) are freed before any trial
subprocess (each ~2.9 GB) is launched.

Fidelity metrics (vs full_reference, by flywire ID): active-set Jaccard,
spike-count ratio, DN_ALL / DNp01 active counts.

Outputs: brain/results/perf_study/{perf_study.json, perf_study.csv, ...}
"""

import gc
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "results" / "perf_study"
PY = sys.executable
sys.path.insert(0, str(HERE))
import brainlib as bl  # noqa: E402

IDS = ROOT / "data" / "io_map" / "ids"
D4_POPS = ["LMC_L1", "LMC_L2", "LMC_L3", "TM_OFF", "MI_ON", "T4", "T5",
           "LPTC_HS", "LPTC_VS", "LPLC2", "LPLC1_LPLC4", "LC4", "LC16",
           "LC17", "LC9", "METU", "GRN_SUGAR"]
D5_POPS = ["KC", "MBON", "PAM", "PPL1", "PPL2", "APL", "DPM", "DAN_DN"]
D6_POPS = ["DN_ALL", "MOTOR_BRAIN"]


def load_ids(pop):
    return set(json.loads((IDS / f"{pop}.json").read_text()))


def run_mode(mode, tag, comp="", con=""):
    spikes = OUT / f"{tag}.txt"
    rec = OUT / f"{tag}.json"
    cmd = [PY, str(HERE / "perf_trial.py"), "--mode", mode,
           "--spikes-out", str(spikes), "--rec-out", str(rec)]
    if comp:
        cmd += ["--comp", comp, "--con", con]
    print(f">>> mode={mode} tag={tag}", flush=True)
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.perf_counter() - t0
    if proc.returncode != 0:
        print(proc.stdout[-3000:])
        print(proc.stderr[-3000:])
        return dict(mode=mode, tag=tag, status="FAILED",
                    outer_wall_s=round(wall, 2), stderr_tail=proc.stderr[-500:])
    print("   ", proc.stdout.strip()[-250:], flush=True)
    r = json.loads(rec.read_text())
    r["status"] = "OK"
    r["outer_wall_s"] = round(wall, 2)
    return r


def parse_active(path):
    s = set()
    for line in Path(path).read_text().splitlines():
        if line.strip():
            s.add(int(line.partition(":")[0]))
    return s


def active_by_flyid(path, i2fly):
    return {i2fly[i] for i in parse_active(path)}


def build_filtered(roi_ids, tag):
    """Streamed write of filtered comp/con files for an ROI (memory-safe)."""
    roi = sorted(set(int(i) for i in roi_ids))
    new_index = {fid: k for k, fid in enumerate(roi)}
    con_pqt = OUT / f"{tag}_connectivity.parquet"
    comp_csv = OUT / f"{tag}_completeness.csv"

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

    pd.DataFrame({"root_id": roi, "Completed": True}).to_csv(comp_csv, index=False)
    df = pd.read_csv(comp_csv, index_col=0)
    assert list(df.index) == roi
    return dict(tag=tag, n_neurons=len(roi), n_connections=n_kept,
                comp=str(comp_csv), con=str(con_pqt))


def compute_rois():
    """All ROI computation happens here; big intermediates freed on exit."""
    entries = set().union(*[load_ids(p) for p in D4_POPS])
    core = set().union(*[load_ids(p) for p in D5_POPS + D6_POPS])
    flyid2i, comp = bl.load_maps("783")
    e_idx = [flyid2i[i] for i in entries if i in flyid2i]
    d_idx = [flyid2i[i] for i in load_ids("DN_ALL") if i in flyid2i]
    adj, _ = bl.adjacency("783")

    def bfs(start, max_h, A):
        n = A.shape[0]
        seen = np.zeros(n, dtype=bool)
        frontier = np.asarray(sorted(set(start)), dtype=np.int64)
        seen[frontier] = True
        for _ in range(max_h):
            nxt = A[frontier].indices
            nxt = nxt[~seen[nxt]]
            if len(nxt) == 0:
                break
            frontier = np.unique(nxt)
            seen[frontier] = True
        return seen

    adjt = adj.T.tocsr()
    fwd3 = bfs(e_idx, 3, adj)
    bwd3 = bfs(d_idx, 3, adjt)
    bridge3 = fwd3 & bwd3
    n_bridge3 = int(bridge3.sum())

    fwd1 = bfs(e_idx, 1, adj)
    bwd1 = bfs(d_idx, 1, adjt)
    bridge1 = fwd1 & bwd1
    bridge_ids = {int(comp.index[i]) for i in np.where(bridge1)[0]}

    # radius ladder for the record (computed while adjacency is loaded)
    fwd2 = bfs(e_idx, 2, adj)
    bwd2 = bfs(d_idx, 2, adjt)
    n_bridge2 = int((fwd2 & bwd2).sum())

    del adj, adjt, fwd1, bwd1, fwd2, bwd2, fwd3, bwd3, bridge1, bridge3
    gc.collect()
    bl.trim_memory()

    roi_sub = entries | core | bridge_ids
    ladder = dict(
        bridge_1plus1=len(bridge_ids), bridge_2plus2=n_bridge2,
        bridge_3plus3=n_bridge3, brain_size=len(flyid2i))
    print(f"ROI(subcircuit 1+1): entries={len(entries)} core={len(core)} "
          f"bridge={len(bridge_ids)} total={len(roi_sub)}")
    print("hop-radius ladder:", ladder)
    Path(OUT / "roi_ladder.json").write_text(json.dumps(ladder))
    return roi_sub


def assemble_only():
    """Rebuild perf_study.json/csv from existing mode records.

    Used when compile-heavy modes (cython/cpp) were run detached while the
    machine was otherwise idle - their trial records are written by
    perf_trial.py exactly as in-process runs would.
    """
    import resource
    flyid2i, comp = bl.load_maps("783")
    i2fly = {v: k for k, v in flyid2i.items()}
    ref = json.loads((OUT / "m1_full_reference.json").read_text())
    ref["status"] = "OK"
    ref_active = active_by_flyid(OUT / "m1_full_reference.txt", i2fly)
    results = dict(full_reference=ref)

    tags = [("subcircuit", "m2_subcircuit"), ("sparse_active", "m3_sparse_active"),
            ("cython_cold", "m4_cython_cold"), ("cython_warm", "m4_cython_warm"),
            ("cpp_standalone", "m5_cpp_standalone")]
    for mode, tag in tags:
        pth = OUT / f"{tag}.json"
        if pth.exists():
            r = json.loads(pth.read_text())
            r["status"] = "OK"
            results[mode] = r
        else:
            results[mode] = dict(mode=mode, tag=tag, status="MISSING")

    # cpp binary-only timing
    bin_main = OUT / "cpp_project" / "main"
    if bin_main.exists():
        t0 = time.perf_counter()
        rc = subprocess.run([str(bin_main)], cwd=str(bin_main.parent),
                            capture_output=True, text=True)
        bin_wall = time.perf_counter() - t0
        child_kb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        results["cpp_standalone"]["binary_only_run_s"] = round(bin_wall, 3)
        results["cpp_standalone"]["binary_only_peak_rss_mb"] = round(child_kb / 1024, 1)
        results["cpp_standalone"]["binary_only_returncode"] = rc.returncode

    dn_ids = load_ids("DN_ALL")
    gf_ids = load_ids("DNp01_GF")

    def counts(active_ids):
        return dict(n_active=len(active_ids),
                    n_in_DN_ALL=len(active_ids & dn_ids),
                    n_in_DNp01=len(active_ids & gf_ids))

    def i2fly_for(tag):
        # filtered modes enumerate their own compact index space; their
        # completeness csv is the ground truth for index -> flywire id
        comp_csv = OUT / f"{tag}_completeness.csv"
        if comp_csv.exists():
            ids = pd.read_csv(comp_csv, index_col=0).index.tolist()
            return {k: int(fid) for k, fid in enumerate(ids)}
        return i2fly

    fidelity = {}
    for mode, tag in [("subcircuit", "m2_subcircuit"),
                      ("sparse_active", "m3_sparse_active"),
                      ("cython", "m4_cython_warm"),
                      ("cpp_standalone", "m5_cpp_standalone")]:
        p = OUT / f"{tag}.txt"
        if not p.exists():
            fidelity[mode] = dict(status="n/a")
            continue
        act = active_by_flyid(p, i2fly_for(tag))
        union = act | ref_active
        inter = act & ref_active
        key = mode if mode != "cython" else "cython_warm"
        fidelity[mode] = dict(
            jaccard_active=round(len(inter) / len(union), 4) if union else None,
            reference=counts(ref_active), mode=counts(act),
            spikes_ref=ref["n_spikes"],
            spikes_mode=results[key]["n_spikes"],
            spike_ratio_vs_ref=round(results[key]["n_spikes"] / ref["n_spikes"], 3),
        )

    rows = []
    for mode, r in results.items():
        rows.append(dict(
            mode=mode, status=r.get("status"),
            n_neurons=r.get("n_neurons_total"),
            wall_build_s=r.get("wall_build_s"),
            wall_run_s=r.get("wall_run_s"),
            bio_s_per_wall_s=round(1.0 / r["wall_run_s"], 4) if r.get("wall_run_s") else None,
            peak_rss_mb=round(r.get("peak_rss_kb", 0) / 1024, 1),
            child_peak_rss_mb=round(r.get("child_peak_rss_kb", 0) / 1024, 1)
            if r.get("child_peak_rss_kb") else None,
            n_active=r.get("n_active_neurons"),
            n_spikes=r.get("n_spikes"),
        ))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "perf_study.csv", index=False)
    print(df.to_string(index=False))
    (OUT / "perf_study.json").write_text(json.dumps(
        dict(results=results, fidelity=fidelity,
             workload="Gate-2 looming stimulus (60 LPLC2 + 20 LC4), seed 11, 1 s, 150 Hz",
             note="full_reference remains the scientific reference; other modes "
                  "are measured accelerations, never silent replacements"),
        indent=1, default=str))
    print("assembled ->", OUT)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--assemble-only":
        assemble_only()
        return
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- reference FIRST (also provides the sparse ROI)
    ref = run_mode("full_reference", "m1_full_reference")
    if ref["status"] != "OK":
        raise SystemExit("reference mode failed - aborting study")

    flyid2i, comp = bl.load_maps("783")
    i2fly = {v: k for k, v in flyid2i.items()}
    ref_active = active_by_flyid(OUT / "m1_full_reference.txt", i2fly)

    # ---- prepare filtered connectomes (parent memory peaks here, then freed)
    roi_sub = compute_rois()
    meta_sub = build_filtered(roi_sub, "m2_subcircuit")
    print("subcircuit files:", meta_sub["n_neurons"], "neurons,",
          meta_sub["n_connections"], "connections")

    adj, _ = bl.adjacency("783")
    a_idx = [flyid2i[i] for i in ref_active if i in flyid2i]
    one = adj[np.asarray(sorted(set(a_idx)), dtype=np.int64)].indices
    roi_sp = set(int(i) for i in ref_active) | {int(comp.index[i]) for i in one}
    del adj
    gc.collect()
    bl.trim_memory()
    print(f"ROI(sparse_active): active={len(ref_active)} "
          f"+1hop={len(roi_sp) - len(ref_active)} total={len(roi_sp)}")
    meta_sp = build_filtered(roi_sp, "m3_sparse_active")
    gc.collect()
    bl.trim_memory()
    # parent must stay LIGHT (subprocess trials each peak near 2.9 GB):
    # drop brianlib's cached CSR graph and the id maps; recomputed later.
    bl._GRAPH.pop("783", None)
    flyid2i.clear(); i2fly.clear()
    gc.collect()
    bl.trim_memory()

    # ---- run remaining modes from a light parent
    results = dict(full_reference=ref)
    sub = run_mode("subcircuit", "m2_subcircuit", meta_sub["comp"], meta_sub["con"])
    sub["roi"] = {k: meta_sub[k] for k in ("n_neurons", "n_connections")}
    results["subcircuit"] = sub

    spa = run_mode("sparse_active", "m3_sparse_active", meta_sp["comp"], meta_sp["con"])
    spa["roi"] = {k: meta_sp[k] for k in ("n_neurons", "n_connections")}
    results["sparse_active"] = spa

    import shutil, os
    cython_cache = Path.home() / ".cython" / "brian_extensions"
    if cython_cache.exists():
        shutil.rmtree(cython_cache)
        print("cleared cython cache for COLD trial")
    cyt = run_mode("cython", "m4_cython_cold")
    results["cython_cold"] = cyt
    cyt2 = run_mode("cython", "m4_cython_warm")
    results["cython_warm"] = cyt2

    cpp_proj = OUT / "cpp_project"
    if cpp_proj.exists():
        shutil.rmtree(cpp_proj)
    cpp = run_mode("cpp_standalone", "m5_cpp_standalone")
    results["cpp_standalone"] = cpp

    # cpp: time the compiled binary alone (steady-state per-trial cost)
    bin_main = OUT / "cpp_project" / "main"
    if bin_main.exists():
        t0 = time.perf_counter()
        rc = subprocess.run([str(bin_main)], cwd=str(bin_main.parent),
                            capture_output=True, text=True)
        bin_wall = time.perf_counter() - t0
        import resource
        child_kb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        results["cpp_standalone"]["binary_only_run_s"] = round(bin_wall, 3)
        results["cpp_standalone"]["binary_only_peak_rss_mb"] = round(child_kb / 1024, 1)
        results["cpp_standalone"]["binary_only_returncode"] = rc.returncode

    # ---- fidelity
    dn_ids = load_ids("DN_ALL")
    gf_ids = load_ids("DNp01_GF")

    def counts(active_ids):
        return dict(n_active=len(active_ids),
                    n_in_DN_ALL=len(active_ids & dn_ids),
                    n_in_DNp01=len(active_ids & gf_ids))

    def i2fly_for(tag):
        # filtered modes enumerate their own compact index space; their
        # completeness csv is the ground truth for index -> flywire id
        comp_csv = OUT / f"{tag}_completeness.csv"
        if comp_csv.exists():
            ids = pd.read_csv(comp_csv, index_col=0).index.tolist()
            return {k: int(fid) for k, fid in enumerate(ids)}
        return i2fly

    fidelity = {}
    for mode, tag in [("subcircuit", "m2_subcircuit"),
                      ("sparse_active", "m3_sparse_active"),
                      ("cython", "m4_cython_warm"),
                      ("cpp_standalone", "m5_cpp_standalone")]:
        p = OUT / f"{tag}.txt"
        if not p.exists():
            fidelity[mode] = dict(status="n/a")
            continue
        act = active_by_flyid(p, i2fly_for(tag))
        union = act | ref_active
        inter = act & ref_active
        fidelity[mode] = dict(
            jaccard_active=round(len(inter) / len(union), 4) if union else None,
            reference=counts(ref_active), mode=counts(act),
            spikes_ref=ref["n_spikes"],
            spikes_mode=results[mode if mode != "cython" else "cython_warm"]["n_spikes"],
            spike_ratio_vs_ref=round(
                results[mode if mode != "cython" else "cython_warm"]["n_spikes"]
                / ref["n_spikes"], 3),
        )

    rows = []
    for mode, r in results.items():
        rows.append(dict(
            mode=mode, status=r.get("status"),
            n_neurons=r.get("n_neurons_total"),
            wall_build_s=r.get("wall_build_s"),
            wall_run_s=r.get("wall_run_s"),
            bio_s_per_wall_s=round(1.0 / r["wall_run_s"], 4) if r.get("wall_run_s") else None,
            peak_rss_mb=round(r.get("peak_rss_kb", 0) / 1024, 1),
            child_peak_rss_mb=round(r.get("child_peak_rss_kb", 0) / 1024, 1)
            if r.get("child_peak_rss_kb") else None,
            n_active=r.get("n_active_neurons"),
            n_spikes=r.get("n_spikes"),
        ))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "perf_study.csv", index=False)
    print(df.to_string(index=False))

    (OUT / "perf_study.json").write_text(json.dumps(
        dict(results=results, fidelity=fidelity,
             workload="Gate-2 looming stimulus (60 LPLC2 + 20 LC4), seed 11, 1 s, 150 Hz",
             note="full_reference remains the scientific reference; other modes "
                  "are measured accelerations, never silent replacements"),
        indent=1, default=str))
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
