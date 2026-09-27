"""Shared harness for the Digital Drosophila brain (D1-D3).

Wraps the pinned third-party model (philshiu/Drosophila_brain_model @
91bdd1e) WITHOUT patching it. Adds what the published code does not have
and our Gate 1 requires:

  * explicit, per-trial RNG seeding (brian2.seed + numpy seed) so that
    runs are reproducible,
  * in-process serial trials (their run_exp uses joblib workers; seeding
    must be under our control),
  * timing + peak-RSS instrumentation for the D3 benchmark,
  * connectome graph utilities (propagation depth) for spike-propagation
    verification.

Terminology (docs/MASTER_SPEC.md section 2): the Shiu et al. code +
FlyWire data = [connectome data + modeled neuronal dynamics]; everything
in THIS file = engineered harness (not part of the brain).
"""

from __future__ import annotations

import copy
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent          # .../brain
MODEL_DIR = ROOT / "third_party" / "Drosophila_brain_model"
RESULTS = ROOT / "results"

# ---------------------------------------------------------------- versions
VERSIONS = {
    "630": {
        "path_comp": MODEL_DIR / "2023_03_23_completeness_630_final.csv",
        "path_con": MODEL_DIR / "2023_03_23_connectivity_630_final.parquet",
        "label": "FlyWire v630 (paper version, 127,400 neurons)",
    },
    "783": {
        "path_comp": MODEL_DIR / "Completeness_783.csv",
        "path_con": MODEL_DIR / "Connectivity_783.parquet",
        "label": "FlyWire v783 (public release, 138,639 neurons) - CANONICAL",
    },
}

# Tutorial stimulus set (example.ipynb): 21 sugar-sensing neurons,
# right hemisphere + the MN9 motor-neuron readout id.
SUGAR_IDS = [
    720575940624963786, 720575940630233916, 720575940637568838,
    720575940638202345, 720575940617000768, 720575940630797113,
    720575940632889389, 720575940621754367, 720575940621502051,
    720575940640649691, 720575940639332736, 720575940616885538,
    720575940639198653, 720575940620900446, 720575940617937543,
    720575940632425919, 720575940633143833, 720575940612670570,
    720575940628853239, 720575940629176663, 720575940611875570,
]
MN9_ID = 720575940660219265

_MODEL_IMPORTED = False


def ensure_model():
    """Import the pinned third-party model module (no patches, ever)."""
    global _MODEL_IMPORTED
    if _MODEL_IMPORTED:
        return
    if not MODEL_DIR.is_dir():
        raise SystemExit(
            f"third-party model missing at {MODEL_DIR}. "
            "Run: python brain/setup/setup_model.py clone"
        )
    sys.path.insert(0, str(MODEL_DIR))
    import model  # noqa: F401  (philshiu/Drosophila_brain_model/model.py)
    import utils  # noqa: F401
    _MODEL_IMPORTED = True


# ------------------------------------------------------------------- maps
_MAPS = {}


def load_maps(version: str):
    """flywire-id <-> brian-index maps for a connectome version."""
    if version not in _MAPS:
        df = pd.read_csv(VERSIONS[version]["path_comp"], index_col=0)
        flyid2i = {int(j): int(i) for i, j in enumerate(df.index)}
        _MAPS[version] = (flyid2i, df)
    return _MAPS[version]


def resolve_ids(fly_ids, version: str):
    """Split a stimulus list into (present indices, missing flywire ids)."""
    flyid2i, _ = load_maps(version)
    idx, missing = [], []
    for f in fly_ids:
        if int(f) in flyid2i:
            idx.append(flyid2i[int(f)])
        else:
            missing.append(int(f))
    return idx, missing


# ------------------------------------------------------------ memory info
def peak_rss_kb():
    """Peak resident set size of this process (VmHWM on Linux)."""
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1])
    except OSError:
        pass
    try:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except Exception:
        return -1


def rss_kb():
    """Current resident set size in kB (Linux)."""
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except OSError:
        pass
    return -1


def trim_memory():
    """Return freed heap to the OS (glibc malloc_trim) + full GC.

    Engineered-harness helper: whole-brain rebuilds peak near 3 GB; without
    trimming, allocator arenas fragment and repeated in-process trials OOM
    small machines. No-op where unsupported.
    """
    import gc
    gc.collect()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def cpu_info():
    info = {"cores": __import__("os").cpu_count()}
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    info["model"] = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemTotal"):
                    info["mem_total_mb"] = round(int(line.split()[1]) / 1024, 1)
    except OSError:
        pass
    return info


# ------------------------------------------------------------- core trial
def run_seeded_trial(seed, version="783", exc_fly_ids=(), exc2_fly_ids=(),
                     slnc_fly_ids=(), t_run_ms=1000, r_poi_hz=150,
                     r_poi2_hz=0, quiet=True):
    """One fully seeded, in-process simulation trial.

    Uses the third-party model's own create_model/poi/silence (unmodified).
    Returns a plain-dict record; spikes are {brian_index: [spike times s]}.
    """
    import brian2 as b2
    from brian2 import Hz, ms

    ensure_model()
    import model as dbm

    flyid2i, _ = load_maps(version)
    cfg = VERSIONS[version]

    exc, miss_exc = resolve_ids(exc_fly_ids, version)
    exc2, miss_exc2 = resolve_ids(exc2_fly_ids, version)
    slnc, miss_slnc = resolve_ids(slnc_fly_ids, version)

    params = copy.copy(dbm.default_params)
    params["t_run"] = t_run_ms * ms
    params["r_poi"] = r_poi_hz * Hz
    params["r_poi2"] = r_poi2_hz * Hz

    if quiet:
        b2.prefs.codegen.runtime.cython.multiprocess_safe = False  # silence fork noise
        import logging
        logging.getLogger("brian2").setLevel(logging.WARNING)

    b2.seed(seed)
    np.random.seed(seed % (2**32))

    t_build0 = time.perf_counter()
    neu, syn, spk_mon = dbm.create_model(cfg["path_comp"], cfg["path_con"], params)
    pois, neu = dbm.poi(neu, exc, exc2, params)
    if slnc:
        syn = dbm.silence(slnc, syn)
    net = b2.Network(neu, syn, spk_mon, *pois)
    t_build = time.perf_counter() - t_build0

    t_run0 = time.perf_counter()
    net.run(params["t_run"], namespace=params)
    t_run_wall = time.perf_counter() - t_run0

    trains = spk_mon.spike_trains()
    spikes = {int(k): [float(t) for t in v] for k, v in trains.items() if len(v)}

    rec = {
        "seed": seed,
        "version": version,
        "t_run_ms": t_run_ms,
        "r_poi_hz": r_poi_hz,
        "n_exc": len(exc),
        "n_exc2": len(exc2),
        "n_slnc": len(slnc),
        "missing_exc": miss_exc,
        "missing_exc2": miss_exc2,
        "missing_slnc": miss_slnc,
        "wall_build_s": round(t_build, 3),
        "wall_run_s": round(t_run_wall, 3),
        "n_spikes": int(sum(len(v) for v in spikes.values())),
        "n_active_neurons": len(spikes),
        "n_neurons_total": None,     # filled by caller (cheap below)
        "peak_rss_kb": peak_rss_kb(),
        "spikes": spikes,
    }
    rec["n_neurons_total"] = len(flyid2i)
    return rec


def spikes_to_df(rec, version):
    """Spike record -> tidy DataFrame (t, trial, flywire_id, brian_index)."""
    flyid2i, _ = load_maps(version)
    i2flyid = {v: k for k, v in flyid2i.items()}
    rows = []
    for bi, ts in rec["spikes"].items():
        fid = i2flyid.get(bi, -1)
        for t in ts:
            rows.append((t, fid, bi))
    return pd.DataFrame(rows, columns=["t", "flywire_id", "brian_index"])


# -------------------------------------------------------- connectome graph
_GRAPH = {}


def adjacency(version: str):
    """CSR adjacency (presynaptic -> postsynaptic) via scipy.sparse.

    Memory discipline: the 15M-row pandas dataframe is FREED after the
    CSR is built (~350 MB kept). The parent must stay light while trial
    subprocesses (each ~2.9 GB) run — coexistence OOMs small machines.
    """
    if version in _GRAPH:
        return _GRAPH[version]
    from scipy.sparse import csr_matrix
    con = pd.read_parquet(VERSIONS[version]["path_con"])
    n = len(load_maps(version)[1])
    adj = csr_matrix(
        (np.ones(len(con), dtype=np.int8),
         (con["Presynaptic_Index"].to_numpy(), con["Postsynaptic_Index"].to_numpy())),
        shape=(n, n),
    )
    del con
    trim_memory()
    _GRAPH[version] = (adj, None)
    return _GRAPH[version]


def hops_from(version: str, source_indices, max_hops=4):
    """Minimum hop distance (<= max_hops) from the stimulus set to every
    neuron, via the directed connectome. Returns an int16 array (-1 beyond)."""
    adj, _ = adjacency(version)
    n = adj.shape[0]
    dist = np.full(n, -1, dtype=np.int16)
    frontier = np.asarray(sorted(set(int(i) for i in source_indices)), dtype=np.int64)
    dist[frontier] = 0
    for h in range(1, max_hops + 1):
        reached = adj[frontier].indices
        new = reached[dist[reached] < 0]
        if len(new) == 0:
            break
        frontier = np.unique(new)
        dist[frontier] = h
    return dist


# ------------------------------------------------------------------- io
def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    def _default(o):
        try:
            return int(o)
        except Exception:
            return str(o)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=_default)


def public_rec(rec):
    """rec without the bulky spikes dict."""
    return {k: v for k, v in rec.items() if k != "spikes"}
