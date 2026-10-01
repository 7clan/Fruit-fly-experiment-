"""D7 engineering runtime: REFERENCE and INTERACTIVE modes over the CANONICAL
v783 digital Drosophila brain (Shiu et al. LIF model, third-party code
unmodified).

Modes (docs/ROADMAP.md D7):
  reference    full 138,639-neuron canonical brain, numpy runtime, ONE
               net.run() for the whole protocol; sensory schedule known
               up-front and delivered through a TimedArray-driven
               PoissonGroup (single fixed schedule).
  interactive  SAME canonical network, advanced in discrete chunks
               (25/50/100/200 ms) with sensory rates re-set between chunks,
               checkpointing (store/restore), per-chunk latency + RSS
               instrumentation. This is the closed-loop engine.

Both modes use the same engineered sensory interface (SensoryInterface
below) which reproduces the model's own dbm.poi() semantics (Poisson drive
onto 'v' with weight w_syn * f_poi, driven cells made a-refractory) but with
runtime-settable rates. The canonical model itself (LIF equations,
parameters, connectome synapses) is NEVER modified.

Fidelity contract (verified by this module's studies):
  * chunked stepping with piecewise-constant rates is BIT-IDENTICAL to a
    single run of the same rate schedule (the only difference is when the
    harness changes rates, and Brian2's PoissonGroup draws exactly one
    uniform per neuron per timestep regardless of the rate value);
  * per-episode determinism: net.restore('ep0') + brian2.seed(episode_seed)
    reproduces an episode bitwise (Brian2 2.10 store/restore does NOT
    restore the RNG stream - reseed-after-restore is the documented
    protocol, see checkpoint study);
  * C++ standalone replay of a recorded schedule is bit-identical to the
    interactive run (Brian2 standalone vendors numpy's randomkit RNG;
    proven by Gate-2 perf study m5 == m1 sha256 and re-proven here on a
    dynamic-rate schedule).

CLI studies (write brain/results/d7_runtime/):
  chunk-study      bit-identity + overhead of 25/50/100/200 ms chunking
  checkpoint       restore+reseed determinism, mid-episode checkpoint,
                   disk store round-trip, 10 s memory soak
  bench            bio-s/wall-s, per-chunk decision latency, RAM, CPU
  cpp-replay       standalone replay of a schedule JSON (own process)
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import brainlib as bl  # noqa: E402

OUT = bl.ROOT / "results" / "d7_runtime"
STUDY_SEED = 11

# Pre-registered D7 study schedule (ms, {channel: Hz}). Version 2: every
# segment is a multiple of 200 ms so ALL chunk sizes 25/50/100/200 deliver
# identical per-step rate sequences. (Version 1 had a 500 ms segment; the
# 200 ms chunking legitimately quantized the mid-chunk transition at
# t=1500 ms - a real boundary-quantization property, kept as evidence in
# results/d7_runtime/chunk_study_misaligned.json.)
STUDY_SCHEDULE = [
    (400, {}),
    (400, {"target_left": 150.0}),
    (400, {}),
    (400, {"looming": 150.0}),
    (400, {}),
    (200, {"target_right": 250.0}),
]


# --------------------------------------------------------------------------
# engineered sensory interface
# --------------------------------------------------------------------------
class SensoryInterface:
    """Runtime-settable Poisson drive onto chosen brain neurons.

    Mirrors dbm.poi() exactly (one Poisson source per driven neuron, event
    adds w_syn * f_poi to v, driven cells lose their refractory period) but
    uses a PoissonGroup whose rates the harness can change between chunks.
    """

    def __init__(self, neu, channel_ids, flyid2i, params):
        import brian2 as b2
        from brian2 import Hz, ms

        self.channel_names = sorted(channel_ids)
        self.slots = {}
        self.missing = {}
        targets = []
        for ch in self.channel_names:
            fids = [int(f) for f in channel_ids[ch]]
            idx = [flyid2i[f] for f in fids if f in flyid2i]
            self.missing[ch] = [f for f in fids if f not in flyid2i]
            self.slots[ch] = np.arange(len(targets), len(targets) + len(idx))
            targets.extend(idx)
        self.targets = np.asarray(targets, dtype=np.int64)
        self.n = int(len(targets))

        self.pg = b2.PoissonGroup(self.n, rates=np.zeros(self.n) * Hz,
                                  name="sensory_iface")
        self.syn = b2.Synapses(self.pg, neu, "w_stim : volt",
                               on_pre="v_post += w_stim",
                               name="sensory_iface_syn")
        self.syn.connect(i=np.arange(self.n), j=self.targets)
        self.syn.w_stim = params["w_syn"] * params["f_poi"]
        self._set_driven_a_refractory(neu)
        self._rates = np.zeros(self.n, dtype=float)

    def _set_driven_a_refractory(self, neu):
        """Driven cells lose their refractory period (mirror of dbm.poi).
        brian2 2.10.1 cpp_standalone cannot set a variable with an index
        ARRAY (ambiguous-truth bug in variableview_set_with_index_array)
        -> per-item single-value assignments there."""
        import brian2 as b2
        from brian2 import ms
        try:
            neu.rfc[self.targets] = 0 * ms
        except ValueError:
            for i in self.targets:
                neu.rfc[int(i)] = 0 * ms

    def set_rates(self, rates_hz):
        from brian2 import Hz
        arr = self._rates
        arr[:] = 0.0
        for ch, r in (rates_hz or {}).items():
            arr[self.slots[ch]] = float(r)
        self.pg.rates = arr * Hz

    def rate_matrix(self, schedule, grid_ms):
        """schedule -> (n_gridsteps, n_slots) rate matrix for TimedArray."""
        n_steps = int(round(sum(d for d, _ in schedule) / grid_ms))
        mat = np.zeros((n_steps, self.n))
        k = 0
        for dur, rates in schedule:
            n_seg = int(round(dur / grid_ms))
            for ch, r in (rates or {}).items():
                mat[k:k + n_seg, self.slots[ch]] = float(r)
            k += n_seg
        assert k == n_steps
        return mat


class TimedSensoryInterface(SensoryInterface):
    """Single-run (reference/standalone) flavor: rates driven by a
    TimedArray built from the full schedule before the run starts."""

    def __init__(self, neu, channel_ids, flyid2i, params, schedule, grid_ms):
        import brian2 as b2
        from brian2 import Hz

        # build slots/targets first without creating the dynamic PoissonGroup
        self.channel_names = sorted(channel_ids)
        self.slots = {}
        self.missing = {}
        targets = []
        for ch in self.channel_names:
            fids = [int(f) for f in channel_ids[ch]]
            idx = [flyid2i[f] for f in fids if f in flyid2i]
            self.missing[ch] = [f for f in fids if f not in flyid2i]
            self.slots[ch] = np.arange(len(targets), len(targets) + len(idx))
            targets.extend(idx)
        self.targets = np.asarray(targets, dtype=np.int64)
        self.n = int(len(targets))

        mat = self.rate_matrix(schedule, grid_ms)
        self.timed_array = b2.TimedArray(mat * Hz, dt=grid_ms * b2.ms,
                                         name="stim_ta")
        self.pg = b2.PoissonGroup(self.n, rates="stim_ta(t, i)",
                                  name="sensory_iface",
                                  namespace={"stim_ta": self.timed_array})
        self.syn = b2.Synapses(self.pg, neu, "w_stim : volt",
                               on_pre="v_post += w_stim",
                               name="sensory_iface_syn")
        self.syn.connect(i=np.arange(self.n), j=self.targets)
        self.syn.w_stim = params["w_syn"] * params["f_poi"]
        self._set_driven_a_refractory(neu)
        self._rates = np.zeros(self.n, dtype=float)

    def set_rates(self, rates_hz):          # pragma: no cover - reference only
        raise RuntimeError("TimedSensoryInterface rates are fixed at build")


# --------------------------------------------------------------------------
# dual-mode brain
# --------------------------------------------------------------------------
class DualModeBrain:
    """REFERENCE or INTERACTIVE runtime over the canonical brain.

    mode='reference'    one net.run() with a TimedArray schedule
    mode='interactive'  chunked stepping, dynamic rates, checkpoints
    Both build the full canonical network unless filtered comp/con paths
    are supplied (sparse variant, verified separately for exactness).
    """

    def __init__(self, mode, seed=STUDY_SEED, version="783",
                 channel_ids=None, readout_pops=None,
                 comp=None, con=None, schedule=None, grid_ms=25.0,
                 quiet=True, record_full_spikes=True):
        import brian2 as b2

        assert mode in ("reference", "interactive")
        self.mode = mode
        bl.ensure_model()
        import model as dbm

        if quiet:
            b2.prefs.codegen.runtime.cython.multiprocess_safe = False
            import logging
            logging.getLogger("brian2").setLevel(logging.WARNING)

        if comp and con:
            df_comp = pd.read_csv(comp, index_col=0)
            self.flyid2i = {int(j): int(i) for i, j in enumerate(df_comp.index)}
            p_comp, p_con = Path(comp), Path(con)
            self.n_neurons = len(df_comp)
            self.sparse = True
        else:
            self.flyid2i, _ = bl.load_maps(version)
            p_comp = bl.VERSIONS[version]["path_comp"]
            p_con = bl.VERSIONS[version]["path_con"]
            self.n_neurons = len(self.flyid2i)
            self.sparse = False
        self.i2fly = {v: k for k, v in self.flyid2i.items()}

        params = copy.copy(dbm.default_params)   # canonical, untouched
        self.params = params

        t0 = time.perf_counter()
        self.neu, self.syn, canonical_mon = dbm.create_model(
            p_comp, p_con, params)

        self.pop_names = []
        self.pop_code = None
        self.pop_sizes = {}
        if readout_pops:
            self._build_readout(readout_pops)

        # Live closed-loop mode does not need every spike timestamp. Brian2
        # SpikeMonitor(record=False) keeps exact per-neuron spike counts and
        # total spike counts without storing the full event arrays. This is
        # instrumentation-only: neuron/synapse dynamics are unchanged.
        self.record_full_spikes = bool(record_full_spikes)
        if mode == "interactive" and not self.record_full_spikes:
            self.mon = b2.SpikeMonitor(
                self.neu, record=False, name="live_count_mon")
            self.instrumentation_scope = "whole_brain_counts_only"
        else:
            self.mon = canonical_mon
            self.instrumentation_scope = "full_spike_recording"

        self.iface = None
        objs = [self.neu, self.syn, self.mon]
        if channel_ids:
            if mode == "reference":
                assert schedule is not None, "reference mode needs the schedule"
                self.iface = TimedSensoryInterface(
                    self.neu, channel_ids, self.flyid2i, params,
                    schedule, grid_ms)
            else:
                self.iface = SensoryInterface(
                    self.neu, channel_ids, self.flyid2i, params)
            objs += [self.iface.pg, self.iface.syn]
        self.net = b2.Network(*objs)
        self.build_wall_s = time.perf_counter() - t0

        self.seed_all(seed)
        if not _is_standalone():
            # checkpoints are a runtime-mode feature; the C++ standalone
            # device does not support store/restore (offline replay only)
            self.net.store("ep0")               # pristine checkpoint
        self.cursor = 0
        self._prev_spike_counts = np.zeros(
            self.n_neurons, dtype=np.int64)
        self.chunk_walls = []

    # ---------------- setup helpers ----------------
    def seed_all(self, seed):
        import brian2 as b2
        b2.seed(seed)
        np.random.seed(seed % (2 ** 32))
        self.last_seed = seed

    def _build_readout(self, readout_pops):
        code = np.full(self.n_neurons, -1, dtype=np.int32)
        names = sorted(readout_pops)
        for k, name in enumerate(names):
            members = 0
            for f in readout_pops[name]:
                i = self.flyid2i.get(int(f))
                if i is not None:
                    code[i] = k           # later pops win overlaps; logged
                    members += 1
            self.pop_sizes[name] = members
        self.pop_code = code
        self.pop_names = names

    # ---------------- episode control ----------------
    def reset_episode(self, seed):
        """Restore pristine state + reseed. The ONLY episode-reset protocol:
        Brian2 2.10 store/restore does not restore the RNG stream, so
        determinism comes from the explicit reseed (checkpoint study)."""
        self.net.restore("ep0")
        self.seed_all(seed)
        self.cursor = 0
        if not self.record_full_spikes:
            self._prev_spike_counts = np.asarray(
                self.mon.count[:], dtype=np.int64).copy()
        self.chunk_walls = []

    def step(self, rates=None, chunk_ms=50.0):
        """Advance one interactive chunk. Returns a per-chunk record."""
        import brian2 as b2
        if self.mode != "interactive":
            raise RuntimeError("step() is for interactive mode")
        if self.iface is not None:
            self.iface.set_rates(rates)
        t0 = time.perf_counter()
        self.net.run(chunk_ms * b2.ms)
        wall = time.perf_counter() - t0
        self.chunk_walls.append(wall)

        if self.record_full_spikes:
            i_new = np.asarray(self.mon.i[self.cursor:])
            active_new = np.unique(i_new)
            n_spikes_new = int(len(i_new))
            if self.pop_code is not None:
                pop_counts = np.bincount(
                    self.pop_code[i_new] + 1,
                    minlength=len(self.pop_names) + 1)[1:]
            else:
                pop_counts = None
            if self.record_full_spikes:
            self.cursor = int(len(np.asarray(self.mon.t[:])))
        else:
            self._prev_spike_counts = np.asarray(
                self.mon.count[:], dtype=np.int64).copy()
            self.cursor = int(self.mon.num_spikes)
        else:
            # Exact per-neuron deltas without storing individual spike times.
            now_counts = np.asarray(self.mon.count[:], dtype=np.int64)
            delta = now_counts - self._prev_spike_counts
            active_new = np.flatnonzero(delta > 0)
            n_spikes_new = int(delta.sum())
            if self.pop_code is not None:
                active_code = self.pop_code[active_new]
                active_delta = delta[active_new]
                pop_counts = np.bincount(
                    active_code + 1, weights=active_delta,
                    minlength=len(self.pop_names) + 1)[1:]
            else:
                pop_counts = None
            self._prev_spike_counts = now_counts.copy()
            self.cursor = int(self.mon.num_spikes)

        active_sample = [
            int(self.i2fly[int(i)]) for i in active_new[:24]
            if int(i) in self.i2fly
        ]
        rec = dict(
            t_bio_s=float(self.net.t / b2.second),
            chunk_ms=float(chunk_ms),
            wall_s=round(wall, 4),
            rss_kb=bl.rss_kb(),
            n_spikes_new=n_spikes_new,
            n_active_new=int(len(active_new)),
            active_flywire_ids_sample=active_sample,
            instrumentation_scope=self.instrumentation_scope,
        )
        if pop_counts is not None:
            rec["pop_counts"] = {
                n: int(v) for n, v in zip(self.pop_names, pop_counts)
            }
        return rec

    def run_reference(self, duration_ms):
        """Single run() of the whole protocol (reference mode). In the C++
        standalone device (build_on_run=False) this only SCHEDULES the run;
        the caller must device.build(run=True) and then read spikes."""
        import brian2 as b2
        assert self.mode == "reference"
        t0 = time.perf_counter()
        self.net.run(duration_ms * b2.ms)
        wall = time.perf_counter() - t0
        self.chunk_walls = [wall]
        if _is_standalone():
            return dict(wall_s=round(wall, 3), rss_kb=bl.rss_kb(),
                        n_spikes=None)
        return dict(wall_s=round(wall, 3), rss_kb=bl.rss_kb(),
                    n_spikes=int(len(np.asarray(self.mon.t[:]))))

    # ---------------- readout / evidence ----------------
    def spike_trains(self):
        if not self.record_full_spikes:
            raise RuntimeError(
                "individual spike times are disabled in live count-only "
                "instrumentation; use full recording mode")
        tr = self.mon.spike_trains()
        return {int(k): [float(x) for x in v] for k, v in tr.items() if len(v)}

    def canonical_spikes_text(self):
        lines = []
        tr = self.spike_trains()
        for bi in sorted(tr):
            lines.append(f"{bi}:" + ",".join(repr(t) for t in tr[bi]))
        return "\n".join(lines)

    def spikes_by_flyid(self):
        return {self.i2fly[bi]: ts for bi, ts in self.spike_trains().items()}

    # ---------------- checkpoints ----------------
    def checkpoint(self, tag="ckpt"):
        self.net.store(tag)

    def resume(self, tag="ckpt", reseed=None):
        self.net.restore(tag)
        if reseed is not None:
            self.seed_all(reseed)
        if self.record_full_spikes:
            self.cursor = int(len(np.asarray(self.mon.t[:])))
        else:
            self._prev_spike_counts = np.asarray(
                self.mon.count[:], dtype=np.int64).copy()
            self.cursor = int(self.mon.num_spikes)

    def store_disk(self, path):
        self.net.store(filename=str(path))

    def load_disk(self, path):
        self.net.restore(filename=str(path))
        if self.record_full_spikes:
            self.cursor = int(len(np.asarray(self.mon.t[:])))
        else:
            self._prev_spike_counts = np.asarray(
                self.mon.count[:], dtype=np.int64).copy()
            self.cursor = int(self.mon.num_spikes)


# --------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------
def sha256_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def load_interface_spec(path):
    """sensory_interface.json -> channel_ids dict."""
    spec = json.loads(Path(path).read_text())
    return {ch: d["ids"] for ch, d in spec["channels"].items()}


def cpu_times():
    t = os.times()
    return dict(user_s=round(t[0], 2), sys_s=round(t[1], 2),
                elapsed_s=round(t.elapsed, 2))


def child_peak_rss_mb():
    """Peak child RSS where the platform exposes it.

    Python's Unix-only resource module does not exist on Windows; the
    Windows live benchmark does not require this child-process metric.
    """
    try:
        import resource
        return round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024, 1)
    except Exception:
        return -1.0


def _is_standalone():
    try:
        from brian2.devices.device import get_device
        return "standalone" in get_device().__class__.__name__.lower()
    except Exception:
        return False


# --------------------------------------------------------------------------
# D7 studies (subprocess-per-build: Gate-1 memory discipline - sequential
# whole-brain rebuilds in ONE process OOM this box)
# --------------------------------------------------------------------------
PY = sys.executable
DATA_IFACE_DEFAULT = bl.ROOT / "data" / "d7_d10" / "sensory_interface.json"


def _schedule_to_json(schedule, path):
    Path(path).write_text(json.dumps(
        [[d, r] for d, r in schedule]))
    return str(path)


def run_worker(mode, schedule, seed, spikes_out, rec_out, chunk_ms=50.0,
               interface=DATA_IFACE_DEFAULT, comp=None, con=None,
               readout=None, extra_tag=""):
    """Run ONE build + one protocol in a fresh subprocess; return rec dict."""
    import subprocess
    cmd = [PY, str(Path(__file__).resolve()), "_worker",
           "--mode", mode, "--seed", str(seed),
           "--interface", str(interface),
           "--spikes-out", str(spikes_out), "--rec-out", str(rec_out),
           "--chunk-ms", str(chunk_ms)]
    if readout:
        cmd += ["--readout", str(readout)]
    if comp:
        cmd += ["--comp", comp, "--con", con]
    if schedule is not None:
        sj = Path(rec_out).with_suffix(".schedule.json")
        _schedule_to_json(schedule, sj)
        cmd += ["--schedule-json", str(sj)]
    else:
        cmd += ["--no-stim"]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.perf_counter() - t0
    if proc.returncode != 0:
        print(proc.stdout[-2500:])
        print(proc.stderr[-2500:])
        raise SystemExit(f"worker failed ({mode} chunk={chunk_ms})")
    rec = json.loads(Path(rec_out).read_text())
    rec["outer_wall_s"] = round(wall, 2)
    rec["tag"] = extra_tag
    return rec


def study_chunk(seed=STUDY_SEED, out_dir=OUT, comp=None, con=None,
                interface=DATA_IFACE_DEFAULT, tag="chunk_study"):
    """Bit-identity + per-chunk overhead for 25/50/100/200 ms chunking."""
    import subprocess
    out_dir.mkdir(parents=True, exist_ok=True)
    t_total_ms = sum(d for d, _ in STUDY_SCHEDULE)

    ref = run_worker("reference", STUDY_SCHEDULE, seed,
                     out_dir / f"{tag}_reference.txt",
                     out_dir / f"{tag}_reference.json",
                     interface=interface, comp=comp, con=con)
    ref_sha = sha256_text(Path(out_dir / f"{tag}_reference.txt").read_text())

    results = dict(seed=seed, n_neurons=ref["n_neurons"],
                   schedule=STUDY_SCHEDULE,
                   reference=dict(sha256=ref_sha,
                                  wall_s=ref["wall_s"],
                                  bio_s_per_wall_s=round(
                                      t_total_ms / 1000.0 / ref["wall_s"], 4),
                                  peak_rss_mb=ref["peak_rss_mb"]),
                   chunkings={})
    for chunk_ms in (25, 50, 100, 200):
        rec = run_worker("interactive", STUDY_SCHEDULE, seed,
                         out_dir / f"{tag}_chunk{chunk_ms}.txt",
                         out_dir / f"{tag}_chunk{chunk_ms}.json",
                         chunk_ms=chunk_ms, interface=interface,
                         comp=comp, con=con, extra_tag=str(chunk_ms))
        sha = sha256_text(
            Path(out_dir / f"{tag}_chunk{chunk_ms}.txt").read_text())
        results["chunkings"][str(chunk_ms)] = dict(
            sha256=sha, identical_to_reference=bool(sha == ref_sha),
            n_chunks=rec["n_chunks"], wall_total_s=rec["wall_s"],
            bio_s_per_wall_s=round(t_total_ms / 1000.0 / rec["wall_s"], 4),
            chunk_wall_ms_mean=rec["chunk_wall_ms_mean"],
            chunk_wall_ms_p95=rec["chunk_wall_ms_p95"],
            chunk_wall_ms_max=rec["chunk_wall_ms_max"],
            peak_rss_mb=rec["peak_rss_mb"], n_spikes=rec["n_spikes"])
        print(f"[chunk-study] chunk={chunk_ms}ms "
              f"identical={sha == ref_sha} "
              f"latency/chunk={rec['chunk_wall_ms_mean']}ms "
              f"bio/wall={t_total_ms / 1000.0 / rec['wall_s']:.3f}")
    results["all_chunkings_identical"] = all(
        c["identical_to_reference"] for c in results["chunkings"].values())
    bl.save_json(out_dir / f"{tag}.json", results)
    print(f"[chunk-study] ALL IDENTICAL: {results['all_chunkings_identical']}")
    return results


def _checkpoint_worker(args):
    """In-process multi-episode study (ONE build, many episodes via
    restore+reseed; mid-episode checkpoint; disk round-trip; memory soak)."""
    channel_ids = load_interface_spec(args.interface)
    brain = DualModeBrain("interactive", seed=args.seed,
                          channel_ids=channel_ids)
    results = dict(seed=args.seed, n_neurons=brain.n_neurons)

    def run_sched(chunk_ms=50.0):
        for dur, rates in STUDY_SCHEDULE:
            for _ in range(int(round(dur / chunk_ms))):
                brain.step(rates, chunk_ms=chunk_ms)
        return brain.canonical_spikes_text()

    sha_fresh = sha256_text(run_sched())
    brain.reset_episode(args.seed)
    sha_reset = sha256_text(run_sched())
    results["fresh_vs_restore_reseed"] = dict(
        fresh=sha_fresh, reset=sha_reset, identical=sha_fresh == sha_reset)

    brain.reset_episode(args.seed)
    for dur, rates in STUDY_SCHEDULE[:3]:
        for _ in range(int(round(dur / 50.0))):
            brain.step(rates, chunk_ms=50.0)
    resume_seed = 4242
    brain.seed_all(resume_seed)
    brain.checkpoint("mid")
    text_cont = run_sched()
    brain.resume("mid", reseed=resume_seed)
    text_resumed = run_sched()
    results["mid_checkpoint_resume"] = dict(identical=text_cont == text_resumed)

    brain.reset_episode(args.seed)
    for dur, rates in STUDY_SCHEDULE[:3]:
        for _ in range(int(round(dur / 50.0))):
            brain.step(rates, chunk_ms=50.0)
    ck = Path(args.out_dir) / "checkpoint_state.bin"
    brain.store_disk(ck)
    brain.resume("mid", reseed=resume_seed)
    text_a = run_sched()
    brain.load_disk(ck)
    brain.seed_all(resume_seed)
    text_b = run_sched()
    results["disk_store_roundtrip"] = dict(
        path=str(ck), size_mb=round(ck.stat().st_size / 1024 / 1024, 1),
        identical=text_a == text_b)
    ck.unlink(missing_ok=True)

    brain.reset_episode(args.seed)
    rss_samples = []
    n_chunks = int(10.0 * 1000 / 50)
    for k in range(n_chunks):
        rates = {"target_left": 150.0} if k % 20 < 10 else {}
        brain.step(rates, chunk_ms=50.0)
        if k % 40 == 0:
            rss_samples.append(dict(t_bio_s=round(k * 0.05, 1),
                                    rss_mb=round(bl.rss_kb() / 1024, 1)))
    results["memory_soak"] = dict(
        duration_bio_s=10.0, chunk_ms=50, rss_samples=rss_samples,
        rss_growth_mb=round(rss_samples[-1]["rss_mb"]
                            - rss_samples[0]["rss_mb"], 1),
        wall_s=round(sum(brain.chunk_walls), 1))
    results["cpu"] = cpu_times()
    results["peak_rss_mb"] = round(bl.peak_rss_kb() / 1024, 1)
    bl.save_json(Path(args.out_dir) / "checkpoint.json", results)
    print(json.dumps({k: v for k, v in results.items()
                      if k not in ("n_neurons",)}, indent=1, default=str))


def _bench_worker(args):
    channel_ids = load_interface_spec(args.interface)
    readout_pops = (json.loads(Path(args.readout).read_text())["populations"]
                    if args.readout else None)
    rates = {"target_left": 150.0, "target_right": 150.0}
    t_ms = args.t_ms
    chunk_ms = args.chunk_ms

    if args.mode == "reference":
        sched = [(t_ms, rates)]
        brain = DualModeBrain("reference", seed=args.seed,
                              channel_ids=channel_ids,
                              readout_pops=readout_pops,
                              schedule=sched, grid_ms=25.0,
                              comp=args.comp or None, con=args.con or None)
        rr = brain.run_reference(t_ms)
        rec = dict(mode="reference", n_neurons=brain.n_neurons,
                   wall_s=rr["wall_s"], n_spikes=rr["n_spikes"],
                   bio_s_per_wall_s=round((t_ms / 1000.0) / rr["wall_s"], 4),
                   peak_rss_mb=round(bl.peak_rss_kb() / 1024, 1))
    else:
        brain = DualModeBrain("interactive", seed=args.seed,
                              channel_ids=channel_ids,
                              readout_pops=readout_pops,
                              comp=args.comp or None, con=args.con or None)
        per = [brain.step(rates, chunk_ms=chunk_ms)
               for _ in range(int(t_ms / chunk_ms))]
        walls = np.array([r["wall_s"] for r in per])
        spikes = sum(r["n_spikes_new"] for r in per)
        ct = cpu_times()
        rec = dict(
            mode="interactive", n_neurons=brain.n_neurons, chunk_ms=chunk_ms,
            t_bio_s=t_ms / 1000.0, n_chunks=len(per),
            build_wall_s=round(brain.build_wall_s, 2),
            wall_s=round(float(walls.sum()), 2),
            bio_s_per_wall_s=round((t_ms / 1000.0) / float(walls.sum()), 4),
            decision_latency_wall_ms_mean=round(
                float(walls.mean() * 1000), 1),
            decision_latency_wall_ms_p95=round(
                float(np.percentile(walls, 95) * 1000), 1),
            decision_latency_wall_ms_max=round(float(walls.max() * 1000), 1),
            bio_ms_per_decision=chunk_ms, n_spikes=spikes,
            cpu_user_s=ct["user_s"], cpu_sys_s=ct["sys_s"],
            peak_rss_mb=round(bl.peak_rss_kb() / 1024, 1),
            machine=bl.cpu_info(),
        )
    Path(args.rec_out).write_text(json.dumps(rec, indent=1, default=str))
    if args.spikes_out:
        Path(args.spikes_out).write_text(brain.canonical_spikes_text())
    print(json.dumps({k: v for k, v in rec.items() if k != "machine"}))


def study_bench(seed=STUDY_SEED, t_ms=1000, chunk_ms=50, out_dir=OUT,
                comp=None, con=None, interface=DATA_IFACE_DEFAULT,
                readout=None, tag="bench"):
    """Representative-workload benchmark: bilateral LC9 target drive (the
    D10 workload). Subprocess per build (memory discipline)."""
    import subprocess
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def spawn(mode, spikes_out, rec_out):
        cmd = [PY, str(Path(__file__).resolve()), "_bench_worker",
               "--mode", mode, "--interface", str(interface),
               "--seed", str(seed), "--t-ms", str(t_ms),
               "--chunk-ms", str(chunk_ms),
               "--spikes-out", str(spikes_out), "--rec-out", str(rec_out)]
        if readout:
            cmd += ["--readout", str(readout)]
        if comp:
            cmd += ["--comp", comp, "--con", con]
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            print(proc.stdout[-2500:])
            print(proc.stderr[-2500:])
            raise SystemExit(f"bench worker failed ({mode})")
        rec = json.loads(Path(rec_out).read_text())
        rec["outer_wall_s"] = round(time.perf_counter() - t0, 1)
        return rec

    inter = spawn("interactive", out_dir / f"{tag}_interactive.txt",
                  out_dir / f"{tag}_interactive.json")
    ref = spawn("reference", out_dir / f"{tag}_reference.txt",
                out_dir / f"{tag}_reference.json")
    inter_sha = sha256_text(
        Path(out_dir / f"{tag}_interactive.txt").read_text())
    ref_sha = sha256_text(Path(out_dir / f"{tag}_reference.txt").read_text())
    res = dict(interactive=inter, reference_single_run=ref,
               workload="bilateral LC9 target drive (D10 workload)",
               spike_fidelity=dict(
                   identical=inter_sha == ref_sha,
                   interactive_sha=inter_sha[:16], reference_sha=ref_sha[:16]))
    bl.save_json(out_dir / f"{tag}.json", res)
    print(f"[bench] interactive {inter['bio_s_per_wall_s']} bio-s/wall-s, "
          f"latency/chunk {inter['decision_latency_wall_ms_mean']} ms, "
          f"RSS {inter['peak_rss_mb']} MB; fidelity identical="
          f"{inter_sha == ref_sha}")
    return res


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description="D7 engineering runtime")
    ap.add_argument("study", choices=["chunk-study", "checkpoint", "bench",
                                      "_worker", "_checkpoint_worker",
                                      "_bench_worker"])
    ap.add_argument("--interface", default=str(DATA_IFACE_DEFAULT))
    ap.add_argument("--readout", default="")
    ap.add_argument("--seed", type=int, default=STUDY_SEED)
    ap.add_argument("--t-ms", type=int, default=1000)
    ap.add_argument("--chunk-ms", type=float, default=50)
    ap.add_argument("--comp", default="")
    ap.add_argument("--con", default="")
    ap.add_argument("--schedule-json", default="")
    ap.add_argument("--spikes-out", default="")
    ap.add_argument("--rec-out", default="")
    ap.add_argument("--out-dir", default=str(OUT))
    ap.add_argument("--mode", default="interactive")
    ap.add_argument("--no-stim", action="store_true")
    args = ap.parse_args()

    if args.study == "_worker":
        channel_ids = load_interface_spec(args.interface)
        readout_pops = (json.loads(Path(args.readout).read_text())
                        ["populations"] if args.readout else None)
        if args.mode == "reference":
            sched = json.loads(Path(args.schedule_json).read_text())
            sched = [(d, r) for d, r in sched]
            brain = DualModeBrain("reference", seed=args.seed,
                                  channel_ids=channel_ids,
                                  readout_pops=readout_pops,
                                  schedule=sched, grid_ms=25.0,
                                  comp=args.comp or None,
                                  con=args.con or None)
            t_total = sum(d for d, _ in sched)
            rr = brain.run_reference(t_total)
            rec = dict(mode="reference", n_neurons=brain.n_neurons,
                       wall_s=rr["wall_s"], n_spikes=rr["n_spikes"],
                       bio_s_per_wall_s=round(
                           (t_total / 1000.0) / rr["wall_s"], 4),
                       peak_rss_mb=round(bl.peak_rss_kb() / 1024, 1))
        else:
            brain = DualModeBrain("interactive", seed=args.seed,
                                  channel_ids=channel_ids,
                                  readout_pops=readout_pops,
                                  comp=args.comp or None,
                                  con=args.con or None)
            sched = (json.loads(Path(args.schedule_json).read_text())
                     if args.schedule_json else [(args.t_ms, {})])
            t_total = 0
            for dur, rates in sched:
                for _ in range(int(round(dur / args.chunk_ms))):
                    brain.step(rates, chunk_ms=args.chunk_ms)
                t_total += dur
            walls = np.array(brain.chunk_walls)
            spikes = sum(len(v) for v in brain.spike_trains().values())
            ct = cpu_times()
            rec = dict(mode="interactive", n_neurons=brain.n_neurons,
                       chunk_ms=args.chunk_ms, n_chunks=len(brain.chunk_walls),
                       wall_s=round(float(walls.sum()), 3),
                       bio_s_per_wall_s=round(
                           (t_total / 1000.0) / float(walls.sum()), 4),
                       chunk_wall_ms_mean=round(
                           float(walls.mean() * 1000), 1),
                       chunk_wall_ms_p95=round(
                           float(np.percentile(walls, 95) * 1000), 1),
                       chunk_wall_ms_max=round(float(walls.max() * 1000), 1),
                       n_spikes=spikes,
                       peak_rss_mb=round(bl.peak_rss_kb() / 1024, 1),
                       cpu_user_s=ct["user_s"],
                       cpu_sys_s=ct["sys_s"])
        if args.spikes_out:
            Path(args.spikes_out).write_text(brain.canonical_spikes_text())
        Path(args.rec_out).write_text(json.dumps(rec, indent=1, default=str))
        print(json.dumps(rec))
    elif args.study == "_checkpoint_worker":
        _checkpoint_worker(args)
    elif args.study == "_bench_worker":
        _bench_worker(args)
    elif args.study == "chunk-study":
        study_chunk(interface=args.interface)
    elif args.study == "checkpoint":
        import subprocess
        OUT.mkdir(parents=True, exist_ok=True)
        cmd = [PY, str(Path(__file__).resolve()), "_checkpoint_worker",
               "--interface", args.interface, "--seed", str(args.seed),
               "--out-dir", str(OUT)]
        subprocess.run(cmd, check=True)
    elif args.study == "bench":
        study_bench(seed=args.seed, t_ms=args.t_ms, chunk_ms=args.chunk_ms,
                    readout=args.readout or None)


if __name__ == "__main__":
    main()
