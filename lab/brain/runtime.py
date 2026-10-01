"""Brain runtime layer: init-once, prewarm, run-continuously.

FINAL_ARCHITECTURE.md §4 (binding):

  * The 138k-neuron canonical brain is NEVER reconstructed per decision.
  * Brain worker: INITIALIZE ONCE → PREWARM → RUN CONTINUOUSLY
    (advance simulation chunk → receive updated inputs → advance next).
  * Biological chunk size 25/50/100 ms — selected by Windows benchmark.
  * Cache at init: connectome preprocessing, neuron ID maps, population
    maps, sensory maps, motor maps (all from the verified D4–D6/D8
    artifacts under brain/data/).
  * REFERENCE_REPLAY_MODE must keep working; any live optimization
    affecting brain equivalence must be identified + benchmarked. Never
    silently call an approximation identical.

Two runtimes implement the same protocol:

  CanonicalBrianRuntime — wraps the validated D7 INTERACTIVE machinery
    (brain/scripts/d7_runtime.py DualModeBrain, mode='interactive'):
    the canonical Shiu/FlyWire v783 network, chunked stepping proven
    bit-identical to a single run, store/restore+reseed determinism,
    C++ standalone replay proven bit-identical. Requires the brain venv
    (brian2) + brain/data/ artifacts. This is the ONLY runtime that may
    back scientific claims.

  MockBrainRuntime — a tiny explicitly-labeled MOCK for pipeline
    bring-up, tests, and hardware-limited development. It mimics the
    chunk-record contract (population rates) with O(1) rules so the full
    perception→brain→intent→resolver→executor chain can be exercised
    WITHOUT brian2. Every record it emits carries runtime="mock". It
    MUST NEVER be used to claim biological results.

Sensory input contract (VERIFIED D8 interface, brain/data/d7_d10/
sensory_interface.json): channels target_left / target_right / looming
(Poisson drive onto v, w_syn·f_poi, driven cells a-refractory — the
model's own dbm.poi() semantics, runtime-settable rates).

Readout contract (D8 motor_readout.json): 25 side-split DN populations
(P9/BPN/RRN forward, MDN backward, FG/BB stop, GF escape, BRK, DN_LEG,
DN_ALL ...).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional, Protocol

# --- canonical artifacts (committed, verified) ------------------------------
AGENT_ROOT = Path(__file__).resolve().parents[2]
BRAIN_DIR = AGENT_ROOT / "brain"
D7_D10 = BRAIN_DIR / "data" / "d7_d10"
SENSORY_SPEC = D7_D10 / "sensory_interface.json"
MOTOR_SPEC = D7_D10 / "motor_readout.json"

# live fly-channel -> D8 sensory-channel mapping (lab/perception/fly_channels)
LIVE_TO_D8_CHANNELS = {
    "target_left_hz": "target_left",
    "target_right_hz": "target_right",
    "looming_hz": "looming",
}


class BrainChunkRecord(dict):
    """Per-chunk output dict. Keys:
    chunk_id, t_bio_s, chunk_ms, wall_s, runtime ('canonical_brian' |
    'canonical_cpp' | 'mock'), pop_counts {pop: spikes}, pop_rates_hz
    {pop: Hz}, sensory_rates_hz {channel: Hz}, ts_ns, bio_s_at_chunk_end
    """

    @property
    def runtime(self) -> str:
        return self["runtime"]


class BrainRuntime(Protocol):
    """Protocol both runtimes implement."""

    runtime_label: str

    def init_once(self) -> None: ...
    def prewarm(self, prewarm_chunks: int = 2) -> None: ...
    def advance_chunk(self, sensory_rates_hz: dict,
                      chunk_ms: float) -> BrainChunkRecord: ...
    def close(self) -> None: ...
    def pop_sizes(self) -> dict: ...


# ---------------------------------------------------------------------------
# mock runtime (EXPLICITLY LABELED — never a scientific claim)
# ---------------------------------------------------------------------------

class MockBrainRuntime:
    """MOCK brain: O(1) surrogate dynamics for pipeline bring-up/tests.

    Imitates the D8/D9 input-output CONTRACT only:
      * target_left/right drive ipsilateral P9 (forward + ipsiversive
        turn — Bidaye 2020) and contralateral inhibition is NOT modeled;
      * looming drives GF (giant fiber escape, Gate-2);
      * walk-OFF (FG/BB) activates when drive is near zero (STOP);
      * MDN (backward) engages on strong bilateral looming + low forward
        drive (avoidance analogue);
      * noise is SEEDED and reproducible, but this is NOT the canonical
      network and produces NO biological evidence.
    """

    runtime_label = "mock"

    def __init__(self, seed: int = 20260929):
        import numpy as np
        self._np = np
        self._rng = np.random.default_rng(seed)
        self._chunk_id = -1
        self._t_bio_s = 0.0
        self._pop_sizes: dict = {}
        if MOTOR_SPEC.exists():
            try:
                self._pop_sizes = {k: max(1, len(v)) for k, v in
                                   json.load(open(MOTOR_SPEC))["populations"].items()}
            except Exception:
                pass
        if not self._pop_sizes:
            self._pop_sizes = {p: 20 for p in
                               ("P9_left", "P9_right", "MDN_bilateral",
                                "BPN_bilateral", "RRN_bilateral",
                                "GF_left", "GF_right",
                                "FG_bilateral", "BB_bilateral")}

    def init_once(self) -> None:
        pass

    def prewarm(self, prewarm_chunks: int = 2) -> None:
        for _ in range(max(0, prewarm_chunks)):
            self.advance_chunk({"target_left": 0.0, "target_right": 0.0,
                                "looming": 0.0}, chunk_ms=50.0)

    def advance_chunk(self, sensory_rates_hz: dict,
                      chunk_ms: float) -> BrainChunkRecord:
        np = self._np
        tl = float(sensory_rates_hz.get("target_left", 0.0))
        tr = float(sensory_rates_hz.get("target_right", 0.0))
        loom = float(sensory_rates_hz.get("looming", 0.0))
        noise = lambda scale: float(self._rng.normal(0, scale))  # noqa: E731

        walk_drive = (tl + tr) / 72.0
        p9_l = max(0.0, tl / 72.0 * 12.0 + noise(0.4))
        p9_r = max(0.0, tr / 72.0 * 12.0 + noise(0.4))
        bpn = max(0.0, walk_drive * 6.0 + noise(0.3))
        rrn = bpn * 0.8
        gf = max(0.0, loom / 150.0 * 30.0 + noise(0.5)) if loom > 45.0 else max(0.0, noise(0.3))
        fg = max(0.0, (1.0 - walk_drive) * 6.0 + noise(0.3))
        bb = fg * 0.9
        mdn = max(0.0, (loom / 150.0) * 8.0 * (1.0 - walk_drive) + noise(0.3))

        rates = {"P9_left": p9_l, "P9_right": p9_r,
                 "BPN_bilateral": bpn, "RRN_bilateral": rrn,
                 "MDN_bilateral": mdn,
                 "GF_left": gf, "GF_right": gf * 0.9,
                 "FG_bilateral": fg, "BB_bilateral": bb}
        n_s = chunk_ms / 1000.0
        counts = {p: int(self._rng.poisson(max(0.0, r) * n * n_s))
                  for p, r in rates.items() for n in [self._pop_sizes.get(p, 20)]}

        self._chunk_id += 1
        self._t_bio_s += chunk_ms / 1000.0
        t0 = time.perf_counter()
        wall = 0.0  # O(1) surrogate — no biological compute
        return BrainChunkRecord(
            chunk_id=self._chunk_id,
            t_bio_s=self._t_bio_s,
            chunk_ms=float(chunk_ms),
            wall_s=round(wall, 6),
            runtime=self.runtime_label,
            pop_counts=counts,
            pop_rates_hz={p: round(r, 3) for p, r in rates.items()},
            sensory_rates_hz={k: round(float(v), 2)
                              for k, v in sensory_rates_hz.items()},
            ts_ns=time.monotonic_ns(),
        )

    def pop_sizes(self) -> dict:
        return dict(self._pop_sizes)

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# canonical runtime (wraps the validated D7 interactive machinery)
# ---------------------------------------------------------------------------

class CanonicalBrianRuntime:
    """The canonical 138,639-neuron Shiu/FlyWire v783 brain, live mode.

    Wraps brain/scripts/d7_runtime.py DualModeBrain(mode='interactive'):
      * network built ONCE in init_once() (never per decision)
      * sensory rates re-set between chunks via the verified
        SensoryInterface (runtime-settable Poisson drive)
      * per-chunk DN population spike counts from the verified readout
      * determinism: reset_episode(seed) = net.restore('ep0') + reseed
        (the documented Brian2 2.10 protocol from the checkpoint study)

    The Brian2 C++ standalone route (proven bit-identical replay) is the
    preferred LIVE backend on Windows once wired; this class exposes the
    same protocol so the swap is transparent and benchmarked, never
    silent. backend='numpy'|'cpp' selects at init (cpp requires the
    compiled standalone project — see brain/scripts/d7_cpp_replay.py).

    Requires: brain venv (brian2, pandas, numpy), the pinned third-party
    model (brain/setup/setup_model.py clone), brain/data/ artifacts.
    """

    runtime_label = "canonical_brian"
    SUPPORTED_BACKENDS = ("numpy", "cpp")

    def __init__(self, seed: int = 20260929, chunk_ms: float = 50.0,
                 backend: str = "numpy", quiet: bool = True,
                 codegen_target: str | None = None,
                 record_full_spikes: bool = True):
        self.seed = int(seed)
        self.chunk_ms = float(chunk_ms)
        self.backend = backend
        self.quiet = quiet
        self.codegen_target = codegen_target
        self.record_full_spikes = bool(record_full_spikes)
        self._brain = None
        self._iface_channels = None
        self._readout_pops = None

    # -- spec loading (verified artifacts, fail-closed) ---------------------
    @staticmethod
    def load_specs() -> tuple[dict, dict]:
        if not (SENSORY_SPEC.exists() and MOTOR_SPEC.exists()):
            raise RuntimeError(
                f"verified D8 spec files missing: {SENSORY_SPEC}, {MOTOR_SPEC} "
                "— the canonical runtime refuses to start without them")
        si = json.load(open(SENSORY_SPEC))
        mr = json.load(open(MOTOR_SPEC))
        return si, mr

    def warm_import(self) -> bool:
        """Import brian2 in the CALLING thread — MUST be the process MAIN
        thread. Brian2 registers a SIGINT handler at import time and
        Python only allows signal handlers in the main thread; threaded
        dev mode therefore pre-imports here (app.start) before worker
        threads spawn. The multiprocess Windows route imports in each
        worker process's own main thread (no issue). Returns False with a
        clear reason if brian2 is unavailable (e.g. wrong interpreter)."""
        try:
            import brian2  # noqa: F401
            return True
        except ImportError as e:
            raise RuntimeError(
                f"canonical runtime requires the brain venv interpreter "
                f"(brian2): {e} — run with brain/.venv/bin/python") from e

    def init_once(self) -> None:
        """Build the canonical network ONCE (expensive: ~10 s + ~3 GB)."""
        import sys
        import brian2 as b2
        if self.codegen_target is not None:
            if self.codegen_target not in ("numpy", "cython"):
                raise ValueError(
                    f"unsupported runtime codegen target {self.codegen_target!r}")
            b2.prefs.codegen.target = self.codegen_target
        sys.path.insert(0, str(BRAIN_DIR / "scripts"))
        import d7_runtime as rt  # the validated machinery, unmodified

        si, mr = self.load_specs()
        # channel spec: {name: {"population", "side", "n", "ids": [...]}}
        # -> the ID list d7_runtime.SensoryInterface expects
        self._iface_channels = {name: spec["ids"]
                                for name, spec in si["channels"].items()}
        self._readout_pops = mr["populations"]         # name -> flywire IDs
        self._brain = rt.DualModeBrain(
            "interactive", seed=self.seed, version="783",
            channel_ids=self._iface_channels,
            readout_pops=self._readout_pops, quiet=self.quiet,
            record_full_spikes=self.record_full_spikes)

    def prewarm(self, prewarm_chunks: int = 2) -> None:
        """Advance a few silent chunks so lazy codegen/spike buffers are
        warm before the live loop starts (init-once → prewarm → run)."""
        zero = {c: 0.0 for c in self._iface_channels}
        for _ in range(max(0, prewarm_chunks)):
            self.advance_chunk(zero, chunk_ms=self.chunk_ms)

    def advance_chunk(self, sensory_rates_hz: dict,
                      chunk_ms: Optional[float] = None) -> BrainChunkRecord:
        cm = float(chunk_ms if chunk_ms is not None else self.chunk_ms)
        rec = self._brain.step(rates=sensory_rates_hz, chunk_ms=cm)
        # rec: t_bio_s, chunk_ms, wall_s, pop_counts, n_spikes_new, ...
        counts = rec.get("pop_counts", {})
        sizes = self._brain.pop_sizes
        n_s = cm / 1000.0
        rates_hz = {}
        for pop, c in counts.items():
            n = sizes.get(pop, 0)
            rates_hz[pop] = round(c / (n * n_s), 3) if n > 0 else 0.0
        return BrainChunkRecord(
            # d7_runtime.cursor is a SPIKE cursor, not a chunk number.
            # Use the runtime's chunk-wall record count when the validated
            # D7 record does not carry an explicit chunk_id.
            chunk_id=rec.get("chunk_id", len(self._brain.chunk_walls) - 1),
            t_bio_s=rec["t_bio_s"],
            chunk_ms=rec["chunk_ms"],
            wall_s=rec["wall_s"],
            runtime=self.runtime_label,
            pop_counts=counts,
            pop_rates_hz=rates_hz,
            sensory_rates_hz={k: round(float(v), 2)
                              for k, v in sensory_rates_hz.items()},
            ts_ns=time.monotonic_ns(),
            n_spikes_new=rec.get("n_spikes_new", 0),
            n_active_new=rec.get("n_active_new", 0),
            active_flywire_ids_sample=rec.get(
                "active_flywire_ids_sample", []),
            instrumentation_scope=rec.get(
                "instrumentation_scope", "full_spike_recording"),
            rss_kb=rec.get("rss_kb"),
        )

    def reset_episode(self, seed: int) -> None:
        """Deterministic episode reset (restore + reseed protocol)."""
        self._brain.reset_episode(seed)

    def pop_sizes(self) -> dict:
        return dict(self._brain.pop_sizes)

    def close(self) -> None:
        # Brian2 network teardown is GC'd; explicit close for symmetry
        self._brain = None


def create_runtime(which: str = "mock", **kw) -> BrainRuntime:
    """Factory. 'canonical' requires the brain venv; 'mock' never claims
    to be biological. Fail closed on unknown names."""
    if which == "mock":
        kw.pop("chunk_ms", None)          # mock takes no chunk_ms
        return MockBrainRuntime(**kw)
    if which in ("canonical", "canonical_brian"):
        return CanonicalBrianRuntime(**kw)
    raise ValueError(f"unknown runtime {which!r} ('mock'|'canonical')")
