# brain/ — the Digital Drosophila brain track (D1–D6)

Governing spec: `docs/MASTER_SPEC.md`. Roadmap: `docs/ROADMAP.md` (Stage I).
Third-party model + data: `THIRD_PARTY.md`.

## Status

| Step | State |
|---|---|
| D1 install/reproduce Shiu model | **DONE** — pinned `91bdd1e`, manifest PASS; tutorial reproduced on v630 (401 active neurons ≈ notebook's "~400"; MN9 driven; silencing changes MN9) |
| D2 FlyWire v783 configured | **DONE** — 138,639 neurons / 15,091,983 connections, SHA-256 manifest verified |
| D1/D2 verification (v783) | **GATE PASS** — `results/repro_783/verdict.json` (cross-process + cross-invocation identical, sha `9e3acaa361f5e477` for seed 11), `results/spikes_783/verdict.json` (negative control, 353 propagated neurons incl. 255 at 2 hops, MN9 80 Hz, silencing causality) |
| D3 benchmark | **DONE** — `results/benchmark/benchmark_783.{json,md}`: init 10.3 s/process, peak RSS 2.92 GB, 0.148 bio-s per wall-s (this 2-core box) |

**GATE 1: PASS** — the connectome-derived digital Drosophila brain runs
locally and produces reproducible neural activity. Full record:
`DIGITAL_BRAIN_REPORT.md`.

Operational notes learned here (matter for every later step):

- One whole-brain network per process, peak ~2.9 GB; NEVER hold big
  pandas/pyarrow buffers in a parent while trial children run (OOM).
- Trial children run serially on machines with < 8 GB RAM.
- The first-ever run pays a one-time Cython codegen compile (~40 s);
  every later run is warm (~10 s per fresh process end-to-end).

## One-time setup (reproducible)

```bash
python setup/setup_model.py clone      # pinned commit 91bdd1e
python setup/setup_model.py verify     # SHA-256 manifest, fails closed
python setup/setup_model.py env        # creates .venv with pinned versions
```

## Commands (all seeded, in-process, serial)

```bash
.venv/bin/python scripts/smoke_test.py              # 200 ms feasibility check
.venv/bin/python scripts/run_example.py             # tutorial (v630, reduced n_run)
.venv/bin/python scripts/verify_reproducibility.py  # GATE 1: identical-seed identity
.venv/bin/python scripts/verify_spikes.py           # GATE 1: propagation + causality
.venv/bin/python scripts/benchmark.py               # D3 metrics
```

## Machine constraints observed here (2 cores, ~4 GB RAM)

- Peak RSS per network build+run: ~2.9–3.0 GB → **serial trials only** on
  this box; the model's default `n_proc=-1` (parallel trials) would OOM.
- ~250 s wall per 1 s stimulated bio-second at v783 (measured; see
  benchmark for the exact numbers).
- Tutorial reproduction therefore uses n_run=3 (not 30) and silences the
  top-1 (not top-3) neuron. Deviations recorded in
  `results/example_630/tutorial_summary.json`.
- The user's Windows laptop will re-run `benchmark.py` for its own
  numbers (script is portable).
