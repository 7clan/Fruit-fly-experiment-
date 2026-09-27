# DIGITAL BRAIN REPORT — D1–D3 (Gate 1)

Date: 2026-09-28. Governing specification: `docs/MASTER_SPEC.md`.

## Verdict

> **GATE 1: PASS — the connectome-derived digital Drosophila brain runs
> locally and produces reproducible neural activity.**

All six pass criteria of MASTER_SPEC §13 were met; every deviation is
listed honestly below. No Roblox access occurred; no game control exists
yet; nothing in this report claims biological equivalence or
consciousness.

## What was done

| # | Criterion | Result | Evidence |
|---|---|---|---|
| 1 | Shiu model installed from pinned commit | PASS | `brain/setup/THIRD_PARTY_MANIFEST.json`: commit `91bdd1e`, 19 files SHA-256 verified (`setup_model.py verify` → PASS) |
| 2 | FlyWire v783 configured | PASS | 138,639 neurons, 15,091,983 connections (bundled in the pinned repo, manifest-verified) |
| 3 | Example/tutorial runs | PASS | `results/example_630/`: 401 active neurons (notebook: "about 400"); MN9 motor neuron driven (82.3 Hz @150 Hz, 74.7 Hz @100 Hz); silencing experiment changes MN9 rate |
| 4 | Spike propagation demonstrated | PASS | `results/spikes_783/verdict.json` — checks c1–c4 all true (below) |
| 5 | Reproducible neural activity | PASS | `results/repro_783/verdict.json` — identical across processes AND script invocations |
| 6 | D3 benchmark | PASS | `results/benchmark/benchmark_783.json` (below) |

## Evidence details

### Tutorial reproduction (v630, as the paper shipped it)

Experiment sequence of `example.ipynb`, each experiment in its own process,
n_run=3: activation of the 21 sugar-sensing neurons (code default 150 Hz),
activation at 100 Hz, silencing of the single most active neuron.

- Active neurons: 401 (150 Hz) / 375 (100 Hz) / 369 (silenced) — matches
  the notebook's "~400 neurons show activity".
- MN9 (motor neuron, flywire 720575940660219265): 82.3 Hz → 74.7 Hz →
  72.7 Hz across the three experiments.
- Silenced neuron: 720575940622695448 (not itself a stimulated sugar
  neuron; the top active one at 100 Hz in our reduced run).

### Reproducibility (canonical v783, engineered seeded harness)

Each trial = fresh process, `brian2.seed` + numpy seed, sugar set (20 of
21 IDs present in v783), 1000 ms, 150 Hz.

| seed | rep0 sha256 | rep1 sha256 | identical |
|---|---|---|---|
| 11 | `9e3acaa361f5e477…` | `9e3acaa361f5e477…` | YES (12,533 spikes, 370 active) |
| 12 | `d6f5457e764d524a…` | `d6f5457e764d524a…` | YES (12,482 spikes, 372 active) |

- Seeds 11 vs 12 differ → the comparison is sensitive, not trivial.
- A second full script invocation reproduced seed 11's spike file
  byte-identically (cross-invocation identical: true).
- Stated plainly: the published model code is UNSEEDED; reproducibility
  comes from the engineered harness (per-trial seeding) around the
  unmodified model.

### Spike propagation (canonical v783, seed 101, 1000 ms, 150 Hz)

| Check | Result |
|---|---|
| c1 negative control (no stimulation) | 0 spikes, 0 active — network silent at rest |
| c2 propagation beyond stimulated set | 353 of 373 active neurons are NOT stimulated; hop depths: 85 at 1 hop, 255 at 2 hops, 13 at 3 hops |
| c3 motor readout | MN9 fires at 80 Hz |
| c4 causality (silence top stimulated neuron, sugar_15 = 720575940617937543) | total spikes 13,105 → 12,120; MN9 80 Hz → 68 Hz |

Multi-synaptic propagation through the biological connectome is therefore
demonstrated: activity evoked in 20 sensory neurons reaches neurons 2–3
synaptic hops away and drives a motor neuron, and cutting one stimulated
neuron's outgoing synapses measurably reduces the downstream response.

### D3 benchmark (canonical v783, this machine: 2 cores, ~4 GB RAM)

| Metric | Value |
|---|---|
| Neurons | 138,639 |
| Connections (synapse rows) | 15,091,983 |
| Data load (pandas) | 0.72 s |
| Network build (per process) | ~1.4 s |
| Full init per fresh process (imports+load+build+run) | 10.26 s mean |
| Peak RSS per process | 2,915.8 MB mean (2,921.5 max) |
| Simulation speed (stimulated, codegen cache warm) | **0.148 bio-s per wall-s** (1 s of biological time ≈ 6.7 s wall) |
| Spikes / active neurons per 1 s trial | ~12.7–13.3k / ~370 |

## Deviations and incidents (all of them)

1. **Stack deviation**: Python 3.12.14 / Brian2 2.10.1 / NumPy 2.5.3 /
   pandas 3.0.6 instead of the model's environment.yml (Python 3.10 /
   Brian2 2.5.1 / NumPy 1.24). Reason: no Python 3.10 available here.
   Consequence: no claim of bitwise equality with the paper's original
   outputs; our reproducibility claim is over THIS pinned stack.
2. **Tutorial reductions**: n_run=3 instead of 30; n_proc=1 instead of -1;
   top-1 instead of top-3 silencing. Reason: 2 cores / 4 GB RAM.
3. **v783 stimulus set**: 20 of 21 tutorial sugar IDs exist in v783 (flywire
   `720575940620900446` was re-rooted between materializations); the
   missing one is skipped and recorded in every trial record.
4. **Process-isolation rewrites**: three silent OOM kills taught us (a)
   repeated whole-brain rebuilds in one process exceed 4 GB; (b) a parent
   holding the 15M-row parquet (pyarrow retains pages) dies while a 2.9 GB
   child runs. The harness now runs every trial in a fresh subprocess and
   keeps parents light. These are engineered-harness engineering facts,
   recorded in `brain/README.md`.
5. **Model README vs code**: the model's README says default stimulation
   is 200 Hz; the code default is `r_poi = 150 Hz`. We follow the code.
6. **Warm-cache assumption**: benchmark timings exclude the one-time Cython
   codegen compile (~40 s, paid once per machine).

## Interpretation (with mandatory labels)

- [connectome data + modeled neuronal dynamics] The canonical brain —
  Shiu et al. LIF model on FlyWire v783 — runs, responds to stimulation,
  propagates activity multi-synaptically, drives motor readout neurons,
  and (under our engineered seeding) reproduces bit-identically.
- [engineered harness] Seeding, process isolation, timing, memory
  discipline, and all verification logic are OUR additions; the
  third-party model is imported unmodified.
- Nothing here demonstrates game-playing, learning, or cognition, and the
  brain does NOT yet receive game-derived sensory input. D4–D6 (population
  inventories) are the next roadmap steps; D7+ builds the artificial
  environment closed loop.
- **Speed implication for D7+**: at 0.148 bio-s/wall-s on this 2-core box,
  full-scale real-time closed loop is not feasible here as-is. Options to
  evaluate at D7 (not decided now): Brian2 C++ standalone mode, coarser
  integration, running the brain faster than biological time with matched
  sensory timescales, or the user's Windows laptop (likely faster; the
  benchmark script is portable and will be run there).

## Reproduction

```bash
cd brain
python setup/setup_model.py clone && python setup/setup_model.py verify
python setup/setup_model.py env
.venv/bin/python scripts/run_example.py --stage exp1   # then exp2, exp3, summarize
.venv/bin/python scripts/verify_reproducibility.py --out results/repro_783
.venv/bin/python scripts/verify_reproducibility.py --seeds 11 --reps 1 \
    --out results/repro_783_check --compare-against results/repro_783
.venv/bin/python scripts/verify_spikes.py
.venv/bin/python scripts/benchmark.py
```

The v630 tutorial stage is stochastic (unseeded by design, see deviation
notes); the v783 verification battery is fully seeded and reproduces
bit-identically on the pinned stack.
