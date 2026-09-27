# THIRD_PARTY — the canonical Digital Drosophila brain

## What it is

The **Shiu et al. whole-brain Drosophila leaky-integrate-and-fire
computational model**:

- Repository: https://github.com/philshiu/Drosophila_brain_model
- Paper: "A leaky integrate-and-fire computational model based on the
  connectome of the entire adult Drosophila brain reveals insights into
  sensorimotor processing" (bioRxiv 2023.05.02.539144)
- License: MIT (Philip Shiu and Nico Spiller)
- **Pinned commit: `91bdd1e7dcf193f3e7ca5a8933497fcef63b7960`**
- All 19 tracked files SHA-256-pinned in
  `brain/setup/THIRD_PARTY_MANIFEST.json` (verify:
  `python brain/setup/setup_model.py verify`)

Terminology (MASTER_SPEC §2): the connectome parquet/csv files are
**biological connectome data** (FlyWire); `model.py`'s LIF equations and
parameters are **modeled neuronal dynamics**. Together they form the
canonical brain. Everything we add around them (encoders, decoders,
learning, planner) is a labeled engineered addition.

## Data bundled in the pinned repo

| File | Content |
|---|---|
| `2023_03_23_completeness_630_final.csv` | FlyWire v630 neuron list (127,400) — paper version |
| `2023_03_23_connectivity_630_final.parquet` | FlyWire v630 connectivity (paper version) |
| `Completeness_783.csv` | FlyWire **v783** neuron list (**138,639**) — canonical for us |
| `Connectivity_783.parquet` | FlyWire **v783** connectivity (**15,091,983** connections) |
| `sez_neurons.pickle` | dict: 106 SEZ cell-type names -> flywire IDs (v630 era) |

D2 status: the v783 files ARE the public FlyWire v783 release as
distributed by the model authors; manifest-verified, no external download
needed.

## Environment — pinned, with an honest deviation

| | Original `environment.yml` | This project |
|---|---|---|
| Python | 3.10 | **3.12.14** (no 3.10 available here; scripts are stack-portable) |
| Brian2 | 2.5.1 | **2.10.1** (newest supporting 3.12) |
| numpy | 1.24 | **2.5.3** |
| pandas | (unpinned) | **3.0.6** |

Consequences, stated plainly:

- We do NOT claim bitwise equality with the paper's original results.
- Our reproducibility claim (Gate 1) is: **two runs of the same seed on
  THIS pinned stack produce identical neural activity** — verified by
  `brain/scripts/verify_reproducibility.py`.
- If the user's Windows machine later needs different pins, re-verify
  Gate 1 on that stack before any D4+ work.

## Zero-patch policy

`model.py` and `utils.py` are imported **unmodified** (the manifest would
fail on any change). What the published code does not have — explicit
RNG seeding, in-process serial trials, timing/RSS instrumentation,
connectome-graph utilities — lives in OUR engineered harness
(`brain/scripts/brainlib.py`) and is labeled as engineered, not as part
of the brain.

Notable behavior of the published code (kept as-is):

- `run_exp()` is **unseeded** (Poisson stimulation is stochastic by
  design); the tutorial reproduction therefore demonstrates functioning,
  not bit-reproducibility.
- Every trial rebuilds the network from the dataframes (no state carry
  over) — convenient for our per-trial seeding.
- `silence()` zeroes outgoing synapses of the silenced neurons.
- The README of the model says default stimulation is 200 Hz; the code
  default is `r_poi = 150 Hz` — we follow the code and note the
  discrepancy.
