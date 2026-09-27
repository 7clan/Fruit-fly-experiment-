# brain/ — the Digital Drosophila brain track (D1–D6)

Governing spec: `docs/MASTER_SPEC.md`. Roadmap: `docs/ROADMAP.md` (Stage I).
Third-party model + data: `THIRD_PARTY.md`.

## Status

| Step | State |
|---|---|
| D1 install/reproduce Shiu model | DONE — pinned `91bdd1e`, manifest PASS, tutorial reproduced (v630) |
| D2 FlyWire v783 configured | DONE — 138,639 neurons / 15,091,983 connections, SHA-256 verified |
| D1/D2 verification runs | `results/repro_783/` + `results/spikes_783/` |
| D3 benchmark | `results/benchmark/benchmark_783.{json,md}` |

(Updated as the runs complete — see `results/` and the gate verdicts.)

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
