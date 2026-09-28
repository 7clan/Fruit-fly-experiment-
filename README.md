# DigitalFlyLab — a connectome-derived Drosophila brain learns to play a game

**A simulated Drosophila neural agent — derived as closely as practical
from real fruit-fly neuroscience — eventually learns and plays Grand Piece
Online (Roblox), with the Drosophila brain model meaningfully in the
control path.**

> ### Project status — read this first
>
> - **There is NO real biological fruit fly** (project correction,
>   2026-09-28). The entire experiment is digital and runs on a laptop.
>   The earlier physical real-fly direction is ARCHIVED under `legacy/`
>   as history — no real-fly data was ever collected, and no game input
>   has ever been sent.
> - **Canonical digital brain:** the Shiu et al. whole-brain Drosophila
>   leaky-integrate-and-fire model
>   (github.com/philshiu/Drosophila_brain_model, pinned commit, MIT) on
>   the public FlyWire v783 adult connectome: **138,639 neurons,
>   15,091,983 connections**. We do NOT create an ordinary neural network
>   and call it a fruit fly. Terminology labels (connectome data /
>   modeled dynamics / engineered sensory interface / engineered motor
>   interface / added learning / external systems) are mandatory in every
>   report. We never claim exact biological equivalence or consciousness.
> - **GATE 1: PASS (2026-09-28)** — the connectome-derived digital
>   Drosophila brain runs locally and produces REPRODUCIBLE neural
>   activity: seeded trials are bit-identical across processes and
>   invocations; stimulation propagates multi-synaptically (255 active
>   neurons at 2 hops) and drives the MN9 motor neuron; benchmark
>   recorded. Full record: `DIGITAL_BRAIN_REPORT.md`.
> - **GATE 2: PASS (2026-09-28)** — a verified SENSORY -> DIGITAL
>   DROSOPHILA -> MOTOR path exists and can be stimulated/read
>   reproducibly: looming (LPLC2+LC4) drives the giant-fiber escape
>   neurons (~110 Hz, 63 DNs active); visual-target (LC9) recruits 92
>   DNs; bit-identical cross-process re-runs; causal entry-off control
>   collapses downstream activity to zero. Neural I/O map for D4
>   (sensory), D5 (mushroom body/dopamine), D6 (descending/motor) with
>   v783 IDs: `D4_D6_NEURAL_IO_MAP.md` + machine-readable
>   `brain/data/io_map/`. Performance study: full-brain numpy reference
>   0.094 bio-s/wall-s @ 2.8 GB; exact sparse-active mode; C++ standalone
>   binary 0.283 bio-s/wall-s @ 678 MB (3.1x, bit-identical).
> - **GATE 3: PASS (2026-09-28)** — FIRST AUTONOMOUS CLOSED-LOOP CONTROL:
>   the intact connectome-derived Drosophila brain (fixed, no learning)
>   perceives a visual target through the verified LC9 sensory interface
>   and steers the agent to it in a REAL closed loop — 9/9 target-approach
>   successes + 3/3 looming escapes, while EVERY control degrades to 0/9
>   (random, shuffled-sensory, shuffled-motor; bootstrap CI [1.00,1.00]).
>   Chunked interactive stepping is bit-identical to the canonical
>   reference at 25/50/100/200 ms; episodes replay bit-exactly (also in
>   the 3x-faster C++ standalone route). Full record:
>   `D7_D10_FIRST_CONTROL_REPORT.md` + `brain/results/d10_closed_loop/`
> - **D11–D13 (2026-09-28): reward-modulated learning implemented and
>   honestly evaluated — GATE 4 NOT PASSED (v1).** The
>   mushroom-body/dopamine plasticity pathway is implemented on top of
>   the Gate-3 runtime (KC→MBON dopamine-gated depression at the
>   connectome's own 62,261 synapses; reward enters ONLY as PAM
>   stimulation; behaviour is read ONLY from DN/MBON spikes). The
>   machinery measurably works (weights + MBON rates change
>   reward-specifically), BUT the pre-registered behavioural criteria
>   FAILED: the frozen brain's innate wiring lean dominated every test
>   choice (B − A = 0.000 [0,0]; extinction flat; reversal 0%).
>   No post-hoc tuning was applied; the mechanistic diagnosis and the
>   pre-registered v2 follow-up are in `D11_D13_LEARNING_REPORT.md`.
>   No Roblox connection was made (blocked before Gate 4).
>   (+ per-episode dashboard videos). Next: D11 (Drosophila-inspired
>   reward-modulated learning). Still no Roblox / CV integration.
> - **Windows target:** the finished application is `DigitalFlyLab.exe`
>   (window-capture mirror of the Roblox process, CV overlays, neural +
>   internal-state visualization, replay). Full addendum:
>   `docs/WINDOWS_RUNTIME_SPEC.md`; implementation gated at D14–D15.
> - **Source policy (corrected):** only the single Grand Piece Online
>   Roblox page may be accessed on Roblox properties; NON-Roblox guides/
>   wikis/videos are allowed as **UNVERIFIED GUIDE KNOWLEDGE**, upgraded
>   to **VERIFIED IN GAME** only by direct observation
>   (`docs/GPO_PLAN.md`).

Governing specification: **`docs/MASTER_SPEC.md`** (reconciles the project
correction with the Windows runtime addendum).

---

## Quick start

```bash
pip install -r requirements.txt

# 1. verify the (legacy-compatible) Phase-1 pipeline still passes
python -m pytest tests/test_pipeline.py -q

# 2. brain model setup + verification (D1-D3, see brain/README.md)
#    clone pinned Shiu model, configure FlyWire v783 data, run examples,
#    verify spike propagation + reproducibility, benchmark
python brain/setup/setup_model.py --help
```

## Repository layout

```
fly-agent/
├── docs/
│   ├── MASTER_SPEC.md          # THE governing spec (reconciled, 2026-09-28)
│   ├── WINDOWS_RUNTIME_SPEC.md # Windows addendum (authoritative for D14+)
│   ├── ROADMAP.md              # D1-D24 development order + gates
│   ├── GPO_PLAN.md             # game source policy + fact/UNKNOWN register
│   └── METHODOLOGY.md          # control-group + attribution methodology
├── brain/                      # ACTIVE: the digital Drosophila brain track
│   ├── README.md               # D1-D3 record: setup, run, reproduce
│   ├── THIRD_PARTY.md          # pinned model commit, license, patches
│   ├── setup/                  # reproducible clone/data scripts
│   ├── scripts/                # spike-propagation / reproducibility / benchmark
│   └── results/                # evidence artifacts + reports
├── flyagent/                   # Phase-1 synthetic apparatus (frozen baseline;
│                               #   2D arena + experiment framework reused at D7/D10)
├── legacy/                     # ARCHIVED physical real-fly direction
├── tests/                      # active test suite
├── main.py                     # Phase-1 CLI (frozen baseline reproduction)
└── runs/                       # immutable run directories
```

## The control path (what the project enforces)

```
GAME SCREEN -> COMPUTER VISION -> SENSORY ENCODER
  -> DIGITAL DROSOPHILA BRAIN (Shiu LIF model on FlyWire v783)
  -> NEURAL ACTIVITY -> MOTOR DECODER -> GAME ACTION
  -> OBSERVED RESULT -> REWARD / MEMORY / LEARNING -> (back to brain)
```

The Drosophila neural system stays meaningfully in the control path;
`COMPUTER VISION -> DIRECT ACTION` bypasses are forbidden. Scientific
control conditions A–E (random / conventional RL / fixed connectome /
adaptive Drosophila / hybrid) measure what the Drosophila-derived
architecture actually contributes (`docs/MASTER_SPEC.md` §8).

## Learning is staged, and everything added is labeled

A — fixed connectome (baseline, no learning) · B — trainable readout only ·
C — Drosophila-inspired plasticity (mushroom bodies, Kenyon cells, MBONs,
PAM/PPL1 dopamine, valence) where scientifically defensible · D — hybrid
RL/planning with the brain as the measured central recurrent computation.
Reports always separate biological connectome data from modeled dynamics
from engineered additions.

## Honesty rules

Never say the system "understands", "feels", or "experiences" anything.
The dashboard (D12/D14) shows actual computational variables (valence,
reward expectation, prediction error, novelty, threat/exploration drives,
uncertainty, arousal-like state) — friendly labels only alongside the raw
variable, never natural-language fabricated thoughts.

## Animal welfare (historical note)

No animals are used. The archived real-fly protocol was non-invasive by
design; it never ran. See `legacy/README.md`.

## Source policy for the target game

Only `https://www.roblox.com/games/1730877806/Grand-Piece-Online` may be
accessed on Roblox properties. Non-Roblox third-party resources are
permitted for learning GPO facts but are stored as UNVERIFIED GUIDE
KNOWLEDGE until verified by direct in-game observation; direct observation
always wins. Full register: `docs/GPO_PLAN.md`.
