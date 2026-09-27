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
>   (github.com/philshiu/Drosophila_brain_model) on the public FlyWire
>   v783 adult connectome. We do NOT create an ordinary neural network
>   and call it a fruit fly. Terminology labels (connectome data /
>   modeled dynamics / engineered sensory interface / engineered motor
>   interface / added learning / external systems) are mandatory in every
>   report. We never claim exact biological equivalence or consciousness.
> - **Current phase: D1–D3** — install/reproduce the brain model,
>   configure FlyWire v783, benchmark. Gate 1: the connectome-derived
>   brain runs locally and produces REPRODUCIBLE neural activity. No
>   Roblox control until Gate 1 passes. See `docs/ROADMAP.md` (D1–D24).
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
