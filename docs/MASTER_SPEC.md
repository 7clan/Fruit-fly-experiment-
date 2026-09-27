# MASTER PROJECT SPECIFICATION (reconciled)

Status: ACTIVE — supersedes every earlier scope statement in this repository.
Date of correction: 2026-09-28. This document reconciles (1) the MASTER
PROJECT CONTEXT correction with (2) the Windows Runtime / User Experience
addendum (preserved verbatim-structured in
`docs/WINDOWS_RUNTIME_SPEC.md`). Where the two overlap, both apply; where
they sequence work, the D1–D24 order below governs implementation priority.

---

## 1. Project correction (read first)

**There is NO real biological fruit fly in this project.**

The entire experiment is DIGITAL and runs on the user's laptop. We do not
build: a physical arena, cameras for observing an animal, LEDs for a real
animal, animal experiments, animal-care workflow, biological recording
apparatus. All prior real-fly / physical-experiment work is preserved under
`legacy/` as history (see `legacy/README.md`) and is no longer active.

**Corrected objective:** build a simulated Drosophila neural agent — derived
as closely as practical from real fruit-fly neuroscience — and allow it to
eventually learn and play a video game (Grand Piece Online on Roblox).

## 2. Terminology discipline (mandatory labeling)

Every component must be labeled as exactly one of:

| Label | Meaning |
|---|---|
| **connectome data** | Biological connectome data (FlyWire v783) — measured, public |
| **neuronal dynamics** | Modeled dynamics (Shiu et al. LIF whole-brain model) |
| **engineered sensory interface** | CV → sensory-population stimulation we design |
| **engineered motor interface** | neural-activity → action decoding we design |
| **added learning** | Plasticity / RL we add (stages A–D, §7) |
| **external systems** | Planner, memory, world model outside the brain |

We never claim this is an exact simulated consciousness or an exact
biological brain. We never create an ordinary neural network and merely
call it a fruit fly. Reports always attribute behavior using these labels.

## 3. Canonical digital Drosophila brain

- **Primary scientific foundation:** Shiu et al. whole-brain Drosophila
  leaky-integrate-and-fire computational model —
  `https://github.com/philshiu/Drosophila_brain_model`.
- **Connectome:** public FlyWire v783 adult Drosophila connectome, where
  supported.
- This model + connectome combination is THE canonical DIGITAL DROSOPHILA
  BRAIN for the experiment.
- Reproducibility policy: the third-party model is pinned by exact commit
  SHA; data files are pinned by SHA-256 manifest; any local patch is
  recorded in `brain/THIRD_PARTY.md` (target: zero patches).

## 4. Target architecture (the control path)

```
GAME SCREEN
  -> COMPUTER VISION
  -> BIOLOGICALLY INSPIRED SENSORY ENCODER
  -> DIGITAL DROSOPHILA BRAIN (Shiu LIF model on FlyWire v783)
  -> NEURAL ACTIVITY
  -> MOTOR DECODER
  -> GAME ACTION
  -> OBSERVED RESULT
  -> REWARD / MEMORY / LEARNING
  -> DIGITAL DROSOPHILA BRAIN  (closed loop)
```

The Drosophila neural system must remain MEANINGFULLY in the control path.
Bypasses such as `COMPUTER VISION -> DIRECT ACTION` are forbidden. External
planner/memory components may propose, but the brain's recurrent activity
remains a measured, central, non-bypassable computation (see control
conditions, §8).

## 5. Perception pipeline (computer vision is core)

The connectome does not process raw 1080p frames neuron-by-neuron. A
perception pipeline must eventually recognize/estimate: player, enemies,
NPCs, objects, movement, optic flow, health, level, progression/EXP,
dialogue, quest indicators, inventory, equipment, weapons, abilities/fruits,
accessories, menus, loading screens, navigation landmarks. Permitted
techniques: object detection, OCR, tracking, segmentation, template
matching, motion analysis. **Every CV output carries confidence
information.** The compact sensory representation is translated into
stimulation of appropriate Drosophila sensory/visual populations wherever
scientifically defensible (documented per population; no defensible mapping
=> flagged, not faked).

## 6. Game source policy (CORRECTED)

- The ONLY Roblox URL permitted:
  `https://www.roblox.com/games/1730877806/Grand-Piece-Online`
  (the original `/fr/` locale form of the same page remains registered as
  the historical permitted source). No other Roblox URL, documentation,
  API, Creator page, support page, forum, or experience may be accessed.
- **Non-Roblox websites and resources ARE allowed** to learn about Grand
  Piece Online: independent GPO wikis, guides, YouTube gameplay/tutorials,
  community discussions, independent databases, articles — covering
  controls, menus, quests, NPCs, enemies, islands, leveling, weapons,
  fighting styles, fruits, accessories, bosses, drops, navigation, combat,
  progression.
- Third-party information is NOT automatically trusted. It is stored as
  **UNVERIFIED GUIDE KNOWLEDGE** and upgraded to **VERIFIED IN GAME** only
  after direct observation confirms it. If a guide conflicts with current
  direct game observation, the direct observation wins. (Knowledge-base
  schema with these two tiers is built at D17.)

## 7. Learning (staged, experimentally)

The published model is not automatically a general-purpose RL gamer.

- **Stage A — fixed connectome.** No learning. Baseline behavior.
- **Stage B — trainable output/readout.** Connectome fixed; train only the
  neural-output-to-action layer.
- **Stage C — Drosophila-inspired plasticity.** Investigate reputable
  research on mushroom bodies, Kenyon cells, MBONs, PAM dopamine neurons,
  PPL1 dopamine neurons, reward, punishment, associative memory,
  approach/avoidance, valence. Add biologically motivated plasticity where
  scientifically defensible.
- **Stage D — hybrid reinforcement learning.** Constrained RL/planning
  components while the Drosophila brain remains a measurable central
  recurrent computation.

Every engineered addition is labeled (§2).

## 8. Scientific control conditions

Maintained as separate, comparable conditions:

| Cond | Controller |
|---|---|
| A | random |
| B | conventional artificial RL agent |
| C | fixed Drosophila connectome controller |
| D | adaptive Drosophila controller |
| E | hybrid Drosophila + CV + planner + memory |

Compared on: learning speed, navigation, task completion, progression,
reward, survival, combat performance, action efficiency, computational
cost. This is what determines what the Drosophila-derived architecture
actually contributes.

## 9. Internal state / "emotion" display rules

The dashboard MAY display internal motivational variables but MUST NOT
pretend the system literally experiences human emotions. Allowed
variables: positive valence, negative valence, reward expectation, reward
prediction error, novelty, avoidance/threat drive, exploration drive,
uncertainty, arousal-like state. Friendly labels ("curious-like",
"threat-like") are allowed ONLY alongside the underlying numerical/
computational variable. Never fabricate natural-language thoughts. Display
actual computational information: OBSERVATION, CURRENT GOAL, MEMORY
RETRIEVED, ACTIVE NEURAL POPULATIONS, POSSIBLE ACTIONS, SELECTED ACTION,
EXPECTED REWARD, ACTUAL REWARD, PREDICTION ERROR.

## 10. Experiment account policy

The user prepares a dedicated game account personally. We never: create
accounts, store passwords, automate login credentials, bypass
authentication. The user logs in normally before experiments. The launcher
may later open ONLY the authorized game URL.

## 11. Windows target (addendum summary — full text in docs/WINDOWS_RUNTIME_SPEC.md)

- Final machine is WINDOWS; the user-facing application is
  **DigitalFlyLab.exe** (self-contained folder variant allowed). Linux dev
  runs portable tests only; Windows capture/input/packaging tests MUST run
  on Windows before those features are considered complete.
- Roblox remains a normal separate Windows process. NO embedding of the
  Roblox renderer; instead: Roblox window -> Windows window capture
  (Windows.Graphics.Capture preferred) -> live frame stream -> dashboard
  game-view panel. ONE capture pipeline serves BOTH computer vision and
  the dashboard preview.
- Isolated platform layer `platform/windows/` (window_finder,
  window_capture, input_controller, process_launcher, focus_manager); all
  brain/CV/memory/planning code stays platform-independent.
- Action path: DROSOPHILA MOTOR OUTPUT -> ACTION DECODER -> WINDOWS INPUT
  ADAPTER -> TARGET GAME WINDOW. Every emitted action is logged. A GLOBAL
  EMERGENCY STOP HOTKEY releases all held keys immediately.
- Start flow: load connectome/brain -> brain self-test -> CV self-test ->
  user START EXPERIMENT -> launcher (authorized URL only, if needed) ->
  detect window -> capture -> dashboard -> CV -> sensory encoding -> brain
  -> action -> input -> result -> reward -> memory/learning -> loop.
- Dashboard: default OBSERVER layout (left 60% live game + CV overlays;
  right 40% brain viz + internal states + goal/action/reward; bottom
  memory / learning curve / progression / event log). Selectable layouts:
  OBSERVER, GAME, BRAIN, SCIENTIST, REPLAY. Dashboard shows REAL internal
  values only — never invented decorative activity.
- Every run records synchronized replay data (frames, CV, sensory
  encodings, neural activity, actions, emitted inputs, rewards,
  motivational state, memories, world-model updates, goals, progression,
  timestamps, configuration, brain/checkpoint version); REPLAY mode
  scrubs to any timestamp.
- Packaging scripts: `setup_windows.ps1`, `build_windows.ps1`,
  `run_dev_windows.ps1`; Windows executables are built AND tested on
  Windows.

## 12. Development order D1–D24 (implementation priority)

Do NOT start by automating Roblox.

| Step | What | Status |
|---|---|---|
| D1 | Install + reproduce the Shiu Drosophila brain model | **IN PROGRESS** |
| D2 | Configure/test FlyWire v783 data | **IN PROGRESS** |
| D3 | Benchmark whole-brain sim (RAM, CPU, init time, sim speed, neurons, synapses) | **IN PROGRESS** |
| D4 | Identify sensory/visual neural populations | pending |
| D5 | Identify mushroom-body/dopamine learning circuitry | pending |
| D6 | Identify descending/motor-related populations | pending |
| D7 | Simple artificial visual environment | pending |
| D8 | Sensory encoder | pending |
| D9 | Motor decoder | pending |
| D10 | Closed-loop target navigation (artificial env) | pending |
| D11 | Reward-modulated learning | pending |
| D12 | Internal-state visualization | pending |
| D13 | Memory + multi-step artificial tasks | pending |
| D14 | Windows dashboard | pending |
| D15 | Windows window capture | pending |
| D16 | Passive game computer vision | pending |
| D17 | Non-Roblox GPO knowledge database (UNVERIFIED/VERIFIED tiers) | pending |
| D18 | Game movement | pending |
| D19 | Interactions/combat | pending |
| D20 | Quest/level progression | pending |
| D21 | Equipment/fruits/accessories | pending |
| D22 | Exploration + strategy selection | pending |
| D23 | Long-term autonomous progression | pending |
| D24 | Optional launcher/menu automation | pending |

Gating rules: no complete dashboard before the brain benchmark (D3);
Windows-specific work only at D14+; no Roblox input before the brain gate
passes (§13) and the user's own ToS/compliance review is recorded.

## 13. Gate 1 (current)

> **THE CONNECTOME-DERIVED DIGITAL DROSOPHILA BRAIN RUNS LOCALLY AND
> PRODUCES REPRODUCIBLE NEURAL ACTIVITY.**

Pass criteria (all required, honestly reported):
1. Shiu model installed from the pinned commit; environment documented.
2. FlyWire v783 connectome data configured (manifest-verified).
3. The model's example/tutorial runs.
4. Spike propagation is demonstrated (stimulus-evoked activity reaches
   downstream neurons; evidence recorded).
5. Reproducibility: two runs with identical seeds produce identical
   (or bit-comparable) neural activity.
6. D3 benchmark metrics recorded (RAM, CPU, init time, simulation speed,
   neuron count, synapse/connection count).

No Roblox control begins until this gate passes.

## 14. Scientific integrity rules (unchanged in spirit)

- Never claim results not demonstrated; every report separates what is
  connectome data / modeled dynamics / engineered interface / added
  learning / external systems.
- Fail closed and report honestly (PASS/FAIL), including third-party
  model failures.
- Reproducibility discipline: pinned SHAs, hashes, seeds, versions.
- No fabricated natural-language consciousness, ever.
