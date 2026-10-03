# AI COMBAT TRAINING PIPELINE

Status: **ACTIVE EXPERIMENTAL WORKSTREAM** on ai-only-autopilot.

This work is intentionally separate from the digital-fly scientific result.
Gemini remains the AI-only branch's high-level gameplay decision owner.

## Why this exists

The 2026-10-03 AI-only runs proved that capture, Gemini planning, target
tracking, Windows input and compound combat can run end-to-end, but they also
showed three structural weaknesses:

1. too much generic recovery during combat;
2. no persistent gameplay trajectory/skill learning across runs;
3. a multi-second cloud planner being asked to rediscover low-level behavior.

The training pipeline addresses those problems without pretending that a
prompt tweak is equivalent to learned skill.

## What is now recorded automatically

Every AI-only run writes a compact trajectory at:

runs/<session>/training/trajectory.jsonl

and a persistent copy at:

runtime_state/training/trajectories/<session>.jsonl

Each sample joins:

- structured world observation;
- quest state;
- Gemini plan;
- local command;
- AI visual track;
- conservative reward components;
- current autonomy metadata.

The persistent root is gitignored so updates/pulls do not erase experience.

## Persistent skill library

runtime_state/training/skills.json

contains stable skill IDs, preconditions, success/failure conditions and
measured segment outcomes. The skill library is injected back into Gemini's
live prompt as **past measured experience**, never as current visual truth.

Initial skills include navigation, quest acceptance, quest combat, block,
evade, jump/climb recovery and combat-target reacquisition.

## Human Teacher Mode

Run:

    .\run_teacher_windows.ps1

You control the character. DigitalFlyLab records only the verified gameplay
controls (W/A/S/D, combat keys, hotbar, M1/RMB state) while the authorized game
window is foreground. It does **not** emit gameplay input and does not record
free-form keyboard text.

This creates state/action demonstrations suitable for behavior cloning and
later DAgger-style correction datasets.

## Reward labels

The online reward file is deliberately conservative:

- +100 observed quest completion;
- +10 observed quest-counter increment;
- -20 observed player death;
- -0.5 generic micro-recovery shaping penalty;
- -1 combat-recovery shaping penalty.

Every component is stored separately so shaping can be reweighted or removed
during offline training.

## Combat recovery change

A close visible enemy is no longer classified as "stuck" merely because
distance stays constant while orbiting.

When a fight target is temporarily lost, the local recovery bridge is now:

1. camera reacquire;
2. short target-centering approach;
3. bounded back-dash/reposition.

Generic CLIMB/JUMP are excluded from automatic combat recovery. They remain
available only when Gemini explicitly selects those skills.

## Export datasets

After collecting runs:

    .\.venv\Scripts\python.exe scripts\export_ai_training.py

Outputs under runtime_state/training/exports/:

- teacher_bc.jsonl — human state -> action examples;
- gemini_sft_candidates.jsonl — positive-outcome state -> plan examples;
- rlft_episodes.jsonl — plan/command/reward records.

These are explicit intermediate datasets. They are not automatically uploaded
to a tuning provider.

## Next training stages

1. Collect 20-60 minutes of clean human demonstrations.
2. Run AI-only sessions and collect successful + failed trajectories.
3. Train a small fast local combat policy from teacher demonstrations.
4. Add takeover/correction logging (DAgger).
5. Evaluate against fixed metrics: quest completions/hour, deaths/hour,
   generic-recovery rate, target uptime, stale-target attacks, XP/minute.
6. Only after the dataset is large/clean enough, export high-quality plan
   examples for Gemini supervised tuning and reward trajectories for RLFT.

The local combat policy should eventually operate at 20-60 Hz while Gemini
selects semantic goals/strategies at the slower cloud cadence.
