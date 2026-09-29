# CURRENT IMPLEMENTATION AUDIT

Date: 2026-09-30

This file separates code that exists today from the target architecture.

## Working today

- Canonical Shiu/FlyWire v783 whole-brain runtime in interactive Brian2 mode.
- Cython code generation for the live interactive brain runtime.
- Windows game-window capture.
- Replay logging and timestamped state channels.
- Fast/heavy perception worker interfaces.
- Fly-channel encoder and intention decoder.
- Planner scaffold and passive action pipeline.
- OpenCV dashboard showing game capture plus fly/helper/action state.
- Canonical Brian2 brain isolated in its own subprocess for Windows live runs.
- One-command persistent passive launcher: `run_live_windows.ps1`.
- Lightweight temporal tracklets for GPO humanoid/quest-NPC candidates.

## Prototype or incomplete

- GPO perception currently recognizes player/camera anchor, quest-marker
  evidence, health/stamina, and tracked humanoid candidates. Hostile-vs-
  nonhostile/boss classification and combat-state estimation are not validated.
- The planner is a small deterministic goal rule set, not a complete
  quest/navigation/progression system.
- The ability resolver exists as infrastructure but is not yet a validated
  live combat system.
- The dashboard is an engineering prototype.
- The canonical brain is still slower than real time on the target laptop
  under the full live workload.

## Missing

- Full worker-process isolation. The canonical brain is now a subprocess,
  while capture/perception/planner/executor/replay/dashboard still share the
  main process.
- Validated hostile-vs-nonhostile/boss classification.
- Validated combat perception.
- Completed Gate-5 real-game assessment.
- Gate-6+ active movement/combat.
- Final one-click packaged application.

## Corrections to older design text

- Brian2 C++ standalone is an offline replay/verification path, not the
  current interactive live brain backend.
- Some older comments describe a future subprocess/multi-process Windows
  runtime that is not implemented yet.

## Next priorities

1. Validate the new GPO tracklets against real gameplay and add role evidence.
2. Add hostile/attack/combat-state classification without inventing unknowns.
3. Gate-5 assessment only after perception is meaningful.
