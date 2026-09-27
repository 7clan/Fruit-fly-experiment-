# LEGACY — the physical real-fly direction (ARCHIVED 2026-09-28)

**Status: ARCHIVED. Not the active project. Do not extend.**

On 2026-09-28 the project was corrected: there is NO real biological fruit
fly. The entire experiment is digital (see `docs/MASTER_SPEC.md`). This
directory preserves the earlier real-fly / physical-experiment work as
history. Nothing here was deleted; it was moved out of the active tree.

## What is archived here

| Path | What it was |
|---|---|
| `phase2/` | The real-fly behavioral-validation apparatus (`flyrec`): camera recorder, arena calibration, tracker, blank-arena checks, welfare gating, pre-registered Phase-2 gate (G1–G5), intake/backup discipline, and the synthetic pipeline-validation results (SA1–SA12) |
| `docs/PHASE2_START_GUIDE.md` | Bench procedure for the physical experiment (shopping list, arena build, camera/LED setup, recording procedure) |
| `docs/HARDWARE.md` | Camera/lens/LED/arena hardware notes for observing an animal |
| `docs/ROADMAP_REAL_FLY.md` | The original 15-phase roadmap (Phases 0–14 with the real fly in the loop from Phase 1 on) |
| `tests/test_phase2_apparatus.py` | The apparatus test suite (offline, synthetic patterns) |

## Historical record (matters for honesty)

- **No real-fly data was EVER collected.** Every real-fly outcome remained
  UNKNOWN; the Phase-2 gate never fired. The pipeline validation used
  synthetic test patterns only.
- **No game input was ever sent.** No game automation exists in the legacy
  code.
- The Phase-2 gate was pre-registered before any data (commit 520eb2f) and
  amended (Amendment 2) before any data — that discipline carries forward
  into the digital project.
- Phase 1 (synthetic fly, 2D environment, control-group experiment
  framework) is NOT archived here: it never touched an animal, it stays
  runnable at the repository root as the frozen baseline
  `PHASE1-BASELINE-1.0.0`, and parts of it (the 2D arena, the experiment
  runner, the control-condition machinery) are reused by the digital
  Drosophila track (D7, D10, control conditions A–E).

## Running the archived tests (optional, historical)

```bash
cd legacy
python -m pytest tests/test_phase2_apparatus.py -q
```

Requires the repository's Python environment (numpy, opencv-python,
PyYAML, pytest). The test file bootstraps its own import path relative to
this directory, so it runs from here unchanged.
