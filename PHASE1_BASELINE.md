# PHASE-1 BASELINE FREEZE

**Baseline ID:** `PHASE1-BASELINE-1.0.0`
**Frozen (UTC):** 2026-09-27T13:08:48Z
**Source revision (frozen tree):** git commit `7296f6bda05cf830f6a045ef6f26d30c32882ce6`
**Git tag:** `phase1-baseline-v1.0.0` (points to the commit that contains this file)

This document is the reproducibility record for the Phase-1 synthetic
validation. Phase 1 is treated as a **frozen scientific baseline**: the
implementation listed below must not be modified. Any change requires a new
baseline ID (`PHASE1-BASELINE-1.x.0`) and a fresh reproduction run; the
original results are never edited.

---

## 1. What is frozen

| Frozen item | Where |
|---|---|
| Controller architecture | `flyagent/controller/` (fly / random / fixed / keyboard, one interface) |
| Reward function | `flyagent/learning/reward.py`, params in `config/experiment.yaml` (`reward:`) |
| Action decoder (both schemes) | `flyagent/biological_agent/action_decoder.py`, params in `config/actions.yaml` |
| Behavior classifier | `flyagent/biological_agent/behavior_classifier.py` |
| Synthetic fly behavioral model | `flyagent/biological_agent/synthetic_fly.py`, params in `config/experiment.yaml` (`synthetic_fly:`) |
| Control groups (5 conditions) | `flyagent/experiments/controls.py` (`run_demo`, `CONDITIONS` in `runner.py`) |
| Trial loop / experiment parameters | `flyagent/experiments/trial.py`, `runner.py` + `config/experiment.yaml` |
| Tracking pipeline | `flyagent/fly_tracker/` + `config/tracking.yaml` |
| Game environment (2D artificial) | `flyagent/environment/arena2d.py` + `game:` params |
| Database schema | `flyagent/data/database.py` |
| Reported results | `runs/2026-09-27 12:16-12:22 session` (see §7) |

**Exception policy:** modifications are permitted ONLY for a genuine
reproducibility or correctness bug. Any such fix must be documented in this
file under "Amendments", with the bug demonstrated (failing reproduction or
test), and must trigger a new baseline ID + full re-run. No silent tuning.

## 2. Configuration files (content-hashed, frozen)

| File | SHA-256 |
|---|---|
| `config/actions.yaml` | `ba71cda744f8cba59795c0feb0038a4ca6cc5f93afd0d60b7a968b9cc5909ac3` |
| `config/experiment.yaml` | `712967395c179838c36add04f48bb152c830e4b328dfe9569cb1f902b2589b9e` |
| `config/tracking.yaml` | `aba1624bb8a15f15fe2e107b16a2633ec645d55b03db77b90f9624b6f32c0614` |

Every run directory also stores its own full `config_snapshot` in
`manifest.json` (config is captured, not just referenced).

## 3. Random seeds (fully deterministic)

All randomness flows through `numpy.random.default_rng` (PCG64). No
wall-clock, no OS entropy, no Python `random` in any behavioral code path.
Timestamps appear only in directory names / metadata, never in behavior.

| Purpose | Seed | Derivation |
|---|---|---|
| Demo master seed | `20260927` | CLI default (`main.py demo --seed`) |
| `random` condition | `20260927` | master + 1000 x 0 |
| `fixed` condition | `20261927` | master + 1000 x 1 |
| `fly_random` condition | `20262927` | master + 1000 x 2 |
| `fly_cue` condition | `20263927` | master + 1000 x 3 (verified against manifest) |
| `fly_cue_shuffled` condition | `20264927` | master + 1000 x 4 |
| Bootstrap resampling | `20260927` | `run_demo` passes the master seed to `write_comparison` |
| Single-condition runs (`main.py run`) | `7` | CLI default (not used in the demo) |

Per-trial variability comes from the single per-condition RNG stream carried
across trials (arena spawn, fly behavior, controller randomness) — the stream
is a pure function of the per-condition seed.

## 4. Software / package versions

| Component | Version |
|---|---|
| FlyAgent software version | `0.1.0` (`flyagent/version.py`) |
| Python | `3.12.14` |
| NumPy | `2.1.3` |
| OpenCV (`cv2`) | `4.13.0` |
| Matplotlib | `3.9.2` |
| PyYAML | `6.0.3` |
| Platform | `Linux-5.10.134-013.15.kangaroo.al8.x86_64-x86_64-with-glibc2.41` |

Reproduction environment matched the original run environment on Python
version and platform string (recorded in every `manifest.json`).

## 5. Experiment parameters (summary of `config/experiment.yaml`)

| Parameter | Value |
|---|---|
| Conditions (order) | `random`, `fixed`, `fly_random`, `fly_cue`, `fly_cue_shuffled` |
| Trials per condition | 30 |
| Frame rate | 30 fps |
| Arena | 640 x 480 px, open_field, 1 obstacle rect defined but unused |
| Player | radius 10 px, speed 80 px/s |
| Target | radius 18 px, spawn distance 260-420 px from start |
| Timeout | 30 s per trial |
| Reward | mode `both`: sparse success +1.0; shaped 0.004/px progress; time penalty 0.002/s; obstacle penalty 0.05 |
| Random controller | action every U(0.5, 1.2) s, P(STOP)=0.10 |
| Fixed controller | greedy axis-dominant action every 0.5 s (ceiling) |
| Synthetic fly | walk bouts U(0.8, 2.5) s; pauses U(0.2, 1.0) s; speed U(16, 34) mm/s; heading noise sigma 1.2 rad/s; cue_bias 0.85 (fly_random forces 0.0); fly arena 90 x 60 mm @ 360 x 280 px |
| Decoder | scheme `motion_direction`; heading dwell 10 frames; pause speed 8 mm/s; pause dwell 12 frames; cooldown 20 frames; action vocabulary FORWARD/BACKWARD/LEFT/RIGHT/STOP |
| Tracker | MOG2 (history 300, varThreshold 32, lr 0.0005) + frame-diff fallback; blob area 25-2500 px; ROI = arena rect; 20 warmup frames |
| Statistics | bootstrap 10,000 iterations; rolling window 10 |
| DB logging | frame rows every 3rd frame (`log_stride: 3`); no JPEG dumps |

## 6. Database schema version

**Schema version 1** (`SCHEMA_VERSION = 1` in `flyagent/data/database.py`):
tables `experiments`, `trials`, `events`, `frames` (full DDL in that file).
One SQLite DB per run inside an immutable timestamped run directory.

## 7. Reported results (frozen original, 2026-09-27 12:16-12:22 UTC+3 session)

Command: `python main.py demo` (defaults: seed 20260927, 30 trials/condition).
Artifact: `runs/demo_20260927-122233/comparison.json`
(SHA-256 `a570c8c0c48d52177d534f310961403169e373947e13354d7f5ddfd7609d6a58`).

| condition | n | success | median TTT | median pathEff | mean actions | mean reward |
|---|---|---|---|---|---|---|
| `random` | 30 | **3.3%** (1/30) | 18.8 s | 0.19 | 34.6 | 0.24 |
| `fixed` | 30 | **100.0%** | 4.8 s | 0.81 | 9.9 | 2.24 |
| `fly_random` | 30 | **6.7%** (2/30) | 25.6 s | 0.27 | 32.1 | -0.01 |
| `fly_cue` | 30 | **76.7%** (23/30) | 12.2 s | 0.36 | 20.5 | 1.96 |
| `fly_cue_shuffled` | 30 | **3.3%** (1/30) | 5.7 s | 0.80 | 31.7 | 0.11 |

Bootstrap 95% CI of success-rate differences (10,000 iters):

| contrast | diff | CI | p(diff<=0) |
|---|---|---|---|
| `fly_cue` - `random` | +0.733 | [+0.567, +0.900] | 0.0000 |
| `fly_cue` - `fly_random` | +0.700 | [+0.533, +0.867] | 0.0000 |
| `fly_cue` - `fly_cue_shuffled` | +0.733 | [+0.567, +0.900] | 0.0000 |
| `fixed` - `random` | +0.967 | [+0.900, +1.000] | 0.0000 |

**PHASE-1 GATE: PASSED** (apparatus can detect a biological-behavior
contribution: both `fly_cue - random` and `fly_cue - fly_cue_shuffled` CIs
exclude zero).

Per-run artifacts (SHA-256 of each `summary.json`):

| Run directory | SHA-256 |
|---|---|
| `runs/20260927-121624-527412_random/` | `9e3815257fc84f750e86882d24b4d63812ad3629f9ae6fdf64245554fbac486d` |
| `runs/20260927-121624-718101_fixed/` | `503652a3885fed62f671fbf8c14cdf42ec2f2b55e87b6b99d1169ee490594924` |
| `runs/20260927-121624-751794_fly_random/` | `3add68fc0353e8972b1abd6a4f4f64cf6bc8c1afe3ca263073fb72d36203f10a` |
| `runs/20260927-121907-419806_fly_cue/` | `e729265b65347ed31920a7d0d0d5733d9ead3c9177469a2f5f226feedc6182d0` |
| `runs/20260927-122026-804472_fly_cue_shuffled/` | `a51dec6959f64390d98ff182ccd3ddaa2d42f3670214f6aedd84ccab659fcf9b` |

## 8. Reproduction protocol (for PHASE1-BASELINE-1.0.0)

```bash
git checkout phase1-baseline-v1.0.0   # or work from this frozen tree
python -m tests.test_pipeline         # pipeline verification first
python main.py demo                   # seed 20260927, 30 trials/condition
python scripts/verify_reproduction.py \
    --original runs/demo_20260927-122233 \
    --new-runs runs --new-demo-dir runs/demo_<new-timestamp>
```

**Materiality criterion (defined in advance):** the reproduction is
*identical* if (a) every per-trial metric of all 5 conditions matches the
original exactly (same seeds, same code => bitwise-equal floats), and (b)
the bootstrap statistics match to printed precision. ANY per-trial mismatch
in outcome or any aggregate success-rate difference > 0 is **material** ->
STOP, diagnose (environment first: numpy/opencv versions), and re-run after
restoring the recorded environment. Parameter tuning to recover a result is
forbidden.

## 9. Amendments

(none)
