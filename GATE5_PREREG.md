# GATE 5 — PRE-REGISTRATION (passive perception on Windows)

**Status:** FROZEN before any Gate-5 assessment run. This document is
committed and pushed BEFORE the assessment sessions. Nothing in it may
change after assessment begins; any deviation discovered mid-run is
recorded as a deviation, not silently applied.

**Governing documents:** `FINAL_ARCHITECTURE.md` (authoritative,
2026-09-29) — Gate-5 definition §12; performance degradation policy §13.
`docs/WINDOWS_RUNTIME_SPEC.md` (W2–W7). This prereg instantiates them
into pass/fail criteria.

**Relationship to the scientific gates:** Gates 1–3 PASS and Gate 4 v1/v2
FAIL are the frozen scientific record. Gate 5 is an **engineering gate**
on the applied Windows system: it makes NO biological claim and changes
NO biological parameter. The biological brain is used exactly as
validated (canonical fixed connectome, interactive chunked runtime).

---

## 1. The ONE question

Can DigitalFlyLab on the target Windows laptop, attached passively to
the authorized GPO target (Roblox, `https://www.roblox.com/games/1730877806/Grand-Piece-Online`),
run the full perception–brain–mirror pipeline **concurrently and
reliably, with measured latency, and ZERO autonomous game input**?

## 2. Non-negotiable constraints (violating any = automatic FAIL)

1. **NO AUTONOMOUS INPUT.** The MotorExecutor must run in SHADOW mode
   (autonomy disabled, SafeNoopBackend). No key/mouse injection into the
   game process at any time during Gate-5 sessions. Any input emission
   event not tagged `shadow: true` in the replay is an automatic FAIL.
2. **Authorized target only.** Only the GPO window (or the synthetic
   dev source in dry runs) may be captured. No other window, screen
   region, or Roblox property may be accessed.
3. **Canonical brain unchanged.** The fly brain runs the pinned Shiu
   v783 model via the validated interactive runtime; no parameter,
   plasticity rule, or structural change. Mode A (PURE_FIXED_FLY)
   semantics; the helper components run but only consume/label — they
   may not alter brain input during Gate 5 beyond the frozen
   `fly-channels-v1` encoder.
4. **Honest measurement.** All latencies from the shared monotonic
   clock; reports come from `benchmark_windows.py` / session replay
   records. Fabricated or post-hoc-"improved" timestamps = automatic
   FAIL (and a project-level integrity violation).

## 3. Prerequisites (all must be GREEN before assessment sessions)

| # | Check | Evidence |
|---|---|---|
| P1 | setup_windows.ps1 completes; self-test PASS (mock, passive) | console + session dir |
| P2 | benchmark_windows.py run on the target laptop; WINDOWS_HARDWARE_REPORT.md + config/live_settings.json generated | files committed |
| P3 | Hardware inventory plausible (CPU/RAM/GPU/Windows detected, not None) | report table |
| P4 | User logged into the prepared experiment account; GPO running in a normal window | human confirmation recorded in session header |

## 4. Assessment protocol (single session series, ≥ 3 sessions)

Each session = 10 minutes of passive observation with the full pipeline:
capture (Windows.Graphics.Capture) → fast vision → world observation →
fly channels → canonical brain (chunk size from live_settings.json) →
intention decoder → SHADOW executor → dashboard mirror (3-column) →
replay recording. Session header records: mode (PASSIVE), runtime label,
chunk_ms, live_settings snapshot, GPO window title, session start/end
wall time.

Dry-run stage (before attaching to the game, same criteria on synthetic
source) is permitted for pipeline bring-up, and is recorded as
`source=synthetic` — it does NOT count toward the game-window sessions.

## 5. Pass criteria (fail-closed; ALL must pass in ≥ 2 of 3 game sessions)

| ID | Criterion | Threshold | Measurement |
|---|---|---|---|
| G5-1 | Target window located + captured reliably | ≥ 95% of scheduled capture ticks publish a frame; no capture restart needed | capture stats vs elapsed |
| G5-2 | Dashboard mirror shows the live game stream | mirror frame age P95 < 250 ms at dashboard_hz; one capture stream only (copy_count ≤ 1/frame) | dashboard + frame_ref accounting |
| G5-3 | Basic game/UI states identified | ≥ 80% of sampled frames produce a structured WorldObservation with non-null player/target fields or an explicit ui state label; vision confidence reported per detection | replay samples (100 stratified frames scored) |
| G5-4 | Important objects tracked | on the synthetic dry-run, target + enemy tracked with ≥ 90% detection and ≤ 15% outlier distance error; on game frames, tracked-or-explicitly-unknown (no fabricated positions) | ground-truth (synthetic) / manual spot-check (game, 30 frames) |
| G5-5 | Fly brain runs CONCURRENTLY with perception | brain chunk stream continuous through the session (no gaps > 5 s); brain.bio_s advances monotonically; runtime label = canonical_brian | brain.events replay |
| G5-6 | Synchronized replay recorded | replay.jsonl reconstructs the decision chain for every sampled brain chunk (frame → observation → channels → chunk → intention → shadow action) with timestamps | replay_decision_chains() on samples |
| G5-7 | CPU/RAM within budget | process tree peak RAM ≤ 80% of physical; sustained CPU ≤ 85%; dashboard renders dropped (not queued) under pressure | benchmark + session report |
| G5-8 | Latency reported honestly | P95 perception→decision latency computed and reported against the 250 ms goal; if not met, the number stands with the degradation plan (§13) | session report |

G5-4 game-frame manual spot-checks are performed by the user on exported
frame+detection pairs (30 frames, majority label agreement); scores are
recorded in the session report.

## 6. Failure handling (pre-registered)

- Any criterion fails → **GATE 5: FAIL**, recorded honestly; fixes are
  engineering-only (per FINAL_ARCHITECTURE §13 order: reduce dashboard
  FPS/visualization/OCR/heavy-detector first; then model
  resolution/batching/provider/memory copies; NEVER remove the fly brain,
  urgent perception, or motor executor to pass).
- Re-run after engineering fixes = a NEW session series (criteria
  unchanged); previous failures remain in the record.
- No biological parameter may be touched to rescue an engineering gate.
- If the laptop cannot meet G5-8's 250 ms despite §13 optimizations,
  Gate 5 may still PASS with the measured value disclosed, PROVIDED the
  degradation order was followed and combat-critical components were not
  removed. This exception does not extend to Gate 6+.

## 7. Deliverables on completion

- `GATE5_REPORT.md` — verdict, per-criterion table, latency/RAM/CPU
  numbers, deviations, session list
- Session directories (`runs/gate5_*`) with replay.jsonl + reports
- Updated `WINDOWS_HARDWARE_REPORT.md` (final laptop numbers)
- Committed + pushed record (fail-closed, like every prior gate)

Gate 6 (basic live movement) requires Gate 5 PASS and its own
pre-registration BEFORE any input emission is armed.
