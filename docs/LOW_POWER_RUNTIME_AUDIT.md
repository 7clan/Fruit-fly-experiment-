# Low-power live runtime audit — i7-5500U / 8 GB target

Date: 2026-10-01

This audit uses the uploaded successful Gate-6F session
`lab_20260930_232120` as the baseline. It does not claim a speedup until a
new live session measures one.

## Baseline evidence

The successful session reported:

- 398 captured frames;
- 372 fast-perception observations;
- 9 canonical brain-worker steps;
- 8 recorded brain events;
- final canonical chunk wall time: 9.336 s;
- canonical biological throughput: 0.0054 bio-s / wall-s;
- fast-vision last step: 109 ms;
- MotorExecutor: 5156 polling steps;
- ReplayRecorder: 3972 polling steps for only 411 recorded events;
- EvidenceRecorder: 555 polling steps for 8 saved brain-chunk frames;
- Dashboard: 71 snapshot steps;
- Win32 SendInput: 14/14 successful, 0 failures;
- total worker errors: 0.

The limiting resource is CPU, with RAM pressure also relevant. The canonical
138,639-neuron / ~15.1M-connection Brian2 simulation is the dominant task,
but several auxiliary workers were still doing unnecessary work around it.

## Changes

### 1. Brian2 live monitor: counts instead of full spike event storage

The canonical third-party model creates a full SpikeMonitor. Offline/reference
studies keep it unchanged.

The live subprocess now substitutes
`SpikeMonitor(record=False)`. Brian2 documents that this keeps exact
per-neuron `count` values and total `num_spikes` while not recording the
individual spike-index/time event arrays.

The live runtime derives, per biological chunk:

- exact total spike count;
- exact number of neurons that spiked;
- exact DN population spike counts;
- a FlyWire-ID sample of active neurons.

Individual spike timestamps are intentionally unavailable in this live
instrumentation mode. This is labeled
`whole_brain_counts_only` in telemetry/dashboard.

Biological neuron/synapse equations, connectome, RNG, sensory interface and
decoder are unchanged.

Reference:
https://brian2.readthedocs.io/en/latest/user/recording.html

### 2. Skip unused live checkpoint copy

D7 scientific/checkpoint studies retain the pristine `net.store("ep0")`.
The game subprocess never calls episode reset/checkpoint functions, so live
mode now disables the initial checkpoint copy. This reduces avoidable memory
pressure on the 8 GB machine without changing network dynamics.

### 3. Navigation-light computer vision

Movement-only navigation needs:

- recommended quest waypoint;
- yellow quest fallback;
- player anchor;
- health/stamina HUD.

It does not need expensive Canny/Sobel/multi-window humanoid proposals on
every frame. Those role proposals are now disabled only in the navigation
profile and remain available for the later combat/quest-role profile.

### 4. Lower auxiliary rates

Navigation launcher:

- Windows capture: 2 Hz
- fast CV / fly-channel encoder: 2 Hz
- detect width: 480 px
- heavy semantic worker: 0.05 Hz
- dashboard snapshot: 0.25 Hz
- executor polling cap: 20 Hz
- planner cap: 0.5 Hz
- replay drain: 10 Hz
- evidence recorder: 1 Hz

The brain still runs continuously as fast as it can; it is not slowed by this
profile.

### 5. Less disk/UI work

- ReplayRecorder no longer flushes the file when no events were drained.
- Autonomous evidence saves one annotated JPEG per new brain chunk instead
  of both raw and annotated copies.
- OpenCV dashboard remains cached/fail-soft.
- capture's legacy stream retains only one frame; live readers use the
  latest-frame state channel.

### 6. Less executor bus churn

`action.selected` is now updated only for a new canonical brain decision,
not on every 20–40 Hz executor polling cycle. Key hold expiry remains
high-rate and independent.

### 7. Better diagnostics

All workers now report cumulative work time and maximum step time so future
ZIPs show which engineered worker is stealing wall time. The canonical brain
also reports its instrumentation scope.

## Validation

Windows CI now performs:

- full Python bytecode compilation for `lab`, `brain/scripts`, and tests;
- neural interface and decoder tests;
- navigation mapping tests;
- Win32 input ABI/filter tests;
- a Brian2 test verifying that a `record=False` SpikeMonitor produces the
  same per-neuron counts and total spike count as a simultaneous full
  SpikeMonitor;
- synthetic end-to-end pipeline smoke test.

The only result CI cannot reproduce is real GPO/Roblox desktop behavior and
the target laptop's actual canonical-brain wall time.

## Expected outcome, not yet claimed as measured

The new configuration should reduce parent-process CPU, disk activity and
RAM pressure. The next real session must determine how much this improves
canonical chunk wall time. No numerical speedup is claimed before that run.


### 8. Downsample before the full BGR ownership copy

The Windows.Graphics.Capture callback now supports a stride downsample before
OpenCV performs the required BGRA->BGR ownership copy. The i7-5500U launcher
uses a factor of 2, so a native ~1920x1030 frame becomes ~960x515 before the
BGR copy, then fast vision works at a maximum width of 480 px.

This reduces memory bandwidth and the size of the shared latest-frame object.
All current navigation/HUD/waypoint geometry uses normalized coordinates.
The original native width/height and applied downsample factor are recorded
with captured frames, and the session header records the capture profile.

This is an engineered perception optimization only; it does not change the
canonical brain or neural decoder.
