# WINDOWS RUNTIME / USER EXPERIENCE SPECIFICATION (addendum, authoritative)

Status: AUTHORITATIVE ADDENDUM to `docs/MASTER_SPEC.md`. Captured 2026-09-28.
Implementation is sequenced by the D1–D24 order (dashboard = D14, window
capture = D15, launcher = D24). Nothing in this document may be weakened;
it is the contract for those steps.

---

## W1. Decision

The final target machine is WINDOWS. The development environment may be
Linux, but the finished project must be packaged and tested on Windows.
The user interacts primarily with ONE application: **DigitalFlyLab.exe**.

## W2. User-facing experience

DigitalFlyLab.exe contains: live mirrored game view; computer-vision
overlays; Drosophila neural visualization; spike/activity visualization;
internal/motivational state; current goal; current observation; selected
action; expected reward; actual reward; memory; learned game knowledge;
progression statistics; experiment controls; replay functionality.

Roblox itself remains a normal separate Windows process. DO NOT attempt
to embed Roblox's renderer directly inside the Qt/application window.
Instead:

```
ROBLOX WINDOW -> Windows window capture -> live frame stream
              -> DigitalFlyLab game-view panel.
```

This gives the appearance of one unified application while maintaining
process isolation.

## W3. Windows modules

An isolated platform layer:

```
platform/windows/
    window_finder.py
    window_capture.py
    input_controller.py
    process_launcher.py
    focus_manager.py
```

All biological simulation, vision, memory and planning code remains
platform-independent.

## W4. Screen capture

Prefer Windows.Graphics.Capture or another robust Windows-native
application-window capture path. Capture ONLY the selected target window.
The captured frame stream serves BOTH (1) computer vision and (2) the
dashboard game preview. Never run separate unrelated capture pipelines.

## W5. Action output

Conceptual path:

```
DROSOPHILA MOTOR OUTPUT -> ACTION DECODER -> WINDOWS INPUT ADAPTER
                       -> TARGET GAME WINDOW.
```

All emitted actions must be logged. Include a GLOBAL EMERGENCY STOP
HOTKEY that releases all held keys immediately.

## W6. Account handling

The user prepares a dedicated experiment account in advance. DO NOT:
create accounts; store account passwords; automate credentials; bypass
authentication. Assume the user has authenticated normally. Later the
launcher may open ONLY the authorized game URL:
`https://www.roblox.com/games/1730877806/Grand-Piece-Online`

## W7. Start experiment flow

```
user opens DigitalFlyLab.exe
 -> application loads connectome / brain
 -> brain self-test
 -> computer-vision self-test
 -> user clicks START EXPERIMENT
 -> launcher opens authorized game URL if necessary
 -> application detects target game window
 -> capture begins
 -> dashboard shows live game
 -> computer vision begins
 -> fly brain receives sensory encoding
 -> fly neural activity selects action
 -> input adapter executes action
 -> result captured
 -> reward calculated
 -> memory / learning updated
 -> loop continues.
```

## W8. Dashboard layout

Default Observer layout:
- LEFT 60%: live game capture with CV overlays.
- RIGHT 40%: brain visualization; internal/motivational states; current
  goal/action/reward.
- BOTTOM: memory, learning curve, progression, event log.

Selectable layouts: OBSERVER, GAME, BRAIN, SCIENTIST, REPLAY.

## W9. Internal state

Never fabricate natural-language consciousness. Show actual computational
variables: positive valence; negative valence; reward expectation; reward
prediction error; novelty; avoidance/threat drive; exploration drive;
uncertainty; arousal-like state. Friendly labels are allowed only
alongside the underlying variable.

## W10. Experiment recording

Every run records enough data for synchronized replay: captured game
frames/video; CV detections; sensory encodings; neural activity; selected
actions; actual emitted inputs; rewards; motivational state; memories
created/retrieved; world-model updates; goals; progression; timestamps;
configuration; brain/checkpoint version. REPLAY mode lets the user scrub
through an experiment and inspect the game, CV, neural state, decision and
reward at the same timestamp.

## W11. Windows packaging

The final user never launches Python modules manually. Packaging route
produces `DigitalFlyLab.exe` or a self-contained:

```
DigitalFlyLab/
    DigitalFlyLab.exe
    resources/
    models/
    connectome/
```

Windows executables are built AND tested on Windows. Provide
`setup_windows.ps1`, `build_windows.ps1`, `run_dev_windows.ps1`. Linux dev
may run brain/CV/unit tests; Windows-specific capture, input and
packaging tests MUST run on Windows before the feature is considered
complete.

## W12. Development priority (superseded by D1–D24)

The original 15-step priority (digital fly runs -> benchmark -> artificial
environment -> sensory encoder -> motor decoder -> closed loop -> basic
dashboard -> CV -> Windows capture -> Windows action adapter -> passive
observation -> game control -> full dashboard -> replay -> launcher/menu
automation) is PRESERVED in intent and superseded in numbering by the
D1–D24 table in `docs/MASTER_SPEC.md` §12. Invariants kept: do NOT build
the complete dashboard before the brain benchmark; the dashboard must
visualize REAL internal values from the running system, never invented
decorative activity.
