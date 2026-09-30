# GATE 6D — PRE-REGISTRATION (properly armed autonomous waypoint navigation)

**Status:** FROZEN before first 6D run.
Date: 2026-09-30

Gate 6D corrects the Gate-6C application-level arming defect. The canonical
brain and intention decoder are unchanged.

## Required arming sequence

1. start capture/perception/canonical brain with real input DISABLED;
2. wait for explicit canonical-brain ready event;
3. focus the captured authorized Roblox window;
4. arm global F12 emergency stop;
5. only then enable the navigation-only executor;
6. start the timed autonomous window.

Any real input before step 5 is an automatic FAIL.

## Navigation cue

The preferred target is the GPO recommended-quest marker identified by the
observed pattern: a bright green roughly circular marker with green distance
text directly below it. The detector excludes the right-edge HUD region.

Yellow QUEST markers remain a fallback local target.

All target geometry is still encoded through fly sensory channels and the
canonical brain. There is no CV-to-key shortcut.

## Allowed outputs

- TURN_LEFT  -> bounded horizontal mouse left + W <= 1.5 s
- TURN_RIGHT -> bounded horizontal mouse right + W <= 1.5 s
- APPROACH   -> W <= 1.25 s
- STOP       -> no input

Hard backend: only W keyboard and horizontal relative mouse movement,
|dx| <= 120. No attacks, block, interact, abilities, menu keys, mouse
buttons, S/A/D, or vertical mouse movement.

## Run

One 90-second autonomous window after proper arming. The user does not move
or steer. F12 aborts immediately.

## Functional success

- no real input before arming;
- canonical brain + subprocess_pipe;
- zero worker errors;
- at least one valid navigation target observed;
- at least one non-STOP canonical intention;
- at least one real W navigation pulse after arming;
- no output outside the hard navigation subset;
- replay and automatic annotated evidence saved;
- visible autonomous displacement in evidence and/or user observation.

Quest interaction and combat are still not authorized.
