# GATE 6B — PRE-REGISTRATION (autonomous quest-waypoint navigation)

**Status:** FROZEN before first 6B run.
Date: 2026-09-30

Gate 6A failed because every canonical intention in its active run was STOP.
Gate 6B tests the corrected navigation loop. It does not alter the biological
network.

## Closed loop

GPO recommended-quest green waypoint
-> engineered visual target
-> `fly-channels-v2-center-bilateral`
-> unchanged canonical Shiu/FlyWire v783 brain
-> DN intention decoder
-> navigation-only MotorExecutor
-> Windows input

There is still no CV-to-key shortcut.

## Allowed real outputs

- APPROACH -> W only, hold <= 1.25 s per new brain decision
- TURN_LEFT / TURN_RIGHT -> bounded relative horizontal mouse movement only
- STOP -> no input

No A/S/D keys, mouse buttons, attack, block, interact, dash, ability,
inventory, menu, or quest-accept input is allowed.

The backend itself hard-filters keyboard output to W and forces all mouse
movement horizontal with absolute magnitude <= 120 per decision.

## Run

The autonomous launcher runs for 180 seconds after canonical brain READY.
The user does not manually steer or move the character. The game window is
focused automatically. F12 is a global emergency stop.

## Success evidence

This is a functional navigation milestone, not a biology claim. A useful run
requires:

1. canonical brain and subprocess_pipe transport;
2. zero worker errors;
3. at least one non-STOP canonical intention;
4. at least one non-empty real navigation command;
5. no output outside the allowed set;
6. replay chain target -> fly sensory rates -> brain -> intention -> input;
7. visible autonomous character/camera movement reported by the user.

Combat, quest interaction, abilities, and unattended long-duration
progression remain outside Gate 6B.
