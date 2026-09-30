# GATE 6C — PRE-REGISTRATION (forward-steering waypoint navigation)

**Status:** FROZEN before first 6C run.
Date: 2026-09-30

Gate 6B demonstrated that the canonical brain can receive the engineered GPO
navigation target and produce TURN intentions, but its turn action rotated the
camera without translating the avatar. Gate 6C tests the corrected
P9 locomotor interpretation.

## Closed loop

GPO quest navigation cue
-> engineered target geometry
-> frozen canonical Shiu/FlyWire v783 brain
-> DN intention decoder
-> movement-only resolver/executor
-> Windows input

No CV-to-key shortcut is introduced.

## Allowed outputs

- TURN_LEFT -> W held <= 1.5 s + bounded horizontal mouse turn left
- TURN_RIGHT -> W held <= 1.5 s + bounded horizontal mouse turn right
- APPROACH -> W held <= 1.25 s
- STOP -> no input

The hard backend allows only W keyboard input and horizontal relative mouse
movement with |dx| <= 120. It cannot emit attack, block, interact, ability,
menu, dash, mouse-button, S/A/D, or vertical mouse input.

## Run

180 seconds after canonical brain READY. Roblox is focused automatically.
The user does not move or steer the avatar. F12 is the global emergency stop.

## Functional success

1. canonical brain + subprocess_pipe;
2. zero worker errors;
3. at least one non-STOP brain intention;
4. at least one real W locomotion event;
5. no output outside the hard navigation subset;
6. complete replay chain;
7. visible autonomous avatar displacement.

This is navigation only. Quest interaction and combat are not authorized.
