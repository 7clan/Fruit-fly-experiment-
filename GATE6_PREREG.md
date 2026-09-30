# GATE 6A — PRE-REGISTRATION (movement-only autonomy)

**Status:** FROZEN before first active-input run.
Date: 2026-09-30

Gate 6A is an engineering test of basic movement only. It does not authorize
combat, attacks, blocking, interaction keys, abilities, quest completion, or
unattended long-duration play.

Gate-5 v2 established the live capture/brain/replay path and zero-input safety,
but hostile-role recognition remains incomplete. Gate 6A is therefore isolated
from that unresolved classifier: hostile labels are NOT used to select or
permit movement. Passing Gate 6A does not retroactively turn Gate-5 v2 into a
full perception PASS and does not authorize combat.

## Question

Can canonical-brain intentions produce observable, logged character movement
in GPO through the existing MotorExecutor/Windows input path without any
CV-to-key shortcut?

## Allowed actions

Only the existing basic movement subset is armed:

- STOP -> no input
- TURN_LEFT -> A
- TURN_RIGHT -> D
- APPROACH -> W

RETREAT and ESCAPE are disabled for this first active test even if emitted by
the decoder. No mouse buttons, attack keys, block key, interaction key, dash,
abilities, menus, or quest actions are allowed.

## Test

One **45-second active window after canonical brain READY** in a low-risk
starter-town area. The user remains present and may abort immediately with
Ctrl+C.

## Automatic fail conditions

- any action outside the allowed movement subset;
- any input event not originating from a new `brain.output` decision;
- any worker error;
- missing replay/report/ZIP;
- input still held after shutdown;
- CV/perception directly selecting a key without the brain intention layer.

## Pass criteria

1. canonical runtime = `canonical_brian`;
2. transport = `subprocess_pipe`;
3. >= 4 completed canonical brain chunks during the active window;
4. >= 1 real movement input emitted;
5. all emitted actions are in {A, D, W};
6. replay records brain intention -> concrete action -> emitted input;
7. user visually confirms at least one observable character displacement;
8. shutdown releases all held keys and total worker errors = 0.

A PASS only authorizes continued movement/navigation engineering. Combat and
interaction remain Gate-7+ work.
