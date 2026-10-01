# GPO control / move catalog

Date: 2026-10-01

This document separates **game controls** from **biological intentions**.

The fly brain does not contain neurons named after GPO attacks. It produces
behavioral categories such as TURN, APPROACH, EVADE, DEFEND, ATTACK and
ESCAPE. The engineered action layer maps those categories onto controls that
are actually available in the current GPO loadout.

## Core control surface now represented

Movement / traversal:

- W / A / S / D movement
- double-tap W sprint
- Space jump
- directional Q dash / roll
- CTRL contextual climb / dive
- repeated airborne Space for Geppo / Sky Walk when unlocked

Combat fundamentals:

- mouse-left basic attack / M1
- repeated M1 strings
- Space held during the M1 sequence for uptilt / air-combo launch
- F block
- timed F perfect block / parry
- R firearm reload

World / utility:

- T interact / talk — retained as community-observed and requires current
  live confirmation before autonomous quest use
- V carry downed player
- B grip / execute downed target
- P sit / attach to ship
- M menu
- J Busoshoku Haki
- G Observation Haki

Evasive while stunned is deliberately recorded as **version-conflicted**:
community documentation has used both CTRL and Q in different versions.
Autonomy must not guess this binding. Current HUD/behavior must verify it.

## Equipped fighting-style / fruit / weapon moves

There is no honest single hard-coded list of every named attack because the
available moves and hotkeys depend on the equipped fighting style, fruit,
weapon, mastery/unlocks and game updates.

Examples in community documentation include Black Leg skills on E/R/Z and
Rokushiki skills on E/Q/R/Z/X, while other styles use keys such as C or N.

Therefore DigitalFlyLab now treats these as **observed equipped abilities**:

1. perception/HUD identifies the current move name + displayed binding;
2. `AbilityRegistry.register_observed_binding(...)` validates and caches it;
3. `AbilityResolver` carries the observed binding with the chosen ability;
4. `MotorExecutor` can emit keyboard or mouse bindings and release them;
5. current autonomy gates still decide whether that category is allowed.

The Windows backend now supports all A-Z keys, 0-9, Space, Ctrl, Shift, Alt,
Tab, Esc, plus left/right/middle mouse buttons.

## Current autonomous authorization

The catalog is broader than the currently authorized live gate.

The existing navigation launcher still uses the movement-only backend, which
hard-blocks everything except W/A/D. Adding the move vocabulary does **not**
silently enable combat.

Combat/quest automation must be enabled by its own validated gate after
hostile recognition, attack timing and quest interaction are reliable.
