# GATE 7B — PRE-REGISTRATION: QUEST FOLLOWING + STARTER PVE LEARNING

**Status:** FROZEN before first active 7B run.
Date: 2026-10-01

Gate 7B follows the successful Gate-6F locomotion run and extends the
quest/PvE engineering layer so the user no longer has to manually jump or
drag the camera.

## Biological / engineered boundary

The canonical Shiu/FlyWire v783 brain remains unchanged.

The fly still owns low-level locomotor intent through the validated decoder:
TURN_LEFT, TURN_RIGHT, APPROACH, RETREAT and escape-related outputs.

The following are explicitly ENGINEERED game affordances because the
validated fly decoder has no direct representation for GPO concepts such as
"talk to quest NPC", "press block", or "use an equipped skill":

- camera recentering;
- obstacle jump/climb recovery;
- quest interaction;
- block/evade learning;
- M1 and loadout-specific ability realization.

Every such command is replay logged separately.

## Direct live quest cues

Current project screenshots establish the loop:

1. green Recommended Quest circle + red arrow path -> travel target;
2. yellow QUEST/! -> quest giver;
3. after a kill quest is accepted, the large red objective marker -> target
   enemy objective.

The red objective detector is compact-circle/text based and excludes generic
red HUD/scenery.

## Camera + obstacle assistance

Camera is treated as part of the fly's sensory apparatus.

When a visible green/yellow/red navigation cue is more than ~0.20 rad from
screen center, the engineered supervisor may apply a small bounded RMB camera
drag toward it. It never chooses a movement key.

If the fly is locomoting but target proximity does not improve:

- jump first;
- repeated stalls -> W+Ctrl climb;
- no target -> bounded camera scan.

For a far green waypoint, double-W sprint is allowed only while the canonical
brain already says APPROACH.

## Physical GPO action surface

The hard Windows backend now supports the complete known character-action
surface needed by future loadouts:

- WASD;
- Space jump / repeated airborne Space (Geppo when unlocked);
- double-W sprint;
- Q + direction dash/roll;
- Ctrl climb/dive/contextual evasive;
- F block / timed F primitive for perfect-block experiments;
- left mouse M1 and air-combo primitive;
- T interaction;
- J/G Haki toggles when acquired;
- 0-9 equipment slots;
- observed skill keys E/R/Z/X/C/V/B/N/Q/F/G/J;
- bounded RMB camera drag.

This capability does NOT mean every action is used blindly. The semantic
supervisor and observed-loadout registry remain the autonomy gate.

Named fruit/style/sword moves are not globally hard-coded. Their keys and
names vary with the equipped loadout, so a generic USE_OBSERVED_ABILITY
primitive accepts only a validated live-HUD binding.

## Starter PvE policy

For the current default Melee loadout:

- confirmed close red quest target -> M1;
- E Gut Punch and R Ground Smash are available with conservative cooldowns;
- health loss -> learn BLOCK vs EVADE_BACK using engineered ValueTable;
- very low health -> escape / critical safety stop.

This learning is engineered action-value learning, not claimed biological
KC-MBON learning.

## Not yet automated in 7B

Stat spending through M -> Stats remains disabled until level/stat OCR and
menu-state confirmation are reliable. It will be a separate strategy layer.

An AI helper is also not in the urgent loop. A later helper may diagnose
persistent failure/stalls, but it must not replace the fly's low-level
locomotor decisions.

## Useful evidence

A useful 7B run should show at least one of:

- autonomous camera recenter;
- autonomous jump/climb recovery;
- yellow quest interaction;
- red enemy-objective transition;
- block/evade/M1 starter-PvE command;

while retaining zero SendInput failures and zero worker errors.
