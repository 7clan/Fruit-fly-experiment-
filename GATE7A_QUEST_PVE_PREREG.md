# GATE 7A — PRE-REGISTRATION: AUTONOMOUS QUEST + STARTER PVE

**Status:** FROZEN before first active 7A run.
Date: 2026-10-01

Gate 6F established autonomous GPO navigation with accepted Windows input.
Gate 7A adds the minimum game-semantic layer needed for the user-requested
loop: follow the recommended quest tracker, interact with a yellow quest NPC,
recover from simple obstacles, recognize the user-observed red quest-enemy
objective marker, and survive/fight starter PvE.

## Scientific boundary

The canonical Shiu/FlyWire v783 brain is unchanged.

The validated DN decoder continues to own low-level locomotor intention:
TURN_LEFT, TURN_RIGHT, APPROACH, RETREAT and escape-related outputs.

Quest concepts such as "talk to NPC", "press the current starter-melee
ability", and "this health loss means try a block" are not represented by the
validated biological decoder. Those actions are implemented by a clearly
labeled ENGINEERED QuestCombatSupervisor and are replay logged separately as
`engineered_command`.

Defense utility learning is an ENGINEERED ValueTable. It is not claimed as
mushroom-body/KC-MBON biological learning.

## Direct live visual cues used

The current project screenshots establish:

- green circular Recommended Quest waypoint + distance text;
- yellow QUEST / ! marker over quest-giver NPCs;
- after accepting a kill objective, a large red tracker marker can identify
  the quest-enemy objective;
- current HUD loadout: default Melee with Gut Punch (E) and Ground Smash (R).

The red target detector is deliberately cheap and conservative: compact red
circle plus nearby red text support in the gameplay region. It excludes the
bottom health-bar ROI and does not turn arbitrary red pixels into enemies.

## GPO controls available to the implementation

The physical Windows backend supports the letter/number row, Space/Ctrl/etc.,
relative mouse motion and mouse buttons. The catalog records stable GPO
controls and dynamically observed loadout keys.

The first 7A autonomous hard gate is intentionally narrower:

- navigation: W/A/S/D;
- jump: Space;
- climb/dive/evasive primitive: Left Ctrl;
- directional dash/roll: Q + direction;
- block: F;
- quest interaction: T;
- M1/basic attack: left mouse;
- current observed default-melee skills: E Gut Punch, R Ground Smash;
- bounded camera look: right-mouse drag internally.

The following remain catalogued but are NOT authorized in this gate:
grip (B), carry (V), ship sit (P), menu/stat spending (M), Haki toggles
(J/G), arbitrary hotbar/equipment switching and unknown fruit/style/sword
skills.

Named fruit/style/sword abilities are not globally hard-coded because their
meaning depends on the equipped loadout and may change with game updates.
They become eligible only after the live HUD/loadout observes their binding.

## Quest loop

1. Green Recommended Quest marker: fly navigates toward it.
2. Yellow QUEST marker: fly navigates to the giver.
3. Close + centered yellow quest giver: engineered supervisor taps T.
4. Red quest-enemy objective marker: fly navigates toward it.
5. Close enemy: conservative starter PvE controller may block and use M1/E/R.
6. After quest completion, return to the next green/yellow tracker cue.

No generic screen coordinate is translated directly to a movement key.
Movement bearing still enters the frozen D8 sensory channels and canonical
brain before W/A/D/S is selected.

## Obstacle recovery

The canonical brain is slower than real time on the target i7-5500U. If the
same visible navigation target stops getting closer for >3.5 seconds while
the fly is trying to locomote:

- first recovery: jump;
- repeated stalls: climb (W + Ctrl);
- if no navigation target is visible: slow bounded camera scan.

These are ENGINEERED locomotor affordance helpers, replay logged separately.
They do not choose left/right path direction.

## Starter PvE policy

When the red quest-enemy objective is close:

- cycle ordinary block, not claimed perfect-block timing;
- otherwise use M1;
- if default Melee is the confirmed loadout and stamina is adequate:
  - E Gut Punch no more often than its approximate observed cooldown;
  - R Ground Smash no more often than its approximate observed cooldown.

After a meaningful health drop, the supervisor chooses between BLOCK and
EVADE_BACK. It deliberately observes each once before preferring the option
with the better Laplace-smoothed short-term health-loss result.

At <=12% detected health, an independent hard safety layer disables the
active agent.

## Performance profile

Designed for i7-5500U / 8 GB:

- capture + fast CV: 3 Hz;
- x2 capture downsample;
- 480 px detector;
- generic humanoid Canny/Sobel proposal pass disabled;
- quest/enemy role cues use cheap yellow/green/red marker masks;
- quest supervisor: 4 Hz simple rules only;
- no LLM, OCR or heavy detector in combat-critical loop;
- dashboard: 0.25 Hz;
- canonical brain remains isolated and runs as fast as possible.

## 7A evidence criteria

A useful run should show:

- zero worker errors;
- SendInput failures = 0;
- autonomous camera correction and/or obstacle recovery if needed;
- at least one semantic quest/PvE command when its visual condition occurs;
- no command outside the hard 7A allowlist;
- replay/report contains quest phase and engineered defense values;
- F12/END RUN still releases every held keyboard/mouse input.

This is the first autonomous quest/PvE engineering gate. It is not evidence
that the fly has biologically learned GPO combat.
