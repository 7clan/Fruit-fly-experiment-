# GPO GAME MODEL AND AGENT PLAN

Status: research/design note. This does not authorize autonomous input.

## Purpose

This document turns current Grand Piece Online (GPO) knowledge into an
engineering model for DigitalFlyLab. The goal is not to encode a brittle
walkthrough. The goal is to define what the agent must perceive, remember,
and do while preserving the project's biological/engineering separation.

## Source discipline

Use this order when sources disagree:

1. Direct observation from the live game window on the target account.
2. The single permitted official game page:
   https://www.roblox.com/games/1730877806/Grand-Piece-Online
3. Current Grand Piece Online community wiki pages.
4. Recent player reports/videos for behavior that is not documented well.

Do not hard-code a number just because one community page says it. Current
sources already disagree on some starter quest counts and older pages still
show obsolete level caps. Record such fields as observed/unknown until the
live client confirms them.

## Current high-confidence game model

### Current build

The permitted official game page identifies Update 13 as current and lists
the current maximum level as 675. The game is an action RPG built around
quests, fighting styles, ability-granting fruits, bosses/raids, arena modes,
battle royale, sailing and multiple seas.

### Town of Beginnings

The starter island is Town of Beginnings. Current community documentation
lists Bandits and the Bandit Boss as the island's hostile inhabitants and
Daph/Ronny as quest-giver NPCs for the Bandit/Bandit Boss progression.
The starter island is documented as PvP-disabled.

The user's live screenshot is ground truth for this project: the humanoid
NPCs visible in the rear combat area are enemy examples, while the foreground
NPC under the yellow QUEST/! marker is a quest-NPC example.

### Core controls relevant to perception/action design

Community documentation currently maps:

- double-W: sprint
- Q + direction: roll/dash
- F: block
- left Ctrl: climb/dive and evasive when stunned
- M: menu
- R: gun reload
- J: Busoshoku Haki after acquisition
- G: Observation Haki after acquisition
- P: attach/sit on a ship

These are interface facts, not permission to emit input. Gate 5 remains
passive/shadow-only.

### Combat mechanics that matter to the agent

GPO combat cannot be modeled as "walk at target and spam attack".

Important documented mechanics include:

- M1/basic attack strings.
- Blocking and guard/block breaking.
- Perfect block/parry: timed block can stun the attacker; NPCs can also
  perfect block.
- Air/uptilt combos.
- Stun, knockdown, i-frames, hyperarmor and evasive states.
- Stamina is consumed by attacks and other movement/combat actions.
- Defense affects health, regeneration and block strength.
- Hostile NPCs can aggro when a player enters their range; multi-NPC aggro
  is a major early-game hazard.
- Bosses can have very different movesets, mobility, summons and mechanics;
  boss logic must not be inferred from normal Bandits.

### Navigation/progression mechanics that matter to the agent

- Recommended Quest tracking exists and is useful as a high-level goal cue.
- Eternal Poses expose a directional arrow to an island.
- Sailing is a real navigation mode; ships have health and can be damaged.
- Rough Waters and sea enemies create a distinct hazard state.
- Second Sea access is a separate progression stage.
- Dungeons, Arena, Battle Royale and PvE raids are separate gameplay modes
  with different rules. They should be modeled as separate world modes, not
  as minor variations of overworld grinding.

## Reliability conflicts already found

Examples of why the project must prefer live observation over hard-coded
wiki numbers:

- Community pages disagree on whether Daph's current Bandit quest asks for
  5 or 10 kills.
- Older community pages show a level cap of 625 while the current official
  game page shows 675.
- Older Bandit pages disagree on HP/damage values.

Therefore the agent should learn/observe quest counters, health changes and
combat outcomes from the live UI rather than assume historical constants.

## Engineering decision

### 1. Do not build "an enemy-color detector"

GPO contains many visually different enemies, bosses, players, quest givers,
merchants and trainers. Clothing/color is not a stable semantic label.

The perception stack should instead produce tracked entities with evidence:

- player_self
- humanoid_unknown
- quest_npc
- hostile_npc
- boss_candidate
- other_player
- merchant_or_trainer
- interactable
- ship
- navigation_marker

Role classification should combine appearance with context and time:

- yellow QUEST/! marker -> strong quest-NPC evidence
- attack/aggro motion toward player -> hostile evidence
- enemy attack animation / damage event -> strong hostile evidence
- boss health/UI or known boss context -> boss evidence
- dialogue/shop UI after interaction -> non-hostile service-NPC evidence
- repeated identity across frames -> tracked entity, not a new detection

Unknown must remain a valid class. The detector must never invent an enemy
position just to satisfy downstream code.

### 2. Separate HUD perception from scene perception

HUD parsing should be deterministic and cheap:

- health fraction
- stamina fraction
- level
- EXP/progression
- currency
- quest/recommended-quest state
- PvP-protection/safe-zone state
- evasive availability
- active ability/cooldown slots when visible
- dialogue/menu/death/loading states

Scene perception should handle:

- player/camera anchor
- humanoids
- quest markers
- hostile motion/attack cues
- boss candidates
- interactables
- navigation/pose direction
- ships and large hazards

### 3. Add temporal tracking before adding more semantic classes

At the measured Windows capture rate, temporal evidence is valuable. Use a
small CPU-friendly tracker/tracklet layer so a Bandit remains the same entity
across frames. Detection gives candidates; temporal behavior gives role
evidence.

Use simple IoU/Kalman-style tracklets first. Sparse Lucas-Kanade optical flow
may be used only for local motion cues where it survives camera movement.
Do not deploy a large tracker/model merely because it is fashionable.

### 4. World modes are explicit state

The world model should expose a small mode enum such as:

- OVERWORLD_IDLE
- NAVIGATING_TO_QUEST
- DIALOGUE
- QUEST_ACTIVE
- COMBAT
- STUNNED_OR_KNOCKED
- RECOVERY
- MENU
- SHIP_NAVIGATION
- SEA_HAZARD
- DEAD_OR_RESPAWNING
- RAID
- DUNGEON
- ARENA_OR_BR

This prevents a planner or motor layer from interpreting the same visual cue
the same way in every context.

### 5. Preserve the fly/helper boundary

The biological fly remains responsible for the low-level behavior channel:
orientation, approach/avoidance, threat response, movement intention and
decoded action intention.

The engineered helper may provide:
- current goal
- entity semantics
- quest/progression context
- ability compatibility/range
- navigation destination
- long-horizon memory

It must not silently replace the fly's intention with a raw key choice.

### 6. Use a skill library, not a giant monolithic policy

Research on embodied game agents supports separating high-level goals from
reusable temporally extended skills. For this project, early skills should
be explicit and testable:

- face_target
- approach_target
- retreat_from_threat
- circle_or_reposition
- interact_with_quest_npc
- acquire_single_enemy
- block_window
- basic_m1_string
- use_evasive
- disengage_and_recover
- return_to_quest_giver
- follow_pose_arrow
- board/control_ship

The AbilityResolver later maps a fly-selected intention into a compatible
skill/ability. Gate rules still determine when any real input backend may be
armed.

## Research references that influenced the architecture

- Google DeepMind SIMA: screen pixels + language goals + keyboard/mouse
  actions, with skills evaluated in 3D game environments.
  https://deepmind.google/blog/sima-generalist-ai-agent-for-3d-virtual-environments/
- Voyager (Wang et al., TMLR 2024): automatic curriculum, reusable skill
  library, environment feedback and long-horizon memory.
  https://voyager.minedojo.org/
- OpenCV optical-flow documentation for lightweight temporal motion cues.
  https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html
- ByteTrack (ECCV 2022) as a reference for tracking-by-detection principles;
  not a requirement to deploy its full implementation on this laptop.
  https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/315_ECCV_2022_paper.php

## Development order from here

### Stage A — Starter-town passive perception

Required before any autonomous input work:

1. robust player/camera anchor
2. quest-marker / quest-NPC detection
3. health + stamina
4. humanoid proposals
5. temporal tracks
6. hostile-vs-nonhostile evidence
7. quest active/completed UI state
8. Bandit Boss candidate when current level/quest context makes it relevant

### Stage B — Passive combat understanding

Collect/validate examples of:

- enemy entering aggro
- enemy M1 wind-up/contact
- player blocking
- perfect-block visual feedback
- player hit / health loss
- enemy hit / knockback
- stun/knockdown
- evasive use/availability
- enemy death/respawn

### Stage C — Passive navigation understanding

Validate:

- Recommended Quest goal cue
- Eternal Pose arrow
- safe-zone/PvP state
- island transition
- ship state
- death/respawn

### Stage D — Re-run Gate 5

Only after the real perception outputs are meaningful. The existing
synthetic-color heuristic is not Gate-5 evidence for GPO perception.

### Stage E — Gate 6 preregistration

Only after Gate 5 passes. This is the first stage where actual movement input
may be considered, under a separately frozen protocol.

## Hardware decision

The target laptop is CPU-constrained. Current evidence supports keeping the
real capture stream capped around 5 FPS during canonical-brain operation and
performing scene detection on a downscaled frame. More perception accuracy
should come from better semantics and temporal evidence, not from returning
to 1920x1030 full-frame masks at 20-30 Hz.

No more repeated micro-benchmarks should be run unless a specific engineering
change requires a measurement.
