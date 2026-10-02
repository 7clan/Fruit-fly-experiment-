# GPO semantic coach playbook — compact runtime knowledge

Purpose: local, cheap context for the DigitalFlyLab semantic coach. This is
ENGINEERED game knowledge. It does not change the biological fly network.
Current direct game observation/HUD always overrides this document when they
conflict.

Last researched: 2026-10-01.

## AI-only runtime contract (binding for the experimental branch)

This section overrides older helper-coach assumptions when `mode=AI_ONLY`.

Perception meanings:
- Yellow QUEST / ! marker = a quest giver is available at that screen location.
- Green recommended marker + distance = travel guidance toward the recommended
  quest/objective area. It does not prove a quest is already active.
- Red quest objective marker = the active quest's target LOCATION/DIRECTION.
  It is not, by itself, proof that an enemy body is in melee range.
- `quest_enemy_actor_visible=true` = the fast tracker has a persistent hostile
  NPC body spatially associated with the red quest objective. This is the
  local evidence required for M1 combat.
- `quest_status=active` is sticky perception state from a red objective/actor.
  Do not walk back to a yellow giver or try to take another quest while active.
- `quest_status=pending_accept` means T was already pressed; wait for the
  quest/dialogue state to change instead of pressing T repeatedly.
- `stuck=true` / `circling=true` means forward navigation is not producing
  measurable visual progress. Recover before repeating the same route.

Quest acceptance sequence:
1. Approach the yellow quest giver until centered and close.
2. Press T ONCE.
3. Wait for the dialogue/UI change.
4. If an in-game Accept/Yes/Confirm-quest button is visibly present, click its
   normalized center with UI_CLICK.
5. Verify acceptance from a red quest objective, changed quest HUD/counter, or
   other explicit current evidence before moving to combat.

Combat sequence:
1. If only the red objective marker is visible, navigate toward it.
2. When a tracked `quest_enemy_actor` is visible, close distance.
3. At melee range use repeated M1/basic attack. Block/evade or a verified HUD
   ability as the current fight requires.
4. Do not attack ordinary players or unrelated humanoids.
5. The on-screen SAFEZONE/PROTECTED label must NOT be used as an excuse to
   postpone quest-NPC PvE. In this experiment it is treated as PvP protection,
   while quest NPC combat remains a separate objective.
6. If defeated, WAIT through respawn, reacquire the active quest/objective and
   resume from current evidence instead of inventing an enemy location.

Movement/recovery binding:
- SPACE = jump.
- Left CTRL while contacting a wall/object = climb. The runtime CLIMB macro
  performs a short jump, then holds forward + CTRL to establish/maintain wall
  contact.
- Double W = sprint.
- Direction + Q = roll/dash.
- Repeated airborne SPACE = Geppo only if the character has it unlocked.
- AI-only target following may rotate the Roblox camera with a bounded RMB drag
  to center the AI-selected target before moving forward. This is actuator
  alignment, not a hidden goal selector.

Equipment/UI:
- 0-9 select hotbar slots. The current screenshot is authoritative for what is
  actually in each slot; never assume a permanent slot after the player changes
  inventory.
- `last_equipped_slot` is the last slot the automation physically selected.
- Default Melee currently exposes Gut Punch (E) and Ground Smash (R) in the
  observed project loadout, but only use loadout abilities when the current HUD
  / verified control catalog supports them.
- UI_CLICK may operate ordinary in-game quest, inventory, menu and Peli-shop
  buttons when the model can see the button and has a reliable normalized
  center. Never automate Robux/gamepass, account/security, external-link or
  trade confirmations.

Shell's Town progression facts relevant to the current level-20 test:
- Robert gives the level-20+ quest to defeat 8 Corrupt Marines.
- Kevin gives the level-25+ Shell's Bandit quest.
- Corrupt Marines are NPC enemies; the red objective should guide travel until
  a real tracked Corrupt Marine body is visible.
Sources: current GPO community NPC/Shell's Town references plus live project
observations from the 2026-10-02 AI-only run.

## 0. Core play loop

1. Understand the current scene before acting.
2. If a yellow QUEST NPC/marker is visible and no active quest is evident:
   approach it, interact with T, and verify that the quest UI/objective changed.
3. If a green recommended-quest waypoint is visible, treat it as the current
   travel objective.
4. If a red quest objective marker is visible, navigate toward it. Treat it
   as a location cue until a persistent quest_enemy_actor body is resolved;
   only that body is eligible for melee attacks.
5. When the quest counter completes, stop attacking, reacquire the next quest
   objective, and repeat.
6. If motion stops changing the scene for several seconds, classify the
   obstruction before repeating inputs: low ledge -> jump, climbable vertical
   surface -> CTRL+forward, open alternate path -> go around/backtrack.
7. Prefer verification after every irreversible/context-changing action:
   quest accepted, item bought, fruit eaten, fighting style changed, ship
   spawned, island reached, boss/quest completed.

## 1. Verified/common controls

Movement:
- W/A/S/D: character movement.
- Double-tap W: sprint.
- Q + a direction: roll/dash/Soru-like contextual dash.
- Left CTRL: climb while touching a wall; dive in water; contextual evasive
  while stunned.
- SPACE: jump; repeated airborne SPACE may Geppo if unlocked.
- P: sit/attach to a ship seat.
- M: game menu.

Combat:
- Left mouse / M1: basic attack.
- F: block; precise timing can perfect-block.
- R: reload when a firearm is equipped. R can also be a loadout move when a
  non-gun loadout exposes it, so live HUD context wins.
- J: Busoshoku Haki when acquired.
- G: Observation Haki when acquired.
- V: carry a downed player.
- B: grip/execute a downed target.

Quest/world:
- T: project-observed quest/NPC interaction control.
- Number row 0-9: hotbar/equipment slots.
- Equipped fruit/style/sword abilities are learned from the live HUD. Never
  assume that E/R/Z/X/C/V/B/N/Q/F/G/J name the same move for every loadout.

Camera:
- Hybrid fly mode does not rotate the user's camera.
- AI-only mode may use bounded RMB camera drags for explicit LOOK actions and
  for target-centering while realizing an AI-selected navigation goal.
- UI clicks are permitted only when the controller explicitly identifies an
  open in-game UI/dialog and supplies a high-confidence normalized location.

Sources:
- https://grand-piece-online.fandom.com/wiki/Controls
- live DigitalFlyLab HUD observations

## 2. Current game scope

Official Roblox listing, Update 13 era:
- Current max level: 675.
- GPO is an action RPG built around quests, fighting styles, weapons, fruits,
  islands, sea travel, bosses, raids and trading.
- Current fruit rarity list from the official Roblox experience:
  Common: Suke, Kilo, Spin, Heal.
  Rare: Bari, Mero, Horo, Gomu, Bomu.
  Epic: Yomi, Spring, Kira.
  Legendary: Mera, Pika, Hie, Magu, Goro, Gura, Zushi, Suna, Ito, Paw, Yuki,
  Kage, Yami, Goru, Smoke, Biscuit.
  Mythical: Tori, Mochi, Ope, Venom, Buddha, Pteranodon, Dragon, Soul, Leopard.
- Fruits may come from natural spawns, fishing, fruit chests, dungeons and
  other activities. Eating a fruit grants a moveset but generally sacrifices
  normal swimming; remove a consumed fruit with the game's removal mechanic
  before changing fruits.

Source:
- https://www.roblox.com/games/1730877806/Grand-Piece-Online
- https://grand-piece-online.fandom.com/wiki/Devil_Fruits

## 3. Level/progression route

Use the player's visible level and current island before choosing travel goals.
A practical level-guide route:

First Sea:
- Town of Beginnings: 0-10. Defeat Bandits / Bandit Boss.
- Sandora: 10-20. Desert Bandits.
- Shell's Town: 20-30. Corrupted Marines.
- Island of Zou: 30-40. Zou inhabitants.
- Baratie: 40-110 in the simple route; Krieg Pirates. Other optional islands
  can be used during this span.
- Skypiea / Sky Castle: 110-160. Castle Guards.
- Gravito's Fort: 160-190. Gravito's Undermen.
- Fishman Island: 190-325. Fishman Karate Users.
- At level 325, prepare for Second Sea.

Second Sea:
- Gain Second Sea access at level 325 by obtaining the World Scroll and using
  Reverse Mountain/gate progression.
- Thriller Bark: approximately 325-425 for straightforward quest grinding.
- Rose Kingdom: approximately 425-675 in the simple route, with bosses,
  Factory/scientist content and other activities as alternatives.

The detailed beginner guide also recommends optional stops for gear, Haki,
fighting styles and bosses rather than blindly grinding one quest forever.

Sources:
- https://grand-piece-online.fandom.com/wiki/Level_Guide
- https://grand-piece-online.fandom.com/wiki/Grinding_For_Beginners
- https://grand-piece-online.fandom.com/wiki/Second_Sea

## 4. Town of Beginnings starter plan

Town of Beginnings contains Bandits, Bandit Boss, starter quest NPCs and
starter shops/materials.

Useful quests:
- Defeat 5 Bandits: available from the beginning.
- Defeat 1 Bandit Boss: level 5+.
- The Bandit Boss can drop the Bandit Eyepatch.
- The island includes basic sailing supplies and starter ranged weapons.

Coach behavior:
- If level < 10-15, favor repeatable Bandit quests over wandering.
- Yellow QUEST -> approach -> T -> verify quest accepted.
- Green tracker -> navigate.
- Red quest marker -> fight only that objective.
- At level ~20, planning should shift toward Shell's Town rather than
  remaining at the starter island.

Source:
- https://grand-piece-online.fandom.com/wiki/Town_of_Beginnings

## 5. Movement / obstacle reasoning

Classify before acting:
- low obstacle/step: JUMP.
- wall with plausible climb surface: CLIMB (CTRL + forward).
- tall/non-climbable wall with free lateral space: GO_AROUND.
- repeated collision/no visual progress: BACKTRACK briefly, then choose a
  different side.
- air gap/high vertical route and Geppo known: GEPPO.
- long clear path: SPRINT.
- water/underwater route: remember CTRL can dive; Devil Fruit users may need
  a bubble/safe traversal depending on location.

Do not repeat jump/climb indefinitely. After 2 failed attempts, change
strategy and reobserve.

Geppo:
- Can be purchased later or supplied by compatible fighting styles.
- In some modes it has a limited number of airborne jumps before cooldown.

Source:
- https://grand-piece-online.fandom.com/wiki/Geppo

## 6. PvE combat policy

Target discipline:
- Only engage an NPC clearly tied to the active quest/red objective or an
  immediate NPC threat. Avoid attacking unrelated players.

Basic loop:
1. Keep target in front by character movement, not autonomous camera takeover.
2. Approach to useful range.
3. Use M1/basic attacks when close.
4. Use observed loadout abilities when their HUD binding and context are known.
5. Block with F against likely incoming melee hits.
6. Use Q-direction evade when a hit is imminent or the current learned defense
   value favors evade.
7. If health is critically low, disengage/pause rather than blindly attacking.
8. After knockdown/death/quest progress, reobserve instead of continuing a
   stale combo.

Starter default Combat loadout already observed by this project:
- M1 basic attack.
- E: Gut Punch.
- R: Ground Smash.
Live HUD overrides this profile after changing styles/items.

Farming tactics from the community guide:
- Terrain can be used to reduce incoming damage: walls, fences, cliffs and
  elevated positions are common PvE farming tools.
- AOE/ranged attacks are valuable when grouping NPCs.
- Against bosses, avoid assuming a low-level solo build can facetank; use
  range, terrain, friends or a known safe strategy.

Source:
- https://grand-piece-online.fandom.com/wiki/Grinding_For_Beginners

## 7. Fighting-style planning

Default Combat:
- Starts available and scales with Strength.

Black Leg:
- Early accessible style, strong practical farming/offense choice.
- Grants/uses Geppo utility and can evolve into Demon Step.
- Community guide commonly prefers it for fruitless early farming.

Rokushiki:
- Utility-oriented; includes movement/defensive tools such as Geppo/Soru/
  Tekkai depending on unlocked skills.
- Often paired with a strong fruit rather than used as the main early farming
  damage source.

Race-linked / later styles include Fishman Karate, Electro evolutions,
Cyborg, and others. Sword styles use Sword Mastery instead of ordinary
Strength scaling.

Planning rule:
- Do not buy/swap a fighting style just because it is available.
- Compare current build, race, level, money, fruit and desired role.
- Prefer unlocks that improve mobility/survivability first on this autonomous
  agent because navigation failure is more expensive than marginal DPS.

Source:
- https://grand-piece-online.fandom.com/wiki/Fighting_Styles

## 8. Stats/build planning

- Players gain 3 stat points per level.
- Strength primarily improves physical/fighting-style/non-sword M1 damage.
- Sword, Gun, Fruit and Fighting Style mastery categories unlock/scale their
  own relevant tools depending on item/style.
- Defense and stamina matter for autonomous survivability.

Coach rule:
- Never spend stats from a screenshot guess alone.
- First identify the current build, fruit/style/weapon, available points, and
  next unlock threshold.
- Ask/verify via the visible menu before committing a large stat allocation.
- Favor enough survivability to avoid repeated deaths, then hit the next
  important move/mastery threshold, then damage.

Source:
- https://grand-piece-online.fandom.com/wiki/Stats

## 9. Haki plan

Busoshoku/Armament:
- J toggles it after acquisition.
- Improves combat and enables reliable interaction with Logia targets.
- V1 is obtained through its quest line; V2 is a later upgrade.

Observation/Kenbunshoku:
- G toggles it after acquisition.
- Treat as a defensive/awareness resource when available.

Coach rule:
- If fighting a Logia-like target and attacks are ineffective, first verify
  whether Busoshoku is owned/active before concluding the target is bugged.
- Avoid wasting a depleted Haki resource outside useful combat.

Sources:
- https://grand-piece-online.fandom.com/wiki/Busoshoku
- https://grand-piece-online.fandom.com/wiki/Busoshoku_V2
- https://grand-piece-online.fandom.com/wiki/Category:Haki

## 10. Weapons and ranged options

Main combat families include fighting styles, swords, guns and fruits.

Swords:
- Typically scale from Sword Mastery, have their own move bindings, and are
  often combined with another combat method.
- Examples in early/mid progression include Katana and boss-drop swords.

Guns:
- Require aiming/reload behavior; R reloads guns.
- Useful for pulling/cheesing NPCs from range even when not the final build.
- The beginner guide specifically notes ranged weapons as a way to fight from
  safer terrain.

Coach rule:
- Learn current hotbar/equipped item from HUD before issuing ability keys.
- Prefer a ranged pull when melee approach repeatedly causes avoidable damage.

Sources:
- https://grand-piece-online.fandom.com/wiki/Swords
- https://grand-piece-online.fandom.com/wiki/Guns

## 11. Devil-fruit planning

General:
- Do not eat a fruit merely because it is rarer than the current one.
- Evaluate farming, mobility, PvE AOE, water-travel downside, current stats
  and whether replacing the current fruit is reversible/desired.
- Strong mobility/water-compatible utility can matter more to an autonomous
  character than theoretical PvP value.
- Community sources mention several fruit acquisition routes: natural spawn,
  fruit chests, dungeons, sea content, ship farming and other events.

Coach rule:
- Before eating/replacing a fruit: identify exact fruit, current fruit, build,
  and explicit user policy. Mark this as a high-impact purchase/use action.
- Fruit moves must be learned from the live HUD after equip/eating rather than
  from assumed fixed hotkeys.

Source:
- https://grand-piece-online.fandom.com/wiki/Devil_Fruits
- official Roblox experience page

## 12. Ships / sea travel

Common actions:
- Spawn/select the owned ship through the appropriate UI.
- Sit/attach with P when at the seat.
- Use ordinary movement steering for ship control when the game accepts it.
- Navigate by objective/compass/pose, not random sea wandering.
- Repair with the appropriate purchased materials/tools when needed.

Known examples:
- Caravel: purchasable starter ship; robust early sea transport.
- Galleon: a useful non-gamepass progression ship available later in the
  early route.
- Hoverboard: compact/faster later option and useful for ship-farming contexts.

Ship farming:
- Marine/Pirate NPC ships can be farmed for bounty/resources/fruit chances,
  but the community guide notes low drop rates and that the captain matters.
- AOE/ranged capability makes ship farming much more practical.
- Do not prioritize ship farming early if quest leveling or basic gear gives
  much better progress.

Sources:
- https://grand-piece-online.fandom.com/wiki/Caravel
- https://grand-piece-online.fandom.com/wiki/Hoverboard
- https://grand-piece-online.fandom.com/wiki/Ship_farming
- https://grand-piece-online.fandom.com/wiki/NPC_Ships

## 13. Accessories / equipment purchases

Accessories occupy slots and grant stat bonuses. Many come from bosses,
dungeons/raids or stores.

Planning rule:
- Compare by useful effective stats for current build: HP, stamina,
  regeneration, damage bonuses, resistances, utility.
- Do not replace an equipped item unless the new item is actually better for
  the relevant slot/build.
- Prefer readily obtainable survival upgrades while leveling; postpone
  difficult low-drop-rate vanity/min-max items until progression supports
  farming them.

Source:
- https://grand-piece-online.fandom.com/wiki/Accessories

## 14. Buying / UI interaction policy

The coach may identify shops, dialog choices, menu buttons and inventory
buttons visually.

For purchases:
1. Read item name and price from the current UI.
2. Verify the item is relevant to the current progression plan.
3. Verify enough currency/resources are visible or known.
4. For expensive/irreversible choices (fruit consumption/removal, race/style
   replacement, large stat reset/allocation), prefer WAIT unless policy
   explicitly authorizes it.
5. For ordinary low-risk progression purchases (starter ship, required
   quest/travel item, cheap potion/repair material), the coach may recommend
   BUY_ITEM.

UI_CLICK is allowed only when:
- an actual dialog/menu/shop is visibly open;
- the coach can point to a specific labeled button;
- confidence is high;
- normalized x/y is inside the Roblox client frame;
- it is not an advertisement, external link, chat, trade confirmation with
  another player, Robux purchase, or account/security UI.

## 15. Trading / player interactions

Trading exchanges player-owned items and takes place in the Trading Hub.
Because values change and another human is involved, autonomous trading is
NOT authorized by the semantic coach. It may explain a trade or recommend
research, but it must not click final Accept/Confirm.

Source:
- https://grand-piece-online.fandom.com/wiki/Trading

## 16. Long-term autonomous planner priorities

At any moment choose the cheapest useful next objective:

A. Survival / locomotion blocker
- Fix repeated death, no ship, no repair materials, no mobility, or a route
  problem before chasing rare drops.

B. Level gate
- If under the next island/quest requirement, grind the best reliable quest.

C. Mobility unlock
- Ship/pose/Geppo/route knowledge can save more time than a small DPS upgrade.

D. Combat breakpoint
- Buy/unlock a move, style, weapon, Haki or stats only if it meaningfully
  improves the current PvE loop.

E. Optional farming
- Boss accessories, rare weapons, fruits, sea content and raids when the build
  can farm them reliably.

F. Second Sea / endgame
- At level 325 prioritize the World Scroll/Reverse Mountain transition.
- At higher levels choose between Rose Kingdom questing, bosses, Factory/
  raid/sea content based on reliability and desired drops.

## 17. Semantic-coach output discipline

The coach is a high-level instructor, not a raw key generator.

It may choose:
- WAIT
- TAKE_QUEST
- NAVIGATE_OBJECTIVE
- FIGHT_QUEST_TARGET
- BLOCK
- EVADE
- JUMP
- CLIMB
- GO_AROUND
- BACKTRACK
- SPRINT
- GEPPO
- INTERACT
- USE_HAKI
- EQUIP_SLOT
- EXEC_CONTROL
- BUY_ITEM
- UI_CLICK
- BOARD_SHIP
- TRAVEL
- REOBSERVE

It must:
- explain what it thinks is happening in one short sentence;
- provide confidence;
- choose only one immediate skill at a time;
- prefer REOBSERVE/WAIT when uncertain;
- never output arbitrary raw keyboard scan codes;
- never invent that an item/fruit/style is owned;
- never attack ordinary players;
- never authorize Robux purchases/trading confirmations/account actions;
- never move/drag the user's camera.

The fruit fly remains the low-level biological orientation/navigation system.
The semantic coach supplies game meaning and high-level objectives.


## 18. Current equipment / style catalog highlights

This is a compact planning index, not a substitute for on-demand wiki lookup.

Fighting styles currently represented by the community catalog include:
Default Combat, Black Leg, Demon Step, Rokushiki, Kamishiki, Dragon Claw,
Electro, Moonlit Electro, Fishman Karate, Abyssal Karate, Cyborg, 1 Sword
Style, 2 Sword Style, 3 Sword Style, Iron Fist, Vampire, Dullahan and related
race/evolution styles.

Useful acquisition examples:
- Black Leg: Baratie trainer; low-cost early general-purpose style and grants
  Geppo utility if not already owned.
- Rokushiki: utility-heavy style; useful alongside a stronger primary damage
  source.
- Fishman Karate: race-gated and resource-gated; strong farming/damage when
  the build supports it.
- Cyborg: Second Sea / Rose Kingdom progression and race/gear gated.
- 1SS -> 2SS -> 3SS is a Sword Mastery progression chain; do not invest in it
  unless the profile is intentionally becoming a sword build.
- Iron Fist and 3SS are later/high-level progression goals.

Early/current sword examples:
- Katana: cheap early Sword Mastery weapon from Roca Island / related quest.
- Kiribachi: Arlong boss drop.
- Ryu's Katana: Fishman Island boss drop.
- Skyblue Katana / Golden Staff: Skypiea-related drops.
- Gravity Blade: Gravito drop.
- Neptune's Trident: Fishman Island boss drop.
- Bisento: Marine Base G-1 boss drop.
The full current sword catalog is large and update-sensitive; request an
on-demand wiki lookup before committing a long farm.

Accessory planning:
- Accessories have separate slots (head/face/forehead/ear/neck/armor/back/
  shoulder/waist/misc and special-eye slots). Compare the actual slot and
  effective stats, not rarity alone.
- Starter example: Bandit Eyepatch gives a small HP increase and drops from
  Bandit Boss, so it is a reasonable free early survivability upgrade.
- Later accessories commonly trade among HP, stamina, regen, elemental
  resistance and build-specific damage.
- If a new accessory conflicts with an equipped slot, compare visible/current
  stats before replacing it.

Ships:
- Rowboat, Caravel and Galleon are normal Peli progression ships.
- Ship HP loss slows travel. Hammers + wooden planks repair normal ships; each
  plank repairs a small amount.
- Rough Waters slow ships and can trigger sea threats.
- When a normal ship is required for progression, prefer a reliable Peli ship
  and repair supplies before rare/Robux mobility.
- Never choose a Robux-only faster repair/item path autonomously.

Sources:
- https://grand-piece-online.fandom.com/wiki/Fighting_Styles
- https://grand-piece-online.fandom.com/wiki/Swords
- https://grand-piece-online.fandom.com/wiki/Accessories
- https://grand-piece-online.fandom.com/wiki/Ships
