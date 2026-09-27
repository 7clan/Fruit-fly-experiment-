# GPO_PLAN — Grand Piece Online integration (source policy + perception + phases)

> **POLICY CORRECTED 2026-09-28** (see `docs/MASTER_SPEC.md` §6). The
> single-permitted-source rule now applies ONLY to Roblox properties.
> Non-Roblox websites and resources (independent GPO wikis, guides,
> YouTube gameplay, community discussions, independent databases,
> articles) ARE permitted for learning GPO facts, but everything obtained
> from them is stored as **UNVERIFIED GUIDE KNOWLEDGE** until confirmed by
> direct game observation (**VERIFIED IN GAME**). If a guide conflicts
> with current direct observation, the direct observation wins. The
> knowledge database with these tiers is built at D17.

## 0. Source policy (binding, corrected)

The ONLY Roblox URLs permitted by this project:

    https://www.roblox.com/games/1730877806/Grand-Piece-Online
    https://www.roblox.com/fr/games/1730877806/Grand-Piece-Online   (historical form, same page)

No other Roblox URL, no Roblox documentation, developer/Creator pages,
APIs, support pages, forums, or experiences. NON-Roblox third-party
resources are permitted under the UNVERIFIED/VERIFIED discipline above.
Anything not obtained from the permitted page, an allowed non-Roblox
source (as UNVERIFIED), or direct observation of the game screen during
experiments is **UNKNOWN — must be discovered experimentally**. No game
mechanic is ever hard-coded from assumption. This document is the living
register.

## 1. Register of page-observable facts (fetched 2026-09-27)

Verbatim/paraphrased from the permitted page only:

| fact | value (as shown on the page) |
|---|---|
| title | [LEOPARD] Grand Piece Online |
| creator | Grand Quest Games |
| genre | RPG (subgenre: Action RPG) |
| maturity | Moderate; content: blood (light/realistic), violence (repeated/light) |
| description | "Set out on a journey on the vast sea in seek of wealth, fame, and power! Gain experience through quests, discover various fighting styles, and find ability-granting fruits to become stronger! Compete with other players through arena modes and battle royale!" |
| max level | 675 (page states "Current max level: 675") |
| fruits | rarity tiers exist: Common (Suke, Kilo, Spin, Heal), Rare (Bari, Mero, Horo, Gomu, Bomu), Epic (Yomi, Spring, Kira), Legendary (Mera, Pika, Hie, Magu, Goro, Gura, Zushi, Suna, Ito, Paw, Yuki, Kage, Yami, Goru, Smoke, Biscuit), Mythical (Tori, Mochi, Ope, Venom, Buddha, Pteranodon, Dragon, Soul, Leopard) |
| fruit acquisition (page) | "natural spawns under trees, fishing, fruit chests, dungeons, and much more" |
| weekend boost (page) | Fri–Sun: 2x Legendary/Mythical fruit rates, 2x boss drop rates, 1.25x EXP |
| premium perks (page) | bonus XP in battle royale battlepass, extra rewards in AFK world, reduced natural fruit spawn timers |
| platforms | Controller, PC, Mobile support |
| live stats at fetch | Active 12,253; Favorites 1,508,107; Visits 1.3B+; likes 121K+; voice chat: not supported; updated 27/09/2026 |
| badges (examples) | "L'eclosion: Oeuf GPO" (collect an egg at the Ville des Commencements / Town of Beginnings, in the first sea, during the hatching event); "Une nouvelle aventure"; "Commercant certifie" |
| systems named on page | quests, fighting styles, fruits, arena modes, battle royale, bosses, dungeons, AFK world, battlepass, events, an in-game store, "first sea" (implies multiple seas) |

## 2. UNKNOWN — must be discovered experimentally

Everything below is NOT available from the permitted page. Each entry has a
discovery protocol (section 4). NOTHING in this list may be hard-coded.

- entry flow: menus, character/mode selection, loading screens, spawn
- UI layout: HUD element positions, health bar, level display, inventory
- controls: keybinds, camera, movement mechanics
- quest system: NPCs, quest list, objectives, rewards, XP per quest
- leveling curve: XP per level (only the MAX LEVEL 675 is known)
- combat: mechanics, damage, blocking, dodging, death/respawn
- items/equipment: shops, currency, inventory, equip mechanics, effects
- fruit mechanics: what each fruit does, how equipping/using works
- dungeons: locations, mechanics, rewards
- bosses: locations, health, drops, respawn timers
- AFK world: access, rules, reward rates
- arena / battle royale: entry points, rules, rewards
- party/trading/social systems
- geography: map layout, islands, "first sea" boundaries and beyond
- server size (page value did not render reliably), private server
  availability (NOT stated on the page)
- anti-cheat / AFK detection behavior (safety-relevant: UNKNOWN)

## 3. Perception architecture (Phase 6+)

SCREEN -> COMPUTER VISION -> OBSERVED STATE. No APIs, no memory reads, no
packet inspection, no undocumented data. The observed state is exactly what
a human player could see on screen, structured:

```json
{
  "screen_state": "UNKNOWN_SCREEN",
  "player_visible": true,
  "health_estimate": 0.72,
  "level_estimate": 12,
  "npc_visible": false,
  "enemy_visible": false,
  "target_direction": 35,
  "target_distance": 250,
  "hud_text_raw": "..."
}
```

UNKNOWN is a valid value everywhere. Values are never invented.

Screen-state machine (calibrated, not hard-coded):
UNKNOWN_SCREEN -> MENU_DETECTED -> MODE_IDENTIFIED -> LOADING -> GAMEPLAY
-> (DEAD | STUCK | OTHER_UI). Region coordinates come from a calibration
procedure (click the HUD elements once), stored per resolution, validated
by state-classifier accuracy — never assumed.

## 4. Discovery protocol for anything UNKNOWN

For each unknown object/mechanic X:

1. OBSERVE passively: record screenshots + state before/after.
2. Probe with ONE safe action at a time: approach / interact / move away /
   attack (only in low-risk contexts), one variable per probe.
3. Record ACTION -> OBSERVED CONSEQUENCE -> REWARD CHANGE into the
   knowledge graph (world_model/).
4. Repeat probe N>=3 times or until consequence is stable; store
   confidence; label remains `entity_X` until the relationship structure
   justifies a human-readable label.

Example (a labeled entity is earned, not assumed):
`entity_A` approached + interacted -> objective text changed, progression
counter increased later -> tentatively "quest_npc" with confidence 0.8.

## 5. Phased integration (maps to ROADMAP phases 6–14)

| GPO phase | content | project phase |
|---|---|---|
| A | screen observation only, state machine calibration | 6 |
| B | player detection, HUD readback, obvious objects | 6 |
| C | fly controls a tiny movement subset (supervised) | 7 |
| D | reliable fly-driven movement + failure recovery | 7 |
| E | target-directed navigation | 8 |
| F | interaction + unknown-object probing | 8 |
| G | simple objectives with progression rewards | 9 |
| H | multi-step objectives (chains) | 10 |
| I | exploration system | 12 |
| J | strategy selection | 13 |
| K | long-term progression optimization | 14 |

## 6. Risk register (honest, unchecked-by-me items marked VERIFY)

1. **Platform terms & automation (VERIFY, user-owned)**: whether input
   automation and screen capture are permitted for this game/platform under
   its current terms is UNKNOWN from the permitted page. Before Phase 7,
   the user must read the current terms directly and decide. This project
   deliberately does not access them (outside the permitted source).
2. **Other players' experience**: the game is multiplayer (page shows
   "Compete with other players"). Mitigations to evaluate at Phase 7:
   low-activity hours; isolated areas; whether private servers exist
   (UNKNOWN — not stated on the page); halt-on-other-player proximity as a
   configurable policy.
3. **Progression economy**: automated play can distort shared-game
   economies. Mitigation: keep sessions short and logged; review impact at
   each phase gate; stop if the game's operators or community signal harm.
4. **AFK/anti-cheat detection (UNKNOWN)**: if the game detects and punishes
   automation, the experiment ends. Detection is itself an observable to
   record, not evade.
5. **Moderation/ban risk to the user's account**: the user accepts this
   risk knowingly; a disposable account is recommended if permitted by the
   terms (VERIFY — account policy is outside the permitted page).
6. **Welfare crossover**: no experiment may ever trade animal welfare for
   game performance. The fly's session limits (docs/METHODOLOGY.md) hold
   regardless of game state.
