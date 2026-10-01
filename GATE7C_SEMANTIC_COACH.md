# Gate 7C — Semantic Coach architecture

Date: 2026-10-01

## Purpose

Gate 7C adds a low-rate multimodal GAME SEMANTICS layer above the canonical
Drosophila brain. It does not replace the fly network.

- Biological fly: left/right/approach/retreat/escape orientation.
- Semantic coach [ENGINEERED]: understand GPO situations, quests, shops,
  progression, obstacles, combat context, ships and visible UI.
- Skill executor [ENGINEERED]: convert a small approved semantic skill into
  verified GPO controls.

The goal is to stop expanding brittle handcrafted visual rules for every
possible game situation while keeping the experiment's biological/engineered
boundary explicit.

## Cloud model

Default for new projects: `gemini-3.5-flash-lite`.

Rationale:
- multimodal image + text input;
- structured JSON output;
- no local model load on the i7-5500U / 8 GB target machine;
- low-rate calls, normally no more often than every 8 seconds;
- Google currently directs new Gemini projects toward 3.5 Flash-Lite.

Existing Gemini projects that still have 2.5 access may set
`GEMINI_MODEL=gemini-2.5-flash-lite` for the lower token price.

The API key is read only from `GEMINI_API_KEY`; setup stores it in the
Windows USER environment, never the repository.

## Knowledge

Runtime knowledge is in:
`knowledge/GPO_SEMANTIC_PLAYBOOK.md`

It includes:
- quest loop and starter progression;
- movement/obstacle reasoning;
- combat/defense tactics;
- level/island route;
- fighting styles;
- stats/build planning;
- Haki;
- weapons;
- current fruit classes and replacement policy;
- ships/sea travel/ship farming;
- accessories/equipment;
- shops and purchases;
- long-term progression priorities;
- verified controls;
- UI interaction policy.

Live screenshot/HUD always overrides static knowledge.

## Coach output

One plan at a time:
- scene
- objective
- target
- skill
- optional verified `control_id`
- optional normalized visible UI click
- confidence
- short explanation
- next step after success

The model cannot directly output raw scan codes.

## Approved semantic skills

WAIT, TAKE_QUEST, NAVIGATE_OBJECTIVE, FIGHT_QUEST_TARGET, BLOCK, EVADE,
JUMP, CLIMB, GO_AROUND, BACKTRACK, SPRINT, GEPPO, INTERACT, USE_HAKI,
EQUIP_SLOT, EXEC_CONTROL, BUY_ITEM, UI_CLICK, BOARD_SHIP, TRAVEL,
REOBSERVE.

## Input constraints

- Character navigation uses W/A/D/S etc.; autonomous camera movement is hard
  disabled in QuestingBackend.
- Basic attack may use M1.
- UI_CLICK is allowed only for a high-confidence explicitly visible in-game
  UI button and only while the captured Roblox HWND is foreground.
- UI click saves the user's cursor, clicks the identified point, then restores
  the cursor immediately.
- Platform-money/account/external/trade-confirmation UI is hard-blocked.
- Coach combat only bridges to ATTACK when the current target is a confirmed
  quest enemy marker.
- User F12 emergency stop remains global.

## Cost/load control

- no local VLM is loaded;
- screenshot sent to coach is capped at 480 px wide JPEG quality 55;
- API calls occur only while agent is enabled;
- minimum call interval: 8 s;
- unchanged situations refresh at most every 30 s;
- coach runs in its own worker thread;
- urgent health-drop defense does NOT wait on the cloud model.

## First test scope

The first Gate-7C run should verify:
1. coach API preflight succeeds before the brain starts;
2. dashboard displays AI COACH skill/confidence/scene;
3. yellow quest -> interact recommendation;
4. green objective -> navigation recommendation while fly supplies direction;
5. red quest enemy -> fight/defense recommendation;
6. no autonomous camera motion;
7. no UI click unless an actual menu/dialog button is visible;
8. replay contains coach.events and the session report contains coach_state.

Do not judge accessory/fruit/ship purchasing from this first starter-island
test; those capabilities should be exercised only after the basic semantic
loop is verified.
