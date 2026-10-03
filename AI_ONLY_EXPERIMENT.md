# AI-only autopilot experiment

This branch is intentionally separate from `main`.

- `main` keeps the digital-fly + semantic-coach research stack unchanged.
- `ai-only-autopilot` removes the fly brain from the **gameplay control path**.
- Gemini Flash-Lite AI is the only gameplay decision-maker in the default launcher.
- Local CV measures screen geometry; the execution adapter only realizes the
  AI-selected skill and does not choose quests, combat, equipment, or routes.
- The canonical Brian2 brain and fly-channel encoder are not started in this
  mode.
- F12 remains the global emergency stop.

## What the AI can decide

The controller receives the current frame when the cloud backend accepts
vision, structured CV state, the full GPO semantic playbook, persistent
character facts learned from visible state, the verified control catalog, and
on-demand GPO wiki context.

It can select quest interaction, objective navigation, combat, block/evade,
jump/climb/dash, explicit camera look left/right, equipment slots, observed
HUD abilities, Haki, ship boarding, ordinary in-game UI clicks/purchases, and
verified movement controls.

The local execution layer does not run the old deterministic quest/PvE
supervisor in this mode. It repeats only the AI's currently selected persistent
skill (for example, follow a waypoint or M1 a confirmed quest enemy) against
fresh visual geometry until the next AI plan arrives.

## Run on Windows

From the repository:

```powershell
git fetch origin
git switch ai-only-autopilot
git reset --hard origin/ai-only-autopilot
.\run_ai_only_windows.ps1
```

The launcher uses the existing Gemini API-key configuration. It verifies both
text and image input before launch, focuses Roblox, and starts the autopilot.
It intentionally does not fall back to the slower Ollama 31B path. No Brian2
prewarm is performed.

Latency mitigation is split into two layers:
- cloud AI decides the semantic goal/action;
- an 8 Hz local tracker follows only the AI-selected bounding box and the
  local executor persists that AI-selected skill between cloud replies.

This means local code reacts quickly without secretly choosing a different
quest, NPC, item, or combat goal.

For a clean experiment, do not manually move, click, equip, or fight after the
run starts. Use F12 to stop; the normal session report/ZIP is still saved.

## Evidence to inspect

A clean AI-only run should show:

- `runtime=disabled_ai_only`
- `brain_chunks=0`
- `[AI->PLAN]` for cloud decisions
- `[AI-ONLY->EXEC]` for commands physically realized from those decisions
- executor `command_only=true`
- AI-only supervisor `decision_owner=cloud_ai`
- no `quest_combat_supervisor` worker in the session report

This makes attribution simple: if an input appears in the replay, it must come
from an AI plan (apart from the explicit F8-F12 safety/control hotkeys).


## Compound combat controller

The cloud AI no longer chooses only one physical key/action during combat.
Every FIGHT plan includes five channels:

- locomotion: approach / orbit left / orbit right / retreat / hold / recovery
- offense: M1 or a move whose binding is visibly grounded in the HUD
- defense: none / guard-between-attacks / evade
- camera: track target / explicit search
- equipment: optional verified hotbar slot

The local executor runs those AI-selected channels concurrently/interleaved at
high frequency while the slow cloud plan remains current. This is intentionally
a latency bridge, not a second combat policy.

## Quest truth

The accepted quest HUD is authoritative. A visible objective/progress panel
(e.g. Defeat X n/m + Rewards/QUIT) latches quest-active state even when the
enemy body is hidden. The red quest dot remains a pursuit cue through walls.
The AI only treats ordinary red UI elements as non-targets.

## Input ownership on Windows

A normal Windows desktop has one shared OS cursor and one foreground keyboard
input stream. This branch therefore cannot create a truly independent second
mouse through SendInput.

Instead, AI camera/UI/M1 operations use short cursor leases: save the user's
cursor location, perform the bounded Roblox interaction, then restore the exact
desktop cursor location. F12 remains the emergency stop.

If truly independent simultaneous human + agent input is required later, use a
separate game session/VM or investigate a controller/virtual-gamepad backend;
Roblox supports gamepad input, but GPO's exact bindings would need to be
verified before replacing the tested keyboard/mouse backend.
