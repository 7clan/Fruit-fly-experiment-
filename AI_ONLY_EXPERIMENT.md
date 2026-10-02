# AI-only autopilot experiment

This branch is intentionally separate from `main`.

- `main` keeps the digital-fly + semantic-coach research stack unchanged.
- `ai-only-autopilot` removes the fly brain from the **gameplay control path**.
- Ollama Cloud AI is the only gameplay decision-maker.
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

The launcher uses the existing Ollama API key configuration. It probes cloud
access, resolves the working hosted model, focuses Roblox, and starts the
autopilot. No Brian2 prewarm is performed.

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
