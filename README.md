# FlyAgent — Phase 1

**A hybrid biological/computational game-playing agent: a fruit fly's behavior
drives a game agent, while the computer provides perception, memory, planning,
and action translation.**

Phase 1 delivers the complete experimental apparatus, validated end-to-end
WITHOUT any animal, against a 2D artificial environment. Nothing in Phase 1
touches the real game (Grand Piece Online). See `docs/ROADMAP.md` for the
full 15-phase plan from here to autonomous long-term game progression.

---

## Quick start

```bash
pip install -r requirements.txt

# 1. verify the whole pipeline (tracker, decoder, game, reward, DB, calibration)
python -m tests.test_pipeline

# 2. run the Phase-1 control-group demonstration (headless, ~5 min)
python main.py demo

# 3. inspect the results
#    runs/demo_<timestamp>/comparison.txt   <- table + bootstrap CIs
#    runs/demo_<timestamp>/comparison.png   <- learning curves
#    runs/<timestamp>_<condition>/          <- per-run SQLite DB + manifest
```

Expected demo outcome (30 trials/condition, synthetic fly):

| condition           | what it is                                  | success rate |
|---------------------|---------------------------------------------|--------------|
| `random`            | floor: random actions                       | ~3–10%       |
| `fixed`             | ceiling: computer picks greedy direction    | 100%         |
| `fly_random`        | fly-controlled, fly ignores the cue         | ~floor       |
| `fly_cue`           | fly-controlled, goal-directed cue           | **~70–90%**  |
| `fly_cue_shuffled`  | same fly, cue position randomized           | ~floor       |

If the bootstrap CIs for `fly_cue − random` and `fly_cue − fly_cue_shuffled`
exclude 0, the apparatus can detect a biological-behavior contribution
(Phase-1 gate). This is the machine-checkable precondition before any real
fly is ever involved.

## Other commands

```bash
python main.py run --controller random  --trials 30       # one condition
python main.py run --controller fly --cue-mode shuffled   # cue control
python main.py run --controller fly --cue-bias 0.0        # random-walk fly
python main.py run --controller fly --source video --video fly.mp4
python main.py run --controller keyboard --show           # human control (display)
python main.py run --controller fly --source webcam --show # live camera (display)
python main.py report --runs runs/<a> runs/<b>             # compare runs
python main.py calibrate                                  # 4-click arena calibration
```

## Repository layout

```
fly-agent/
├── main.py                    # CLI (demo / run / report / calibrate)
├── config/
│   ├── actions.yaml           # fly behavior -> action mapping (CHANGE HERE)
│   ├── experiment.yaml        # game, reward, controllers, demo conditions
│   └── tracking.yaml          # camera / tracker / calibration parameters
├── flyagent/
│   ├── fly_tracker/           # sources (synthetic/webcam/video), MOG2+diff
│   │                          #   tracker, 4-point homography, trajectory
│   ├── biological_agent/      # behavior classifier, action decoder,
│   │                          #   synthetic Drosophila (burst-pause + taxis)
│   ├── environment/           # 2D arena (player, target, obstacles) + render
│   ├── controller/            # fly / random / fixed / keyboard controllers
│   ├── learning/              # sparse + shaped reward
│   ├── experiments/           # trial loop, runner, control groups, bootstrap
│   ├── data/                  # SQLite schema, immutable run directories
│   └── visualization/         # live dashboard (display) + learning curves
├── tests/test_pipeline.py     # pipeline smoke tests
├── scripts/calibrate_arena.py # interactive 4-click calibration tool
├── docs/                      # ROADMAP, METHODOLOGY, HARDWARE, GPO_PLAN
└── runs/                      # every run: manifest.json + experiment.db + summary.json
```

## The scientific rule this repo enforces

Never say "the fly learned X" unless the experiment establishes it. The
correct phrasing is: *"the fly-controlled system showed improved performance
under condition X"* — and then the control conditions (random, fixed,
shuffled-cue) rule out the alternative explanations (computer algorithm,
cue regularity, environmental structure). All conditions run through the
SAME trial loop, the SAME metrics, and the SAME database, so any performance
difference is attributable to the controller alone. See `docs/METHODOLOGY.md`.

## Attribution (what does what)

- **Computer provides**: cue placement (goal selection), action translation,
  reward, metrics, logging, statistics.
- **Fly provides**: movement behavior that selects among the presented
  options (approach the cued edge or not).
- **Hard-coded**: the action vocabulary, the decoder rules, the reward
  definition. Nothing is "learned" in Phase 1 — the synthetic fly's cue
  preference is a MODEL PARAMETER, not a learned skill.
- **Learned**: nothing yet. Learning enters in Phase 4+ (`docs/ROADMAP.md`).

## Animal welfare

Phase 1 uses NO animal. When real flies arrive (Phase 2), the protocol is
observation + benign visual stimuli only: no surgery, no implants, no
chemicals, no aversive stimuli, no injury. Drosophila melanogaster from a
registered stock center, sessions <= 30 min, rest days between sessions.
If any procedure would require institutional oversight, the protocol stops
and says so. See `docs/METHODOLOGY.md` (welfare section) and `docs/HARDWARE.md`.

## Source policy for the target game

The ONLY game-related source permitted is
`https://www.roblox.com/fr/games/1730877806/Grand-Piece-Online`.
Everything not observable on that page or by direct observation of the game
screen is UNKNOWN and must be discovered experimentally. The full register
of page-observable facts vs UNKNOWNs is in `docs/GPO_PLAN.md`. Game
integration begins at Phase 6, after the 2D milestones.
