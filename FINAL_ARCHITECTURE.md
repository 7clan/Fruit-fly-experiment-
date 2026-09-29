# FINAL ARCHITECTURE — Digital Drosophila → Windows → GPO

Status: **AUTHORITATIVE** (2026-09-29). This document supersedes previous
implementation-order suggestions where they conflict. It captures the final
execution specification for the applied (game-playing) phase of the project.

Scientific record so far (immutable, do not redo):

| Gate | Verdict |
|---|---|
| Gate 1 — digital Drosophila brain reproducibly runs | **PASS** |
| Gate 2 — sensory / learning / motor neural I/O mapping | **PASS** |
| Gate 3 — fixed connectome controls artificial environment | **PASS** |
| Gate 4 v1 — behavioral reward learning | **FAIL** |
| Gate 4 v2 — G4A FAIL / G4B FAIL / **OVERALL FAIL** |

A cue-specific KC→MBON synaptic trace (v2 result A2) is a positive
scientific result, but it is **not** behavioral associative learning.

> **BIOLOGICAL PARAMETER TUNING IS FROZEN.** No further biological-learning
> tuning. The biological-learning workstream is closed as an experimental
> record. The applied system below is explicitly HYBRID and every
> engineered component is labeled ENGINEERED.

---

## 1. System overview

The practical agent is a **HYBRID**:

```
ROBLOX / GPO  (separate normal Windows process)
        |
WINDOW CAPTURE  (Windows.Graphics.Capture; ONE stream feeds vision + dashboard)
        |
PERCEPTION  (fast vision 20–30 Hz; heavy vision / OCR 5–20 Hz)
        |
STRUCTURED WORLD OBSERVATION  (compact typed state)
        |
WORLD MODEL / MEMORY / VALUE / GOAL CONTEXT   [ENGINEERED helper]
        |
COMPACT CONTEXT + SENSORY SIGNALS  (fly input channels)
        |
DIGITAL DROSOPHILA BRAIN  (canonical 138,639-neuron Shiu/FlyWire v783)
        |
DESCENDING / MOTOR / DRIVE ACTIVITY  (DN population rates)
        |
INTENTION DECODER  (STOP / TURN / APPROACH / EVADE / ATTACK / …)
        |
ABILITY RESOLVER  [ENGINEERED]  (intention → concrete game ability)
        |
WINDOWS INPUT EXECUTOR  (MotorExecutor; timestamped, logged, gated)
        |
ROBLOX / GPO
```

Two hard rules:

1. **The helper must NOT normally directly choose raw keyboard/mouse
   actions.** It outputs GOALS / CONTEXT / VALUE / RELEVANCE. The fly
   brain stays meaningfully involved in low-level behavior.
2. **No CV→input shortcut.** Computer vision may not bypass the
   fly/intention layer to press keys (exceptions: human emergency stop,
   technical failsafe, application recovery — each logged separately).

---

## 2. Three agent modes (all must remain available)

| Mode | Brain | Plasticity | Helper stack | Use |
|---|---|---|---|---|
| **A: PURE_FIXED_FLY** | canonical fixed connectome | none | none beyond I/O conversion | canonical scientific control |
| **B: BIOLOGICAL_PLASTICITY_FLY** | canonical | frozen Gate-4 v2 experimental plasticity (DO NOT tune) | none | frozen experimental record |
| **C: HYBRID_FLY** | canonical | none (biological) | CV + semantic world model + episodic memory + engineered value learning + goal/planner + ability resolver | practical game-playing configuration |

Every ENGINEERED component is labeled as such in code (docstring header)
and on the dashboard. Mode is recorded in every replay event and log line.

---

## 3. Concurrency model — speed is a first-class requirement

NEVER build `capture → wait → vision → wait → planner → wait → brain →
wait → input`. The system is an **asynchronous multi-process pipeline**
(`lab/bus.py`):

- **Bounded queues** (drop-oldest) for frame/command streams — an
  unbounded frame queue is a defect.
- **Latest-state slots** for state that is continuously overwritten
  (world observation, brain input, dashboard snapshot). **NEWER DATA
  WINS**: if vision produces frame 106 while frame 103 waits, 103 is
  discarded, not queued.
- Nothing in the real-time path may block on: dashboard rendering, disk
  logging, OCR, guide research, or memory persistence.

Approximate process boundaries (tunable with profiling evidence):

| # | Process | Target rate |
|---|---|---|
| 1 | Windows capture | 30–60 FPS |
| 2 | Fast perception | 20–30 Hz |
| 3 | Heavy perception / OCR | 5–20 Hz (workload-dependent) |
| 4 | Digital Drosophila brain decision loop | 5–10 Hz (hardware permitting) |
| 5 | World model / memory / planner | ~0.5–2 Hz or event-driven |
| 6 | Motor + ability executor | 20–60 Hz |
| 7 | Dashboard / logging / replay | 10–30 Hz (snapshot reader) |

Rates are **targets, not fabricated guarantees** — they are selected on
the real Windows laptop by `benchmark_windows.py` (Section 8) and
recorded in `WINDOWS_HARDWARE_REPORT.md` + `config/live_settings.json`.

### Latency budget

Timestamped chain measured end-to-end
(frame captured → detection → world observation → fly input → brain
output → intention → concrete ability → Windows input emitted):

- Initial goal: **P95 urgent sensory-to-action latency ≤ 250 ms**.
- 500+ ms persistent combat-reaction latency triggers optimization work.
- If the laptop cannot meet a target, **report the real numbers** —
  never fake faster timestamps.

---

## 4. Digital brain runtime (live mode)

The biological reference is always the canonical full Shiu/FlyWire v783
brain (138,639 neurons, 15,091,983 connections).

- **INITIALIZE ONCE → PREWARM → RUN CONTINUOUSLY.** The 138k-neuron
  network is never reconstructed per decision.
- The brain worker advances simulation chunks (biological 25 / 50 / 100 ms
  — select by Windows benchmark), receives updated sensory rates between
  chunks, and publishes descending/motor activity. This is exactly the
  D7 INTERACTIVE runtime contract (chunked stepping bit-identical to a
  single run; store/restore + reseed determinism; C++ standalone replay
  proven bit-identical).
- Preferred live route: the validated **Brian2 C++ standalone** binary
  (compile once, reuse) with the numpy interactive runtime as fallback.
- Cached once at init: connectome preprocessing, neuron ID maps,
  population maps (D4–D6 `brain/data/io_map/`), sensory maps, motor maps.
- `REFERENCE_REPLAY_MODE` must keep working: important episodes re-run
  against the canonical full brain; any live-mode approximation affecting
  brain equivalence must be identified and benchmarked. **Never silently
  call an approximation identical.**

---

## 5. Perception

One capture stream feeds both CV and the dashboard mirror.

**Fast vision (immediate)**: player / target / enemy position, approach,
motion, attack wind-up cues, health changes, large UI-state transitions,
looming/threat, distance estimates.

**Heavy vision (slow semantic)**: OCR (dialogue, quests, inventory,
equipment, item/ability names, level/EXP, menus), detailed NPC
recognition. **Combat must never wait for OCR.**

Vision runs behind an inference-backend abstraction (ONNX Runtime /
WinML preferred on Windows; CUDA when installed; CPU fallback). Detect
hardware rather than assuming NVIDIA; fixed model input sizes;
pre-load models — **never load a detector during combat**.

Structured world observation (typed, compact — never giant blobs past
to every component):

```
timestamp; player(position, heading, health, stamina);
target(type, direction, distance, confidence);
enemy(direction, distance, attacking, threat, confidence);
ui(loading, dialogue, menu); abilities(available, cooldown, inferred range)
```

---

## 6. Fly interface

**Input channels (compact, ENGINEERED)**: TARGET_LEFT / TARGET_RIGHT /
TARGET_CENTER / TARGET_DISTANCE; THREAT_LEFT / THREAT_RIGHT /
THREAT_INTENSITY; ATTACK_OPPORTUNITY; AVOIDANCE_VALUE; APPROACH_VALUE;
GOAL_RELEVANCE; LOW_HEALTH_CONTEXT. The helper says WHAT MATTERS —
not which key to press.

**Output — intention vector** (initial combat set; smaller validated
subsets fine to start): STOP, TURN_LEFT, TURN_RIGHT, APPROACH, RETREAT,
EVADE_LEFT, EVADE_RIGHT, DEFEND, ATTACK_LIGHT, ATTACK_HEAVY,
ATTACK_RANGED, SPECIAL, ESCAPE.

The intention decoder builds on the verified D9 explicit-rule decoder
(thresholds/hysteresis/min-duration/STOP-failsafe over side-split DN
population readouts — P9/BPN/RRN forward, MDN backward, FG/BB stop,
giant fiber DNp01 escape) and NEVER sees arena/game state — only
neural population spike counts.

---

## 7. Ability resolver + registry [ENGINEERED]

A fruit fly has no neuron for a named GPO move, so:

- **AbilityRegistry** caches structured ability records: id, name, input
  binding, intent tags, range, startup, recovery, cooldown, resource
  cost, damage estimate, mobility/defensive properties, last result,
  learned utility, confidence. Static properties are NEVER re-derived
  each frame.
- **AbilityResolver** converts a fly INTENTION into a concrete available
  ability. Fly-intention compatibility is a **HARD GATE** (dominant
  constraint): the resolver may choose *between* abilities compatible
  with the intention but must not casually override DEFEND into
  ATTACK_HEAVY.
- Selection is cheap numeric scoring (compatibility gate, availability,
  cooldown, distance fit, resource, risk, recent success, learned
  utility) — milliseconds, never seconds.
- Populated progressively from observation + allowed GPO knowledge
  sources (non-Roblox sources stored as UNVERIFIED until direct game
  observation confirms VERIFIED / REJECTED / OUTDATED — direct current
  observation wins on conflict).

---

## 8. High-level helper, world model, memory, value [ALL ENGINEERED]

- **Helper** handles what the fly cannot represent: semantics, quests,
  world knowledge, long-term goals, equipment comparison, route and
  progression planning. It outputs GOALS / CONTEXT / VALUE / RELEVANCE —
  not key-by-key scripts.
- **World model**: structured entities (PLAYER, NPC, ENEMY, QUEST,
  OBJECT, ISLAND, LANDMARK, WEAPON, FRUIT, ABILITY, ACCESSORY) and
  relations (located_at, requires, rewards, dangerous_to, useful_for,
  leads_to, equipped, cooldown, quest_target). Labeled ENGINEERED
  semantic understanding — the fly brain does not "understand quests".
- **Memory**: semantic (facts/relations) + episodic (events/outcomes);
  writes asynchronous (in-memory cache; periodic/event-driven
  persistence; never large synchronous DB writes during combat).
- **Engineered value learning** (explicitly NOT fly mushroom-body
  learning): per (enemy type × move) success, per-route outcomes; feeds
  AbilityResolver and Planner.
- **No LLM in the combat loop.** Blocking/dodging/turning/attacking use
  cached structured data. High-level reasoning (quest interpretation,
  strategy, build planning, long-term goals, novel situations) runs
  asynchronously; if it is delayed the fast controller keeps operating.

---

## 9. Motor executor + urgent reflex path

- MotorExecutor owns key-down/key-up, mouse movement, action duration,
  cooldown timers, action lock, held-key state. The brain need not
  recompute while merely holding W for 300 ms.
- Every emitted input is timestamped and logged.
- **Urgent reflex path** (attack wind-up → threat signal → next fly tick
  → DEFEND/EVADE → resolver → input) must never wait on quest planner,
  OCR, memory persistence, dashboard, or language reasoning.
- Global emergency-stop hotkey releases all held keys immediately.

---

## 10. DigitalFlyLab application

User workflow: user already logged into the dedicated experiment
account → open **DigitalFlyLab.exe** → self-test → detect/launch
authorized GPO target (only `https://www.roblox.com/games/1730877806/Grand-Piece-Online`)
→ detect game window → capture starts → dashboard shows live mirror →
fly brain starts → passive observation → control enabled only when the
relevant gate passes.

Dashboard: three clearly separated columns — **FLY BRAIN** (sensory
activity, neural activity, descending populations, current intention,
threat/approach/avoid signals) / **HYBRID HELPER** (recognized objects,
world state, goal, memory, learned value, strategy) / **ACTION SYSTEM**
(eligible abilities, selected intention, selected concrete ability,
cooldowns, emitted input) — so the user can see WHO made each decision.

Dashboard **reads snapshots only**; it never owns the simulation clock;
if rendering drops frames, control continues; slow charts are dropped,
the fly is never paused for a graph.

**Replay**: shared monotonic clock; record frame reference, CV
observation, world state, goal, memory retrieved, brain input, brain
output, intention, ability candidates + scores, selected ability,
emitted input, game outcome, value updates — reconstructing the complete
decision chain.

Security: no account creation, no stored passwords, no automated
credentials. The user logs in normally.

---

## 11. Development tracks (run in parallel; commit often)

| Track | Content |
|---|---|
| A | Windows runtime / dashboard |
| B | window capture + hardware benchmark |
| C | vision / OCR |
| D | world model / GPO knowledge |
| E | hybrid value / memory / planner |
| F | brain-runtime optimization |

Sandbox resets must not lose hours of work again — incremental commits
are mandatory.

---

## 12. Gate ladder (applied phase)

| Gate | Content | Autonomous input? |
|---|---|---|
| **5 — passive perception** | locate window, capture, mirror, basic UI/object states, structured observations, fly brain concurrent, synchronized replay, acceptable CPU/RAM/latency | **NO** |
| **6 — basic live movement** | FORWARD / TURN_LEFT / TURN_RIGHT / STOP / camera; low-risk env first; measure latency, direction accuracy, stuck detection, recovery, input reliability | gated |
| **7 — combat** (staged: single attack → block/evade → light vs heavy → ranged/special → multiple abilities) | detect enemy state → fly combat intention → compatible ability → execute → observe → update engineered value, without planner latency blocking combat | gated |
| **8 — quest / progression** | NPC recognition, dialogue/OCR, quest detection, objective tracking, navigation + combat goals, return-to-NPC, progression recognition; planner selects goals, fly keeps lower-level sensorimotor behavior | gated |
| **9 — long-term autonomy** | equipment/weapons/abilities/fruits/accessories, island navigation, boss encounters, strategy adaptation, optional launcher/menu automation last | gated |

Each gate has a pre-registered, fail-closed protocol (see
`GATE5_PREREG.md` and successors).

---

## 13. Performance degradation policy (binding order)

If the laptop cannot hold target rates:

1. FIRST reduce: dashboard FPS, visualization detail, OCR frequency,
   heavy-detector frequency.
2. THEN optimize: model resolution, batching, provider selection,
   memory copies.
3. NEVER first remove: fly brain, urgent perception, motor executor —
   those are central to the experiment.

---

## 14. Code map (new, this phase)

```
lab/                     DigitalFlyLab application package
  bus.py                 async bus: bounded drop-oldest queues + latest-wins slots
  clock.py               shared monotonic clock (replay timestamps)
  modes.py               PURE_FIXED_FLY / BIOLOGICAL_PLASTICITY_FLY / HYBRID_FLY
  capture/               capture adapters (interface, synthetic dev source,
                         guarded Windows.Graphics.Capture adapter)
  perception/            fast/heavy vision workers, WorldObservation schema,
                         fly input channels
  brain/                 brain worker (init-once/chunked), runtime protocol,
                         mock + canonical runtimes, intention schema/decoder
  action/                AbilityRegistry, AbilityResolver, MotorExecutor,
                         input backends (noop-safe until gates pass)
  world/                 world model, semantic+episodic memory, engineered
                         value, planner (all ENGINEERED)
  dashboard/             3-column snapshot dashboard (never blocks control)
  replay.py              timestamped decision-chain recorder
  benchmark/             hardware inventory + latency benchmarks
  app.py                 DigitalFlyLab entry point / pipeline assembly
benchmark_windows.py     Windows hardware benchmark CLI → WINDOWS_HARDWARE_REPORT.md
setup_windows.ps1        environment setup
run_dev_windows.ps1      dev launcher
benchmark_windows.ps1    benchmark launcher
build_windows.ps1        PyInstaller build → DigitalFlyLab.exe
GATE5_PREREG.md          Gate-5 pre-registration (passive perception)
```

Platform isolation follows `docs/WINDOWS_RUNTIME_SPEC.md` W3: all
Windows-specific code lives behind guarded imports; biology, vision,
memory and planning remain platform-independent.
