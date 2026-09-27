# METHODOLOGY — measurement, statistics, controls, welfare

## 1. The claim ladder (scientific language discipline)

Every result is stated at the LOWEST defensible level of the ladder:

1. "The fly-controlled system showed performance X under condition Y."
2. "Performance differed between conditions Y1 and Y2 (statistic, CI)."
3. "The difference is attributable to the fly's cue-approach behavior
   (because the shuffled-cue and no-preference controls isolate it)."
4. "Behavioral reliability changed across sessions (learning-candidate
   evidence)."
5. ONLY after ruling out alternatives: "The fly learned Z."

Level 5 additionally requires the session-structure controls in section 4.
Words like "understands", "knows", "wants" are never used about the fly.

## 2. Metrics (all computed per trial, per session, per condition)

| metric | definition |
|---|---|
| success rate | trials with TARGET_REACHED / total trials |
| time to target | seconds until reach (successes only) |
| path efficiency | straight-line distance / actual path length (<= 1) |
| action count | emissions per trial (work measure) |
| action changes | consecutive-different emissions (turnover measure) |
| mean inter-action interval | emission cadence (cadence-fairness check) |
| reward total | sparse + shaped components logged separately |
| tracking rate / lost rate | apparatus health (per run, per trial) |

Learning curves plot trial number vs rolling success rate (window 10),
time-to-target, path efficiency, and error count. Before/after training,
after rest (retention), and in a NEW environment (transfer) are separate
session phases recorded in the `experiments.phase` column.

## 3. Control conditions (Phase 1 demo, same loop for all)

| condition | controller | cue | fly taxis | isolates |
|---|---|---|---|---|
| random | uniform random actions | - | - | floor |
| fixed | greedy computer policy | - | - | ceiling (computer alone) |
| fly_random | fly | goal-directed | OFF (bias 0) | fly-as-noise |
| fly_cue | fly | goal-directed | ON | the real hybrid |
| fly_cue_shuffled | fly | RANDOMIZED every ~2 s | ON | closed-loop goal signal |

Logic: fly_cue > random shows the hybrid works. fly_cue > fly_random shows
the fly's cue preference (not mere motion) drives it. fly_cue >
fly_cue_shuffled shows the goal-directed CLOSED LOOP (not fly locomotion
alone) drives it. fixed bounds everything from above.

Confounds actively managed:
- Cadence fairness: controllers emit on comparable intervals; inter-action
  intervals are REPORTED per condition. If cadences differ systematically,
  the comparison is repeated with matched cadence before any claim.
- Apparatus health: tracking lost-rate is logged; trials with lost-rate
  > 10% are flagged and excluded in a pre-registered sensitivity analysis.
- Spurious structure: the 2D arena spawn distribution is uniform and
  seeded; seeds are recorded; every run is reproducible from manifest.json.

## 4. Session design for real-fly learning experiments (Phase 3+)

Pre-registered primary metric: success rate per session (30+ trials).
Between-subject design: each fly experiences ONE condition; >= 12 flies per
condition for the headline comparisons (power note: detecting 0.35 vs 0.70
success at alpha 0.05, power 0.8 needs ~23 flies/condition by two-proportion
test; 12 flies is a pilot scale — the roadmap states which is which).

Session structure per fly: baseline (no cue contingency) -> training
(closed loop) -> test (closed loop, new target layouts) -> reversal
(cue->action mapping swapped) -> retention (24–48 h later). The REVERSAL
test is the key learning discriminator: if performance follows the NEW
mapping quickly, contingency control is demonstrated; if performance
persists on the old mapping, habituation — not contingency — is implicated.

Yoked control: for each trained fly, a yoked fly/replay receives the SAME
action sequence replayed non-contingently (no closed loop). Improvement in
yoked animals would indicate non-associative explanations (handling,
exposure, maturation).

Attribution freeze: decoder parameters, thresholds, and cue-placement
policy are FROZEN for the duration of a learning experiment. Any apparatus
change invalidates cross-session comparison and is logged as a new
experiment generation.

## 5. Statistics

- Success-rate differences: percentile bootstrap 95% CI (10,000 resamples,
  trial-level resampling within condition) + bootstrap p(diff<=0).
- Time-to-target / path efficiency: medians with bootstrap CIs (robust to
  the success-only selection); Mann–Whitney U as a secondary check.
- Multiple comparisons: Holm correction when more than the three
  pre-registered contrasts are reported.
- Pre-registration: the demo's three contrasts are fixed in
  `experiments/controls.py::COMPARISON_PAIRS`; new contrasts are labeled
  exploratory.

## 6. Data management and reproducibility

- Every run: `runs/<timestamp>_<condition>/` with manifest.json (software
  version, full config snapshot, seed, platform), experiment.db (events +
  stride-sampled frames), summary.json (aggregates + per-trial rows).
- Raw vs processed: optional frame dumps (`trial.save_frames: true`) go to
  `frames/`; the DB is the processed layer; nothing is ever overwritten.
- Reproduce any run: same code version + same seed + same config snapshot.

## 7. Animal welfare (binding for all phases)

Allowed: observation, markerless tracking, benign visual stimuli (dark
stripes / dim diffuse light patterns), standard husbandry.

Never allowed in this project: surgery, implants, heating/cooling beyond
comfortable rearing range, chemicals, aversive stimuli of any kind
(shock, vibration startle paradigms, puffs), starvation/sleep-deprivation
manipulations, injury, lethal endpoints.

Husbandry: D. melanogaster from a registered stock center or education
supplier; 22–25 C; 12:12 light:dark; standard medium; sexes handled per
standard practice (single-fly assays to avoid mating confounds); sessions
<= 30 min; >= 1 rest day between sessions per fly; flies monitored; any
fly showing abnormal behavior is withdrawn from the study.

Oversight: insect work is not typically covered by vertebrate-animal
regulations, BUT institutional policies vary — before Phase 1, confirm
with the relevant institution/authority if working under one. If any
future direction would require facilities or procedures beyond benign
observation (e.g., optogenetic strains, temperature-shift genetics), that
work is OUT of scope for this project as stated, and would require proper
professional facilities, trained personnel, and formal approval.

## 8. Known apparatus limitations (honest list)

- MOG2 with a near-frozen learning rate assumes stable lighting: any slow
  ambient drift will produce false foreground. The light-tight enclosure
  (docs/HARDWARE.md) is therefore part of the apparatus, not an accessory.
- Single-fly assumption: the tracker picks the nearest plausible blob; two
  flies in the arena invalidate decoding. Single-fly sessions only.
- Body-orientation is not yet used: heading comes from observed velocity;
  a motionless fly has no heading. Ellipse-orientation tracking is a
  planned Phase-1 improvement for wall-press decoding robustness.
- The synthetic fly is a MODEL: its cue taxis stands in for "a fly with a
  measured stimulus preference" (Phase 2 measures the real thing). No
  result from Phase 0 says anything about real fly behavior.
