# GATE 6E — PRE-REGISTRATION (D8-corrected autonomous navigation)

**Status:** FROZEN before first 6E run.
Date: 2026-09-30

Gate 6E follows the live-navigation triple audit. It corrects implementation
drift between the validated D8/D10 sensory interface and the game-facing live
adapter. It does **not** change the canonical brain, connectome, synapses,
neuron parameters, or motor decoder thresholds.

## Corrections entering 6E

- live target LC9 drive matches frozen D8: 0..150 Hz;
- live target angular tuning matches frozen D8 theta0=15 deg,
  theta_w=30 deg;
- recommended-quest marker can occur at the real right edge if its green
  circle + green distance-text evidence is present;
- Windows SendInput return values are verified and reported.

## Allowed active outputs

Unchanged navigation-only subset:

- TURN_LEFT  -> bounded horizontal mouse left + W pulse;
- TURN_RIGHT -> bounded horizontal mouse right + W pulse;
- APPROACH   -> W pulse;
- STOP       -> no input.

Hard backend still allows only W plus bounded horizontal mouse movement.
No attack, block, interact, ability, dash, menu, mouse-button, S/A/D, or
vertical mouse commands.

## Run

90 seconds after explicit brain READY. User may enable/disable movement from
the dashboard. F12 remains the global emergency stop.

## What counts as useful evidence

- canonical brain + subprocess_pipe;
- zero unrelated worker errors;
- visible valid target sensory rates;
- at least one non-STOP neural intention if the target is visible;
- if a real command is attempted, SendInput success/failure is recorded;
- no active output outside the hard navigation subset;
- replay + automatic evidence saved.

This run is intended to identify the remaining layer, not to hide a failure:
neural recruitment, target perception, or Windows input.
