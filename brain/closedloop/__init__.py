"""Closed-loop digital Drosophila package (D7R-D13).

Tier labels (single source of truth, docs/MASTER_SPEC.md section 2):

  BIOLOGICAL  : connectome data (FlyWire v783) + the pinned Shiu et al. LIF
                dynamics + published circuit assignments (Aso 2014 MBON
                valence, PAM/PPL1 reinforcement roles).
  MODELED     : rules we add to mimic biology that is observed but not
                simulated by the pinned model (KC->MBON dopamine-gated
                plasticity; weight decay).
  ENGINEERED  : task-side machinery (arena, sensory encoder rates, motor
                decoder thresholds, reward delivery schedule, external
                episodic memory in D13 condition B). Never a hidden
                target_position -> correct_action shortcut.

Every deliverable that uses these circuits must cite its tier.
"""
__version__ = "0.1.0"

TIER_BIOLOGICAL = "biological"
TIER_MODELED = "modeled"
TIER_ENGINEERED = "engineered"
