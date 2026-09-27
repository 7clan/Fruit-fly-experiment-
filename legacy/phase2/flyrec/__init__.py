"""flyrec — Phase-2 real-fly behavioral recording & analysis apparatus.

Layer map (see PHASE2_PROTOCOL.md section 9):

    recording/   camera -> immutable session dirs (video + metadata)
    tracking/    video -> per-frame position + confidence + stimulus readback
    analysis/    tracks -> kinematics, bouts/pauses, walls, circular stats,
                 variability, stimulus response, master statistics
    vocabulary/  behavior -> candidate action states + reduction ladder
    gate/        pre-registered PHASE2-GATE-1.0.0 evaluation
    validation/  synthetic ground-truth videos -> SOFTWARE acceptance only

Nothing in this package connects a real fly (or any controller) to game
input. Phase 2 is observation only.
"""
__version__ = "0.2.0"
PHASE = "Phase 2 - real-fly behavioral validation (non-invasive observation)"
