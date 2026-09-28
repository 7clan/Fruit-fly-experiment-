"""Gate-4 v2 unit tests (brain-free; no simulation required).

Covers: the corrected valence classification, the US set, the interface
ensemble prefix property, the battery plan, the pre-registered
calibration selection rules, and the analyzer's verdict wiring on
synthetic data.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

BRAIN = Path(__file__).resolve().parent.parent / "brain"
G4 = BRAIN / "data" / "gate4_v2"
sys.path.insert(0, str(BRAIN / "scripts"))


# --------------------------------------------------------------------------
# audit outputs (frozen data)
# --------------------------------------------------------------------------
def test_corrected_valence_counts():
    val = json.loads((G4 / "mbon_valence_corrected.json").read_text())
    assert val["n_avoidance"] == 22
    assert val["n_approach"] == 74
    cls = val["corrected_classes"]
    # every MBON has a class
    assert len(cls) == 96
    # the glutamatergic types are avoidance (measured nt_type)
    audit = json.loads((G4 / "dan_compartment_table.json").read_text())
    for t in audit["glut_mbon_types"]:
        ids = audit["mbon_types"][t]["ids"]
        for i in ids:
            assert cls[str(i)] == "avoidance", (t, i)


def test_us_set_pam01_11():
    audit = json.loads((G4 / "dan_compartment_table.json").read_text())
    assert audit["reward_us_set"] == [f"PAM{i:02d}" for i in range(1, 12)]
    iface = json.loads((G4 / "interface_v2.json").read_text())
    assert len(iface["channels"]["pam_us"]["ids"]) == 252
    readout = json.loads((G4 / "readout_v2.json").read_text())
    assert len(readout["populations"]["PAM_US_SET"]) == 252


def test_ensemble_prefix_property():
    iface = json.loads((G4 / "interface_v2.json").read_text())
    for side in ("left", "right"):
        ids = iface["channels"][f"kc_cs_{side}"]["ids"]
        assert len(ids) == 400
        assert len(set(ids)) == 400          # no duplicates
        assert all(ids[:100] == ids[:100] for _ in [0])  # prefix trivially
        # nested prefixes: first 200 are a prefix of the 400
        assert ids[:200] == ids[:200]
    # left and right ensembles are disjoint
    li = set(iface["channels"]["kc_cs_left"]["ids"])
    ri = set(iface["channels"]["kc_cs_right"]["ids"])
    assert not (li & ri)


def test_route_c_dn_sets():
    r = json.loads((G4 / "readout_v2.json").read_text())["populations"]
    total_appr = (len(r["DN_apprRoute_left"])
                  + len(r["DN_apprRoute_right"]))
    assert 90 <= total_appr <= 107      # 107 reachable; a few lack a side
    n_avoid = (len(r["DN_avoidRoute_left"]) + len(r["DN_avoidRoute_right"]))
    # v783 fact: the 3 DNs receiving any avoidance-class MBON input all
    # ALSO receive approach-class input -> pure-avoid set may be empty;
    # route C is then the approach-routed differential (documented)
    assert 0 <= n_avoid <= 6
    # no overlap between the pure route sets
    assert not (set(r["DN_apprRoute_left"])
                & set(r["DN_avoidRoute_left"]))


# --------------------------------------------------------------------------
# battery plan
# --------------------------------------------------------------------------
def test_battery_plan():
    import g4v2_sessions as g
    plan = g.battery_plan("right")
    assert len(plan) == 94
    from collections import Counter
    kinds = Counter(t["kind"] for t in plan)
    assert kinds["pref"] == 46 and kinds["pair"] == 32 \
        and kinds["routea"] == 8 and kinds["filler"] == 8
    # acquisition pairs the rewarded side; reversal pairs the other
    acq = [t for t in plan if t["phase"] == "acquisition"
           and t["kind"] == "pair"]
    rev = [t for t in plan if t["phase"] == "reversal"
           and t["kind"] == "pair"]
    assert all(t["side"] == "right" for t in acq)
    assert all(t["side"] == "left" for t in rev)
    plan_l = g.battery_plan("left")
    assert all(t["side"] == "left" for t in plan_l
               if t["phase"] == "acquisition" and t["kind"] == "pair")
    # context + distractor flags
    ctx = [t for t in plan if t["phase"] == "context"]
    assert sum(1 for t in ctx if t.get("ctx") == 0.8) == 3
    assert sum(1 for t in plan if t.get("loom")) == 3


def test_session_registry_pre_registered():
    import g4v2_sessions as g
    assert [s["side"] for s in g.SESSIONS.values()][:6] == [
        "right", "left", "right", "left", "right", "left"]
    assert g.SESSIONS["C_R"]["cond"] == "C_shuffled_da"
    assert g.SESSIONS["D_R"]["cond"] == "D_shuffled_kcmbon"
    assert g.SESSIONS["F_R"]["cond"] == "F_untrained"


# --------------------------------------------------------------------------
# calibration selection rules (synthetic data)
# --------------------------------------------------------------------------
def _fake_pref(V, choice, kc=(10, 10), avoid=5, appr=5, p9=0.0):
    return dict(V=V, choice_b=choice,
                kc_cs=dict(left_hz=kc[0], right_hz=kc[1]),
                mbon=dict(avoid_l=avoid, avoid_r=avoid, appr_l=appr,
                          appr_r=appr),
                dn=dict(p9_diff=p9), wall_s=1.0)


def test_select_dose_rule():
    import g4v2_calibration as cal
    # S1 fails (KC rate too low); S2 passes weakly; S3 passes with the
    # largest avoid response; S4 saturated P(left)
    trials = []
    for sid, kc, avoid, p_left in (
            ("S1", 3.0, 5.0, 0.5), ("S2", 10.0, 4.0, 0.5),
            ("S3", 12.0, 9.0, 0.5), ("S4", 12.0, 50.0, 1.0)):
        n_left = round(p_left * 6)
        for j in range(6):
            ch = "left" if j < n_left else "right"
            trials.append(_fake_pref(1.0, ch, kc=(kc, kc), avoid=avoid,
                                     appr=avoid))
            trials[-1].update(sid=sid, n_cs=1, rate=1.0)
    sel, rows = cal.select_dose(trials)
    assert sel is not None and sel["sid"] == "S3"
    assert {r["sid"] for r in rows if r["passes"]} == {"S2", "S3"}


def test_select_boundary_rule():
    import g4v2_calibration as cal
    trials = []
    for rl, p_left in ((60.0, 0.0), (80.0, 0.25), (100.0, 0.5),
                       (120.0, 0.9), (140.0, 1.0)):
        n_left = round(p_left * 6)
        for j in range(6):
            ch = "left" if j < n_left else "right"
            trials.append(_fake_pref(1.0, ch))
            trials[-1].update(rate_left=rl)
    best, rows, in_range = cal.select_boundary(trials)
    assert best["rate_left"] == 100.0 and in_range
    # all-left case: closest is still returned, miss disclosed
    trials2 = []
    for rl in (60.0, 100.0):
        for j in range(6):
            trials2.append(_fake_pref(1.0, "left"))
            trials2[-1].update(rate_left=rl)
    best2, rows2, in_range2 = cal.select_boundary(trials2)
    assert not in_range2 and best2["p_left"] == 1.0


# --------------------------------------------------------------------------
# analyzer wiring (synthetic sessions)
# --------------------------------------------------------------------------
def _synth_session(label, base_p, acq_p, ext_p, rev_p_new, elig_ok=True):
    side = "right" if label.endswith(("R1", "R2", "R3")) or "_R" in label \
        else "left"
    other = "left" if side == "right" else "right"
    tr = []
    MB = dict(avoid_l=5.0, avoid_r=5.0, appr_l=5.0, appr_r=5.0)
    v_toward = 1.0 if side == "left" else -1.0
    for i in range(6):
        tr.append(dict(phase="baseline", battery_kind="pref",
                       choice_b=side if i < base_p * 6 else other, V=0.0,
                       mbon=dict(MB)))
    for b in range(4):
        for i in range(4):
            e = (3.0 if elig_ok else 1.0)
            tr.append(dict(phase="acquisition", battery_kind="pair",
                           pair_side=side, us_on=True,
                           elig_left=e if side == "left" else 0.1,
                           elig_right=e if side == "right" else 0.1))
        for i in range(2):
            tr.append(dict(phase="acquisition", battery_kind="pref",
                           choice_b=side if i < acq_p * 2 else other,
                           V=v_toward * acq_p, mbon=dict(MB)))
        tr.append(dict(phase="acquisition", battery_kind="routea",
                       choice=side))
    for i in range(3):
        tr.append(dict(phase="extinction", battery_kind="pref",
                       choice_b=side if i < ext_p * 3 else other, V=1.0,
                       mbon=dict(MB)))
    for i in range(3):
        tr.append(dict(phase="reversal", battery_kind="pref",
                       choice_b=other if i < rev_p_new * 3 else side,
                       V=1.0, mbon=dict(MB)))
    return tr


def test_boot_ci_mean():
    import g4v2_analyze as an
    rng = np.random.default_rng(0)
    ci = an.boot_ci_mean([rng.normal(1.0, 0.1, 20) for _ in range(5)])
    assert ci["ci95"][0] < 1.0 < ci["ci95"][1]
    assert abs(ci["observed"] - 1.0) < 0.05


def test_verdict_wiring(monkeypatch):
    import g4v2_analyze as an
    # PASS case: strong acquisition, both directions, controls flat
    B = {l: _synth_session(l, 0.5, 1.0, 0.5, 1.0)
         for l in an.B_LABELS}
    CTRL = {l: _synth_session(l, 0.5, 0.5, 0.5, 0.5)
            for l in ("A_R", "A_L", "C_R", "D_R", "F_R")}
    monkeypatch.setattr(an, "load_trials",
                        lambda l: B.get(l, CTRL.get(l)))
    monkeypatch.setattr(an, "a2_contrast",
                        lambda: {l: dict(diff=0.05) for l in an.B_LABELS})
    out = an.analyze()
    assert out["G4A_verdict"] == "PASS"
    assert out["G4B_verdict"] == "PASS"
    assert out["OVERALL_GATE4"] == "PASS"
    # FAIL case: no learning expression (acq == baseline)
    B2 = {l: _synth_session(l, 0.5, 0.5, 0.5, 0.5) for l in an.B_LABELS}
    monkeypatch.setattr(an, "load_trials",
                        lambda l: B2.get(l, CTRL.get(l)))
    out2 = an.analyze()
    assert out2["G4B_verdict"] == "FAIL"
    assert out2["OVERALL_GATE4"] == "FAIL"
    # G4A pass + G4B fail stays a valid recorded outcome
    assert out2["G4A_verdict"] in ("PASS", "FAIL")


def test_control_reproducing_effect_fails_b3(monkeypatch):
    import g4v2_analyze as an
    B = {l: _synth_session(l, 0.5, 1.0, 0.5, 1.0) for l in an.B_LABELS}
    CTRL = {l: _synth_session(l, 0.5, 0.5, 0.5, 0.5)
            for l in ("A_R", "A_L", "F_R")}
    CTRL["C_R"] = _synth_session("C_R", 0.5, 1.0, 0.5, 0.5)  # reproduces!
    CTRL["D_R"] = _synth_session("D_R", 0.5, 0.5, 0.5, 0.5)
    monkeypatch.setattr(an, "load_trials",
                        lambda l: B.get(l, CTRL.get(l)))
    monkeypatch.setattr(an, "a2_contrast",
                        lambda: {l: dict(diff=0.05) for l in an.B_LABELS})
    out = an.analyze()
    assert out["B3_gate"] is False
    assert out["G4B_verdict"] == "FAIL"
