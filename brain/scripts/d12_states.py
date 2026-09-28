"""D12 - internal-state visualization: honest computational states derived
ONLY from measured model variables.

MANDATE (D12): the dashboard gains real computed internal-state variables -
positive valence, negative valence, reward expectation, reward-prediction
error, novelty, exploration drive, threat/avoidance drive, uncertainty,
arousal-like activation.  Every state below is a documented transform of
MEASURED network variables (MBON population rates, DAN rates, DN rates, KC
ensemble eligibility, choices, US delivery).  NO fake human emotion, NO
natural-language thought; friendly labels sit next to the underlying
numeric variables, never replace them.

State -> underlying variable (single source of truth):
  positive_valence        mean(MBON_approach_left, MBON_approach_right)
                          rates at end of choice window [Hz]
  negative_valence        mean(MBON_avoidance_left, MBON_avoidance_right)
                          [Hz]
  valence_bias            the D11 decoder bias term / beta
                          [(apprL-avoidL)-(apprR-avoidR)] [Hz]
  reward_expectation_L/R  appr_side/(appr_side+avoid_side+1Hz) - the brain's
                          own learned valence of that context [0..1]
  reward_prediction_error us_delivered(1/0) - reward_expectation(chosen
                          side) [dimensionless]
  novelty                 1 - cosine(KC-ensemble eligibility pattern,
                          EMA of past patterns) [0..1]
  exploration_drive       normalized Shannon entropy of last 8 choices
                          [0..1]
  threat_drive            looming-channel drive + giant-fiber rate [Hz]
                          (0 throughout the D11 task - no loom; live in
                          D13/GPO)
  uncertainty             EW std (half-life 8 trials) of recent RPEs
  arousal_like            z-score of (PAM+PPL1 choice-window rate
                          + DN_ALL choice-window rate)

Usage:
  python d12_states.py states  <session_dir>          # -> states.jsonl
  python d12_states.py dashboard <session_dir>        # -> dashboard.html
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import brainlib as bl  # noqa: E402

RESULTS = bl.ROOT / "results" / "d12_states"


# --------------------------------------------------------------------------
# state computation
# --------------------------------------------------------------------------
def load_trials(session_dir):
    p = Path(session_dir) / "trials.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines()]


def load_chunks(session_dir):
    p = Path(session_dir) / "chunks.jsonl"
    if not p.exists():
        return None
    return [json.loads(l) for l in p.read_text().splitlines()]


def _ew_std(xs, half_life):
    """Exponentially weighted std (recursive, half-life in samples)."""
    if not xs:
        return None
    alpha = 1.0 - math.pow(0.5, 1.0 / half_life)
    mean = xs[0]
    var = 0.0
    out = []
    for x in xs:
        delta = x - mean
        mean += alpha * delta
        var = (1 - alpha) * (var + alpha * delta * delta)
        out.append(math.sqrt(max(var, 0.0)))
    return out


def compute_states(trials, chunks=None):
    """Per-trial internal-state records (documented transforms only)."""
    # --- threat drive from chunk log (0 if absent / no loom) -----------
    threat = {}
    gf = {}
    if chunks:
        for c in chunks:
            k = c["trial"]
            threat.setdefault(k, []).append(
                c.get("rates", {}).get("looming", 0.0))
            gf.setdefault(k, []).append(c.get("gf_hz", 0.0))
    threat_mean = {k: float(np.mean(v)) for k, v in threat.items()}
    gf_mean = {k: float(np.mean(v)) for k, v in gf.items()}

    # --- novelty: KC-ensemble pattern vs EMA of past patterns ---------
    ema = None
    novelty = []
    beta_ema = 0.3
    for t in trials:
        pat = t.get("kc_elig_ensemble")
        if not pat:
            novelty.append(None)
            continue
        v = np.asarray(pat, dtype=float)
        nrm = np.linalg.norm(v)
        if ema is None or nrm == 0:
            novelty.append(0.0 if nrm == 0 else 1.0)
        else:
            e = np.asarray(ema)
            denom = np.linalg.norm(v) * np.linalg.norm(e)
            cos = float(np.dot(v, e) / denom) if denom > 0 else 0.0
            novelty.append(round(max(0.0, 1.0 - cos), 4))
        if nrm > 0:
            ema = v if ema is None else (1 - beta_ema) * np.asarray(ema) \
                + beta_ema * v

    # --- exploration: entropy of last 8 choices -----------------------
    exploration = []
    hist = []
    for t in trials:
        c = t.get("choice")
        if c is not None:
            hist.append(c)
        win = hist[-8:]
        if len(win) < 2:
            exploration.append(None)
            continue
        pL = sum(1 for x in win if x == "left") / len(win)
        pR = 1 - pL
        h = 0.0
        for p in (pL, pR):
            if p > 0:
                h -= p * math.log2(p)
        exploration.append(round(h, 4))

    # --- RPE + uncertainty --------------------------------------------
    rpe = []
    for t in trials:
        v = t.get("mbon_valence_end")
        c = t.get("choice")
        if not v or c is None:
            rpe.append(None)
            continue
        appr = v[f"appr_{'l' if c == 'left' else 'r'}"]
        avoid = v[f"avoid_{'l' if c == 'left' else 'r'}"]
        expect = appr / (appr + avoid + 1.0)
        rpe.append(round((1.0 if t.get("us_delivered") else 0.0)
                         - expect, 4))
    rpe_vals = [x for x in rpe if x is not None]
    rpe_ew = _ew_std([x for x in rpe if x is not None], 8) \
        if rpe_vals else []
    uncertainty = []
    j = 0
    for x in rpe:
        if x is None:
            uncertainty.append(None)
        else:
            uncertainty.append(round(rpe_ew[j], 4) if j < len(rpe_ew)
                               else None)
            j += 1

    # --- arousal: z-scored composite ----------------------------------
    dan = [t.get("pam_choice_hz", 0.0) + t.get("ppl1_choice_hz", 0.0)
           for t in trials]
    dn = [t.get("dn_all_choice_hz", 0.0) for t in trials]
    comp = np.array(dan, dtype=float) + np.array(dn, dtype=float)
    z = ((comp - comp.mean()) / comp.std()) if comp.std() > 0 \
        else np.zeros_like(comp)

    states = []
    for i, t in enumerate(trials):
        v = t.get("mbon_valence_end") or {}
        appr_l, appr_r = v.get("appr_l"), v.get("appr_r")
        avoid_l, avoid_r = v.get("avoid_l"), v.get("avoid_r")
        rec = dict(
            trial=t["trial"], phase=t["phase"], choice=t.get("choice"),
            us_delivered=t.get("us_delivered"),
            # --- states (friendly label : numeric variable) ---
            positive_valence_hz=round(
                float(np.mean([x for x in (appr_l, appr_r)
                               if x is not None])), 3)
            if (appr_l is not None and appr_r is not None) else None,
            negative_valence_hz=round(
                float(np.mean([x for x in (avoid_l, avoid_r)
                               if x is not None])), 3)
            if (avoid_l is not None and avoid_r is not None) else None,
            valence_bias_hz=t.get("valence_bias_end"),
            reward_expectation_left=round(
                appr_l / (appr_l + avoid_l + 1.0), 4)
            if (appr_l is not None and avoid_l is not None) else None,
            reward_expectation_right=round(
                appr_r / (appr_r + avoid_r + 1.0), 4)
            if (appr_r is not None and avoid_r is not None) else None,
            reward_prediction_error=rpe[i],
            novelty=novelty[i],
            exploration_drive=exploration[i],
            threat_drive_hz=round(threat_mean.get(t["trial"], 0.0)
                                  + gf_mean.get(t["trial"], 0.0), 3),
            uncertainty=uncertainty[i],
            arousal_like_z=round(float(z[i]), 3),
        )
        states.append(rec)
    return states


# --------------------------------------------------------------------------
# dashboard (self-contained HTML, matplotlib panels embedded base64)
# --------------------------------------------------------------------------
def _panel(fig):
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def render_dashboard(session_dir, states=None, out=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    for _f in ("/usr/share/fonts/truetype/chinese/NotoSansSC-Regular.ttf",
               "/usr/share/fonts/truetype/chinese/NotoSansSC[wght].ttf",
               "/usr/share/fonts/truetype/chinese/SarasaMonoSC-Regular.ttf"):
        try:
            fm.fontManager.addfont(_f)
            break
        except Exception:
            continue
    plt.rcParams["font.sans-serif"] = ["Noto Sans SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    session_dir = Path(session_dir)
    trials = load_trials(session_dir)
    if states is None:
        states = compute_states(trials, load_chunks(session_dir))
    summary = json.loads((session_dir / "session.json").read_text()) \
        if (session_dir / "session.json").exists() else {}

    ks = [s["trial"] for s in states]

    # panel 1: internal-state time series (z-scored where numeric)
    fig, ax = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
    series = [("positive_valence_hz", "positive valence (approach-MBON)"),
              ("negative_valence_hz", "negative valence (avoidance-MBON)"),
              ("valence_bias_hz", "valence bias (decoder term)"),
              ("reward_prediction_error", "reward-prediction error"),
              ("novelty", "novelty (KC-ensemble pattern)"),
              ("exploration_drive", "exploration drive (choice entropy)"),
              ("threat_drive_hz", "threat drive (loom+GF)"),
              ("uncertainty", "uncertainty (EW RPE std)"),
              ("arousal_like_z", "arousal-like (z DAN+DN)")]
    for key, label in series:
        xs = [s.get(key) for s in states]
        good = [(k, x) for k, x in zip(ks, xs) if x is not None]
        if not good:
            continue
        vals = np.array([x for _, x in good], dtype=float)
        sd = vals.std()
        zz = (vals - vals.mean()) / sd if sd > 1e-9 else vals * 0
        ax.plot([k for k, _ in good], zz, label=label, lw=1.4, alpha=0.85)
    for ph, col in (("acquisition", "tab:green"), ("extinction", "tab:red"),
                    ("reversal", "tab:purple"), ("pilot_acquisition",
                                                 "tab:gray")):
        idx = [s["trial"] for s in states if s["phase"] == ph]
        if idx:
            ax.axvspan(min(idx) - 0.5, max(idx) + 0.5, color=col,
                       alpha=0.06)
            ax.text(np.mean(idx), ax.get_ylim()[1] * 0.97, ph, ha="center",
                    va="top", fontsize=8, color=col)
    ax.set(title=f"INTERNAL STATES (z-scored) - {session_dir.name} - "
                 "every trace = a documented transform of measured model "
                 "variables",
           xlabel="trial", ylabel="z")
    ax.legend(fontsize=8, ncol=2, loc="upper left", bbox_to_anchor=(1.01, 1))
    p1 = _panel(fig)

    # panel 2: reward timeline + choices + DAN rates
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True,
                             constrained_layout=True)
    ax = axes[0]
    ch = [(s["trial"], s["choice"]) for s in states]
    for k, c in ch:
        if c == "left":
            ax.plot(k, 1, "|", ms=14, color="tab:blue")
        elif c == "right":
            ax.plot(k, 2, "|", ms=14, color="tab:orange")
        else:
            ax.plot(k, 1.5, "x", ms=6, color="gray")
    us = [s["trial"] for s in states if s.get("us_delivered")]
    for k in us:
        ax.axvline(k, color="tab:green", alpha=0.3, lw=2)
    ax.set_yticks([1, 2], ["LEFT", "RIGHT"])
    ax.set(title=f"choices (green band = US delivered) - "
                 f"{summary.get('condition', '?')}, "
                 f"reward side(s): see phases")
    ax = axes[1]
    ax.plot(ks, [t.get("pam_us_hz", 0) for t in trials], ".-", label="PAM (US window)")
    ax.plot(ks, [t.get("ppl1_us_hz", 0) for t in trials], ".-",
            label="PPL1 (US window)")
    ax.plot(ks, [t.get("pam_choice_hz", 0) for t in trials], ".--",
            label="PAM (choice window)", alpha=0.6)
    ax.set_ylabel("Hz"); ax.legend(fontsize=8)
    ax = axes[2]
    v = trials[0].get("mbon_valence_end") or {}
    for key, label in (("appr_l", "approach-MBON L"), ("appr_r", "approach-MBON R"),
                       ("avoid_l", "avoidance-MBON L"), ("avoid_r", "avoidance-MBON R")):
        xs = [(t["trial"], (t.get("mbon_valence_end") or {}).get(key))
              for t in trials]
        good = [(k, x) for k, x in xs if x is not None]
        if good:
            ax.plot([k for k, _ in good], [x for _, x in good], ".-",
                    label=label, ms=4)
    ax.set_ylabel("Hz"); ax.set_xlabel("trial"); ax.legend(fontsize=8)
    p2 = _panel(fig)

    # panel 3: reward expectation + RPE + uncertainty
    fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
    for key, label in (("reward_expectation_left", "expectation LEFT"),
                       ("reward_expectation_right", "expectation RIGHT")):
        good = [(s["trial"], s.get(key)) for s in states
                if s.get(key) is not None]
        if good:
            ax.plot([k for k, _ in good], [x for _, x in good], ".-",
                    label=label, ms=4)
    good = [(s["trial"], s.get("reward_prediction_error")) for s in states
            if s.get("reward_prediction_error") is not None]
    if good:
        ax.plot([k for k, _ in good], [x for _, x in good], ".", ms=5,
                color="k", alpha=0.5, label="RPE")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylim(-1.05, 1.05)
    ax.set(title="reward expectation (brain's own MBON valence ratio) and "
                 "reward-prediction error", xlabel="trial")
    ax.legend(fontsize=8)
    p3 = _panel(fig)

    # panel 4: synaptic weights
    fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
    for key, label in (("avoidance_class_mean", "KC->avoid-MBON mean scale"),
                       ("approach_class_mean", "KC->appr-MBON mean scale"),
                       ("mean_scale", "all KC->MBON mean scale")):
        good = [(t["trial"], (t.get("weight_stats") or {}).get(key))
                for t in trials]
        good = [(k, x) for k, x in good if x is not None]
        if good:
            ax.plot([k for k, _ in good], [x for _, x in good], ".-",
                    label=label, ms=4)
    ax.set_ylim(0, 1.02)
    ax.set(title="MODELED synaptic scales at the canonical KC->MBON site "
                 "(1.0 = canonical connectome)", xlabel="trial")
    ax.legend(fontsize=8)
    p4 = _panel(fig)

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>D12 internal states - {session_dir.name}</title>
<style>
 body {{ font-family: 'Noto Sans SC', sans-serif; margin: 24px;
        background: #f6f7f9; color: #1c2733; }}
 h1 {{ font-size: 20px; }} h2 {{ font-size: 15px; margin-top: 28px; }}
 .box {{ background: white; padding: 16px 20px; border-radius: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,.08); margin-bottom: 18px; }}
 img {{ max-width: 100%; }}
 table {{ border-collapse: collapse; font-size: 12px; }}
 td, th {{ border: 1px solid #d8dee6; padding: 4px 10px;
         text-align: left; }}
 .tier {{ font-size: 12px; color: #5a6b7d; }}
 .warn {{ color: #8a4b08; }}
</style></head><body>
<h1>Digital Drosophila - internal-state dashboard (D12)</h1>
<p class="tier">Session <b>{session_dir.name}</b> &middot; condition
<b>{summary.get('condition', '?')}</b> &middot; {summary.get('n_trials',
len(trials))} trials &middot; wall {summary.get('session_wall_s','?')} s
&middot; peak RSS {summary.get('peak_rss_mb','?')} MB</p>
<div class="box"><b>HONESTY NOTE.</b> Every "internal state" below is a
documented numeric transform of MEASURED model variables (MBON/DAN/DN
population rates, KC ensemble eligibility, choices, US delivery).  These
are <i>computational state variables</i>, not claims about subjective
experience; friendly labels are captions, the underlying variables are the
science.  Tier labels: substrate = BIOLOGICAL connectome; plasticity rule
= MODELED; task/arena/interface/readout = ENGINEERED (see
brain/data/d11_d13/preregistration.json).</div>
<div class="box"><h2>1. Internal-state time series</h2><img
 src="data:image/png;base64,{p1}"></div>
<div class="box"><h2>2. Choices, US delivery, DAN rates, MBON rates</h2>
<img src="data:image/png;base64,{p2}"></div>
<div class="box"><h2>3. Reward expectation + prediction error</h2><img
 src="data:image/png;base64,{p3}"></div>
<div class="box"><h2>4. Synaptic scales (KC&rarr;MBON, MODELED)</h2><img
 src="data:image/png;base64,{p4}"></div>
<div class="box"><h2>State &rarr; underlying variable</h2>
<table><tr><th>friendly label</th><th>numeric variable (measured)</th></tr>
<tr><td>positive valence</td><td>mean approach-MBON rate (L+R), Hz</td></tr>
<tr><td>negative valence</td><td>mean avoidance-MBON rate (L+R), Hz</td></tr>
<tr><td>valence bias</td><td>(apprL&minus;avoidL)&minus;(apprR&minus;avoidR), Hz</td></tr>
<tr><td>reward expectation (side)</td><td>appr/(appr+avoid+1Hz) of that side's MBONs</td></tr>
<tr><td>reward-prediction error</td><td>US(1/0) &minus; expectation(chosen side)</td></tr>
<tr><td>novelty</td><td>1 &minus; cosine(KC-ensemble eligibility, EMA of past)</td></tr>
<tr><td>exploration drive</td><td>Shannon entropy of last 8 choices</td></tr>
<tr><td>threat drive</td><td>loom-channel drive + giant-fiber rate, Hz</td></tr>
<tr><td>uncertainty</td><td>EW std (half-life 8) of RPE</td></tr>
<tr><td>arousal-like</td><td>z-score(PAM+PPL1+DN_ALL choice-window rates)</td></tr>
</table></div>
</body></html>"""
    out = Path(out) if out else session_dir / "dashboard_d12.html"
    out.write_text(html)
    print(f"[d12] dashboard -> {out}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["states", "dashboard"])
    ap.add_argument("session_dir")
    ap.add_argument("--out")
    args = ap.parse_args()
    trials = load_trials(args.session_dir)
    states = compute_states(trials, load_chunks(args.session_dir))
    if args.cmd == "states":
        out = Path(args.session_dir) / "states.jsonl"
        out.write_text("\n".join(json.dumps(s) for s in states))
        print(f"[d12] {len(states)} state records -> {out}")
    else:
        render_dashboard(args.session_dir, states, args.out)


if __name__ == "__main__":
    main()
