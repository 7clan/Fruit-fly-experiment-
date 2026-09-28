# Gate-4 v2 — calibration report (ALL tested values)

Pre-registered procedure: GATE4_V2_PREREG.md section 4. No reward side existed during calibration; no plasticity except the uscheck pairing trial; no outcome data.

## Phase G — CS dose-response (seed c1, 6 preference trials/setting, symmetric rates)

| setting | n KC/side | rate (Hz) | KC L/R (Hz) | appr (Hz) | avoid (Hz) | P9 diff (Hz) | P(left|B) | mean V | std V | passes |
|---|---|---|---|---|---|---|---|---|---|---|
| S1 | 100 | 75 | 84.368/79.849 | 39.095 | 44.539 | 0.0 | 0.6666666666666666 | 9.708 | 14.928 | True |
| S2 | 200 | 75 | 81.744/78.725 | 47.291 | 44.489 | 0.0 | 0.3333333333333333 | -1.56 | 21.14 | True |
| S3 | 200 | 100 | 101.01/100.42 | 44.82 | 38.163 | 0.0 | 0.16666666666666666 | -6.707 | 6.474 | True |
| S4 | 400 | 100 | 119.844/111.187 | 101.852 | 140.991 | 0.0 | 1.0 | 20.433 | 0.845 | False |

**Selected (pre-registered rule): S1 (n=100/side, R=75 Hz)**

## Phase F — balanced-boundary sweep (seeds c1+c2, 12 preference trials/value)

| kc_cs_left (Hz) | P(left|B) | mean V | std V | KC L (Hz) |
|---|---|---|---|---|
| 35 | 0.5 | 8.519 | 19.346 | 50.961 |
| 75 | 0.6666666666666666 | 13.725 | 16.039 | 87.578 |
| 95 | 0.3333333333333333 | 5.935 | 15.517 | 100.418 |
| 115 | 0.3333333333333333 | 3.585 | 11.724 | 116.954 |
| 55 | 0.8333333333333334 | 18.496 | 14.709 | 75.236 |

**Selected kc_cs_left = 35 Hz (P(left)=0.5)** — neutral range [0.35, 0.65]: YES

## US feasibility check

```json
{
 "gate_hz": 148.46,
 "gate_ok": true,
 "update_ok": true,
 "n_synapses_changed": 360,
 "avoid_r_base": 0.0,
 "avoid_r_pairing": 73.106,
 "response_change_hz": 73.106,
 "response_ok": true,
 "uscheck_pass": true
}
```

## Freeze

See `FROZEN_PARAMS.json`. Reward-side assignment (prereg section 7) happens only after this freeze.
