## Verdicts (pre-registered criteria, fail-closed)

- **G4A — NEURAL ASSOCIATIVE LEARNING: FAIL**
- **G4B — BEHAVIOURAL EXPRESSION: FAIL**
- **OVERALL GATE 4 (v2): FAIL**

## G4A evidence

- A1 eligibility lateralization: mean fraction of pairing trials with paired-side eligibility > 2x unpaired = **0.8073** (gate >= 0.90: FAIL); per session [0.812, 0.844, 0.812, 0.812, 0.781, 0.781]
- A2 cue-specific synaptic change (paired- vs unpaired-context KC->GLUT-MBON scales): pooled 0.686 CI95 [0.608, 0.7564] (PASS)
- A3 (descriptive) avoidance-class MBON response to the paired CS, baseline -> acquisition: B_R1: 13.977->65.341 (+51.364); B_L1: 25.467->51.402 (+25.935); B_R2: 34.95->67.595 (+32.645); B_L2: 41.01->44.593 (+3.583); B_R3: 50.998->53.712 (+2.714); B_L3: 40.909->62.32 (+21.411)
- A4 learned valence signal (toward-rewarded, Hz): pooled -4.0998 CI95 [-16.1142, 7.9147] (FAIL); per session {'B_R1': -26.0741, 'B_L1': 14.2825, 'B_R2': -19.1982, 'B_L2': 3.3905, 'B_R3': -9.5313, 'B_L3': 12.5319}

## G4B evidence

- B1 route-B preference shift toward the rewarded side (pooled 6 sessions): -0.0486 CI95 [-0.25, 0.1667]; per session {'B_R1': -0.417, 'B_L1': 0.125, 'B_R2': -0.25, 'B_L2': 0.0, 'B_R3': -0.125, 'B_L3': 0.375}
- B2 both directions: RIGHT-rewarded mean -0.264 [-0.417, -0.25, -0.125]; LEFT-rewarded mean 0.167 [0.125, 0.0, 0.375]
- B3 controls (pooled shifts, CI must overlap 0):
  - A_R: -0.0833 CI95 [-0.0833, -0.0833]
  - A_L: 0.0 CI95 [0.0, 0.0]
  - C_R: -0.375 CI95 [-0.375, -0.375] (vs B: 0.326)
  - D_R: -0.4167 CI95 [-0.4167, -0.4167] (vs B: 0.368)
  - F_R: -0.1667 CI95 [-0.1667, -0.1667]

## Descriptive battery (B sessions)

| session | extinction first3 -> last3 (rewarded side) | reversal P(new side) | context x0.8 / x1.2 | distractor | route-A p(left) |
|---|---|---|---|---|---|
| B_R1 | 0.0 -> 0.3333333333333333 | 0.875 | 1.0 / 0.0 | 0.0 | 0.429 (7/8) |
| B_L1 | 0.3333333333333333 -> 0.3333333333333333 | 0.125 | 0.3333333333333333 / 1.0 | 1.0 | 0.125 (8/8) |
| B_R2 | 0.0 -> 0.0 | 1.0 | 0.3333333333333333 / 0.0 | 0.0 | 0.714 (7/8) |
| B_L2 | 0.3333333333333333 -> 0.6666666666666666 | 0.375 | 0.3333333333333333 / 1.0 | 0.3333333333333333 | 0.25 (8/8) |
| B_R3 | 0.0 -> 0.0 | 0.875 | 0.6666666666666666 / 0.0 | 0.0 | 0.375 (8/8) |
| B_L3 | 0.3333333333333333 -> 0.6666666666666666 | 0.125 | 1.0 / 0.6666666666666666 | 0.3333333333333333 | 0.25 (8/8) |

- E_rl B_R1 (NON-biological Q-learner): baseline 0.167 -> acquisition 0.75
- E_rl B_L1 (NON-biological Q-learner): baseline 1.0 -> acquisition 1.0

## Performance (Section N)

| session | wall s | bio s | peak RSS MB | CPU s | mean pref wall s | trials |
|---|---|---|---|---|---|---|
| B_R1 | 2095.4 | 202.3 | 3018.8 | 2093.4 | 24.7 | 94 |
| B_L1 | 2067.1 | 196.4 | 3018.3 | 2065.4 | 24.9 | 94 |
| B_R2 | 2178.9 | 200.0 | 3013.4 | 2176.7 | 25.9 | 94 |
| B_L2 | 1958.0 | 197.4 | 3006.9 | 1956.7 | 23.3 | 94 |
| B_R3 | 2024.4 | 198.4 | 3017.3 | 2022.9 | 24.2 | 94 |
| B_L3 | 2060.4 | 197.1 | 3025.1 | 2058.5 | 24.6 | 94 |
| A_R | 2020.2 | 196.7 | 3025.9 | 2018.3 | 24.4 | 94 |
| A_L | 2001.3 | 197.4 | 3023.2 | 1999.8 | 24.1 | 94 |
| C_R | 2048.9 | 197.8 | 3018.5 | 2047.3 | 24.7 | 94 |
| D_R | 2006.8 | 197.0 | 3014.5 | 2005.3 | 24.2 | 94 |
| F_R | 1944.4 | 196.5 | 3020.1 | 1942.7 | 23.5 | 94 |

Frozen calibration: n_cs=100/side, kc_cs_right=75.0 Hz, kc_cs_left=35.0 Hz (boundary P(left)=0.5), US=PAM01-11 (252 cells) @ 150.0 Hz.
