# GATE 5 v1 — assessment record

Date: 2026-09-30

Verdict: **FAIL / NOT COMPLETED under the frozen v1 preregistration.**

This is an engineering verdict only. It changes no biological result.

## Evidence

Windows development sessions successfully demonstrated:

- authorized GPO window capture and live dashboard mirror;
- canonical `canonical_brian` runtime in an isolated `subprocess_pipe`;
- synchronized perception / brain / shadow-action replay;
- zero autonomous game input;
- clean session `lab_20260930_025436` with 635 captured frames,
  589 fast-perception events, 9 canonical brain chunks, and 0 worker errors;
- automatic evidence frames and uploadable session ZIPs.

## Why v1 cannot be called PASS

The frozen `GATE5_PREREG.md` requires all of the following:

1. a series of at least three 10-minute game sessions;
2. in at least two of three sessions, a continuous canonical brain stream
   with no gap greater than 5 seconds;
3. the remaining manual/object-tracking and performance criteria.

The current target laptop executes a 50 ms canonical biological chunk in
roughly 6.8–11.0 wall-seconds in the latest clean session. Therefore the
frozen 5-second continuity threshold is not met even though the brain worker
runs correctly and bio-time advances monotonically. The required three
10-minute assessment sessions were also not performed.

Per the v1 fail-closed rule, these criteria are not weakened retroactively.
The v1 record remains FAIL / incomplete.

## Engineering follow-up

A separately pre-registered Gate-5 v2 may use hardware-realistic continuity
criteria while preserving the important safety requirements:

- canonical brain unchanged;
- zero autonomous input;
- honest measured latency;
- replay completeness;
- explicit perception-role validation;
- no hidden CV-to-input shortcut.
