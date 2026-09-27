"""SQLite experiment database.

Principles (per spec):
  * every action is timestamped
  * raw data (optional frame dumps) lives on disk, events/metrics in SQLite
  * nothing is ever overwritten - each run gets its own database file
    inside a timestamped run directory
  * forward-compatible columns (objective, strategy, phase, subject_id)
    so real-fly and later-phase experiments slot in without migration
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id   TEXT PRIMARY KEY,
    started_at      TEXT NOT NULL,
    controller      TEXT NOT NULL,
    condition       TEXT,
    subject_id      TEXT,
    session_index   INTEGER,
    phase           TEXT,               -- baseline | training | test | reversal | na
    scenario        TEXT,
    software_version TEXT,
    config_json     TEXT NOT NULL,
    seed            INTEGER,
    notes           TEXT
);
CREATE TABLE IF NOT EXISTS trials (
    trial_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id   TEXT NOT NULL,
    trial_number    INTEGER NOT NULL,
    started_at_ms   INTEGER,
    duration_s      REAL,
    outcome         TEXT,               -- SUCCESS | TIMEOUT
    time_to_target_s REAL,
    path_length     REAL,
    straight_dist   REAL,
    path_efficiency REAL,
    n_actions       INTEGER,
    n_action_changes INTEGER,
    mean_inter_action_s REAL,
    reward_total    REAL,
    target_x        REAL, target_y REAL,
    start_x         REAL, start_y  REAL
);
CREATE TABLE IF NOT EXISTS events (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    trial_id INTEGER NOT NULL,
    t_ms     INTEGER NOT NULL,
    kind     TEXT NOT NULL,
    payload  TEXT
);
CREATE TABLE IF NOT EXISTS frames (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    trial_id INTEGER NOT NULL,
    t_ms     INTEGER NOT NULL,
    fly_x REAL, fly_y REAL, fly_speed REAL, fly_heading REAL,
    behavior_state TEXT, zone TEXT, decoded_action TEXT,
    player_x REAL, player_y REAL, dist_to_target REAL,
    reward REAL, track_confidence REAL
);
CREATE INDEX IF NOT EXISTS idx_trials_exp ON trials(experiment_id);
CREATE INDEX IF NOT EXISTS idx_events_trial ON events(trial_id);
CREATE INDEX IF NOT EXISTS idx_frames_trial ON frames(trial_id);
"""


class Database:
    def __init__(self, path):
        self.path = Path(path)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)

    def close(self) -> None:
        self.conn.commit()
        self.conn.close()

    # ------------------------------------------------------------------
    def insert_experiment(self, experiment_id: str, started_at: str,
                          controller: str, condition: str, config: dict,
                          software_version: str, seed: int,
                          subject_id: str = "synthetic",
                          session_index: int = 0, phase: str = "na",
                          scenario: str = "open_field",
                          notes: str = "") -> None:
        cols = ("experiment_id, started_at, controller, condition, subject_id,"
                " session_index, phase, scenario, software_version, config_json,"
                " seed, notes")
        marks = ",".join(["?"] * 12)
        self.conn.execute(
            f"INSERT INTO experiments ({cols}) VALUES ({marks})",
            (experiment_id, started_at, controller, condition, subject_id,
             session_index, phase, scenario, software_version,
             json.dumps(config, default=str), int(seed), notes))

    _TRIAL_COLS = ("experiment_id, trial_number, started_at_ms, duration_s,"
                   " outcome, time_to_target_s, path_length, straight_dist,"
                   " path_efficiency, n_actions, n_action_changes,"
                   " mean_inter_action_s, reward_total, target_x, target_y,"
                   " start_x, start_y")

    def insert_trial(self, experiment_id: str, r: dict) -> int:
        cols = self._TRIAL_COLS
        n = len([c for c in cols.split(",")])
        cur = self.conn.execute(
            f"INSERT INTO trials ({cols}) VALUES ({','.join(['?'] * n)})",
            (experiment_id, r["trial_number"], r["started_at_ms"],
             r["duration_s"], r["outcome"], r["time_to_target_s"],
             r["path_length"], r["straight_dist"], r["path_efficiency"],
             r["n_actions"], r["n_action_changes"], r["mean_inter_action_s"],
             r["reward_total"], r["target_x"], r["target_y"],
             r["start_x"], r["start_y"]))
        return int(cur.lastrowid)

    def insert_event(self, trial_id: int, t_ms: int, kind: str,
                     payload: dict | None = None) -> None:
        self.conn.execute("INSERT INTO events (trial_id, t_ms, kind, payload)"
                          " VALUES (?,?,?,?)",
                          (trial_id, int(t_ms), kind,
                           json.dumps(payload or {}, default=str)))

    def insert_frame(self, trial_id: int, t_ms: int, fly: dict | None,
                     player: tuple[float, float], dist: float,
                     reward: float) -> None:
        f = fly or {}
        cols = ("trial_id, t_ms, fly_x, fly_y, fly_speed, fly_heading,"
                " behavior_state, zone, decoded_action, player_x, player_y,"
                " dist_to_target, reward, track_confidence")
        n = len(cols.split(","))
        self.conn.execute(
            f"INSERT INTO frames ({cols}) VALUES ({','.join(['?'] * n)})",
            (trial_id, int(t_ms), f.get("x"), f.get("y"), f.get("speed"),
             f.get("heading"), f.get("state"), f.get("zone"),
             f.get("action"), player[0], player[1], dist, reward,
             f.get("confidence")))

    def commit(self) -> None:
        self.conn.commit()

    # ------------------------------------------------------------------
    @staticmethod
    def fetch_trials(db_path) -> list[dict]:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT trial_number, outcome, duration_s, time_to_target_s,"
                " path_efficiency, n_actions, n_action_changes, reward_total"
                " FROM trials ORDER BY trial_number").fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()
