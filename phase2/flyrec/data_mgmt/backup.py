"""Immutable session backup — never overwrites, always verifies copies.

Backs up every session directory under `data/sessions/` into
<to>/flyagent-backup/ with a SHA-256 manifest per file:

  * a session already in the manifest with MATCHING hashes  -> skipped;
  * a session already in the manifest with DIFFERENT hashes -> ERROR
    (raw recordings are immutable — this is a data-integrity alarm, not
    something the backup silently "fixes");
  * a new session -> copied, every copied byte re-hashed and compared,
    then added to the manifest.

A repo snapshot (current gate.yaml / tracking.yaml / apparatus_checks.yaml
hashes + git revision + UTC time) is written alongside so every backup is
self-describing. Nothing is ever deleted from the backup target.
"""
from __future__ import annotations

import datetime
import json
import os
import shutil
from pathlib import Path

from ..config import CONFIG_DIR
from ..rig import (SESSIONS_DIR, config_hashes, git_revision, sha256_file,
                   utc_now_iso)

MANIFEST_NAME = "backup_manifest.json"
BACKUP_SUBDIR = "flyagent-backup"


def _load_manifest(backup_root: Path) -> dict:
    p = backup_root / MANIFEST_NAME
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"sessions": {}, "repo_snapshots": []}


def _atomic_write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _hash_dir(d: Path) -> dict:
    return {f.name: sha256_file(f) for f in sorted(d.iterdir()) if f.is_file()}


def backup(to_dir: str | Path, sessions_root: Path | None = None,
           repo_root: Path | None = None) -> dict:
    to_dir = Path(to_dir).expanduser()
    backup_root = to_dir / BACKUP_SUBDIR
    backup_root.mkdir(parents=True, exist_ok=True)
    sessions_root = Path(sessions_root) if sessions_root else SESSIONS_DIR

    manifest = _load_manifest(backup_root)
    report = {"utc": utc_now_iso(), "target": str(backup_root),
              "skipped": [], "backed_up": [], "errors": [], "verified": []}

    session_dirs = sorted(d for d in sessions_root.iterdir()
                          if d.is_dir() and next(d.glob("video.*"), None))
    if not session_dirs:
        report["errors"].append(f"no session directories under {sessions_root}")

    for sdir in session_dirs:
        name = sdir.name
        src_hashes = _hash_dir(sdir)
        known = manifest["sessions"].get(name)
        if known:
            same = all(known.get("files", {}).get(f) == h
                       for f, h in src_hashes.items()) and \
                set(known.get("files", {})) == set(src_hashes)
            if same:
                report["skipped"].append(name)
                continue
            report["errors"].append(
                f"{name}: files changed since the last backup (raw "
                "recordings must be immutable) - NOT overwritten; inspect "
                "manually")
            continue
        dest = backup_root / "sessions" / name
        if dest.exists():
            report["errors"].append(
                f"{name}: target exists but is not in the manifest - "
                "inspect manually (NOT overwritten)")
            continue
        dest.mkdir(parents=True)
        copied = {}
        failed = False
        for fname, _h in src_hashes.items():
            shutil.copy2(sdir / fname, dest / fname)
            got = sha256_file(dest / fname)
            if got != src_hashes[fname]:
                report["errors"].append(
                    f"{name}/{fname}: copy verification FAILED "
                    f"({got[:12]} != {src_hashes[fname][:12]})")
                failed = True
                continue
            copied[fname] = got
            report["verified"].append(f"{name}/{fname}")
        if copied and not failed:
            manifest["sessions"][name] = {
                "files": copied, "backed_up_utc": utc_now_iso()}
            report["backed_up"].append(name)

    # self-describing snapshot of the current config/code state
    snapshot = {"utc": utc_now_iso(), "git_revision": git_revision(),
                "config_hashes": config_hashes(),
                "config_files": {f.name: sha256_file(f)
                                 for f in sorted(CONFIG_DIR.glob("*.yaml"))}}
    manifest["repo_snapshots"].append(snapshot["utc"])
    _atomic_write_json(backup_root / MANIFEST_NAME, manifest)
    (backup_root / ("repo_snapshot_" +
                    snapshot["utc"].replace(":", "").replace("+", "") +
                    ".json")).write_text(
        json.dumps(snapshot, indent=2), encoding="utf-8")

    report["manifest"] = str(backup_root / MANIFEST_NAME)
    report["n_sessions_in_manifest"] = len(manifest["sessions"])
    report["ok"] = not report["errors"]
    return report


def render_backup(report: dict) -> str:
    lines = ["=" * 70, "SESSION BACKUP", "=" * 70]
    lines.append(f"target: {report.get('target')}")
    lines.append(f"backed up this run: {len(report.get('backed_up', []))}")
    for v in report.get("backed_up", []):
        lines.append(f"    {v}")
    lines.append(f"already up to date: {len(report.get('skipped', []))}")
    lines.append(f"files verified:     {len(report.get('verified', []))}")
    for e in report.get("errors", []):
        lines.append(f"ERROR: {e}")
    lines.append(f"manifest: {report.get('manifest')}")
    lines.append("OVERALL: " + ("OK" if report.get("ok") else
                                "ERRORS - inspect before continuing"))
    lines.append("=" * 70)
    return "\n".join(lines)
