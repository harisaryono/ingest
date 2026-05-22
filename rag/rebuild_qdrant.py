#!/usr/bin/env python3
"""Rebuild the local Qdrant database from source JSON.

The script archives the current qdrant_db once, then runs the normal ingest
pipeline against a fresh database. A small journal in DATABASE_DIR keeps the
rebuild resumable so reruns do not archive the database again.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from config import DATABASE_DIR, QDRANT_PATH  # noqa: E402

DATABASE_PATH = Path(DATABASE_DIR)
REBUILD_STATE_PATH = DATABASE_PATH / "rebuild_qdrant_state.json"
BACKUP_ROOT = DATABASE_PATH / "backups"
DEFAULT_VENV_PYTHON = Path("/home/harry/venv/rag-buku/bin/python")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str) -> None:
    print(message, flush=True)


def load_rebuild_state() -> dict:
    if not REBUILD_STATE_PATH.exists():
        return {}
    try:
        with REBUILD_STATE_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def save_rebuild_state(state: dict) -> None:
    REBUILD_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = REBUILD_STATE_PATH.with_suffix(".json.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(tmp_path, REBUILD_STATE_PATH)


def unique_backup_path() -> Path:
    BACKUP_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    candidate = BACKUP_ROOT / f"qdrant_db_{stamp}"
    if not candidate.exists():
        return candidate
    suffix = 1
    while True:
        alt = BACKUP_ROOT / f"qdrant_db_{stamp}_{suffix}"
        if not alt.exists():
            return alt
        suffix += 1


def archive_existing_db() -> Path | None:
    qdrant_path = Path(QDRANT_PATH)
    if not qdrant_path.exists():
        return None
    backup_path = unique_backup_path()
    shutil.move(str(qdrant_path), str(backup_path))
    return backup_path


def choose_python() -> str:
    override = os.environ.get("RAG_PYTHON", "").strip()
    if override:
        return override
    if DEFAULT_VENV_PYTHON.exists():
        return str(DEFAULT_VENV_PYTHON)
    return sys.executable


def run_ingest() -> int:
    env = os.environ.copy()
    env.setdefault("EMBED_BATCH_SIZE", "128")
    env.setdefault("EMBED_TIMEOUT", "120")
    env.pop("INGEST_SKIP_BOOTSTRAP", None)
    python_bin = choose_python()
    log(f"Using Python interpreter: {python_bin}")
    proc = subprocess.run(
        [python_bin, str(SCRIPT_DIR / "ingest.py")],
        cwd=str(SCRIPT_DIR),
        env=env,
    )
    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild the local Qdrant DB")
    parser.add_argument(
        "--force",
        action="store_true",
        help="archive the current qdrant_db again and restart from scratch",
    )
    args = parser.parse_args()

    state = load_rebuild_state()
    status = state.get("status", "")

    if status == "complete" and not args.force:
        log(f"Rebuild already complete at {state.get('finished_at', 'unknown time')}")
        return 0

    if args.force:
        log("Force rebuild requested.")
        archived = archive_existing_db()
        state = {
            "status": "prepared",
            "backup_path": str(archived) if archived else "",
            "prepared_at": utc_now(),
            "updated_at": utc_now(),
        }
        save_rebuild_state(state)
    elif status not in {"prepared", "ingesting", "failed"}:
        archived = archive_existing_db()
        state = {
            "status": "prepared",
            "backup_path": str(archived) if archived else "",
            "prepared_at": utc_now(),
            "updated_at": utc_now(),
        }
        save_rebuild_state(state)
        if archived:
            log(f"Archived existing Qdrant DB to {archived}")
        else:
            log("No existing Qdrant DB found; starting fresh rebuild")
    else:
        log(f"Resuming rebuild from status={status or 'unknown'}")

    state["status"] = "ingesting"
    state["started_at"] = state.get("started_at") or utc_now()
    state["updated_at"] = utc_now()
    save_rebuild_state(state)

    log("Running ingest pipeline...")
    code = run_ingest()
    if code == 0:
        state["status"] = "complete"
        state["finished_at"] = utc_now()
        state["updated_at"] = utc_now()
        save_rebuild_state(state)
        log("Rebuild complete.")
    else:
        state["status"] = "failed"
        state["last_error_at"] = utc_now()
        state["updated_at"] = utc_now()
        save_rebuild_state(state)
        log(f"Rebuild failed with exit code {code}.")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
