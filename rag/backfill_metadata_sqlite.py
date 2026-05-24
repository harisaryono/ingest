#!/usr/bin/env python3
"""Backfill the review metadata SQLite index from existing JSON corpus files."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from config import DATABASE_DIR, JSON_DIR, METADATA_DB_PATH  # noqa: E402
from metadata_store import connect, upsert_book, upsert_pages  # noqa: E402


def resolve_index_json_path(record: dict, json_dir: str) -> str:
    json_path = record.get("json_path", "")
    if json_path:
        if os.path.isabs(json_path):
            return json_path
        normalized = os.path.normpath(json_path)
        prefix = f"json_output{os.sep}"
        if normalized == "json_output":
            return json_dir
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]
        return os.path.join(json_dir, normalized)
    filename = record.get("filename", "")
    return os.path.join(json_dir, f"{filename}.json")


def load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return default
    return data if isinstance(data, type(default)) else default


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill metadata SQLite from json_output")
    parser.add_argument("--index-path", default=str(Path(JSON_DIR) / "_index.json"))
    parser.add_argument("--json-dir", default=str(JSON_DIR))
    parser.add_argument("--db-path", default=str(METADATA_DB_PATH))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=25)
    args = parser.parse_args()

    index_path = Path(args.index_path)
    json_dir = Path(args.json_dir)
    index = load_json(index_path, {"files": []})
    records = index.get("files", []) if isinstance(index, dict) else []
    if args.limit and args.limit > 0:
        records = records[: args.limit]

    processed = 0
    imported = 0
    skipped = 0
    errors = 0
    batch = 0

    with connect(args.db_path) as conn:
        for record in records:
            if not isinstance(record, dict):
                continue
            json_path = Path(resolve_index_json_path(record, str(json_dir)))
            if not json_path.exists():
                skipped += 1
                continue
            try:
                with json_path.open("r", encoding="utf-8") as f:
                    book = json.load(f)
                upsert_book(conn, book, record=record)
                upsert_pages(conn, book)
                imported += 1
                batch += 1
            except Exception as exc:
                skipped += 1
                errors += 1
                print(f"WARN skip {json_path}: {exc}", flush=True)
            processed += 1
            if batch >= args.batch_size:
                conn.commit()
                batch = 0
            if processed % 100 == 0:
                print(
                    f"progress processed={processed} imported={imported} skipped={skipped} errors={errors}",
                    flush=True,
                )
        if batch:
            conn.commit()

    print(f"processed={processed} imported={imported} skipped={skipped} errors={errors} db={args.db_path}", flush=True)


if __name__ == "__main__":
    main()
