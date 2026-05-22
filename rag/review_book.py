#!/usr/bin/env python3
"""Update review status for a book JSON record and its index entry."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional


def load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return default
    return data if isinstance(data, type(default)) else default


def resolve_index_json_path(record: Dict, json_dir: str) -> str:
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


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(tmp, path)


def find_record(index: Dict, book_id: Optional[str], json_path: Optional[str]) -> Optional[Dict]:
    files = index.get("files", [])
    if book_id:
        for record in files:
            if record.get("book_id") == book_id:
                return record
    if json_path:
        target = os.path.normpath(json_path)
        for record in files:
            if os.path.normpath(str(record.get("json_path", ""))) == target:
                return record
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Approve or reject a book for ingestion")
    parser.add_argument("--book-id", help="book_id from _index.json")
    parser.add_argument("--json-path", help="relative json path from _index.json")
    parser.add_argument(
        "--status",
        required=True,
        choices=[
            "approved_manual",
            "approved_lease",
            "rejected",
            "pending_review",
        ],
        help="new review status",
    )
    parser.add_argument("--reviewed-by", default="manual", help="reviewer label to store")
    parser.add_argument("--note", default="", help="optional note")
    args = parser.parse_args()

    if not args.book_id and not args.json_path:
        raise SystemExit("Provide --book-id or --json-path")

    repo_dir = Path(__file__).resolve().parent.parent
    database_dir = Path(os.getenv("DATABASE_DIR", str(repo_dir.parent.parent / "DATABASE")))
    output_dir = database_dir / "json_output"
    index_path = output_dir / "_index.json"
    content_index_path = output_dir / "_content_index.json"

    index = load_json(index_path, {"total_files": 0, "languages": {}, "files": []})
    record = find_record(index, args.book_id, args.json_path)
    if not record:
        raise SystemExit("Book record not found in index")

    json_path = Path(resolve_index_json_path(record, str(output_dir)))
    if not json_path.exists():
        raise SystemExit(f"Book JSON not found: {json_path}")

    now = datetime.now(timezone.utc).isoformat()
    payload = load_json(json_path, {})
    if not isinstance(payload, dict):
        raise SystemExit("Book JSON is malformed")

    payload["review_status"] = args.status
    payload["review_required"] = args.status == "pending_review"
    payload["review_route"] = "manual_or_lease_coordinator" if args.status == "pending_review" else args.reviewed_by
    payload["reviewed_by"] = args.reviewed_by if args.status != "pending_review" else ""
    payload["reviewed_at"] = now if args.status != "pending_review" else ""
    payload["review_note"] = args.note
    payload["ingest_ready"] = args.status.startswith("approved")

    save_json(json_path, payload)

    for item in index.get("files", []):
        if item.get("book_id") == record.get("book_id") or os.path.normpath(str(item.get("json_path", ""))) == os.path.normpath(str(record.get("json_path", ""))):
            item["review_status"] = payload["review_status"]
            item["review_required"] = payload["review_required"]
            item["review_route"] = payload["review_route"]
            item["reviewed_by"] = payload["reviewed_by"]
            item["reviewed_at"] = payload["reviewed_at"]
            item["review_note"] = payload["review_note"]
            item["ingest_ready"] = payload["ingest_ready"]
            break

    save_json(index_path, index)

    content_index = load_json(content_index_path, {"total_files": 0, "entries": []})
    if isinstance(content_index, dict):
        for entry in content_index.get("entries", []):
            if entry.get("json_path") == record.get("json_path"):
                entry["review_status"] = payload["review_status"]
                entry["review_required"] = payload["review_required"]
                entry["review_route"] = payload["review_route"]
                entry["reviewed_by"] = payload["reviewed_by"]
                entry["reviewed_at"] = payload["reviewed_at"]
                entry["review_note"] = payload["review_note"]
                entry["ingest_ready"] = payload["ingest_ready"]
                break
        save_json(content_index_path, content_index)

    print(f"Updated {record.get('json_path')} -> {payload['review_status']}")


if __name__ == "__main__":
    main()
