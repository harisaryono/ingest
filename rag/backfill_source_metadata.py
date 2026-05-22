#!/usr/bin/env python3
"""Backfill source metadata into existing JSON corpus files and indexes.

This updates historical JSON files that were created before source metadata
fields were added, using the authoritative values from _index.json.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Iterable, Tuple


def load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return default
    return data if isinstance(data, type(default)) else default


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(tmp, path)


def source_type_from_record(record: Dict) -> str:
    source_type = str(record.get("source_type", "") or "").strip().lower()
    if source_type:
        return source_type

    source_ext = str(record.get("source_ext", "") or "").strip().lower()
    if source_ext in {".htm", ".html"}:
        return "html"
    if source_ext:
        return source_ext.lstrip(".")

    source_path = str(record.get("source_path", "") or record.get("filename", "") or "").strip()
    if source_path:
        ext = os.path.splitext(source_path)[1].lower()
        if ext in {".htm", ".html"}:
            return "html"
        if ext:
            return ext.lstrip(".")
    return "unknown"


def source_ext_from_record(record: Dict) -> str:
    source_ext = str(record.get("source_ext", "") or "").strip().lower()
    if source_ext:
        return source_ext

    source_path = str(record.get("source_path", "") or record.get("filename", "") or "").strip()
    if source_path:
        return os.path.splitext(source_path)[1].lower()
    return ""


def document_type_from_record(record: Dict) -> str:
    document_type = str(record.get("document_type", "") or "").strip().lower()
    if document_type:
        return document_type
    return "html_document" if source_type_from_record(record) == "html" else "book"


def build_record_lookup(index: Dict) -> Dict[str, Dict]:
    lookup = {}
    for record in index.get("files", []):
        raw = str(record.get("json_path", "") or "")
        variants = []
        if raw:
            normalized = os.path.normpath(raw)
            variants.append(normalized)
            prefix = f"json_output{os.sep}"
            if normalized.startswith(prefix):
                variants.append(normalized[len(prefix) :])
            if normalized == "json_output":
                variants.append("")
        for key in variants:
            if key:
                lookup.setdefault(key, record)
    return lookup


def update_book_json(json_path: Path, index_record: Dict) -> bool:
    payload = load_json(json_path, {})
    if not isinstance(payload, dict) or not payload:
        return False

    changed = False

    def set_if_missing(key: str, value):
        nonlocal changed
        if key not in payload or payload.get(key) in ("", None):
            payload[key] = value
            changed = True

    set_if_missing("source_root", index_record.get("source_root", ""))
    set_if_missing("source_path", index_record.get("source_path", ""))
    set_if_missing("source_relpath", index_record.get("source_relpath", ""))
    set_if_missing("source_ext", source_ext_from_record(index_record))
    set_if_missing("source_type", source_type_from_record(index_record))
    set_if_missing("document_type", document_type_from_record(index_record))
    set_if_missing("review_status", index_record.get("review_status", payload.get("review_status", "approved_auto")))
    set_if_missing("review_required", bool(index_record.get("review_required", payload.get("review_required", False))))
    set_if_missing("review_route", index_record.get("review_route", payload.get("review_route", "auto")))
    set_if_missing("reviewed_by", index_record.get("reviewed_by", payload.get("reviewed_by", "")))
    set_if_missing("reviewed_at", index_record.get("reviewed_at", payload.get("reviewed_at", "")))
    set_if_missing("review_note", index_record.get("review_note", payload.get("review_note", "")))
    set_if_missing("ingest_ready", bool(index_record.get("ingest_ready", payload.get("ingest_ready", True))))
    set_if_missing("quality_status", index_record.get("quality_status", payload.get("quality_status", "ok")))
    set_if_missing("quality_reasons", index_record.get("quality_reasons", payload.get("quality_reasons", [])))
    set_if_missing("quality_warnings", index_record.get("quality_warnings", payload.get("quality_warnings", [])))

    if changed:
        save_json(json_path, payload)
    return changed


def update_index_records(index: Dict) -> Tuple[int, int]:
    files = index.get("files", [])
    changed = 0
    for record in files:
        before = dict(record)
        record["source_path"] = record.get("source_path", "") or ""
        record["source_relpath"] = record.get("source_relpath", "") or ""
        record["source_ext"] = source_ext_from_record(record)
        record["source_type"] = source_type_from_record(record)
        record["document_type"] = document_type_from_record(record)
        if record != before:
            changed += 1

    index["source_types"] = {}
    index["document_types"] = {}
    for record in files:
        st = source_type_from_record(record)
        dt = document_type_from_record(record)
        index["source_types"][st] = index["source_types"].get(st, 0) + 1
        index["document_types"][dt] = index["document_types"].get(dt, 0) + 1

    return changed, len(files)


def update_content_index(content_index: Dict, lookup: Dict[str, Dict]) -> int:
    changed = 0
    for entry in content_index.get("entries", []):
        key = os.path.normpath(str(entry.get("json_path", "")))
        record = lookup.get(key)
        if not record:
            continue
        updates = {
            "source_path": record.get("source_path", ""),
            "source_relpath": record.get("source_relpath", ""),
            "source_ext": source_ext_from_record(record),
            "source_type": source_type_from_record(record),
            "document_type": document_type_from_record(record),
        }
        for k, v in updates.items():
            if entry.get(k) != v:
                entry[k] = v
                changed += 1
    return changed


def update_unindexed_empty_json(json_path: Path, output_dir: Path) -> bool:
    payload = load_json(json_path, {})
    if not isinstance(payload, dict) or not payload:
        return False

    source_name = json_path.name[:-5] if json_path.name.endswith(".json") else json_path.name
    source_path = (Path("/media/harry/DATA250/txt") / source_name).resolve()
    source_ext = os.path.splitext(source_name)[1].lower()
    source_type = source_ext.lstrip(".") if source_ext else "unknown"

    updates = {
        "source_root": str(Path("/media/harry/DATA250/txt")),
        "source_path": str(source_path),
        "source_relpath": source_name,
        "source_ext": source_ext,
        "source_type": source_type,
        "document_type": "empty",
    }

    changed = False
    for key, value in updates.items():
        if payload.get(key) != value:
            payload[key] = value
            changed = True

    if changed:
        save_json(json_path, payload)
    return changed


def main() -> None:
    repo_dir = Path(__file__).resolve().parent.parent
    database_dir = Path(os.getenv("DATABASE_DIR", str(repo_dir.parent.parent / "DATABASE")))
    output_dir = database_dir / "json_output"
    index_path = output_dir / "_index.json"
    content_index_path = output_dir / "_content_index.json"

    index = load_json(index_path, {"total_files": 0, "languages": {}, "files": []})
    content_index = load_json(content_index_path, {"total_files": 0, "entries": []})
    lookup = build_record_lookup(index)

    json_files = [p for p in output_dir.rglob("*.json") if p.name not in {"_index.json", "_content_index.json"}]
    json_updates = 0
    skipped = 0
    for path in sorted(json_files):
        rel = os.path.normpath(os.path.relpath(path, output_dir))
        record = lookup.get(rel)
        if not record:
            record = lookup.get(f"json_output{os.sep}{rel}")
        if not record:
            if rel.startswith(f"_empty{os.sep}") and update_unindexed_empty_json(path, output_dir):
                json_updates += 1
                continue
            skipped += 1
            continue
        if update_book_json(path, record):
            json_updates += 1

    index_updates, total_index_records = update_index_records(index)
    content_updates = update_content_index(content_index, lookup)

    save_json(index_path, index)
    save_json(content_index_path, content_index)

    print(f"JSON files scanned   : {len(json_files)}")
    print(f"JSON files updated   : {json_updates}")
    print(f"JSON files skipped   : {skipped}")
    print(f"Index records total  : {total_index_records}")
    print(f"Index records touched: {index_updates}")
    print(f"Content index updates: {content_updates}")
    print(f"Wrote index          : {index_path}")
    print(f"Wrote content index  : {content_index_path}")


if __name__ == "__main__":
    main()
