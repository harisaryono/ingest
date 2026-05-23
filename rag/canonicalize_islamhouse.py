#!/usr/bin/env python3
"""Canonicalize Islamhouse JSON outputs by family.

The script keeps one canonical record per exact family key:
language + document_type + normalized filename stem.
Duplicate variants are archived out of the active json_output tree, the
index/content index are rewritten, and optional Qdrant/state pruning keeps the
search corpus aligned.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from config import COLLECTION_NAME, DATABASE_DIR, INGEST_STATE_PATH, JSON_DIR, LEXICAL_INDEX_PATH, QDRANT_PATH  # noqa: E402

DATABASE_PATH = Path(DATABASE_DIR)
INDEX_PATH = Path(JSON_DIR) / "_index.json"
CONTENT_INDEX_PATH = Path(JSON_DIR) / "_content_index.json"
STATE_PATH = DATABASE_PATH / "canonicalize_islamhouse_state.json"
REPORT_PATH = DATABASE_PATH / "canonicalize_islamhouse_report.jsonl"
ARCHIVE_ROOT = Path(JSON_DIR) / "_archive" / "canonical_duplicates"

QUALITY_RANK = {
    "ok": 3,
    "warn": 2,
    "quarantine": 0,
}
CONVERSION_RANK = {
    "good": 3,
    "degraded": 2,
    "failed": 0,
}
REVIEW_RANK = {
    "approved_manual": 4,
    "approved_lease": 4,
    "approved_auto": 3,
    "pending_review": 1,
    "rejected": 0,
}
SOURCE_TYPE_RANK = {
    "docx": 6,
    "pdf": 5,
    "doc": 4,
    "txt": 3,
    "html": 2,
    "ibooks": 1,
    "unknown": 0,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str) -> None:
    print(message, flush=True)


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
    fd, tmp_path = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=path.suffix, dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def _empty_state() -> Dict[str, Dict]:
    return {"books": {}, "chunks": {}}


def normalize_state(raw: Dict | None) -> Dict[str, Dict]:
    if not isinstance(raw, dict):
        return _empty_state()
    if "books" in raw or "chunks" in raw:
        books = raw.get("books", {})
        chunks = raw.get("chunks", {})
    else:
        books = raw
        chunks = {}

    normalized = _empty_state()

    if isinstance(books, dict):
        for book_id, book_state in books.items():
            if not isinstance(book_state, dict):
                continue
            normalized["books"][book_id] = {
                "source_hash": book_state.get("source_hash", ""),
                "next_batch_index": int(book_state.get("next_batch_index", 0) or 0),
                "complete": bool(book_state.get("complete", False)),
                "chunk_hashes": list(book_state.get("chunk_hashes", [])),
                "point_count": int(book_state.get("point_count", 0) or 0),
                "updated_at": float(book_state.get("updated_at", 0.0) or 0.0),
            }

    if isinstance(chunks, dict):
        for chunk_hash, chunk_state in chunks.items():
            if not isinstance(chunk_state, dict):
                continue
            normalized["chunks"][chunk_hash] = {
                "point_id": chunk_state.get("point_id", ""),
                "ref_count": int(chunk_state.get("ref_count", 0) or 0),
                "first_seen_at": float(chunk_state.get("first_seen_at", 0.0) or 0.0),
            }

    return normalized


def load_state(path: str = INGEST_STATE_PATH) -> Dict[str, Dict]:
    path_obj = Path(path)
    if not path_obj.exists():
        return _empty_state()
    try:
        with path_obj.open("r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        return _empty_state()
    return normalize_state(raw)


def save_state(state: Dict[str, Dict], path: str = INGEST_STATE_PATH) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".canonicalize_state.", suffix=".json", dir=str(path_obj.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2, sort_keys=True)
        os.replace(tmp_path, path_obj)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def delete_point_ids(client, point_ids: List[str], batch_size: int = 256) -> None:
    for i in range(0, len(point_ids), batch_size):
        client.delete(collection_name=COLLECTION_NAME, points_selector=point_ids[i : i + batch_size])


def release_book_from_state(state: Dict[str, Dict], client, book_id: str) -> int:
    book_state = state["books"].get(book_id)
    if not book_state:
        return 0

    point_ids_to_delete: List[str] = []
    for chunk_hash in book_state.get("chunk_hashes", []):
        chunk_state = state["chunks"].get(chunk_hash)
        if not chunk_state:
            continue
        chunk_state["ref_count"] = max(0, int(chunk_state.get("ref_count", 0)) - 1)
        if chunk_state["ref_count"] <= 0:
            point_ids_to_delete.append(chunk_state["point_id"])
            del state["chunks"][chunk_hash]

    if point_ids_to_delete:
        delete_point_ids(client, point_ids_to_delete)

    del state["books"][book_id]
    return len(point_ids_to_delete)


def append_jsonl(path: Path, entry: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def normalize_stem(filename: str) -> str:
    stem = Path(filename).stem
    stem = stem.replace("_", " ").replace("-", " ").replace(".", " ")
    stem = re.sub(r"[^a-z0-9]+", " ", stem.lower()).strip()
    stem = re.sub(r"\s+", " ", stem)
    return stem


def family_key(record: Dict) -> str:
    language = str(record.get("language", "unknown") or "unknown").strip().lower() or "unknown"
    document_type = str(record.get("document_type", "book") or "book").strip().lower() or "book"
    return f"{language}||{document_type}||{normalize_stem(record.get('filename', ''))}"


def score_record(record: Dict) -> Tuple[int, int, int, int, int, int, int, str]:
    quality = QUALITY_RANK.get(str(record.get("quality_status", "")).strip().lower(), 0)
    conversion = CONVERSION_RANK.get(str(record.get("conversion_status", "")).strip().lower(), 0)
    review = REVIEW_RANK.get(str(record.get("review_status", "")).strip().lower(), 0)
    ingest_ready = 1 if bool(record.get("ingest_ready", False)) else 0
    pages = int(record.get("total_pages", 0) or 0)
    source_type = SOURCE_TYPE_RANK.get(str(record.get("source_type", "unknown")).strip().lower() or "unknown", 0)
    size_bytes = int(record.get("size_bytes", 0) or 0)
    json_path = str(record.get("json_path", ""))
    return (quality, conversion, review, ingest_ready, pages, source_type, size_bytes, json_path)


def choose_canonical(records: List[Dict]) -> Dict:
    ranked = sorted(records, key=score_record, reverse=True)
    return ranked[0]


def family_sort_key(record: Dict) -> Tuple[str, str, str]:
    return (
        str(record.get("language", "unknown") or "unknown"),
        str(record.get("document_type", "book") or "book"),
        family_key(record),
    )


def resolve_active_json_path(json_path: str) -> Path:
    path = str(json_path or "")
    if not path:
        return Path(JSON_DIR)
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    normalized = os.path.normpath(path)
    prefix = f"json_output{os.sep}"
    if normalized == "json_output":
        return Path(JSON_DIR)
    if normalized.startswith(prefix):
        normalized = normalized[len(prefix) :]
    return Path(JSON_DIR) / normalized


def load_index_records() -> Dict:
    index = load_json(INDEX_PATH, {"total_files": 0, "languages": {}, "files": []})
    if "files" not in index or not isinstance(index.get("files"), list):
        index["files"] = []
    return index


def load_content_index() -> Dict:
    content_index = load_json(CONTENT_INDEX_PATH, {"entries": [], "total_files": 0})
    if "entries" not in content_index or not isinstance(content_index.get("entries"), list):
        content_index["entries"] = []
    return content_index


def group_families(records: List[Dict]) -> Dict[str, List[Dict]]:
    families: Dict[str, List[Dict]] = defaultdict(list)
    for record in records:
        families[family_key(record)].append(record)
    return families


def rebuild_index(records: List[Dict]) -> Dict:
    files = sorted(records, key=family_sort_key)
    index = {
        "files": files,
        "total_files": len(files),
    }
    languages: Counter[str] = Counter()
    source_types: Counter[str] = Counter()
    document_types: Counter[str] = Counter()
    conversion_status_counts: Counter[str] = Counter()
    quality_status_counts: Counter[str] = Counter()
    review_status_counts: Counter[str] = Counter()
    for record in files:
        languages[str(record.get("language", "unknown") or "unknown")] += 1
        source_types[str(record.get("source_type", "unknown") or "unknown")] += 1
        document_types[str(record.get("document_type", "book") or "book")] += 1
        conversion_status_counts[str(record.get("conversion_status", "unknown") or "unknown")] += 1
        quality_status_counts[str(record.get("quality_status", "ok") or "ok")] += 1
        review_status_counts[str(record.get("review_status", "approved_auto") or "approved_auto")] += 1
    index["languages"] = dict(sorted(languages.items()))
    index["source_types"] = dict(sorted(source_types.items()))
    index["document_types"] = dict(sorted(document_types.items()))
    index["conversion_status_counts"] = dict(sorted(conversion_status_counts.items()))
    index["quality_status_counts"] = dict(sorted(quality_status_counts.items()))
    index["review_status_counts"] = dict(sorted(review_status_counts.items()))
    return index


def rebuild_content_index(entries: List[Dict]) -> Dict:
    ordered = sorted(entries, key=lambda r: (r.get("json_path", ""), r.get("book_id", "")))
    return {
        "entries": ordered,
        "total_files": len(ordered),
    }


def write_report(entry: Dict) -> None:
    append_jsonl(REPORT_PATH, entry)


def archive_duplicate_file(record: Dict, archive_root: Path) -> str:
    json_path = record.get("json_path", "")
    if not json_path:
        return ""
    source_path = resolve_active_json_path(json_path)
    if not source_path.exists():
        return ""
    dest = archive_root / json_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()
    shutil.move(str(source_path), str(dest))
    return str(dest.relative_to(Path(JSON_DIR))).replace("\\", "/")


def delete_qdrant_book(client, book_id: str, state: Dict[str, Dict]) -> int:
    if book_id in state.get("books", {}):
        return release_book_from_state(state, client, book_id)

    from qdrant_client.http.models import FieldCondition, Filter, MatchValue

    q_filter = Filter(
        must=[FieldCondition(key="book_id", match=MatchValue(value=book_id))]
    )
    client.delete(collection_name=COLLECTION_NAME, points_selector=q_filter, wait=True)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Canonicalize Islamhouse JSON records by family")
    parser.add_argument("--dry-run", action="store_true", help="show planned changes without writing anything")
    parser.add_argument(
        "--archive-duplicates",
        action="store_true",
        default=True,
        help="move duplicate JSON files out of the active json_output tree",
    )
    parser.add_argument(
        "--no-archive-duplicates",
        dest="archive_duplicates",
        action="store_false",
        help="keep duplicate JSON files in place",
    )
    parser.add_argument(
        "--prune-qdrant",
        action="store_true",
        help="remove duplicate books from Qdrant and ingest state",
    )
    parser.add_argument(
        "--no-prune-qdrant",
        dest="prune_qdrant",
        action="store_false",
        help="keep Qdrant untouched",
    )
    parser.set_defaults(prune_qdrant=False)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    state = load_json(STATE_PATH, {})

    index = load_index_records()
    content_index = load_content_index()
    records = list(index.get("files", []))
    families = group_families(records)

    canonical_records: List[Dict] = []
    removed_records: List[Dict] = []
    family_summaries: List[Dict] = []
    kept_json_paths = set()

    for fam_key, fam_records in sorted(families.items(), key=lambda item: item[0]):
        canonical = choose_canonical(fam_records)
        canonical = dict(canonical)
        canonical["family_key"] = fam_key
        canonical_records.append(canonical)
        kept_json_paths.add(canonical.get("json_path", ""))

        fam_records_sorted = sorted(fam_records, key=score_record, reverse=True)
        family_summaries.append(
            {
                "family_key": fam_key,
                "canonical_json_path": canonical.get("json_path", ""),
                "canonical_book_id": canonical.get("book_id", ""),
                "members": len(fam_records),
                "canonical_score": list(score_record(canonical)),
                "members_json_paths": [r.get("json_path", "") for r in fam_records_sorted],
            }
        )
        for record in fam_records_sorted[1:]:
            removed_records.append(
                {
                    **record,
                    "family_key": fam_key,
                    "canonical_json_path": canonical.get("json_path", ""),
                    "canonical_book_id": canonical.get("book_id", ""),
                }
            )

    active_entries = [entry for entry in content_index.get("entries", []) if entry.get("json_path") in kept_json_paths]
    new_index = rebuild_index(canonical_records)
    new_content_index = rebuild_content_index(active_entries)

    planned_state = {
        "status": "planned",
        "created_at": state.get("created_at") or utc_now(),
        "updated_at": utc_now(),
        "index_total_files": index.get("total_files", len(records)),
        "canonical_total_files": len(canonical_records),
        "family_count": len(families),
        "removed_count": len(removed_records),
        "removed_books": [r.get("book_id", "") for r in removed_records],
        "removed_json_paths": [r.get("json_path", "") for r in removed_records],
        "archive_root": str(ARCHIVE_ROOT),
        "qdrant_pruned": False,
    }
    save_json(STATE_PATH, planned_state)

    if args.dry_run:
        log(f"Families        : {len(families)}")
        log(f"Canonical files : {len(canonical_records)}")
        log(f"Removed records : {len(removed_records)}")
        for fam in family_summaries[:20]:
            log(f"- {fam['family_key']} -> {fam['canonical_json_path']} ({fam['members']} variants)")
        return 0

    save_json(INDEX_PATH, new_index)
    save_json(CONTENT_INDEX_PATH, new_content_index)
    log(f"Rewrote index to {len(canonical_records)} canonical records")

    for record in removed_records:
        archive_path = ""
        if args.archive_duplicates:
            archive_path = archive_duplicate_file(record, ARCHIVE_ROOT)
        write_report(
            {
                "kind": "canonical_duplicate",
                "family_key": record.get("family_key", ""),
                "removed_book_id": record.get("book_id", "") or os.path.splitext(record.get("filename", ""))[0],
                "removed_json_path": record.get("json_path", ""),
                "canonical_book_id": record.get("canonical_book_id", ""),
                "canonical_json_path": record.get("canonical_json_path", ""),
                "archived_to": archive_path,
                "source_path": record.get("source_path", ""),
                "source_relpath": record.get("source_relpath", ""),
                "source_type": record.get("source_type", "unknown"),
                "document_type": record.get("document_type", "book"),
                "conversion_status": record.get("conversion_status", "unknown"),
                "quality_status": record.get("quality_status", "ok"),
                "review_status": record.get("review_status", "approved_auto"),
                "ingest_ready": bool(record.get("ingest_ready", True)),
                "score": list(score_record(record)),
            }
        )

    current_state = load_state(INGEST_STATE_PATH)
    qdrant_deleted = 0
    if args.prune_qdrant and removed_records:
        from qdrant_client import QdrantClient  # noqa: WPS433,E402
        from qdrant_client.http.models import FieldCondition, Filter, MatchValue  # noqa: WPS433,E402

        client = QdrantClient(path=QDRANT_PATH)
        deleted_books = 0
        for record in removed_records:
            deleted_books += 1
            qdrant_deleted += delete_qdrant_book(client, record.get("book_id", ""), current_state)
        save_state(current_state, INGEST_STATE_PATH)
        log(f"Pruned {deleted_books} duplicate book ids from Qdrant/state")
    elif args.prune_qdrant:
        log("No duplicate books to prune from Qdrant/state")

    if os.path.exists(LEXICAL_INDEX_PATH):
        os.remove(LEXICAL_INDEX_PATH)
        log("Removed lexical cache so it can rebuild against the canonical index")

    final_state = {
        **planned_state,
        "status": "done",
        "updated_at": utc_now(),
        "finished_at": utc_now(),
        "qdrant_pruned": bool(args.prune_qdrant),
        "qdrant_deleted_points": qdrant_deleted,
    }
    save_json(STATE_PATH, final_state)

    log(f"Canonical families : {len(families)}")
    log(f"Kept records       : {len(canonical_records)}")
    log(f"Removed records    : {len(removed_records)}")
    log(f"Report             : {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
