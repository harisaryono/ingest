from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

from config import METADATA_DB_PATH


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS books (
  book_id TEXT PRIMARY KEY,
  json_path TEXT NOT NULL,
  filename TEXT,
  source_root TEXT,
  source_path TEXT,
  source_relpath TEXT,
  source_ext TEXT,
  source_type TEXT,
  document_type TEXT,
  language TEXT,
  title TEXT,
  size_bytes INTEGER DEFAULT 0,
  total_pages INTEGER DEFAULT 0,
  source_hash TEXT,
  content_hash TEXT,
  text_hash TEXT,
  text_simhash INTEGER DEFAULT 0,
  import_stage INTEGER DEFAULT 1,
  pdf_pipeline TEXT,
  pdf_ocr_strategy TEXT,
  extractor TEXT,
  conversion_status TEXT,
  quality_status TEXT,
  quality_reasons_json TEXT,
  quality_warnings_json TEXT,
  quality_metrics_json TEXT,
  quality_expected_language TEXT,
  quality_expected_arabic INTEGER DEFAULT 0,
  quality_detected_script TEXT,
  review_status TEXT,
  review_required INTEGER DEFAULT 0,
  review_route TEXT,
  reviewed_by TEXT,
  reviewed_at TEXT,
  review_note TEXT,
  ingest_ready INTEGER DEFAULT 0,
  page_quality_summary_json TEXT,
  page_ocr_candidates_json TEXT,
  updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_books_review_status ON books(review_status);
CREATE INDEX IF NOT EXISTS idx_books_quality_status ON books(quality_status);
CREATE INDEX IF NOT EXISTS idx_books_conversion_status ON books(conversion_status);
CREATE INDEX IF NOT EXISTS idx_books_ingest_ready ON books(ingest_ready);
CREATE INDEX IF NOT EXISTS idx_books_language ON books(language);
CREATE INDEX IF NOT EXISTS idx_books_source_type ON books(source_type);
CREATE INDEX IF NOT EXISTS idx_books_document_type ON books(document_type);
CREATE INDEX IF NOT EXISTS idx_books_source_hash ON books(source_hash);
CREATE INDEX IF NOT EXISTS idx_books_content_hash ON books(content_hash);

CREATE TABLE IF NOT EXISTS pages (
  book_id TEXT NOT NULL,
  page_num INTEGER NOT NULL,
  content_len INTEGER DEFAULT 0,
  page_hash TEXT,
  page_text_hash TEXT,
  page_simhash INTEGER DEFAULT 0,
  import_stage INTEGER DEFAULT 1,
  first_pass_done INTEGER DEFAULT 1,
  page_quality_status TEXT,
  page_quality_score REAL DEFAULT 0,
  page_quality_reasons_json TEXT,
  page_quality_warnings_json TEXT,
  page_ocr_needed INTEGER DEFAULT 0,
  page_review_status TEXT,
  page_reviewed_by TEXT,
  page_reviewed_at TEXT,
  page_review_note TEXT,
  ocr_attempted INTEGER DEFAULT 0,
  ocr_done INTEGER DEFAULT 0,
  ocr_engine TEXT,
  ocr_status TEXT,
  updated_at TEXT NOT NULL,
  PRIMARY KEY (book_id, page_num),
  FOREIGN KEY (book_id) REFERENCES books(book_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_pages_review_status ON pages(page_review_status);
CREATE INDEX IF NOT EXISTS idx_pages_quality_status ON pages(page_quality_status);
CREATE INDEX IF NOT EXISTS idx_pages_ocr_needed ON pages(page_ocr_needed);
CREATE INDEX IF NOT EXISTS idx_pages_first_pass ON pages(first_pass_done);
CREATE INDEX IF NOT EXISTS idx_pages_book_id ON pages(book_id);

CREATE TABLE IF NOT EXISTS book_families (
  family_key TEXT PRIMARY KEY,
  canonical_book_id TEXT,
  canonical_json_path TEXT,
  source_hash TEXT,
  content_hash TEXT,
  text_hash TEXT,
  text_simhash INTEGER DEFAULT 0,
  member_count INTEGER DEFAULT 0,
  reviewed_member_count INTEGER DEFAULT 0,
  updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_book_families_canonical_book_id ON book_families(canonical_book_id);

CREATE TABLE IF NOT EXISTS review_actions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  book_id TEXT NOT NULL,
  page_num INTEGER,
  scope TEXT NOT NULL,
  action TEXT NOT NULL,
  reviewed_by TEXT,
  note TEXT,
  payload_json TEXT,
  created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_review_actions_book_id ON review_actions(book_id);
CREATE INDEX IF NOT EXISTS idx_review_actions_page_num ON review_actions(page_num);
CREATE INDEX IF NOT EXISTS idx_review_actions_scope ON review_actions(scope);
CREATE INDEX IF NOT EXISTS idx_review_actions_created_at ON review_actions(created_at);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _int(value, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _bool(value) -> int:
    return 1 if bool(value) else 0


def _sqlite_int64(value) -> int:
    value = _int(value, 0)
    if value >= 1 << 63:
        value -= 1 << 64
    if value < -(1 << 63):
        value = -(1 << 63)
    return value


def _coalesce(*values, default=""):
    for value in values:
        if value is None:
            continue
        if isinstance(value, str):
            if value.strip():
                return value
            continue
        return value
    return default


@contextmanager
def connect(db_path: str | None = None) -> Iterator[sqlite3.Connection]:
    path = Path(db_path or METADATA_DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def _page_metrics(page: Dict) -> Dict[str, object]:
    content = str(page.get("content", "") or "")
    content_norm = " ".join(content.replace("\x0c", " ").replace("\xa0", " ").split())
    page_hash = _sha256_text(content_norm)
    page_text_hash = _sha256_text(content.strip())
    page_simhash = _sqlite_int64(_simhash(content_norm.split()))
    return {
        "content_len": len(content),
        "page_hash": page_hash,
        "page_text_hash": page_text_hash,
        "page_simhash": page_simhash,
    }


def _sha256_text(text: str) -> str:
    import hashlib

    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def _simhash(tokens: Iterable[str]) -> int:
    import hashlib

    counts: Dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    if not counts:
        return 0
    vector = [0] * 64
    for token, weight in counts.items():
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        fingerprint = int.from_bytes(digest, "big")
        for bit in range(64):
            if fingerprint & (1 << bit):
                vector[bit] += weight
            else:
                vector[bit] -= weight
    value = 0
    for bit, score in enumerate(vector):
        if score > 0:
            value |= 1 << bit
    return value


def upsert_book(conn: sqlite3.Connection, book: Dict, record: Dict | None = None) -> None:
    record = record or {}
    now = _now()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO books (
          book_id, json_path, filename, source_root, source_path, source_relpath, source_ext,
          source_type, document_type, language, title, size_bytes, total_pages, source_hash,
          content_hash, text_hash, text_simhash, import_stage, pdf_pipeline, pdf_ocr_strategy,
          extractor, conversion_status, quality_status, quality_reasons_json, quality_warnings_json,
          quality_metrics_json, quality_expected_language, quality_expected_arabic, quality_detected_script,
          review_status, review_required, review_route, reviewed_by, reviewed_at, review_note,
          ingest_ready, page_quality_summary_json, page_ocr_candidates_json, updated_at
        ) VALUES (
          :book_id, :json_path, :filename, :source_root, :source_path, :source_relpath, :source_ext,
          :source_type, :document_type, :language, :title, :size_bytes, :total_pages, :source_hash,
          :content_hash, :text_hash, :text_simhash, :import_stage, :pdf_pipeline, :pdf_ocr_strategy,
          :extractor, :conversion_status, :quality_status, :quality_reasons_json, :quality_warnings_json,
          :quality_metrics_json, :quality_expected_language, :quality_expected_arabic, :quality_detected_script,
          :review_status, :review_required, :review_route, :reviewed_by, :reviewed_at, :review_note,
          :ingest_ready, :page_quality_summary_json, :page_ocr_candidates_json, :updated_at
        )
        ON CONFLICT(book_id) DO UPDATE SET
          json_path=excluded.json_path,
          filename=excluded.filename,
          source_root=excluded.source_root,
          source_path=excluded.source_path,
          source_relpath=excluded.source_relpath,
          source_ext=excluded.source_ext,
          source_type=excluded.source_type,
          document_type=excluded.document_type,
          language=excluded.language,
          title=excluded.title,
          size_bytes=excluded.size_bytes,
          total_pages=excluded.total_pages,
          source_hash=excluded.source_hash,
          content_hash=excluded.content_hash,
          text_hash=excluded.text_hash,
          text_simhash=excluded.text_simhash,
          import_stage=excluded.import_stage,
          pdf_pipeline=excluded.pdf_pipeline,
          pdf_ocr_strategy=excluded.pdf_ocr_strategy,
          extractor=excluded.extractor,
          conversion_status=excluded.conversion_status,
          quality_status=excluded.quality_status,
          quality_reasons_json=excluded.quality_reasons_json,
          quality_warnings_json=excluded.quality_warnings_json,
          quality_metrics_json=excluded.quality_metrics_json,
          quality_expected_language=excluded.quality_expected_language,
          quality_expected_arabic=excluded.quality_expected_arabic,
          quality_detected_script=excluded.quality_detected_script,
          review_status=excluded.review_status,
          review_required=excluded.review_required,
          review_route=excluded.review_route,
          reviewed_by=excluded.reviewed_by,
          reviewed_at=excluded.reviewed_at,
          review_note=excluded.review_note,
          ingest_ready=excluded.ingest_ready,
          page_quality_summary_json=excluded.page_quality_summary_json,
          page_ocr_candidates_json=excluded.page_ocr_candidates_json,
          updated_at=excluded.updated_at
        """,
        {
            "book_id": str(
                _coalesce(
                    book.get("book_id"),
                    record.get("book_id"),
                    os.path.splitext(str(record.get("filename", "")))[0],
                    default="",
                )
            ),
            "json_path": str(_coalesce(book.get("json_path"), record.get("json_path"), default="")),
            "filename": str(_coalesce(book.get("filename"), record.get("filename"), default="")),
            "source_root": str(_coalesce(book.get("source_root"), record.get("source_root"), default="")),
            "source_path": str(_coalesce(book.get("source_path"), record.get("source_path"), default="")),
            "source_relpath": str(_coalesce(book.get("source_relpath"), record.get("source_relpath"), default="")),
            "source_ext": str(_coalesce(book.get("source_ext"), record.get("source_ext"), default="")),
            "source_type": str(_coalesce(book.get("source_type"), record.get("source_type"), default="unknown")),
            "document_type": str(_coalesce(book.get("document_type"), record.get("document_type"), default="book")),
            "language": str(_coalesce(book.get("language"), record.get("language"), default="unknown")),
            "title": str(_coalesce(book.get("title"), record.get("title"), default="")),
            "size_bytes": _int(book.get("size_bytes", record.get("size_bytes", 0))),
            "total_pages": _int(book.get("total_pages", record.get("total_pages", 0))),
            "source_hash": str(_coalesce(book.get("source_hash"), record.get("source_hash"), default="")),
            "content_hash": str(_coalesce(book.get("content_hash"), record.get("content_hash"), default="")),
            "text_hash": str(_coalesce(book.get("text_hash"), record.get("text_hash"), default="")),
            "text_simhash": _sqlite_int64(_coalesce(book.get("text_simhash"), record.get("text_simhash"), default=0)),
            "import_stage": _int(_coalesce(book.get("import_stage"), record.get("import_stage"), default=1), 1),
            "pdf_pipeline": str(_coalesce(book.get("pdf_pipeline"), record.get("pdf_pipeline"), default="")),
            "pdf_ocr_strategy": str(_coalesce(book.get("pdf_ocr_strategy"), record.get("pdf_ocr_strategy"), default="")),
            "extractor": str(_coalesce(book.get("extractor"), record.get("extractor"), default="")),
            "conversion_status": str(_coalesce(book.get("conversion_status"), record.get("conversion_status"), default="unknown")),
            "quality_status": str(_coalesce(book.get("quality_status"), record.get("quality_status"), default="ok")),
            "quality_reasons_json": _json(_coalesce(book.get("quality_reasons"), record.get("quality_reasons"), default=[])),
            "quality_warnings_json": _json(_coalesce(book.get("quality_warnings"), record.get("quality_warnings"), default=[])),
            "quality_metrics_json": _json(_coalesce(book.get("quality_metrics"), record.get("quality_metrics"), default={})),
            "quality_expected_language": str(_coalesce(book.get("quality_expected_language"), record.get("quality_expected_language"), default="")),
            "quality_expected_arabic": _bool(_coalesce(book.get("quality_expected_arabic"), record.get("quality_expected_arabic"), default=False)),
            "quality_detected_script": str(_coalesce(book.get("quality_detected_script"), record.get("quality_detected_script"), default="")),
            "review_status": str(_coalesce(book.get("review_status"), record.get("review_status"), default="approved_auto")),
            "review_required": _bool(_coalesce(book.get("review_required"), record.get("review_required"), default=False)),
            "review_route": str(_coalesce(book.get("review_route"), record.get("review_route"), default="auto")),
            "reviewed_by": str(_coalesce(book.get("reviewed_by"), record.get("reviewed_by"), default="")),
            "reviewed_at": str(_coalesce(book.get("reviewed_at"), record.get("reviewed_at"), default="")),
            "review_note": str(_coalesce(book.get("review_note"), record.get("review_note"), default="")),
            "ingest_ready": _bool(_coalesce(book.get("ingest_ready"), record.get("ingest_ready"), default=True)),
            "page_quality_summary_json": _json(book.get("page_quality_summary", {})),
            "page_ocr_candidates_json": _json(book.get("page_ocr_candidates", [])),
            "updated_at": now,
        },
    )


def upsert_pages(conn: sqlite3.Connection, book: Dict) -> None:
    now = _now()
    book_id = str(book.get("book_id", "") or "")
    if not book_id:
        return
    rows = []
    for page in book.get("pages", []) or []:
        page_num = _int(page.get("page", 0))
        metrics = _page_metrics(page)
        rows.append(
            (
                book_id,
                page_num,
                metrics["content_len"],
                metrics["page_hash"],
                metrics["page_text_hash"],
                metrics["page_simhash"],
                _int(page.get("page_import_stage", book.get("import_stage", 1)), 1),
                1,
                str(page.get("page_quality_status", "ok") or "ok"),
                float(page.get("page_quality_score", 1.0) or 0.0),
                _json(page.get("page_quality_reasons", [])),
                _json(page.get("page_quality_warnings", [])),
                _bool(page.get("page_ocr_needed", False)),
                str(page.get("page_review_status", book.get("page_review_status", "")) or ""),
                str(page.get("page_reviewed_by", book.get("page_reviewed_by", "")) or ""),
                str(page.get("page_reviewed_at", book.get("page_reviewed_at", "")) or ""),
                str(page.get("page_review_note", book.get("page_review_note", "")) or ""),
                _bool(page.get("ocr_attempted", False)),
                _bool(page.get("ocr_done", False)),
                str(page.get("ocr_engine", "") or ""),
                str(page.get("ocr_status", "") or ""),
                now,
            )
        )

    if not rows:
        return

    conn.executemany(
        """
        INSERT INTO pages (
          book_id, page_num, content_len, page_hash, page_text_hash, page_simhash,
          import_stage, first_pass_done, page_quality_status, page_quality_score,
          page_quality_reasons_json, page_quality_warnings_json, page_ocr_needed,
          page_review_status, page_reviewed_by, page_reviewed_at, page_review_note,
          ocr_attempted, ocr_done, ocr_engine, ocr_status, updated_at
        ) VALUES (
          ?, ?, ?, ?, ?, ?,
          ?, ?, ?, ?,
          ?, ?, ?,
          ?, ?, ?, ?,
          ?, ?, ?, ?, ?
        )
        ON CONFLICT(book_id, page_num) DO UPDATE SET
          content_len=excluded.content_len,
          page_hash=excluded.page_hash,
          page_text_hash=excluded.page_text_hash,
          page_simhash=excluded.page_simhash,
          import_stage=excluded.import_stage,
          first_pass_done=excluded.first_pass_done,
          page_quality_status=excluded.page_quality_status,
          page_quality_score=excluded.page_quality_score,
          page_quality_reasons_json=excluded.page_quality_reasons_json,
          page_quality_warnings_json=excluded.page_quality_warnings_json,
          page_ocr_needed=excluded.page_ocr_needed,
          page_review_status=excluded.page_review_status,
          page_reviewed_by=excluded.page_reviewed_by,
          page_reviewed_at=excluded.page_reviewed_at,
          page_review_note=excluded.page_review_note,
          ocr_attempted=excluded.ocr_attempted,
          ocr_done=excluded.ocr_done,
          ocr_engine=excluded.ocr_engine,
          ocr_status=excluded.ocr_status,
          updated_at=excluded.updated_at
        """,
        rows,
    )


def upsert_book_and_pages(book: Dict, record: Dict | None = None, db_path: str | None = None) -> None:
    with connect(db_path) as conn:
        upsert_book(conn, book, record=record)
        upsert_pages(conn, book)


def record_review_action(
    book_id: str,
    action: str,
    *,
    scope: str = "book",
    page_num: int | None = None,
    reviewed_by: str = "",
    note: str = "",
    payload: Dict | None = None,
    db_path: str | None = None,
) -> None:
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO review_actions (
              book_id, page_num, scope, action, reviewed_by, note, payload_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(book_id),
                page_num,
                str(scope),
                str(action),
                str(reviewed_by),
                str(note),
                _json(payload or {}),
                _now(),
            ),
        )
