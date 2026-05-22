#!/usr/bin/env python3
"""Import the Islamhouse corpus into json_output with content dedupe.

This script scans a source directory, extracts text from supported formats,
checks whether the extracted content already exists in the current JSON corpus,
and writes only unique books into DATABASE_DIR/json_output.

Supported inputs:
- .txt
- .html / .htm
- .docx
- .doc (via LibreOffice HTML fallback)
- .pdf
- .epub / .ibooks (ZIP-based XHTML extraction)

Skipped inputs are reported to a JSONL duplicates file so the caller can audit
what was already present in the corpus.
"""

from __future__ import annotations

import argparse
import hashlib
import html as html_lib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from bs4 import BeautifulSoup
from docx import Document
from pdfminer.high_level import extract_text as pdf_extract_text

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = Path("/media/harry/DATA250/Islamhouse")
DATABASE_DIR = Path(
    os.getenv(
        "DATABASE_DIR",
        str(Path(__file__).resolve().parent.parent.parent / "DATABASE"),
    )
)
OUTPUT_DIR = DATABASE_DIR / "json_output"
CONTENT_INDEX_PATH = OUTPUT_DIR / "_content_index.json"
DUPLICATE_REPORT_PATH = OUTPUT_DIR / "_duplicates.jsonl"
INDEX_PATH = OUTPUT_DIR / "_index.json"

SUPPORTED_EXTS = {
    ".txt",
    ".htm",
    ".html",
    ".docx",
    ".doc",
    ".pdf",
    ".epub",
    ".ibooks",
}

ARABIC_DIACRITICS_RE = re.compile(r"[\u064b-\u065f\u0670\u06d6-\u06ed]")
LANG_PREFIX_RE = re.compile(r"^([a-z]{2})(?:[_-]|$)", re.IGNORECASE)
PAGE_BREAK_RE = re.compile(r"\f+")


@dataclass
class BookSignature:
    page_hashes: List[str]
    content_hash: str
    text_hash: str
    text_simhash: int
    page_count: int
    size_bytes: int


def log(message: str) -> None:
    print(message, flush=True)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def simhash(tokens: Iterable[str]) -> int:
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


def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


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


def detect_language(filename: str) -> str:
    match = LANG_PREFIX_RE.match(filename)
    if match:
        return match.group(1).lower()
    return "unknown"


def clean_title(filename: str) -> str:
    name = filename
    match = LANG_PREFIX_RE.match(name)
    if match:
        name = name[match.end() :]
    name = name.replace("_", " ").replace("-", " ")
    name = re.sub(r"\s+", " ", name).strip()
    words = []
    for word in name.split():
        if word.upper() == word and len(word) > 1:
            words.append(word)
        elif word.lower() in {"and", "the", "of", "in", "to", "for", "a", "an", "or", "by", "is", "on", "its"}:
            words.append(word.lower() if words else word.capitalize())
        elif word.startswith("al-") or word.startswith("Al-"):
            words.append(word[:1].upper() + word[1:])
        else:
            words.append(word.capitalize())
    if words:
        words[0] = words[0].capitalize()
    return " ".join(words) if words else filename


def make_book_id(source_label: str, relpath: str) -> str:
    normalized = relpath.replace("/", "__").replace("\\", "__")
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return f"{source_label}__{normalized}"


def normalize_text(text: str, language: str) -> str:
    text = html_lib.unescape(text or "")
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\x0c", " ").replace("\xa0", " ")
    if language == "ar" or re.search(r"[\u0600-\u06ff]", text):
        text = ARABIC_DIACRITICS_RE.sub("", text)
    text = text.lower()
    text = re.sub(r"[^\w\s\u0600-\u06ff]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_pages_from_text(text: str) -> List[str]:
    if not text or not text.strip():
        return []
    pages = PAGE_BREAK_RE.split(text)
    pages = [p.strip() for p in pages if p and p.strip()]
    return pages


def read_text_file(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        content = f.read()
    pages = split_pages_from_text(content)
    return pages or [content.strip()]


def html_to_pages(html_text: str) -> List[str]:
    soup = BeautifulSoup(html_text, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text("\n")
    pages = split_pages_from_text(text)
    return pages or [text.strip()]


def read_html_file(path: Path) -> List[str]:
    return html_to_pages(path.read_text(encoding="utf-8", errors="replace"))


def read_docx_file(path: Path) -> List[str]:
    doc = Document(str(path))
    pages: List[str] = []
    current: List[str] = []

    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            current.append(text)
        para_xml = para._p.xml
        if ("w:type=\"page\"" in para_xml or "w:lastRenderedPageBreak" in para_xml) and current:
            pages.append("\n".join(current).strip())
            current = []

    if current:
        pages.append("\n".join(current).strip())

    if not pages:
        all_text = "\n".join(p.text.strip() for p in doc.paragraphs if p.text.strip())
        return split_pages_from_text(all_text) or [all_text.strip()]

    return pages


def read_pdf_file(path: Path) -> List[str]:
    text = pdf_extract_text(str(path))
    pages = split_pages_from_text(text)
    return pages or [text.strip()]


def read_zip_xhtml_file(path: Path) -> List[str]:
    pages: List[str] = []
    with zipfile.ZipFile(path) as zf:
        members = sorted(
            name for name in zf.namelist()
            if name.lower().endswith((".xhtml", ".html", ".htm"))
        )
        for name in members:
            with zf.open(name) as handle:
                raw = handle.read().decode("utf-8", errors="replace")
            page_text = html_to_pages(raw)
            pages.extend(page_text or [""])
    return [p for p in pages if p.strip()]


def read_doc_with_libreoffice(path: Path) -> List[str]:
    if shutil.which("libreoffice") is None:
        raise RuntimeError("libreoffice not available for .doc fallback")

    with tempfile.TemporaryDirectory(prefix="islamhouse-lo-") as tmp:
        tmpdir = Path(tmp)
        result = subprocess.run(
            [
                "libreoffice",
                "--headless",
                "--convert-to",
                "html",
                "--outdir",
                str(tmpdir),
                str(path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        html_files = sorted(tmpdir.glob("*.html"))
        if result.returncode != 0 or not html_files:
            raise RuntimeError(
                f"LibreOffice conversion failed for {path.name}: {result.stderr.strip() or result.stdout.strip()}"
            )
        html_text = html_files[0].read_text(encoding="utf-8", errors="replace")
        pages = html_to_pages(html_text)
        return pages or [html_text.strip()]


def extract_pages(path: Path) -> Tuple[List[str], str]:
    ext = path.suffix.lower()
    if ext == ".txt":
        return read_text_file(path), "txt"
    if ext in {".htm", ".html"}:
        return read_html_file(path), "html"
    if ext == ".docx":
        return read_docx_file(path), "docx"
    if ext == ".pdf":
        return read_pdf_file(path), "pdf"
    if ext in {".epub", ".ibooks"}:
        return read_zip_xhtml_file(path), "zip-xhtml"
    if ext == ".doc":
        return read_doc_with_libreoffice(path), "libreoffice-html"
    raise RuntimeError(f"Unsupported extension: {ext}")


def build_signature(pages: List[str], language: str, size_bytes: int) -> BookSignature:
    page_hashes: List[str] = []
    full_normalized_parts: List[str] = []
    for page in pages:
        normalized = normalize_text(page, language)
        if not normalized:
            continue
        page_hashes.append(sha256_text(normalized))
        full_normalized_parts.append(normalized)
    text = " ".join(full_normalized_parts)
    content_hash = sha256_text("\n".join(page_hashes))
    text_hash = sha256_text(text)
    text_simhash = simhash(text.split())
    return BookSignature(
        page_hashes=page_hashes,
        content_hash=content_hash,
        text_hash=text_hash,
        text_simhash=text_simhash,
        page_count=len(page_hashes),
        size_bytes=size_bytes,
    )


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


def build_content_catalog(index_records: List[Dict], output_dir: Path) -> List[Dict]:
    catalog: List[Dict] = []
    for record in index_records:
        json_path = resolve_index_json_path(record, str(output_dir))
        path = Path(json_path)
        if not path.exists():
            continue
        try:
            with path.open("r", encoding="utf-8") as f:
                book = json.load(f)
            pages = [p.get("content", "") for p in book.get("pages", [])]
            sig = build_signature(pages, book.get("language", "unknown"), int(book.get("size_bytes", 0) or 0))
            catalog.append({
                "book_id": record.get("book_id") or book.get("book_id") or os.path.splitext(record["filename"])[0],
                "filename": record["filename"],
                "json_path": record["json_path"],
                "language": record.get("language", book.get("language", "unknown")),
                "title": record.get("title", book.get("title", "")),
                "total_pages": int(record.get("total_pages", len(book.get("pages", []))) or 0),
                "size_bytes": int(record.get("size_bytes", book.get("size_bytes", 0)) or 0),
                "content_hash": sig.content_hash,
                "text_hash": sig.text_hash,
                "text_simhash": sig.text_simhash,
                "page_hashes": sig.page_hashes,
                "source_path": record.get("source_path", ""),
                "source_hash": record.get("source_hash", ""),
            })
        except Exception as e:
            log(f"  WARN  content catalog skip {record.get('filename')}: {e}")
    return catalog


def load_catalog(output_dir: Path) -> Tuple[Dict, List[Dict]]:
    index = load_json(INDEX_PATH, {"total_files": 0, "languages": {}, "files": []})
    entries = build_content_catalog(index.get("files", []), output_dir)
    return index, entries


def page_overlap_score(a: List[str], b: List[str]) -> float:
    if not a or not b:
        return 0.0
    sa = set(a)
    sb = set(b)
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def find_duplicate(candidate: Dict, catalog: List[Dict], threshold: float) -> Tuple[Optional[Dict], float, str]:
    if not candidate["page_hashes"]:
        return None, 0.0, "empty"
    candidate_set = set(candidate["page_hashes"])
    best: Optional[Dict] = None
    best_score = 0.0
    best_reason = "none"
    candidate_title_tokens = set(normalize_text(candidate.get("title", ""), candidate["language"]).split())

    for entry in catalog:
        if candidate["language"] != "unknown" and entry.get("language") not in {"unknown", candidate["language"]}:
            continue
        if candidate["page_count"] and entry.get("total_pages"):
            ratio = min(candidate["page_count"], int(entry["total_pages"])) / max(candidate["page_count"], int(entry["total_pages"]))
            if ratio < 0.45:
                continue
        existing_set = set(entry.get("page_hashes", []))
        if not existing_set:
            continue
        if candidate["content_hash"] == entry.get("content_hash"):
            return entry, 1.0, "exact"
        if candidate.get("text_hash") and candidate.get("text_hash") == entry.get("text_hash"):
            return entry, 1.0, "text_exact"

        simhash_distance = None
        if candidate.get("text_simhash") is not None and entry.get("text_simhash") is not None:
            simhash_distance = hamming_distance(int(candidate["text_simhash"]), int(entry["text_simhash"]))

        title_tokens = set(normalize_text(entry.get("title", ""), entry.get("language", "unknown")).split())
        title_score = 0.0
        if candidate_title_tokens or title_tokens:
            title_score = len(candidate_title_tokens & title_tokens) / max(len(candidate_title_tokens | title_tokens), 1)

        page_score = page_overlap_score(list(candidate_set), list(existing_set))
        score = page_score
        if simhash_distance is not None:
            score = max(score, 1.0 - (simhash_distance / 64.0))
        score = max(score, title_score)
        if score > best_score:
            best = entry
            best_score = score
            best_reason = "page_overlap" if score == page_score else "text_similarity"

    if best and best_score >= threshold:
        return best, best_score, best_reason
    return None, best_score, best_reason


def scan_files(input_dir: Path, recursive: bool) -> List[Path]:
    if recursive:
        paths = [p for p in input_dir.rglob("*") if p.is_file()]
    else:
        paths = [p for p in input_dir.iterdir() if p.is_file()]
    return sorted(
        p for p in paths
        if p.suffix.lower() in SUPPORTED_EXTS
        and not p.name.startswith(".~lock.")
    )


def make_output_path(source_label: str, input_dir: Path, path: Path) -> Tuple[Path, str]:
    relpath = path.relative_to(input_dir).as_posix()
    rel_json = Path(source_label) / f"{relpath}.json"
    outpath = OUTPUT_DIR / rel_json
    return outpath, relpath


def process_file(
    path: Path,
    input_dir: Path,
    source_label: str,
    catalog: List[Dict],
    threshold: float,
    keep_duplicates: bool,
) -> Tuple[Optional[Dict], Optional[Dict], Optional[Dict], str]:
    pages, extractor = extract_pages(path)
    if not pages:
        return None, None, None, extractor

    language = detect_language(path.name)
    signature = build_signature(pages, language, path.stat().st_size)
    outpath, relpath = make_output_path(source_label, input_dir, path)
    book_id = make_book_id(source_label, relpath)
    duplicate_of, score, reason = find_duplicate(
        {
            "page_hashes": signature.page_hashes,
            "content_hash": signature.content_hash,
            "text_hash": signature.text_hash,
            "text_simhash": signature.text_simhash,
            "page_count": signature.page_count,
            "language": language,
            "title": clean_title(path.stem),
        },
        catalog,
        threshold,
    )

    record = {
        "book_id": book_id,
        "filename": path.name,
        "language": language,
        "title": clean_title(path.stem),
        "size_bytes": path.stat().st_size,
        "total_pages": len(pages),
        "json_path": str(outpath.relative_to(OUTPUT_DIR)),
        "source_root": str(input_dir),
        "source_path": str(path),
        "source_relpath": relpath,
        "source_hash": sha256_file(path),
        "content_hash": signature.content_hash,
        "text_hash": signature.text_hash,
        "text_simhash": signature.text_simhash,
        "extractor": extractor,
    }

    if duplicate_of and not keep_duplicates:
        return record, duplicate_of, {"score": score, "reason": reason}, extractor

    book_json = {
        **record,
        "pages": [{"page": i + 1, "content": page} for i, page in enumerate(pages)],
    }
    return book_json, duplicate_of, {"score": score, "reason": reason}, extractor


def update_index(index: Dict, record: Dict) -> None:
    files = index.setdefault("files", [])
    files = [r for r in files if r.get("json_path") != record["json_path"]]
    files.append(record)
    files.sort(key=lambda r: (r.get("language", ""), r.get("json_path", "")))
    index["files"] = files
    index["total_files"] = len(files)
    languages: Dict[str, int] = {}
    for r in files:
        lang = r.get("language", "unknown")
        languages[lang] = languages.get(lang, 0) + 1
    index["languages"] = languages


def update_content_index(entries: List[Dict], record: Dict, signature: BookSignature) -> List[Dict]:
    entries = [e for e in entries if e.get("json_path") != record["json_path"]]
    entries.append({
        "book_id": record["book_id"],
        "filename": record["filename"],
        "language": record["language"],
        "title": record["title"],
        "json_path": record["json_path"],
        "source_path": record["source_path"],
        "source_relpath": record["source_relpath"],
        "source_hash": record["source_hash"],
        "content_hash": signature.content_hash,
        "text_hash": signature.text_hash,
        "text_simhash": signature.text_simhash,
        "page_count": signature.page_count,
        "size_bytes": signature.size_bytes,
        "page_hashes": signature.page_hashes,
    })
    entries.sort(key=lambda r: r.get("json_path", ""))
    return entries


def write_duplicate_report(path: Path, entry: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Import Islamhouse sources into json_output")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR), help="source directory to scan")
    parser.add_argument("--source-label", default="islamhouse", help="output subdirectory and book_id prefix")
    parser.add_argument("--recursive", action="store_true", help="scan source directory recursively")
    parser.add_argument("--keep-duplicates", action="store_true", help="write duplicate books instead of skipping them")
    parser.add_argument("--duplicate-threshold", type=float, default=0.92, help="page overlap threshold for duplicate detection")
    parser.add_argument("--limit", type=int, default=0, help="process at most N files for testing")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.exists():
        raise SystemExit(f"Input directory not found: {input_dir}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.joinpath(args.source_label).mkdir(parents=True, exist_ok=True)

    index, catalog = load_catalog(OUTPUT_DIR)
    files = scan_files(input_dir, args.recursive)
    if args.limit and args.limit > 0:
        files = files[: args.limit]

    log(f"Source dir       : {input_dir}")
    log(f"Output dir       : {OUTPUT_DIR}")
    log(f"Source label     : {args.source_label}")
    log(f"Files discovered : {len(files)}")
    log(f"Existing books   : {len(index.get('files', []))}")
    log(f"Catalog entries  : {len(catalog)}")

    processed = 0
    imported = 0
    skipped_duplicates = 0
    unsupported = 0
    duplicate_hits = 0

    for i, path in enumerate(files, 1):
        processed += 1
        try:
            book_json, duplicate_of, dup_info, extractor = process_file(
                path=path,
                input_dir=input_dir,
                source_label=args.source_label,
                catalog=catalog,
                threshold=args.duplicate_threshold,
                keep_duplicates=args.keep_duplicates,
            )
        except Exception as e:
            unsupported += 1
            log(f"[{i:04d}] ERROR  {path.name}: {e}")
            continue

        if book_json is None:
            unsupported += 1
            log(f"[{i:04d}] SKIP   {path.name} (empty after extraction)")
            continue

        if duplicate_of and not args.keep_duplicates:
            skipped_duplicates += 1
            duplicate_hits += 1
            report_entry = {
                "source_path": book_json["source_path"],
                "source_relpath": book_json["source_relpath"],
                "duplicate_of": duplicate_of.get("json_path"),
                "duplicate_title": duplicate_of.get("title"),
                "score": dup_info["score"] if dup_info else 0.0,
                "reason": dup_info["reason"] if dup_info else "duplicate",
                "content_hash": book_json["content_hash"],
                "language": book_json["language"],
                "title": book_json["title"],
                "extractor": extractor,
            }
            write_duplicate_report(DUPLICATE_REPORT_PATH, report_entry)
            log(
                f"[{i:04d}] DUP    {path.name} -> {duplicate_of.get('json_path')} "
                f"score={report_entry['score']:.3f}"
            )
            continue

        outpath = OUTPUT_DIR / book_json["json_path"]
        outpath.parent.mkdir(parents=True, exist_ok=True)
        with outpath.open("w", encoding="utf-8") as f:
            json.dump(book_json, f, ensure_ascii=False, indent=2)

        index_record = {
            "book_id": book_json["book_id"],
            "filename": book_json["filename"],
            "language": book_json["language"],
            "title": book_json["title"],
            "total_pages": book_json["total_pages"],
            "size_bytes": book_json["size_bytes"],
            "json_path": book_json["json_path"],
            "source_root": book_json["source_root"],
            "source_path": book_json["source_path"],
            "source_relpath": book_json["source_relpath"],
            "source_hash": book_json["source_hash"],
            "content_hash": book_json["content_hash"],
        }

        update_index(index, index_record)
        signature = build_signature([p["content"] for p in book_json["pages"]], book_json["language"], book_json["size_bytes"])
        catalog = update_content_index(catalog, index_record, signature)
        imported += 1

        save_json(INDEX_PATH, index)
        save_json(CONTENT_INDEX_PATH, {"total_files": len(catalog), "entries": catalog})

        log(
            f"[{i:04d}] OK     {path.name} -> {book_json['json_path']} "
            f"pages={book_json['total_pages']} extractor={extractor}"
        )

    save_json(INDEX_PATH, index)
    save_json(CONTENT_INDEX_PATH, {"total_files": len(catalog), "entries": catalog})

    log("")
    log("=== DONE ===")
    log(f"Processed          : {processed}")
    log(f"Imported unique    : {imported}")
    log(f"Skipped duplicates  : {skipped_duplicates}")
    log(f"Unsupported/empty   : {unsupported}")
    log(f"Duplicate matches   : {duplicate_hits}")
    log(f"Index path          : {INDEX_PATH}")
    log(f"Content index path  : {CONTENT_INDEX_PATH}")
    log(f"Duplicate report    : {DUPLICATE_REPORT_PATH}")


if __name__ == "__main__":
    main()
