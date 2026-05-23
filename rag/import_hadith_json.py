#!/usr/bin/env python3
"""Convert AhmedBaset/hadith-json into local reference_data/hadith files.

The upstream dataset contains per-book JSON files with Arabic text and English
metadata. This importer normalizes those files into a local, offline-friendly
layout that can be used by marker replacement and review tooling.

Default output:
- DATABASE/reference_data/hadith/by_book/<collection>.json
- DATABASE/reference_data/hadith/index.json

Optional output:
- DATABASE/reference_data/hadith/by_chapter/<collection>/<chapter_id>.json
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from config import HADITH_REFERENCE_DIR


DEFAULT_SOURCE_DIR = Path(
    os.getenv("HADITH_JSON_SOURCE_DIR", "/tmp/hadith-json")
)


def log(message: str) -> None:
    print(message, flush=True)


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict JSON at {path}")
    return data


def discover_source_files(source_dir: Path) -> List[Path]:
    by_book_root = source_dir / "db" / "by_book"
    if not by_book_root.exists():
        raise SystemExit(f"Source by_book directory not found: {by_book_root}")
    return sorted(p for p in by_book_root.rglob("*.json") if p.is_file())


def slug_from_path(path: Path) -> str:
    return path.stem.strip().lower()


def normalize_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def format_english(entry: Dict) -> Tuple[str, str]:
    english = entry.get("english")
    narrator = ""
    text = ""
    if isinstance(english, dict):
        narrator = normalize_text(english.get("narrator"))
        text = normalize_text(english.get("text"))
    elif isinstance(english, str):
        text = normalize_text(english)
    translation = "\n\n".join([part for part in [narrator, text] if part]).strip()
    return narrator, translation


def build_hadith_record(
    *,
    collection: str,
    source_path: Path,
    source_root: Path,
    dataset: Dict,
    item: Dict,
    chapter_map: Dict[int, Dict],
) -> Dict:
    number = item.get("idInBook") or item.get("numberInBook") or item.get("number") or item.get("id")
    if number is None:
        raise ValueError("missing hadith number")
    chapter_id = item.get("chapterId") or 0
    book_id = item.get("bookId") or dataset.get("id") or 0
    arabic = normalize_text(item.get("arabic"))
    english_narrator, english_text = format_english(item)
    chapter_meta = chapter_map.get(int(chapter_id), {})

    translation_text = english_text or normalize_text(item.get("translation")) or normalize_text(item.get("text"))
    if english_narrator and translation_text and not translation_text.startswith(english_narrator):
        translation_full = "\n\n".join([part for part in [english_narrator, translation_text] if part]).strip()
    else:
        translation_full = translation_text

    key = f"{collection}:{int(number)}"
    return {
        "key": key,
        "collection": collection,
        "book_alias": collection,
        "book_id": int(book_id) if str(book_id).isdigit() else book_id,
        "number": int(number) if str(number).isdigit() else str(number),
        "id": item.get("id"),
        "chapter_id": int(chapter_id) if str(chapter_id).isdigit() else chapter_id,
        "chapter_title_ar": normalize_text(chapter_meta.get("arabic")),
        "chapter_title_en": normalize_text(chapter_meta.get("english")),
        "arabic": arabic,
        "translation_id": translation_full,
        "translation_en": translation_full,
        "english_narrator": english_narrator,
        "english_text": translation_text,
        "source_path": str(source_path),
        "source_relpath": str(source_path.relative_to(source_root)),
    }


def iter_hadith_items(node) -> Iterable[Dict]:
    if isinstance(node, dict):
        if {"id", "chapterId", "bookId"}.issubset(node.keys()) and "arabic" in node:
            yield node
        for value in node.values():
            yield from iter_hadith_items(value)
    elif isinstance(node, list):
        for value in node:
            yield from iter_hadith_items(value)


def convert_book_file(
    path: Path,
    output_dir: Path,
    source_root: Path,
    write_by_chapter: bool = False,
) -> Dict:
    dataset = load_json(path)
    collection = slug_from_path(path)
    chapters = dataset.get("chapters") or []
    chapter_map = {}
    for chapter in chapters if isinstance(chapters, list) else []:
        if not isinstance(chapter, dict):
            continue
        chapter_id = chapter.get("id")
        if chapter_id is None:
            continue
        chapter_map[int(chapter_id)] = chapter

    entries: Dict[str, Dict] = {}
    chapter_buckets: Dict[int, Dict[str, Dict]] = defaultdict(dict)
    for item in iter_hadith_items(dataset):
        try:
            record = build_hadith_record(
                collection=collection,
                source_path=path,
                source_root=source_root,
                dataset=dataset,
                item=item,
                chapter_map=chapter_map,
            )
        except Exception:
            continue
        entries[str(record["number"])] = record
        chapter_buckets[int(record["chapter_id"])][str(record["number"])] = record

    meta = {
        "collection": collection,
        "source_path": str(path),
        "source_relpath": str(path.relative_to(source_root)),
        "source_file": path.name,
        "book_id": dataset.get("id"),
        "length": dataset.get("metadata", {}).get("length"),
        "arabic_title": normalize_text(dataset.get("metadata", {}).get("arabic", {}).get("title")),
        "arabic_author": normalize_text(dataset.get("metadata", {}).get("arabic", {}).get("author")),
        "english_title": normalize_text(dataset.get("metadata", {}).get("english", {}).get("title")),
        "english_author": normalize_text(dataset.get("metadata", {}).get("english", {}).get("author")),
        "chapter_count": len(chapter_map),
        "hadith_count": len(entries),
    }

    out_book_dir = output_dir / "by_book"
    out_book_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_book_dir / f"{collection}.json"
    payload = {
        "_meta": meta,
        **dict(sorted(entries.items(), key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else kv[0])),
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    chapter_outputs = []
    if write_by_chapter:
        out_chapter_dir = output_dir / "by_chapter" / collection
        out_chapter_dir.mkdir(parents=True, exist_ok=True)
        for chapter_id, chapter_entries in sorted(chapter_buckets.items(), key=lambda kv: kv[0]):
            chapter_path = out_chapter_dir / f"{chapter_id}.json"
            chapter_payload = {
                "_meta": {
                    **meta,
                    "chapter_id": chapter_id,
                    "chapter_title_ar": normalize_text(chapter_map.get(chapter_id, {}).get("arabic")),
                    "chapter_title_en": normalize_text(chapter_map.get(chapter_id, {}).get("english")),
                    "hadith_count": len(chapter_entries),
                },
                **dict(sorted(chapter_entries.items(), key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else kv[0])),
            }
            chapter_path.write_text(json.dumps(chapter_payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
            chapter_outputs.append(str(chapter_path))

    return {
        "collection": collection,
        "source_path": str(path),
        "output_path": str(out_path),
        "hadith_count": len(entries),
        "chapter_count": len(chapter_map),
        "chapter_outputs": chapter_outputs,
        "arabic_title": meta["arabic_title"],
        "english_title": meta["english_title"],
    }


def write_index(output_dir: Path, summaries: List[Dict]) -> Path:
    index = {
        "total_books": len(summaries),
        "total_hadiths": sum(int(item.get("hadith_count", 0)) for item in summaries),
        "books": summaries,
    }
    index_path = output_dir / "index.json"
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return index_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert AhmedBaset/hadith-json into local reference_data/hadith")
    parser.add_argument("--source-dir", default=str(DEFAULT_SOURCE_DIR), help="path to the cloned hadith-json repository")
    parser.add_argument(
        "--output-dir",
        default=HADITH_REFERENCE_DIR,
        help="local hadith reference directory, defaults to DATABASE/reference_data/hadith",
    )
    parser.add_argument("--write-by-chapter", action="store_true", help="also write per-chapter normalized files")
    args = parser.parse_args()

    source_dir = Path(args.source_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not source_dir.exists():
        raise SystemExit(f"Source directory not found: {source_dir}")

    files = discover_source_files(source_dir)
    if not files:
        raise SystemExit(f"No source JSON files found under {source_dir / 'db' / 'by_book'}")

    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: List[Dict] = []
    log(f"Source dir : {source_dir}")
    log(f"Output dir : {output_dir}")
    log(f"Files      : {len(files)}")
    for i, path in enumerate(files, 1):
        try:
            summary = convert_book_file(
                path,
                output_dir,
                source_root=source_dir,
                write_by_chapter=args.write_by_chapter,
            )
            summaries.append(summary)
            log(f"[{i:03d}/{len(files):03d}] {summary['collection']} hadiths={summary['hadith_count']}")
        except Exception as exc:
            log(f"[{i:03d}/{len(files):03d}] ERROR {path.name}: {exc}")

    index_path = write_index(output_dir, summaries)
    log(f"Index written : {index_path}")
    log(f"Books         : {len(summaries)}")
    log(f"Hadiths       : {sum(int(item.get('hadith_count', 0)) for item in summaries)}")


if __name__ == "__main__":
    main()
