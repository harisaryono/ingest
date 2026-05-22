import json
import os
import re
from typing import List, Dict, Optional

from config import CHUNK_MAX_CHARS, CHUNK_OVERLAP, CHUNK_MIN_CHARS


def is_metadata_page(text: str) -> bool:
    if len(text) < 30:
        return True
    lines = text.strip().split("\n")
    if len(lines) <= 3 and any("cover" in ln.lower() for ln in lines):
        return True
    if "HYPERLINK" in text and len(text) < 300:
        return True
    if re.match(r'^\d{4}\s*[Hh]\s*$', text.strip()):
        return True
    return False


def is_chapter_heading(text: str) -> bool:
    text = text.strip()
    if len(text) > 120:
        return False
    lines = [ln for ln in text.split("\n") if ln.strip()]
    if not lines:
        return False
    first_line = lines[0].strip()
    if not first_line:
        return False
    if re.match(r'^\d+[\.\)]\s', first_line):
        return True
    if re.match(r'^[IVXLCDM]+\.?\s', first_line):
        return True
    if len(first_line) < 80 and first_line.isupper():
        return True
    if len(lines) == 1 and len(first_line) < 80 and first_line[0].isupper():
        return True
    return False


def split_long_text(text: str, max_chars: int = CHUNK_MAX_CHARS, overlap: int = CHUNK_OVERLAP) -> List[str]:
    if len(text) <= max_chars:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        if end >= len(text):
            chunks.append(text[start:])
            break
        bound = text.rfind("\n", start, end)
        if bound > start + max_chars // 2:
            end = bound + 1
        elif bound == -1:
            space = text.rfind(" ", start, end)
            if space > start + max_chars // 2:
                end = space + 1
        chunks.append(text[start:end])
        start = end - overlap
        if start < 0:
            start = 0
    return chunks


def chunk_book(book_json_path: str) -> List[Dict]:
    with open(book_json_path, "r", encoding="utf-8") as f:
        book = json.load(f)

    book_id = os.path.splitext(book["filename"])[0]
    title = book["title"]
    language = book["language"]
    filename = book["filename"]
    pages = book["pages"]

    raw_chunks = []
    for p in pages:
        raw_chunks.append({
            "text": p["content"],
            "page": p["page"],
        })

    processed = []
    buffer_text = ""
    buffer_page_start = None
    buffer_page_end = None

    def flush_buffer():
        nonlocal buffer_text, buffer_page_start, buffer_page_end
        if not buffer_text.strip():
            buffer_text = ""
            buffer_page_start = None
            buffer_page_end = None
            return
        parts = split_long_text(buffer_text)
        for idx, part in enumerate(parts):
            processed.append({
                "text": part,
                "page_start": buffer_page_start,
                "page_end": buffer_page_end,
                "chunk_idx": idx,
            })
        buffer_text = ""
        buffer_page_start = None
        buffer_page_end = None

    for rc in raw_chunks:
        text = rc["text"].strip()
        page_num = rc["page"]

        if not text or is_metadata_page(text):
            continue

        if len(text) < CHUNK_MIN_CHARS:
            if is_chapter_heading(text):
                flush_buffer()
                if processed and processed[-1]["page_end"] == page_num - 1:
                    processed[-1]["text"] = processed[-1]["text"] + "\n\n" + text
                    processed[-1]["page_end"] = page_num
                elif text:
                    buffer_text = text
                    buffer_page_start = page_num
                    buffer_page_end = page_num
                continue
            else:
                if buffer_page_start is not None:
                    buffer_text += "\n\n" + text
                    buffer_page_end = page_num
                elif processed:
                    processed[-1]["text"] = processed[-1]["text"] + "\n\n" + text
                    processed[-1]["page_end"] = page_num
                continue

        flush_buffer()

        parts = split_long_text(text)
        for idx, part in enumerate(parts):
            processed.append({
                "text": part,
                "page_start": page_num,
                "page_end": page_num,
                "chunk_idx": idx,
            })

    flush_buffer()

    result = []
    for c in processed:
        result.append({
            "text": c["text"],
            "metadata": {
                "book_id": book_id,
                "filename": filename,
                "title": title,
                "language": language,
                "page_start": c["page_start"],
                "page_end": c["page_end"],
                "chunk_idx": c["chunk_idx"],
            },
        })

    return result
