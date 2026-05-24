from __future__ import annotations

import json
import os
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import fitz

from config import (
    ARABIC_OCR_BLOCK_LANG,
    ARABIC_OCR_PSM,
    ARABIC_OCR_TSV_LANG,
    ARABIC_REVIEW_DIR,
    TESSERACT_BIN,
)

ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
BLOCK_CACHE_VERSION = 1


def arabic_ratio(text: str) -> float:
    chars = [c for c in (text or "") if c.isalpha() or ARABIC_RE.match(c)]
    if not chars:
        return 0.0
    arabic = sum(1 for c in chars if ARABIC_RE.match(c))
    return arabic / len(chars)


def _page_dir(book_id: str, page_num: int) -> Path:
    return Path(ARABIC_REVIEW_DIR) / str(book_id) / f"page_{int(page_num):03d}"


def _cache_path(book_id: str, page_num: int) -> Path:
    return _page_dir(book_id, page_num) / "blocks.json"


def _page_image_path(book_id: str, page_num: int) -> Path:
    return _page_dir(book_id, page_num) / f"page_{int(page_num):03d}.png"


def _render_pdf_page(pdf_path: str, page_num: int, out_path: Path, zoom: float = 2.0) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with fitz.open(pdf_path) as doc:
        if len(doc) <= 0:
            raise ValueError("empty PDF")
        page_index = max(0, min(int(page_num) - 1, len(doc) - 1))
        page = doc.load_page(page_index)
        zoom = max(1.0, min(float(zoom or 2.0), 4.0))
        matrix = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        pix.save(str(out_path))


def _run_tesseract_tsv(image_path: Path) -> str:
    cmd = [
        TESSERACT_BIN,
        str(image_path),
        "stdout",
        "-l",
        ARABIC_OCR_TSV_LANG,
        "--psm",
        str(ARABIC_OCR_PSM),
        "tsv",
        "-c",
        "preserve_interword_spaces=1",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return proc.stdout or ""


def _run_tesseract_text(image_path: Path, lang: str) -> str:
    cmd = [
        TESSERACT_BIN,
        str(image_path),
        "stdout",
        "-l",
        lang,
        "--psm",
        str(ARABIC_OCR_PSM),
        "-c",
        "preserve_interword_spaces=1",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return (proc.stdout or "").strip()


def _parse_tsv(tsv_text: str) -> List[Dict]:
    lines = (tsv_text or "").splitlines()
    if len(lines) <= 1:
        return []
    header = lines[0].split("\t")
    rows: List[Dict] = []
    for line in lines[1:]:
        cols = line.split("\t")
        if len(cols) < len(header):
            continue
        row = dict(zip(header, cols))
        rows.append(row)
    return rows


def _group_rows(rows: Iterable[Dict]) -> List[Dict]:
    groups: Dict[Tuple[int, int, int], List[Dict]] = defaultdict(list)
    for row in rows:
        if str(row.get("level", "")).strip() != "5":
            continue
        text = str(row.get("text", "") or "").strip()
        if not text:
            continue
        try:
            block_num = int(float(row.get("block_num", 0) or 0))
            par_num = int(float(row.get("par_num", 0) or 0))
            line_num = int(float(row.get("line_num", 0) or 0))
        except Exception:
            continue
        groups[(block_num, par_num, line_num)].append(row)

    blocks: List[Dict] = []
    for key, items in groups.items():
        items_sorted = sorted(items, key=lambda r: int(float(r.get("word_num", 0) or 0)))
        texts = [str(r.get("text", "") or "").strip() for r in items_sorted if str(r.get("text", "") or "").strip()]
        if not texts:
            continue
        xs1: List[int] = []
        ys1: List[int] = []
        xs2: List[int] = []
        ys2: List[int] = []
        confs: List[float] = []
        for row in items_sorted:
            try:
                left = int(float(row.get("left", 0) or 0))
                top = int(float(row.get("top", 0) or 0))
                width = int(float(row.get("width", 0) or 0))
                height = int(float(row.get("height", 0) or 0))
                conf = float(row.get("conf", -1) or -1)
            except Exception:
                continue
            xs1.append(left)
            ys1.append(top)
            xs2.append(left + max(0, width))
            ys2.append(top + max(0, height))
            if conf >= 0:
                confs.append(conf)
        if not xs1 or not ys1 or not xs2 or not ys2:
            continue
        text = " ".join(texts).strip()
        blocks.append(
            {
                "group_key": key,
                "bbox": [min(xs1), min(ys1), max(xs2), max(ys2)],
                "text": text,
                "confidence": round(sum(confs) / len(confs), 2) if confs else None,
                "arabic_ratio": round(arabic_ratio(text), 3),
            }
        )
    blocks.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    return blocks


def _build_text_layer_blocks(pdf_path: str, page_num: int) -> List[Dict]:
    blocks: List[Dict] = []
    with fitz.open(pdf_path) as doc:
        if len(doc) <= 0:
            return blocks
        page_index = max(0, min(int(page_num) - 1, len(doc) - 1))
        page = doc.load_page(page_index)
        for item in page.get_text("blocks"):
            if len(item) < 5:
                continue
            x0, y0, x1, y1, text = item[:5]
            text = str(text or "").strip()
            if not text:
                continue
            ratio = arabic_ratio(text)
            blocks.append(
                {
                    "bbox": [int(x0), int(y0), int(x1), int(y1)],
                    "text": text,
                    "confidence": None,
                    "arabic_ratio": round(ratio, 3),
                }
            )
    blocks.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    return blocks


def _crop_pdf_region(pdf_path: str, page_num: int, bbox: List[int], out_path: Path, zoom: float = 2.0, padding: int = 8) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    x0, y0, x1, y1 = [int(v) for v in bbox]
    with fitz.open(pdf_path) as doc:
        if len(doc) <= 0:
            raise ValueError("empty PDF")
        page_index = max(0, min(int(page_num) - 1, len(doc) - 1))
        page = doc.load_page(page_index)
        zoom = max(1.0, min(float(zoom or 2.0), 4.0))
        scale = 1.0 / zoom
        rect = fitz.Rect(
            max(0, x0 - padding) * scale,
            max(0, y0 - padding) * scale,
            max(x0 + 1, x1 + padding) * scale,
            max(y0 + 1, y1 + padding) * scale,
        )
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=rect, alpha=False)
        pix.save(str(out_path))


def _load_cache(cache_path: Path) -> Dict | None:
    if not cache_path.exists():
        return None
    try:
        with cache_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _save_cache(cache_path: Path, data: Dict) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
    tmp.replace(cache_path)


def _source_signature(pdf_path: str) -> Dict[str, object]:
    stat = os.stat(pdf_path)
    return {
        "source_path": str(pdf_path),
        "source_mtime": stat.st_mtime,
        "source_size": stat.st_size,
    }


def detect_arabic_blocks(
    pdf_path: str,
    book_id: str,
    page_num: int,
    *,
    zoom: float = 2.0,
    force: bool = False,
) -> Dict:
    pdf_path = str(pdf_path)
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(pdf_path)

    page_dir = _page_dir(book_id, page_num)
    cache_path = _cache_path(book_id, page_num)
    page_image = _page_image_path(book_id, page_num)
    source_sig = _source_signature(pdf_path)
    cache = None if force else _load_cache(cache_path)

    if cache and cache.get("cache_version") == BLOCK_CACHE_VERSION:
        cached_sig = {
            "source_path": cache.get("source_path"),
            "source_mtime": cache.get("source_mtime"),
            "source_size": cache.get("source_size"),
        }
        if cached_sig == source_sig and float(cache.get("zoom", zoom) or zoom) == float(zoom):
            return cache

    _render_pdf_page(pdf_path, page_num, page_image, zoom=zoom)
    tsv_rows = _parse_tsv(_run_tesseract_tsv(page_image))
    blocks = _group_rows(tsv_rows)
    if not blocks:
        blocks = _build_text_layer_blocks(pdf_path, page_num)

    arabic_blocks = 0
    for idx, block in enumerate(blocks, 1):
        text = str(block.get("text", "") or "").strip()
        ratio = float(block.get("arabic_ratio", arabic_ratio(text)) or 0.0)
        script_guess = "arabic" if ratio >= 0.25 else "latin"
        if ratio >= 0.12 and script_guess != "arabic":
            script_guess = "mixed"
        block_id = f"p{int(page_num):03d}_b{idx:03d}"
        crop_path = page_dir / f"{block_id}.png"
        ocr_text = text
        ocr_status = "skipped"
        if script_guess in {"arabic", "mixed"}:
            arabic_blocks += 1
            try:
                _crop_pdf_region(pdf_path, page_num, block.get("bbox", [0, 0, 0, 0]), crop_path, zoom=zoom)
                cropped_text = _run_tesseract_text(crop_path, ARABIC_OCR_BLOCK_LANG)
                if cropped_text:
                    ocr_text = cropped_text
                    ocr_status = "ok"
                else:
                    ocr_status = "empty"
            except Exception as exc:
                ocr_status = f"error: {exc}"
        block.update(
            {
                "block_id": block_id,
                "page": int(page_num),
                "crop_path": str(crop_path),
                "crop_url": f"/books/{book_id}/pages/{int(page_num)}/arabic-blocks/{block_id}/crop",
                "script_guess": script_guess,
                "needs_arabic_ocr": script_guess in {"arabic", "mixed"},
                "ocr_text": ocr_text if ocr_status == "ok" or script_guess not in {"arabic", "mixed"} else ocr_text,
                "ocr_status": ocr_status,
            }
        )

    result = {
        "cache_version": BLOCK_CACHE_VERSION,
        **source_sig,
        "book_id": str(book_id),
        "page": int(page_num),
        "zoom": float(zoom),
        "page_image": str(page_image),
        "blocks": blocks,
        "counts": {
            "total": len(blocks),
            "arabic_or_mixed": arabic_blocks,
        },
    }
    _save_cache(cache_path, result)
    return result


def load_cached_arabic_blocks(book_id: str, page_num: int) -> Dict | None:
    return _load_cache(_cache_path(book_id, page_num))


def cache_arabic_blocks(book_id: str, page_num: int, data: Dict) -> None:
    _save_cache(_cache_path(book_id, page_num), data)


def update_cached_arabic_block(book_id: str, page_num: int, block_id: str, *, ocr_text: str, ocr_status: str) -> Dict | None:
    cache = load_cached_arabic_blocks(book_id, page_num)
    if not cache:
        return None
    for block in cache.get("blocks", []):
        if block.get("block_id") == block_id:
            block["ocr_text"] = ocr_text
            block["ocr_status"] = ocr_status
            cache_arabic_blocks(book_id, page_num, cache)
            return cache
    return None


def get_cached_crop_path(book_id: str, page_num: int, block_id: str) -> str | None:
    cache = load_cached_arabic_blocks(book_id, page_num)
    if not cache:
        return None
    for block in cache.get("blocks", []):
        if block.get("block_id") == block_id:
            return str(block.get("crop_path") or "")
    return None


def ocr_arabic_crop(crop_path: str) -> str:
    return _run_tesseract_text(Path(crop_path), ARABIC_OCR_BLOCK_LANG)
