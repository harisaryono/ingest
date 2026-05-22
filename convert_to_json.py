import json
import re
import os
from pathlib import Path

INPUT_DIR = Path("/media/harry/DATA250/txt")
OUTPUT_DIR = Path(__file__).resolve().parent / "json_output"
COVERS_DIR = OUTPUT_DIR / "_covers"
EMPTY_DIR = OUTPUT_DIR / "_empty"

SKIP_DIRS = {"json_output", "_covers", "_empty"}
SKIP_EXTENSIONS = {".py", ".sh", ".bat", ".exe"}

def detect_language(filename):
    if filename.startswith("en_") or filename.startswith("en-"):
        return "en"
    if filename.startswith("id_") or filename.startswith("id-"):
        return "id"
    if filename.startswith("ru_") or filename.startswith("ru-"):
        return "ru"
    return "unknown"

def clean_title(filename):
    name = filename
    lang_prefixes = ["en_", "en-", "id_", "id-", "ru_", "ru-"]
    for p in lang_prefixes:
        if name.startswith(p):
            name = name[len(p):]
            break

    name = name.replace("_", " ")
    name = name.replace("-", " ")
    name = re.sub(r'\s+', ' ', name).strip()

    words = name.split()
    processed = []
    for w in words:
        if w.upper() == w and len(w) > 1:
            processed.append(w)
        elif w.lower() in ("and", "the", "of", "in", "to", "for", "a", "an", "or", "by", "is", "on", "its", "wa", "fi", "bi", "li"):
            processed.append(w.lower() if processed else w.capitalize())
        elif w.startswith("al-") or w.startswith("Al-"):
            processed.append(w[0].upper() + w[1:])
        else:
            processed.append(w.capitalize())
    if processed:
        processed[0] = processed[0].capitalize()
    return " ".join(processed)


def split_pages(content):
    """Split content by form feed character. Returns list of page text strings."""
    if not content or not content.strip():
        return []

    pages = content.split("\f")
    pages = [p.strip() for p in pages if p.strip()]
    return pages


def process_file(filepath):
    filename = os.path.basename(filepath)
    stem, ext = os.path.splitext(filename)
    ext = ext or ""

    if not os.path.isfile(filepath):
        return None

    filesize = os.path.getsize(filepath)

    if filesize == 0:
        return {
            "type": "empty",
            "filename": filename,
            "language": detect_language(filename),
            "title": clean_title(stem),
        }

    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except Exception as e:
        print(f"  ERROR reading {filename}: {e}")
        return None

    pages_raw = split_pages(content)

    if not pages_raw:
        pages_raw = [content.strip()]

    return {
        "type": "normal",
        "filename": filename,
        "language": detect_language(filename),
        "title": clean_title(stem),
        "size_bytes": filesize,
        "total_pages": len(pages_raw),
        "pages": [
            {"page": i + 1, "content": p}
            for i, p in enumerate(pages_raw)
        ],
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    COVERS_DIR.mkdir(parents=True, exist_ok=True)
    EMPTY_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted([
        f for f in os.listdir(INPUT_DIR)
        if os.path.isfile(INPUT_DIR / f)
        and f not in SKIP_DIRS
        and os.path.splitext(f)[1].lower() not in SKIP_EXTENSIONS
    ])

    print(f"Total files ditemukan: {len(files)}")
    index_records = []

    for i, fname in enumerate(files, 1):
        fpath = INPUT_DIR / fname
        result = process_file(fpath)

        if result is None:
            continue

        if result["type"] == "empty":
            outpath = EMPTY_DIR / f"{fname}.json"
            del result["type"]
            with open(outpath, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"[{i:03d}] EMPTY  {fname} -> _empty/")
            continue

        is_cover = result["total_pages"] <= 1 and result["size_bytes"] < 20000

        del result["type"]

        if is_cover:
            outpath = COVERS_DIR / f"{fname}.json"
        else:
            outpath = OUTPUT_DIR / f"{fname}.json"

        with open(outpath, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        label = "COVER " if is_cover else "OK    "
        print(f"[{i:03d}] {label} {fname} -> {os.path.relpath(outpath, INPUT_DIR)} ({result['total_pages']} halaman)")

        index_records.append({
            "filename": fname,
            "language": result["language"],
            "title": result["title"],
            "total_pages": result["total_pages"],
            "size_bytes": result["size_bytes"],
            "json_path": str(os.path.relpath(outpath, INPUT_DIR)),
        })

    # Write index
    index_path = OUTPUT_DIR / "_index.json"
    index_data = {
        "total_files": len(index_records),
        "languages": {},
        "files": index_records,
    }
    for r in index_records:
        lang = r["language"]
        if lang not in index_data["languages"]:
            index_data["languages"][lang] = 0
        index_data["languages"][lang] += 1

    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index_data, f, indent=2, ensure_ascii=False)

    print(f"\n=== SELESAI ===")
    print(f"Total file diproses: {len(index_records)}")
    print(f"Rincian bahasa: {index_data['languages']}")
    print(f"Output utama: {OUTPUT_DIR}/")
    print(f"File cover:   {COVERS_DIR}/")
    print(f"File kosong:  {EMPTY_DIR}/")
    print(f"Index:        {index_path}")


if __name__ == "__main__":
    main()
