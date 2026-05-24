import json
import os
import re
import unicodedata
from functools import lru_cache
from typing import Dict, List, Tuple

from config import (
    HADITH_REFERENCE_DIR,
    QURAN_REFERENCE_SOURCE_PATH,
    QURAN_REFERENCE_PATH,
    QURAN_TRANSLATION_EN_PATH,
    QURAN_TRANSLATION_PATH,
)
from dorar_client import search_dorar_hadith

DELETE_BLOCK_RE = re.compile(r"\[\[DELETE_START\]\].*?\[\[DELETE_END\]\]", re.I | re.S)
QURAN_RE = re.compile(r"\[\[(?:FIX_)?QS\s+(\d{1,3}):(\d{1,3})(?:\s+([a-z0-9+_-]+))?\]\]", re.I)
ARABIC_BLOCK_RE = re.compile(r"\[\[(?:ARABIC_BLOCK|FIX_ARABIC_BLOCK)\s+([a-z0-9_.-]+)\]\]", re.I | re.S)
DORAR_LOCAL_HADITH_RE = re.compile(
    r"\[\[(?:FIX_HADITH|FIX_HADITH_LOCAL|HADITH)\s+([a-z0-9_.-]+)\s*:\s*([0-9a-zA-Z_.-]+)(?:\s+([a-z0-9+_-]+))?\]\]",
    re.I | re.S,
)
DORAR_RE = re.compile(r"\[\[(?:DORAR_SEARCH|FIX_HADITH_DORAR|CANDIDATE_HADITH_DORAR)\s+(.+?)\]\]", re.I | re.S)
HADITH_INDEX_PATH = os.path.join(HADITH_REFERENCE_DIR, "index.json")
HADITH_COLLECTION_ALIASES = {
    "sahih_bukhari": "bukhari",
    "sahih_al_bukhari": "bukhari",
    "bukhari": "bukhari",
    "sahih_muslim": "muslim",
    "sahih_al_muslim": "muslim",
    "muslim": "muslim",
    "sunan_abi_dawud": "abudawud",
    "sunan_abu_dawud": "abudawud",
    "abu_dawud": "abudawud",
    "abudawud": "abudawud",
    "abu_daud": "abudawud",
    "sunan_at_tirmidhi": "tirmidhi",
    "jami_at_tirmidhi": "tirmidhi",
    "jami_tirmidhi": "tirmidhi",
    "tirmidhi": "tirmidhi",
    "at_tirmidhi": "tirmidhi",
    "sunan_an_nasai": "nasai",
    "sunan_an_nasa_i": "nasai",
    "sunan_an_nasa": "nasai",
    "nasai": "nasai",
    "an_nasai": "nasai",
    "an_nasa_i": "nasai",
    "sunan_ibn_majah": "ibnmajah",
    "ibn_majah": "ibnmajah",
    "ibnmajah": "ibnmajah",
    "muwatta_malik": "malik",
    "al_muwatta": "malik",
    "muwatta": "malik",
    "malik": "malik",
    "musnad_ahmad": "ahmed",
    "musnad_ahmed": "ahmed",
    "ahmad": "ahmed",
    "ahmed": "ahmed",
    "darimi": "darimi",
    "sunan_darimi": "darimi",
    "riyad_as_salihin": "riyad_assalihin",
    "riyadh_as_salihin": "riyad_assalihin",
    "riyad_assalihin": "riyad_assalihin",
    "riyadussalihin": "riyad_assalihin",
    "shamail_muhammadiyah": "shamail_muhammadiyah",
    "bulugh_al_maram": "bulugh_almaram",
    "bulugh_almaram": "bulugh_almaram",
    "al_adab_al_mufrad": "aladab_almufrad",
    "aladab_al_mufrad": "aladab_almufrad",
    "aladab_almufrad": "aladab_almufrad",
    "mishkat_al_masabih": "mishkat_almasabih",
    "mishkat_almasabih": "mishkat_almasabih",
    "forty_hadith_nawawi": "nawawi40",
    "nawawi40": "nawawi40",
    "forty_hadith_qudsi": "qudsi40",
    "qudsi40": "qudsi40",
    "forty_hadith_shah_waliullah": "shahwaliullah40",
    "shahwaliullah40": "shahwaliullah40",
}
HADITH_COLLECTION_PRIORITY = {
    "bukhari": 0,
    "muslim": 1,
    "abudawud": 2,
    "tirmidhi": 3,
    "nasai": 4,
    "ibnmajah": 5,
    "malik": 6,
    "ahmed": 7,
    "darimi": 8,
    "riyad_assalihin": 20,
    "mishkat_almasabih": 21,
    "bulugh_almaram": 22,
    "aladab_almufrad": 23,
    "shamail_muhammadiyah": 24,
    "nawawi40": 30,
    "qudsi40": 31,
    "shahwaliullah40": 32,
}


def _load_json(path: str, default):
    if not path or not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return default
    return data if isinstance(data, type(default)) else default


def _load_pipe_quran(path: str, *, translation_field: str = "") -> Dict[str, Dict]:
    if not path or not os.path.exists(path):
        return {}
    out: Dict[str, Dict] = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("|", 2)
                if len(parts) < 3:
                    continue
                surah, ayah, text = parts[0].strip(), parts[1].strip(), parts[2].strip()
                if not surah or not ayah:
                    continue
                key = _normalize_quran_key(surah, ayah)
                entry = out.setdefault(
                    key,
                    {
                        "key": key,
                        "surah": int(surah),
                        "ayah": int(ayah),
                        "arabic": "",
                        "translation_id": "",
                        "translation_en": "",
                        "raw": {},
                    },
                )
                if translation_field == "translation_en":
                    entry["translation_en"] = text
                elif translation_field == "translation_id":
                    entry["translation_id"] = text
                else:
                    entry["arabic"] = text
    except Exception:
        return {}
    return out


def _normalize_quran_key(surah: str, ayah: str) -> str:
    return f"{int(surah)}:{int(ayah)}"


def _coerce_quran_entry(entry: Dict, key: str) -> Dict:
    if not isinstance(entry, dict):
        return {}
    arabic = entry.get("arabic") or entry.get("text") or entry.get("ayah_ar") or entry.get("ayah") or ""
    translation = (
        entry.get("translation_id")
        or entry.get("translation")
        or entry.get("id_translation")
        or entry.get("terjemah")
        or ""
    )
    translation_en = entry.get("translation_en") or entry.get("translation_en_sahih") or entry.get("en_translation") or ""
    return {
        "key": entry.get("key", key),
        "surah": int(entry.get("surah", key.split(":")[0]) or key.split(":")[0]),
        "ayah": int(entry.get("ayah", key.split(":")[1]) or key.split(":")[1]),
        "arabic": str(arabic).strip(),
        "translation_id": str(translation).strip(),
        "translation_en": str(translation_en).strip(),
        "raw": entry,
    }


def _persist_quran_entry(key: str, entry: Dict) -> None:
    try:
        os.makedirs(os.path.dirname(QURAN_REFERENCE_PATH), exist_ok=True)
        quran = _load_json(QURAN_REFERENCE_PATH, {})
        if not isinstance(quran, dict):
            quran = {}
        existing = quran.get(key, {}) if isinstance(quran.get(key, {}), dict) else {}
        quran[key] = {
            "key": key,
            "surah": entry.get("surah"),
            "ayah": entry.get("ayah"),
            "arabic": entry.get("arabic", existing.get("arabic", "")),
            "translation_id": entry.get("translation_id", existing.get("translation_id", "")),
            "translation_en": entry.get("translation_en", existing.get("translation_en", "")),
        }
        with open(QURAN_REFERENCE_PATH, "w", encoding="utf-8") as f:
            json.dump(quran, f, ensure_ascii=False, indent=2, sort_keys=True)
    except Exception:
        pass


@lru_cache(maxsize=1024)
def load_quran_reference() -> Dict[str, Dict]:
    merged: Dict[str, Dict] = {}
    sources = [
        _load_pipe_quran(QURAN_REFERENCE_SOURCE_PATH, translation_field=""),
        _load_json(QURAN_REFERENCE_PATH, {}),
        _load_pipe_quran(QURAN_TRANSLATION_PATH, translation_field="translation_id"),
        _load_pipe_quran(QURAN_TRANSLATION_EN_PATH, translation_field="translation_en"),
    ]

    for source in sources:
        if isinstance(source, dict):
            for key, entry in source.items():
                if not isinstance(entry, dict):
                    continue
                key = str(key)
                current = merged.get(key, {})
                coerced = _coerce_quran_entry({**current, **entry}, key)
                if key in merged:
                    for field, value in coerced.items():
                        if field == "raw":
                            continue
                        if value in {"", None}:
                            continue
                        merged[key][field] = value
                else:
                    merged[key] = coerced

    # Ensure source TXT data wins over cache for arabic/translation text.
    source_txt = _load_pipe_quran(QURAN_REFERENCE_SOURCE_PATH, translation_field="")
    if isinstance(source_txt, dict):
        for key, entry in source_txt.items():
            if not isinstance(entry, dict):
                continue
            merged.setdefault(key, _coerce_quran_entry({"key": key}, key))
            merged[key]["arabic"] = str(entry.get("arabic", "") or "").strip()

    translation_id_txt = _load_pipe_quran(QURAN_TRANSLATION_PATH, translation_field="translation_id")
    if isinstance(translation_id_txt, dict):
        for key, entry in translation_id_txt.items():
            if not isinstance(entry, dict):
                continue
            merged.setdefault(key, _coerce_quran_entry({"key": key}, key))
            merged[key]["translation_id"] = str(entry.get("translation_id", "") or "").strip()

    translation_en_txt = _load_pipe_quran(QURAN_TRANSLATION_EN_PATH, translation_field="translation_en")
    if isinstance(translation_en_txt, dict):
        for key, entry in translation_en_txt.items():
            if not isinstance(entry, dict):
                continue
            merged.setdefault(key, _coerce_quran_entry({"key": key}, key))
            merged[key]["translation_en"] = str(entry.get("translation_en", "") or "").strip()

    return merged


def _normalize_hadith_collection_name(name: str) -> str:
    value = (name or "").strip().lower()
    value = unicodedata.normalize("NFKC", value)
    value = value.replace("'", "").replace("`", "")
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value


@lru_cache(maxsize=1)
def _load_hadith_index() -> Dict[str, Dict]:
    data = _load_json(HADITH_INDEX_PATH, {})
    if not isinstance(data, dict):
        return {}
    books = data.get("books") if isinstance(data.get("books"), list) else []
    out: Dict[str, Dict] = {}
    for book in books:
        if not isinstance(book, dict):
            continue
        collection = _normalize_hadith_collection_name(str(book.get("collection") or ""))
        if collection:
            out[collection] = book
    return out


def resolve_hadith_collection(collection: str) -> str:
    normalized = _normalize_hadith_collection_name(collection)
    if not normalized:
        return ""
    alias = HADITH_COLLECTION_ALIASES.get(normalized, normalized)
    index = _load_hadith_index()
    if alias in index:
        return alias
    if normalized in index:
        return normalized
    alias_compact = alias.replace("_", "")
    for key in index.keys():
        if key.replace("_", "") in {alias_compact, normalized.replace("_", "")}:
            return key
    return alias


def _hadith_collection_record(collection: str) -> Dict:
    resolved = resolve_hadith_collection(collection)
    index = _load_hadith_index()
    return index.get(resolved) or index.get(_normalize_hadith_collection_name(collection)) or {}


def lookup_quran_entry(surah: int | str, ayah: int | str) -> Dict:
    key = _normalize_quran_key(surah, ayah)
    local = load_quran_reference().get(key)
    return local or {}


@lru_cache(maxsize=32)
def load_hadith_collection(collection: str) -> Dict:
    collection = resolve_hadith_collection(collection)
    if not collection:
        return {}
    record = _hadith_collection_record(collection)
    candidate_paths = []
    for key in ("output_path", "path", "json_path"):
        value = str(record.get(key) or "").strip()
        if value:
            candidate_paths.append(value)
    candidates = [
        os.path.join(HADITH_REFERENCE_DIR, f"{collection}.json"),
        os.path.join(HADITH_REFERENCE_DIR, f"{collection}.jsonl"),
        os.path.join(HADITH_REFERENCE_DIR, "by_book", f"{collection}.json"),
        os.path.join(HADITH_REFERENCE_DIR, "by_book", f"{collection}.jsonl"),
    ]
    seen_paths = []
    for path in candidate_paths + candidates:
        if not path or path in seen_paths:
            continue
        seen_paths.append(path)
        if os.path.exists(path):
            if path.endswith(".jsonl"):
                out = {}
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            item = json.loads(line)
                            if isinstance(item, dict):
                                key = item.get("id") or item.get("key") or item.get("number")
                                if key is not None:
                                    out[str(key)] = item
                    return out
                except Exception:
                    continue
            data = _load_json(path, {})
            if isinstance(data, dict):
                return data
            if isinstance(data, list):
                out = {}
                for item in data:
                    if isinstance(item, dict):
                        key = item.get("id") or item.get("key") or item.get("number")
                        if key is not None:
                            out[str(key)] = item
                return out
    return {}


def lookup_hadith_entry(collection: str, number: str) -> Dict:
    collection = resolve_hadith_collection(collection)
    number = str(number or "").strip()
    if not collection or not number:
        return {}
    data = load_hadith_collection(collection)
    if not data:
        return {}

    candidates = [
        f"{collection}:{number}",
        f"{collection}_{number}",
        f"{collection}-{number}",
        number,
    ]
    if number.isdigit():
        candidates.insert(1, str(int(number)))

    for key in candidates:
        if key in data and isinstance(data[key], dict):
            return data[key]

    for item in data.values():
        if not isinstance(item, dict):
            continue
        entry_key = str(item.get("key") or item.get("id") or item.get("number") or "").strip()
        if entry_key in {number, f"{collection}:{number}", f"{collection}_{number}"}:
            return item
        item_number = str(item.get("number") or item.get("hadith_number") or item.get("no") or "").strip()
        if item_number and item_number == number:
            return item
    return {}


def _contains_arabic(text: str) -> bool:
    return bool(re.search(r"[\u0600-\u06FF]", text or ""))


def _normalize_search_text(text: str) -> str:
    value = unicodedata.normalize("NFKC", str(text or ""))
    value = value.lower().replace("\u0640", "")
    value = re.sub(r"[\u064B-\u065F\u0670]", "", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _normalize_search_tokens(text: str) -> List[str]:
    tokens = re.findall(r"[\w\u0600-\u06FF]+", _normalize_search_text(text))
    out: List[str] = []
    for token in tokens:
        token = token.strip()
        if len(token) < 2:
            continue
        out.append(token)
    return out


def _hadith_source_label(collection: str, entry: Dict) -> str:
    record = _hadith_collection_record(collection)
    english_title = str(record.get("english_title") or "").strip()
    arabic_title = str(record.get("arabic_title") or "").strip()
    title = english_title or arabic_title or collection
    chapter = str(entry.get("chapter_title_en") or entry.get("chapter_title_ar") or "").strip()
    if chapter and chapter != title:
        return f"{collection} · {title} · {chapter}"
    return f"{collection} · {title}"


def _hadith_preview_text(entry: Dict) -> str:
    parts = [
        str(entry.get("arabic") or entry.get("text") or "").strip(),
        str(entry.get("translation_id") or entry.get("translation") or entry.get("english_text") or "").strip(),
        str(entry.get("translation_en") or "").strip(),
    ]
    parts = [part for part in parts if part]
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    return "\n\n".join(parts[:2]).strip()


def search_local_hadith(query: str, limit: int = 10, collection: str = "") -> List[Dict]:
    query = str(query or "").strip()
    limit = max(1, min(int(limit or 10), 50))
    if not query:
        return []

    requested_collection = resolve_hadith_collection(collection)
    parsed_collection = ""
    parsed_number = ""
    match = re.match(r"^\s*([a-z0-9_]+)\s*:\s*([0-9a-zA-Z_.-]+)\s*$", query, re.I)
    if match:
        parsed_collection = resolve_hadith_collection(match.group(1))
        parsed_number = str(match.group(2)).strip()

    candidate_collections: List[str]
    if requested_collection:
        candidate_collections = [requested_collection]
    elif parsed_collection:
        candidate_collections = [parsed_collection]
    else:
        candidate_collections = sorted(_load_hadith_index().keys())

    if not candidate_collections:
        return []

    normalized_query = _normalize_search_text(query)
    tokens = _normalize_search_tokens(query)
    arabic_query = _contains_arabic(query)
    scored: List[Tuple[float, Dict]] = []

    for coll in candidate_collections:
        data = load_hadith_collection(coll)
        if not data:
            continue
        for key, entry in data.items():
            if not isinstance(entry, dict) or str(key).startswith("_"):
                continue

            hadith_key = str(entry.get("key") or entry.get("id") or entry.get("number") or key).strip()
            number = str(entry.get("number") or entry.get("hadith_number") or entry.get("no") or "").strip()
            entry_text = _normalize_search_text(
                " \n ".join(
                    part
                    for part in [
                        entry.get("arabic", ""),
                        entry.get("translation_id", ""),
                        entry.get("translation_en", ""),
                        entry.get("english_text", ""),
                        entry.get("text", ""),
                        entry.get("narrator", ""),
                        entry.get("english_narrator", ""),
                        entry.get("chapter_title_ar", ""),
                        entry.get("chapter_title_en", ""),
                    ]
                    if part
                )
            )
            if not entry_text:
                continue

            score = 0.0
            exact_match = False
            exact_reason = ""
            if normalized_query and normalized_query == entry_text:
                score += 1000.0
                exact_match = True
                exact_reason = "teks"
            if normalized_query and normalized_query in entry_text:
                score += 500.0
            if tokens:
                hits = sum(1 for token in tokens if token in entry_text)
                if hits:
                    score += hits * 40.0
                    score += (hits / max(len(tokens), 1)) * 30.0
            if arabic_query and _contains_arabic(str(entry.get("arabic") or "")):
                score += 20.0
            if parsed_number and number and parsed_number == number:
                score += 160.0
                exact_match = True
                exact_reason = exact_reason or "nomor"
            if parsed_collection and coll == parsed_collection:
                score += 25.0
            if query.isdigit() and number == query:
                score += 140.0
                exact_match = True
                exact_reason = exact_reason or "nomor"
            if normalized_query and hadith_key and normalized_query == _normalize_search_text(hadith_key):
                score += 150.0
                exact_match = True
                exact_reason = exact_reason or "kode"
            if normalized_query and hadith_key and normalized_query in _normalize_search_text(hadith_key):
                score += 40.0
            if parsed_collection and parsed_number and coll == parsed_collection and number == parsed_number:
                exact_match = True
                exact_reason = exact_reason or "koleksi+nomor"
            if score <= 0:
                continue

            preview = _hadith_preview_text(entry)
            scored.append(
                (
                    score,
                    {
                        "collection": coll,
                        "number": number or str(entry.get("number") or entry.get("id") or ""),
                        "key": hadith_key,
                        "text": preview or str(entry.get("arabic") or entry.get("translation_en") or entry.get("translation_id") or ""),
                        "source": _hadith_source_label(coll, entry),
                        "grade": str(entry.get("grade") or entry.get("hukm") or entry.get("verdict") or "").strip(),
                        "exact_match": exact_match,
                        "exact_reason": exact_reason,
                        "raw": entry,
                    },
                )
            )

    scored.sort(
        key=lambda item: (
            -item[0],
            HADITH_COLLECTION_PRIORITY.get(str(item[1].get("collection", "") or ""), 100),
            str(item[1].get("collection", "") or ""),
            item[1].get("number", ""),
            item[1].get("key", ""),
        )
    )
    results = [item[1] for item in scored[:limit]]
    return results


def format_quran(entry: Dict, mode: str = "ar") -> str:
    arabic = str(entry.get("arabic", "") or "").strip()
    translation = str(entry.get("translation_id", "") or "").strip()
    translation_en = str(entry.get("translation_en", "") or "").strip()
    mode = (mode or "ar").lower()
    if mode == "ar":
        return arabic
    if mode == "id":
        return translation
    if mode == "en":
        return translation_en
    if mode in {"ar+id", "id+ar"}:
        parts = [part for part in [arabic, translation] if part]
        return "\n\n".join(parts).strip()
    if mode in {"ar+en", "en+ar"}:
        parts = [part for part in [arabic, translation_en] if part]
        return "\n\n".join(parts).strip()
    if mode in {"id+en", "en+id"}:
        parts = [part for part in [translation, translation_en] if part]
        return "\n\n".join(parts).strip()
    if mode in {"ar+id+en", "ar+en+id", "id+ar+en", "id+en+ar", "en+ar+id", "en+id+ar"}:
        parts = [part for part in [arabic, translation, translation_en] if part]
        return "\n\n".join(parts).strip()
    return arabic


def _format_hadith_entry(entry: Dict, mode: str = "ar") -> str:
    arabic = str(entry.get("arabic", "") or entry.get("text", "") or "").strip()
    translation = str(
        entry.get("translation_id")
        or entry.get("translation")
        or entry.get("text_id")
        or entry.get("translation_en")
        or entry.get("english_text")
        or ""
    ).strip()
    grade = str(entry.get("grade") or entry.get("hukm") or "").strip()
    mode = (mode or "ar").lower()
    if mode == "ar":
        return arabic
    if mode == "id":
        parts = [part for part in [translation, f"Derajat: {grade}" if grade else ""] if part]
        return "\n\n".join(parts).strip()
    if mode in {"ar+id", "id+ar"}:
        parts = [part for part in [arabic, translation, f"Derajat: {grade}" if grade else ""] if part]
        return "\n\n".join(parts).strip()
    return arabic


def search_dorar_candidates(query: str, limit: int = 5) -> List[Dict]:
    return search_dorar_hadith(query, limit=limit)


def apply_reference_markers(
    text: str,
    *,
    dorar_limit: int = 5,
    dorar_policy: str = "preserve",
    dorar_choices: Dict[str, int] | None = None,
    arabic_blocks: List[Dict] | None = None,
) -> Dict:
    """
    Apply reference markers to a text draft.

    - DELETE_START/END blocks are removed.
    - QS markers are replaced from local Quran reference data.
    - ARABIC_BLOCK markers are replaced from cached Arabic OCR blocks.
    - DORAR markers are searched and can optionally be replaced by selected candidates.
    """
    source_text = text or ""
    stages: List[Dict] = []
    unresolved: List[Dict] = []
    dorar_candidates: Dict[str, List[Dict]] = {}
    arabic_block_map: Dict[str, Dict] = {}
    if isinstance(arabic_blocks, list):
        for block in arabic_blocks:
            if not isinstance(block, dict):
                continue
            block_id = str(block.get("block_id") or "").strip()
            if block_id:
                arabic_block_map[block_id.lower()] = block

    def _delete_blocks(value: str) -> str:
        removed = 0

        def repl(match):
            nonlocal removed
            removed += 1
            return ""

        value = DELETE_BLOCK_RE.sub(repl, value)
        if removed:
            stages.append({"type": "delete_block", "count": removed})
        return value

    def _replace_quran(value: str) -> str:
        replaced = 0

        def repl(match):
            nonlocal replaced
            key = _normalize_quran_key(match.group(1), match.group(2))
            mode = (match.group(3) or "ar").lower()
            entry = lookup_quran_entry(match.group(1), match.group(2))
            if not entry:
                unresolved.append({"type": "quran", "key": key, "marker": match.group(0)})
                return f"[[ERROR_QS_NOT_FOUND {key}]]"
            replaced += 1
            return format_quran(entry, mode)

        value = QURAN_RE.sub(repl, value)
        if replaced:
            stages.append({"type": "quran", "count": replaced})
        return value

    def _replace_local_hadith(value: str) -> str:
        replaced = 0

        def repl(match):
            nonlocal replaced
            collection = (match.group(1) or "").strip().lower()
            number = (match.group(2) or "").strip()
            mode = (match.group(3) or "ar").lower()
            entry = lookup_hadith_entry(collection, number)
            if not entry:
                unresolved.append(
                    {
                        "type": "hadith_local",
                        "collection": collection,
                        "number": number,
                        "marker": match.group(0),
                    }
                )
                return match.group(0)
            replaced += 1
            return _format_hadith_entry(entry, mode=mode)

        value = DORAR_LOCAL_HADITH_RE.sub(repl, value)
        if replaced:
            stages.append({"type": "hadith_local", "count": replaced})
        return value

    def _replace_arabic_blocks(value: str) -> str:
        replaced = 0

        def repl(match):
            nonlocal replaced
            block_id = (match.group(1) or "").strip()
            block = arabic_block_map.get(block_id.lower())
            if not block:
                unresolved.append({"type": "arabic_block", "block_id": block_id, "marker": match.group(0)})
                return match.group(0)
            replaced += 1
            text_value = str(block.get("ocr_text") or block.get("text") or "").strip()
            if not text_value:
                unresolved.append({"type": "arabic_block", "block_id": block_id, "marker": match.group(0), "reason": "empty"})
                return match.group(0)
            return text_value

        value = ARABIC_BLOCK_RE.sub(repl, value)
        if replaced:
            stages.append({"type": "arabic_block", "count": replaced})
        return value

    def _replace_dorar(value: str) -> str:
        found = 0

        def repl(match):
            nonlocal found
            raw_query = match.group(1).strip()
            mode = "ar"
            if "::" in raw_query:
                raw_query, mode = [part.strip() for part in raw_query.split("::", 1)]
            local_candidates = search_local_hadith(raw_query, limit=dorar_limit)
            candidates = local_candidates
            search_source = "local"
            if not candidates:
                candidates = search_dorar_candidates(raw_query, limit=dorar_limit)
                search_source = "dorar"
            dorar_candidates[raw_query] = candidates
            found += 1

            selected_index = None
            if dorar_choices and raw_query in dorar_choices:
                try:
                    selected_index = int(dorar_choices[raw_query])
                except Exception:
                    selected_index = None

            if selected_index is not None and 0 <= selected_index < len(candidates):
                return _format_hadith_entry(candidates[selected_index], mode=mode)

            policy = (dorar_policy or "preserve").lower()
            if policy in {"first", "auto", "replace_first"} and candidates:
                return _format_hadith_entry(candidates[0], mode=mode)
            if policy in {"candidates", "list"} and candidates:
                unresolved.append({"type": search_source, "query": raw_query, "marker": match.group(0), "candidates": candidates})
                lines = [f"[[DORAR_CANDIDATES {raw_query}]]"]
                for idx, cand in enumerate(candidates, start=1):
                    preview = cand.get("text", "")[:220].replace("\n", " ")
                    lines.append(f"{idx}. {preview}")
                return "\n".join(lines)
            unresolved.append({"type": search_source, "query": raw_query, "marker": match.group(0), "candidates": candidates})
            return match.group(0)

        value = DORAR_RE.sub(repl, value)
        if found:
            stages.append({"type": "dorar", "count": found})
        return value

    resolved = _delete_blocks(source_text)
    resolved = _replace_quran(resolved)
    resolved = _replace_local_hadith(resolved)
    resolved = _replace_arabic_blocks(resolved)
    resolved = _replace_dorar(resolved)

    return {
        "resolved_text": resolved,
        "stages": stages,
        "unresolved": unresolved,
        "dorar_candidates": dorar_candidates,
        "quran_reference_path": QURAN_REFERENCE_PATH,
        "hadith_reference_dir": HADITH_REFERENCE_DIR,
    }
