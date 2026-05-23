import json
import os
import re
from functools import lru_cache
from typing import Dict, List, Tuple

import requests

from config import (
    HADITH_REFERENCE_DIR,
    QURAN_REFERENCE_PATH,
    QURAN_TRANSLATION_PATH,
)
from dorar_client import search_dorar_hadith

DELETE_BLOCK_RE = re.compile(r"\[\[DELETE_START\]\].*?\[\[DELETE_END\]\]", re.I | re.S)
QURAN_RE = re.compile(r"\[\[(?:FIX_)?QS\s+(\d{1,3}):(\d{1,3})(?:\s+([a-z0-9+_-]+))?\]\]", re.I)
DORAR_RE = re.compile(r"\[\[(?:DORAR_SEARCH|FIX_HADITH_DORAR|CANDIDATE_HADITH_DORAR)\s+(.+?)\]\]", re.I | re.S)
QURAN_ONLINE_BASE = "https://api.alquran.cloud/v1"


def _load_json(path: str, default):
    if not path or not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return default
    return data if isinstance(data, type(default)) else default


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
    return {
        "key": entry.get("key", key),
        "surah": int(entry.get("surah", key.split(":")[0]) or key.split(":")[0]),
        "ayah": int(entry.get("ayah", key.split(":")[1]) or key.split(":")[1]),
        "arabic": str(arabic).strip(),
        "translation_id": str(translation).strip(),
        "raw": entry,
    }


def _persist_quran_entry(key: str, entry: Dict) -> None:
    try:
        os.makedirs(os.path.dirname(QURAN_REFERENCE_PATH), exist_ok=True)
        quran = _load_json(QURAN_REFERENCE_PATH, {})
        if not isinstance(quran, dict):
            quran = {}
        quran[key] = {
            "key": key,
            "surah": entry.get("surah"),
            "ayah": entry.get("ayah"),
            "arabic": entry.get("arabic", ""),
            "translation_id": entry.get("translation_id", ""),
        }
        with open(QURAN_REFERENCE_PATH, "w", encoding="utf-8") as f:
            json.dump(quran, f, ensure_ascii=False, indent=2, sort_keys=True)
    except Exception:
        pass


@lru_cache(maxsize=1024)
def _fetch_quran_ayah_online(key: str) -> Dict:
    try:
        surah, ayah = key.split(":", 1)
        response = requests.get(
            f"{QURAN_ONLINE_BASE}/ayah/{int(surah)}:{int(ayah)}/editions/quran-uthmani,id.indonesian",
            timeout=20,
            headers={"User-Agent": "rag-review-workspace/1.0"},
        )
        response.raise_for_status()
        data = response.json()
    except Exception:
        return {}

    arabic = ""
    translation = ""
    items = data.get("data") if isinstance(data, dict) else []
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue
            identifier = str(item.get("edition", {}).get("identifier", "") or "").lower()
            text = str(item.get("text", "") or "").strip()
            if identifier == "quran-uthmani":
                arabic = text
            elif identifier in {"id.indonesian", "id.muntakhab", "id.jalalayn"} and not translation:
                translation = text

    if not arabic and isinstance(items, list) and items:
        first = items[0] if isinstance(items[0], dict) else {}
        arabic = str(first.get("text", "") or "").strip()

    if not arabic:
        return {}

    surah_no, ayah_no = key.split(":", 1)
    entry = {
        "key": key,
        "surah": int(surah_no),
        "ayah": int(ayah_no),
        "arabic": arabic,
        "translation_id": translation,
    }
    _persist_quran_entry(key, entry)
    return entry


@lru_cache(maxsize=8)
def load_quran_reference() -> Dict[str, Dict]:
    primary = _load_json(QURAN_REFERENCE_PATH, {})
    translation = _load_json(QURAN_TRANSLATION_PATH, {})
    merged: Dict[str, Dict] = {}

    if isinstance(primary, dict):
        for key, entry in primary.items():
            if isinstance(entry, dict):
                merged[str(key)] = _coerce_quran_entry(entry, str(key))
    elif isinstance(primary, list):
        for entry in primary:
            if not isinstance(entry, dict):
                continue
            key = entry.get("key") or entry.get("ref") or entry.get("verse_key")
            if not key and "surah" in entry and "ayah" in entry:
                key = _normalize_quran_key(entry["surah"], entry["ayah"])
            if key:
                merged[str(key)] = _coerce_quran_entry(entry, str(key))

    if isinstance(translation, dict):
        for key, entry in translation.items():
            key = str(key)
            if key not in merged:
                merged[key] = _coerce_quran_entry({"key": key}, key)
            if isinstance(entry, dict):
                merged[key]["translation_id"] = str(
                    entry.get("translation_id")
                    or entry.get("translation")
                    or entry.get("text")
                    or ""
                ).strip()
            elif isinstance(entry, str):
                merged[key]["translation_id"] = entry.strip()

    return merged


def lookup_quran_entry(surah: int | str, ayah: int | str) -> Dict:
    key = _normalize_quran_key(surah, ayah)
    local = load_quran_reference().get(key)
    if local and local.get("arabic"):
        return local
    online = _fetch_quran_ayah_online(key)
    if online:
        return online
    return local or {}


@lru_cache(maxsize=32)
def load_hadith_collection(collection: str) -> Dict:
    collection = (collection or "").strip().lower()
    if not collection:
        return {}
    candidates = [
        os.path.join(HADITH_REFERENCE_DIR, f"{collection}.json"),
        os.path.join(HADITH_REFERENCE_DIR, f"{collection}.jsonl"),
    ]
    for path in candidates:
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


def format_quran(entry: Dict, mode: str = "ar") -> str:
    arabic = str(entry.get("arabic", "") or "").strip()
    translation = str(entry.get("translation_id", "") or "").strip()
    mode = (mode or "ar").lower()
    if mode == "ar":
        return arabic
    if mode == "id":
        return translation
    if mode in {"ar+id", "id+ar"}:
        parts = [part for part in [arabic, translation] if part]
        return "\n\n".join(parts).strip()
    return arabic


def _format_hadith_entry(entry: Dict, mode: str = "ar") -> str:
    arabic = str(entry.get("arabic", "") or entry.get("text", "") or "").strip()
    translation = str(
        entry.get("translation_id")
        or entry.get("translation")
        or entry.get("text_id")
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
) -> Dict:
    """
    Apply reference markers to a text draft.

    - DELETE_START/END blocks are removed.
    - QS markers are replaced from local Quran reference data.
    - DORAR markers are searched and can optionally be replaced by selected candidates.
    """
    source_text = text or ""
    stages: List[Dict] = []
    unresolved: List[Dict] = []
    dorar_candidates: Dict[str, List[Dict]] = {}

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

    def _replace_dorar(value: str) -> str:
        found = 0

        def repl(match):
            nonlocal found
            raw_query = match.group(1).strip()
            mode = "ar"
            if "::" in raw_query:
                raw_query, mode = [part.strip() for part in raw_query.split("::", 1)]
            candidates = search_dorar_candidates(raw_query, limit=dorar_limit)
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
                unresolved.append({"type": "dorar", "query": raw_query, "marker": match.group(0), "candidates": candidates})
                lines = [f"[[DORAR_CANDIDATES {raw_query}]]"]
                for idx, cand in enumerate(candidates, start=1):
                    preview = cand.get("text", "")[:220].replace("\n", " ")
                    lines.append(f"{idx}. {preview}")
                return "\n".join(lines)
            unresolved.append({"type": "dorar", "query": raw_query, "marker": match.group(0), "candidates": candidates})
            return match.group(0)

        value = DORAR_RE.sub(repl, value)
        if found:
            stages.append({"type": "dorar", "count": found})
        return value

    resolved = _delete_blocks(source_text)
    resolved = _replace_quran(resolved)
    resolved = _replace_dorar(resolved)

    return {
        "resolved_text": resolved,
        "stages": stages,
        "unresolved": unresolved,
        "dorar_candidates": dorar_candidates,
        "quran_reference_path": QURAN_REFERENCE_PATH,
        "hadith_reference_dir": HADITH_REFERENCE_DIR,
    }
