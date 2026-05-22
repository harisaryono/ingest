import json
import re
import requests
from typing import List, Dict, Tuple

from config import (
    OLLAMA_BASE,
    LLM_MODEL,
    GENERATION_BACKEND,
    LEASE_COORDINATOR_URL,
    LEASE_MODEL,
)

OLLAMA_GENERATE_URL = f"{OLLAMA_BASE}/api/generate"


def build_prompt(
    query: str,
    context_chunks: List[Dict],
    strict: bool = True,
) -> str:
    context_lines = []
    for c in context_chunks:
        meta = c.get("payload", c.get("metadata", {}))
        page_range = f"{meta.get('page_start', '?')}"
        if meta.get("page_start") != meta.get("page_end"):
            page_range += f"–{meta.get('page_end', '')}"
        context_lines.append(f"[Buku: {meta.get('title', '?')}, Halaman {page_range}]")
        context_lines.append(c.get("text", ""))
        context_lines.append("")

    context_str = "\n".join(context_lines)

    if strict:
        rules = """ATURAN WAJIB:
1. Jangan memakai pengetahuan di luar referensi.
2. Jangan menambah ayat, hadis, nama ulama, atau hukum jika tidak muncul dalam referensi.
3. Jangan menyimpulkan status hukum seperti wajib, sunnah, haram, makruh, bid'ah, sah, atau batal kecuali istilah itu disebut jelas dalam referensi.
4. Jika referensi tidak cukup menjawab, jawab:
   "Tidak ditemukan secara cukup dalam buku referensi."
5. Jawab ringkas maksimal 3 paragraf.
6. Setelah setiap poin penting, cantumkan rujukan: [Nama Buku, hlm. X–Y].
7. Jika ada perbedaan antar referensi, sebutkan perbedaannya secara netral."""
    else:
        rules = """ATURAN:
1. Jawab ringkas maksimal 3 paragraf.
2. Setelah setiap poin penting, cantumkan rujukan: [Nama Buku, hlm. X–Y].
3. Jika informasi tidak cukup, katakan tidak tahu.
4. Gunakan bahasa Indonesia."""

    return f"""Anda adalah asisten RAG. Jawaban wajib hanya berdasarkan INFORMASI REFERENSI di bawah.

{rules}

INFORMASI REFERENSI:
{context_str}

PERTANYAAN: {query}

JAWABAN:"""


def select_mode(query: str, strict: bool) -> str:
    if GENERATION_BACKEND == "ollama":
        return "local"
    if GENERATION_BACKEND == "lease":
        return "large"

    hukum_keywords = [
        "hukum", "wajib", "sunnah", "haram", "makruh", "mubah",
        "bid'ah", "sah", "batal", "dalil", "hadis", "ayat",
        "khilaf", "akidah", "tauhid", "syirik", "kafir",
        "shalat", "puasa", "zakat", "haji", "wudhu", "nikah",
    ]
    query_lower = query.lower()
    if strict and any(k in query_lower for k in hukum_keywords):
        return "large"
    if len(query) < 50 and not strict:
        return "local"
    return "local"


def generate_local(prompt: str) -> str:
    resp = requests.post(
        OLLAMA_GENERATE_URL,
        json={
            "model": LLM_MODEL,
            "prompt": prompt,
            "stream": False,
        },
        timeout=300,
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("response", "").strip()


def generate_remote(prompt: str, model: str = None) -> str:
    if model is None:
        model = LEASE_MODEL
    resp = requests.post(
        LEASE_COORDINATOR_URL,
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
        },
        timeout=300,
    )
    resp.raise_for_status()
    data = resp.json()
    if "choices" in data:
        return data["choices"][0]["message"]["content"].strip()
    if "response" in data:
        return data["response"].strip()
    return str(data)


def format_chunks_only(context_chunks: List[Dict]) -> str:
    lines = []
    for c in context_chunks:
        meta = c.get("payload", c.get("metadata", {}))
        page_range = f"{meta.get('page_start', '?')}"
        if meta.get("page_start") != meta.get("page_end"):
            page_range += f"–{meta.get('page_end', '')}"
        lines.append(f"[{meta.get('title', '?')}, hlm. {page_range}]")
        lines.append(c.get("text", ""))
        lines.append("---")
    return "\n".join(lines)


def generate(
    query: str,
    context_chunks: List[Dict],
    strict: bool = True,
    mode: str = "auto",
) -> Tuple[str, str, List[Dict], str]:
    if mode == "search_only":
        return format_chunks_only(context_chunks), "search_only", context_chunks

    if mode == "auto":
        mode = select_mode(query, strict)

    prompt = build_prompt(query, context_chunks, strict=strict)

    if mode == "local":
        try:
            answer = generate_local(prompt)
            return answer, "ollama", mode, context_chunks
        except Exception:
            pass

    if mode == "large":
        try:
            answer = generate_remote(prompt)
            return answer, "lease_coordinator", mode, context_chunks
        except Exception:
            pass

    try:
        answer = generate_local(prompt)
        return answer, "ollama (fallback)", "local", context_chunks
    except Exception as e:
        return f"ERROR generating answer: {e}", "none", mode, context_chunks


def extract_sources(answer: str, context_chunks: List[Dict]) -> List[Dict]:
    sources = []
    seen = set()
    for c in context_chunks:
        meta = c.get("payload", c.get("metadata", {}))
        key = f"{meta.get('book_id', '')}_{meta.get('page_start', '')}"
        if key not in seen:
            seen.add(key)
            sources.append({
                "title": meta.get("title", ""),
                "page_start": meta.get("page_start", 0),
                "page_end": meta.get("page_end", 0),
                "book_id": meta.get("book_id", ""),
                "filename": meta.get("filename", ""),
            })
    return sources
