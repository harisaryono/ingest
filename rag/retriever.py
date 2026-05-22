import requests
import re
from threading import Lock
from typing import List, Dict, Optional

from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

from config import (
    OLLAMA_BASE,
    EMBED_MODEL,
    QDRANT_PATH,
    COLLECTION_NAME,
    DEFAULT_TOP_K,
    RETRIEVAL_CANDIDATES_PER_QUERY,
)

EMBED_URL = f"{OLLAMA_BASE}/api/embed"

_client: Optional[QdrantClient] = None
_client_lock = Lock()


def _get_client() -> QdrantClient:
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = QdrantClient(path=QDRANT_PATH)
    return _client


def embed_query(query: str) -> List[float]:
    resp = requests.post(
        EMBED_URL,
        json={"model": EMBED_MODEL, "input": [query]},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["embeddings"][0]


def _extract_keywords(text: str, min_len: int = 3) -> List[str]:
    words = re.findall(r"[a-zA-Z0-9']+", text.lower())
    stopwords = {
        "apa", "bagaimana", "bagi", "dalam", "dengan", "dan", "dari", "di",
        "ke", "kepada", "oleh", "pada", "para", "sebagai", "sebab", "sesuai",
        "untuk", "yang", "atau", "itu", "ini", "cara", "tentang", "dalam",
        "agar", "maka", "serta", "karena",
    }
    keywords = [w for w in words if len(w) >= min_len and w not in stopwords]
    return keywords


QUERY_EXPANSIONS = {
    "sholat": {"sholat", "shalat", "salat"},
    "shalat": {"sholat", "shalat", "salat"},
    "salat": {"sholat", "shalat", "salat"},
    "nabi": {"nabi", "rasul", "muhammad", "sunnah", "hadis", "hadits"},
    "rasul": {"nabi", "rasul", "muhammad", "sunnah", "hadis", "hadits"},
    "muhammad": {"nabi", "rasul", "muhammad", "sunnah", "hadis", "hadits"},
    "hadis": {"hadis", "hadits"},
    "hadits": {"hadis", "hadits"},
    "wudhu": {"wudhu", "wudlu"},
    "wudlu": {"wudhu", "wudlu"},
}


def _build_query_groups(query: str) -> List[List[str]]:
    keywords = _extract_keywords(query)
    groups: List[List[str]] = []
    seen = set()

    for kw in keywords:
        group = sorted(QUERY_EXPANSIONS.get(kw, {kw}))
        key = tuple(group)
        if key in seen:
            continue
        seen.add(key)
        groups.append(group)

    return groups


def _build_query_variants(query: str) -> List[str]:
    groups = _build_query_groups(query)
    if not groups:
        return [query]

    variants = [query]
    merged_terms = []
    seen_terms = set()
    for group in groups:
        for term in group:
            if term not in seen_terms:
                merged_terms.append(term)
                seen_terms.add(term)

    merged_variant = " ".join([query] + merged_terms)
    if merged_variant not in variants:
        variants.append(merged_variant)

    for group in groups:
        if len(group) > 1:
            variant = " ".join(group)
            if variant not in variants:
                variants.append(variant)

    return variants


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _match_group(text: str, group: List[str]) -> bool:
    haystack = _normalize(text)
    return any(term in haystack for term in group)


def _count_group_hits(text: str, groups: List[List[str]]) -> int:
    return sum(1 for group in groups if _match_group(text, group))


def rerank(results: List[Dict], query: str) -> List[Dict]:
    query_groups = _build_query_groups(query)
    query_keywords = _extract_keywords(query, min_len=3)

    for r in results:
        score = r["_qdrant_score"]

        payload = r.get("payload", {})
        title = payload.get("title", "")
        text = payload.get("text", "")
        haystack = f"{title} {text}"

        group_hits = _count_group_hits(haystack, query_groups)
        if query_groups:
            score += 0.22 * group_hits
            score += 0.05 * (group_hits / len(query_groups))

        title_hits = _count_group_hits(title, query_groups)
        if title_hits:
            score += 0.08 * title_hits

        if query_keywords:
            word_hits = sum(1 for kw in query_keywords if kw in _normalize(haystack))
            score += 0.01 * word_hits

        if len(text) < 50:
            score -= 0.10

        if "HYPERLINK" in text:
            score -= 0.05

        r["score"] = round(score, 4)

    results.sort(key=lambda r: r["score"], reverse=True)
    return results


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    language: str = "id",
) -> List[Dict]:
    client = _get_client()

    qdrant_filter = None
    if language != "all":
        qdrant_filter = Filter(
            must=[FieldCondition(key="language", match=MatchValue(value=language))]
        )

    candidate_map: Dict[str, Dict] = {}
    query_variants = _build_query_variants(query)

    for variant in query_variants:
        query_embedding = embed_query(variant)
        raw_response = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding,
            limit=RETRIEVAL_CANDIDATES_PER_QUERY,
            query_filter=qdrant_filter,
            with_payload=True,
        )

        raw_results = getattr(raw_response, "points", raw_response)
        for hit in raw_results:
            current = candidate_map.get(str(hit.id))
            candidate = {
                "id": hit.id,
                "text": hit.payload.get("text", ""),
                "payload": hit.payload,
                "_qdrant_score": hit.score,
            }
            if current is None or hit.score > current["_qdrant_score"]:
                candidate_map[str(hit.id)] = candidate

    results = list(candidate_map.values())

    results = rerank(results, query)
    return results[:top_k]
