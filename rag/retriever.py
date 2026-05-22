import heapq
import json
import math
import os
import pickle
import re
import unicodedata
from collections import Counter, defaultdict
from threading import Lock
from typing import List, Dict, Optional, Tuple

import requests
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

from config import (
    OLLAMA_BASE,
    EMBED_MODEL,
    QDRANT_PATH,
    COLLECTION_NAME,
    DEFAULT_TOP_K,
    RETRIEVAL_CANDIDATES,
    RETRIEVAL_CANDIDATES_PER_QUERY,
    LEXICAL_INDEX_PATH,
    JSON_DIR,
)
from chunker import chunk_book
from ingest_common import (
    chunk_hash,
    infer_source_ext,
    infer_document_type,
    infer_source_type,
    point_id_for_hash,
    resolve_index_json_path,
)


EMBED_URL = f"{OLLAMA_BASE}/api/embed"
LEXICAL_INDEX_VERSION = 1
BM25_K1 = 1.5
BM25_B = 0.75

_client: Optional[QdrantClient] = None
_client_lock = Lock()
_lexical_index: Optional[Dict] = None
_lexical_index_lock = Lock()

ARABIC_DIACRITICS_RE = re.compile(r"[\u064b-\u065f\u0670\u06d6-\u06ed]")
TOKEN_RE = re.compile(r"[A-Za-z0-9\u0600-\u06ff']+")


def _get_client() -> QdrantClient:
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = QdrantClient(path=QDRANT_PATH)
    return _client


def _load_json(path: str, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return default
    return data if isinstance(data, type(default)) else default


def _save_pickle(path: str, data: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def _load_pickle(path: str):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def _normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("\x0c", " ").replace("\xa0", " ")
    text = ARABIC_DIACRITICS_RE.sub("", text)
    text = text.lower()
    text = re.sub(r"[^\w\s\u0600-\u06ff]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokenize(text: str) -> List[str]:
    return [tok for tok in TOKEN_RE.findall(_normalize_text(text)) if len(tok) >= 2]


def embed_query(query: str) -> List[float]:
    resp = requests.post(
        EMBED_URL,
        json={"model": EMBED_MODEL, "input": [query]},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["embeddings"][0]


def _extract_keywords(text: str, min_len: int = 3) -> List[str]:
    words = _tokenize(text)
    stopwords = {
        "apa", "bagaimana", "bagi", "dalam", "dengan", "dan", "dari", "di",
        "ke", "kepada", "oleh", "pada", "para", "sebagai", "sebab", "sesuai",
        "untuk", "yang", "atau", "itu", "ini", "cara", "tentang", "agar",
        "maka", "serta", "karena", "adalah", "ada", "berapa", "siapa",
        "what", "how", "the", "and", "of", "to", "in", "for", "is", "are",
    }
    return [w for w in words if len(w) >= min_len and w not in stopwords]


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
    "bidah": {"bidah", "bid'ah", "bidaah"},
    "bid'ah": {"bidah", "bid'ah", "bidaah"},
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


def _build_query_terms(query: str) -> List[str]:
    keywords = _extract_keywords(query)
    terms: List[str] = []
    seen = set()
    for kw in keywords:
        expanded = QUERY_EXPANSIONS.get(kw, {kw})
        for term in expanded:
            if term not in seen:
                terms.append(term)
                seen.add(term)
    return terms or keywords or [query]


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


def _match_group(text: str, group: List[str]) -> bool:
    haystack = _normalize_text(text)
    return any(term in haystack for term in group)


def _count_group_hits(text: str, groups: List[List[str]]) -> int:
    return sum(1 for group in groups if _match_group(text, group))


def _canonical_group_label(group: List[str]) -> str:
    return group[0] if group else ""


def _load_lexical_index() -> Optional[Dict]:
    global _lexical_index
    if _lexical_index is not None:
        return _lexical_index

    with _lexical_index_lock:
        if _lexical_index is not None:
            return _lexical_index

        index_path = os.path.join(JSON_DIR, "_index.json")
        index_data = _load_json(index_path, {"total_files": 0, "languages": {}, "files": []})
        try:
            stat = os.stat(index_path)
        except FileNotFoundError:
            return None

        cached = _load_pickle(LEXICAL_INDEX_PATH)
        if (
            isinstance(cached, dict)
            and cached.get("version") == LEXICAL_INDEX_VERSION
            and cached.get("source_index_mtime_ns") == stat.st_mtime_ns
            and cached.get("source_index_size") == stat.st_size
        ):
            _lexical_index = cached
            return _lexical_index

        docs: Dict[str, Dict] = {}
        postings: Dict[str, Dict[str, int]] = defaultdict(dict)
        doc_lengths: Dict[str, int] = {}
        total_docs = 0
        total_terms = 0
        total_books = 0

        for record in index_data.get("files", []):
            if not bool(record.get("ingest_ready", True)):
                continue
            json_path = resolve_index_json_path(record, JSON_DIR)
            if not os.path.exists(json_path):
                continue
            try:
                chunks = chunk_book(json_path)
            except Exception:
                continue
            if not chunks:
                continue

            total_books += 1
            for chunk in chunks:
                payload = dict(chunk.get("metadata", {}))
                payload["text"] = chunk.get("text", "")
                payload["json_path"] = record.get("json_path", "")
                payload["quality_status"] = record.get("quality_status", "ok")
                payload["review_status"] = record.get("review_status", "approved_auto")
                payload["review_required"] = bool(record.get("review_required", False))
                payload["review_route"] = record.get("review_route", "auto")
                payload["reviewed_by"] = record.get("reviewed_by", "")
                payload["reviewed_at"] = record.get("reviewed_at", "")
                payload["review_note"] = record.get("review_note", "")
                payload["ingest_ready"] = bool(record.get("ingest_ready", True))
                payload["source_path"] = record.get("source_path", "")
                payload["source_hash"] = record.get("source_hash", "")
                payload["source_relpath"] = record.get("source_relpath", "")
                payload["source_ext"] = infer_source_ext(record)
                payload["source_type"] = infer_source_type(record)
                payload["document_type"] = infer_document_type(record)

                text = f"{payload.get('title', '')} {payload['text']}"
                tokens = _tokenize(text)
                if not tokens:
                    continue

                doc_id = point_id_for_hash(chunk_hash(chunk))
                tf = Counter(tokens)
                docs[doc_id] = {
                    "payload": payload,
                    "length": len(tokens),
                }
                doc_lengths[doc_id] = len(tokens)
                total_docs += 1
                total_terms += len(tokens)
                for term, freq in tf.items():
                    postings[term][doc_id] = freq

        avgdl = total_terms / max(total_docs, 1)
        cached = {
            "version": LEXICAL_INDEX_VERSION,
            "source_index_mtime_ns": stat.st_mtime_ns,
            "source_index_size": stat.st_size,
            "source_index_path": index_path,
            "doc_count": total_docs,
            "book_count": total_books,
            "avgdl": avgdl,
            "docs": docs,
            "doc_lengths": doc_lengths,
            "postings": dict(postings),
        }
        try:
            _save_pickle(LEXICAL_INDEX_PATH, cached)
        except Exception:
            pass
        _lexical_index = cached
        return _lexical_index


def _bm25_candidates(query_terms: List[str]) -> Tuple[Dict[str, float], Dict[str, List[str]]]:
    lexical_index = _load_lexical_index()
    if not lexical_index:
        return {}, {}

    doc_count = max(int(lexical_index.get("doc_count", 0) or 0), 1)
    avgdl = float(lexical_index.get("avgdl", 0.0) or 0.0) or 1.0
    postings = lexical_index.get("postings", {})
    doc_lengths = lexical_index.get("doc_lengths", {})

    scores: Dict[str, float] = defaultdict(float)
    matched_terms: Dict[str, List[str]] = defaultdict(list)

    for term in query_terms:
        term_postings = postings.get(term)
        if not term_postings:
            continue
        df = len(term_postings)
        idf = math.log(1.0 + ((doc_count - df + 0.5) / (df + 0.5)))
        for doc_id, tf in term_postings.items():
            dl = int(doc_lengths.get(doc_id, 0) or 0)
            denom = tf + BM25_K1 * (1.0 - BM25_B + BM25_B * (dl / avgdl))
            score = idf * (tf * (BM25_K1 + 1.0)) / max(denom, 1e-9)
            scores[doc_id] += score
            if term not in matched_terms[doc_id]:
                matched_terms[doc_id].append(term)

    return scores, matched_terms


def _score_candidate(candidate: Dict, query: str, query_groups: List[List[str]], query_terms: List[str]) -> Dict:
    payload = candidate.get("payload", {})
    title = payload.get("title", "")
    text = candidate.get("text", "")
    haystack = f"{title} {text}"
    normalized_haystack = _normalize_text(haystack)

    dense_score = float(candidate.get("dense_score", 0.0) or 0.0)
    bm25_score = float(candidate.get("bm25_score", 0.0) or 0.0)
    bm25_norm = bm25_score / (bm25_score + 4.0) if bm25_score > 0 else 0.0

    concept_hits = _count_group_hits(haystack, query_groups)
    matched_concepts = [_canonical_group_label(group) for group in query_groups if _match_group(haystack, group)]
    title_hits = _count_group_hits(title, query_groups)
    query_term_hits = sum(1 for term in query_terms if term in normalized_haystack)

    title_bonus = min(0.08, 0.03 * title_hits)
    lexical_bonus = 0.28 * bm25_norm
    concept_bonus = 0.14 * (concept_hits / max(len(query_groups), 1)) if query_groups else 0.0
    query_bonus = min(0.04, 0.01 * query_term_hits)
    short_penalty = 0.10 if len(text) < 50 else 0.0
    hyperlink_penalty = 0.05 if "HYPERLINK" in text else 0.0
    mismatch_penalty = 0.0
    if len(query_groups) >= 2 and concept_hits == 0:
        mismatch_penalty += 0.10
    elif len(query_groups) >= 2 and concept_hits == 1:
        mismatch_penalty += 0.04

    final_score = (
        0.56 * dense_score
        + lexical_bonus
        + concept_bonus
        + title_bonus
        + query_bonus
        - short_penalty
        - hyperlink_penalty
        - mismatch_penalty
    )

    score_components = {
        "dense_score": round(dense_score, 4),
        "bm25_score": round(bm25_score, 4),
        "bm25_norm": round(bm25_norm, 4),
        "concept_hits": concept_hits,
        "concept_total": len(query_groups),
        "matched_concepts": matched_concepts,
        "title_hits": title_hits,
        "query_term_hits": query_term_hits,
        "title_bonus": round(title_bonus, 4),
        "lexical_bonus": round(lexical_bonus, 4),
        "concept_bonus": round(concept_bonus, 4),
        "query_bonus": round(query_bonus, 4),
        "short_penalty": round(short_penalty, 4),
        "hyperlink_penalty": round(hyperlink_penalty, 4),
        "mismatch_penalty": round(mismatch_penalty, 4),
        "final_score": round(final_score, 4),
    }

    candidate["score"] = round(final_score, 4)
    candidate["score_components"] = score_components
    candidate["matched_concepts"] = matched_concepts
    candidate["query"] = query
    return candidate


def rerank(
    results: List[Dict],
    query: str,
    bm25_scores: Optional[Dict[str, float]] = None,
    matched_terms: Optional[Dict[str, List[str]]] = None,
) -> List[Dict]:
    query_groups = _build_query_groups(query)
    query_terms = _build_query_terms(query)

    if bm25_scores is None or matched_terms is None:
        bm25_scores, matched_terms = _bm25_candidates(query_terms)

    for r in results:
        doc_id = str(r["id"])
        r["dense_score"] = float(r.get("dense_score", 0.0) or 0.0)
        if doc_id in bm25_scores:
            r["bm25_score"] = bm25_scores[doc_id]
            r["bm25_terms"] = matched_terms.get(doc_id, [])
        else:
            r["bm25_score"] = 0.0
            r["bm25_terms"] = []
        _score_candidate(r, query, query_groups, query_terms)

    results.sort(key=lambda r: r["score"], reverse=True)
    return results


def _merge_candidate(candidate_map: Dict[str, Dict], doc_id: str, payload: Dict, text: str = "") -> Dict:
    current = candidate_map.get(doc_id)
    if current is None:
        current = {
            "id": doc_id,
            "text": text or payload.get("text", ""),
            "payload": payload,
            "dense_score": 0.0,
            "bm25_score": 0.0,
            "bm25_terms": [],
        }
        candidate_map[doc_id] = current
    else:
        if text and not current.get("text"):
            current["text"] = text
        if payload and not current.get("payload"):
            current["payload"] = payload
    return current


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
            doc_id = str(hit.id)
            candidate = _merge_candidate(candidate_map, doc_id, hit.payload or {}, text=(hit.payload or {}).get("text", ""))
            if hit.score > candidate.get("dense_score", 0.0):
                candidate["dense_score"] = float(hit.score)

    lexical_index = _load_lexical_index()
    bm25_scores = {}
    bm25_matched_terms = {}
    if lexical_index:
        query_terms = _build_query_terms(query)
        bm25_scores, bm25_matched_terms = _bm25_candidates(query_terms)
        top_lexical = heapq.nlargest(
            max(RETRIEVAL_CANDIDATES * 4, top_k * 8, 50),
            bm25_scores.items(),
            key=lambda item: item[1],
        )
        docs = lexical_index.get("docs", {})
        for doc_id, score in top_lexical:
            doc = docs.get(doc_id)
            if not doc:
                continue
            payload = doc.get("payload", {})
            candidate = _merge_candidate(candidate_map, doc_id, payload, text=payload.get("text", ""))
            if score > candidate.get("bm25_score", 0.0):
                candidate["bm25_score"] = float(score)
            candidate["bm25_terms"] = bm25_matched_terms.get(doc_id, candidate.get("bm25_terms", []))

    results = list(candidate_map.values())
    if not results:
        return []

    results = rerank(results, query, bm25_scores=bm25_scores if lexical_index else None, matched_terms=bm25_matched_terms if lexical_index else None)
    return results[:top_k]
