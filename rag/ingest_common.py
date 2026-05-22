import hashlib
import json
import os
import tempfile
import time
import uuid
from typing import Dict, Iterator, List

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from config import COLLECTION_NAME, INGEST_STATE_PATH, VECTOR_DIM


def hash_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def infer_source_type(record: Dict) -> str:
    source_type = str(record.get("source_type", "") or "").strip().lower()
    if source_type:
        return source_type

    source_ext = str(record.get("source_ext", "") or "").strip().lower()
    if source_ext in {".htm", ".html"}:
        return "html"
    if source_ext:
        return source_ext.lstrip(".")

    source_path = str(record.get("source_path", "") or record.get("filename", "") or "").strip()
    if source_path:
        ext = os.path.splitext(source_path)[1].lower()
        if ext in {".htm", ".html"}:
            return "html"
        if ext:
            return ext.lstrip(".")
    return "unknown"


def infer_source_ext(record: Dict) -> str:
    source_ext = str(record.get("source_ext", "") or "").strip().lower()
    if source_ext:
        return source_ext

    source_path = str(record.get("source_path", "") or record.get("filename", "") or "").strip()
    if source_path:
        return os.path.splitext(source_path)[1].lower()
    return ""


def infer_document_type(record: Dict) -> str:
    document_type = str(record.get("document_type", "") or "").strip().lower()
    if document_type:
        return document_type
    return "html_document" if infer_source_type(record) == "html" else "book"


def chunk_identity_key(chunk: Dict) -> str:
    metadata = chunk.get("metadata", {})
    book_id = metadata.get("book_id", "")
    page_start = metadata.get("page_start", "")
    page_end = metadata.get("page_end", "")
    chunk_idx = metadata.get("chunk_idx", "")
    text = chunk.get("text", "")
    return f"{book_id}|{page_start}|{page_end}|{chunk_idx}|{text}"


def chunk_hash(chunk: Dict) -> str:
    return hashlib.sha256(chunk_identity_key(chunk).encode("utf-8")).hexdigest()


def point_id_for_hash(chunk_hash_value: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"rag-chunk:{chunk_hash_value}"))


def _empty_state() -> Dict[str, Dict]:
    return {"books": {}, "chunks": {}}


def normalize_state(raw: Dict | None) -> Dict[str, Dict]:
    if not isinstance(raw, dict):
        return _empty_state()
    if "books" in raw or "chunks" in raw:
        books = raw.get("books", {})
        chunks = raw.get("chunks", {})
    else:
        books = raw
        chunks = {}

    normalized = _empty_state()

    if isinstance(books, dict):
        for book_id, book_state in books.items():
            if not isinstance(book_state, dict):
                continue
            normalized["books"][book_id] = {
                "source_hash": book_state.get("source_hash", ""),
                "next_batch_index": int(book_state.get("next_batch_index", 0) or 0),
                "complete": bool(book_state.get("complete", False)),
                "chunk_hashes": list(book_state.get("chunk_hashes", [])),
                "point_count": int(book_state.get("point_count", 0) or 0),
                "updated_at": float(book_state.get("updated_at", 0.0) or 0.0),
            }

    if isinstance(chunks, dict):
        for chunk_hash, chunk_state in chunks.items():
            if not isinstance(chunk_state, dict):
                continue
            normalized["chunks"][chunk_hash] = {
                "point_id": chunk_state.get("point_id", point_id_for_hash(chunk_hash)),
                "ref_count": int(chunk_state.get("ref_count", 0) or 0),
                "first_seen_at": float(chunk_state.get("first_seen_at", 0.0) or 0.0),
            }

    return normalized


def load_state(path: str = INGEST_STATE_PATH) -> Dict[str, Dict]:
    if not os.path.exists(path):
        return _empty_state()
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        return _empty_state()
    return normalize_state(raw)


def save_state(state: Dict[str, Dict], path: str = INGEST_STATE_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=".ingest_state.",
        suffix=".json",
        dir=os.path.dirname(path),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False, sort_keys=True)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def ensure_collection(client: QdrantClient) -> bool:
    try:
        client.get_collection(COLLECTION_NAME)
        return False
    except Exception:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        )
        return True


def iter_batches(items: List[Dict], batch_size: int) -> Iterator[List[Dict]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def delete_point_ids(client: QdrantClient, point_ids: List[str], batch_size: int = 256) -> None:
    for i in range(0, len(point_ids), batch_size):
        client.delete(collection_name=COLLECTION_NAME, points_selector=point_ids[i : i + batch_size])


def release_book(state: Dict[str, Dict], client: QdrantClient, book_id: str) -> int:
    book_state = state["books"].get(book_id)
    if not book_state:
        return 0

    point_ids_to_delete: List[str] = []
    for chunk_hash in book_state.get("chunk_hashes", []):
        chunk_state = state["chunks"].get(chunk_hash)
        if not chunk_state:
            continue
        chunk_state["ref_count"] = max(0, int(chunk_state.get("ref_count", 0)) - 1)
        if chunk_state["ref_count"] <= 0:
            point_ids_to_delete.append(chunk_state["point_id"])
            del state["chunks"][chunk_hash]

    if point_ids_to_delete:
        delete_point_ids(client, point_ids_to_delete)

    del state["books"][book_id]
    return len(point_ids_to_delete)


def get_book_state(state: Dict[str, Dict], book_id: str) -> Dict:
    return state["books"].get(book_id, {})


def should_skip_book(book_state: Dict, source_hash: str) -> bool:
    return bool(book_state) and book_state.get("source_hash") == source_hash and bool(book_state.get("complete"))


def resume_batch_index(book_state: Dict, source_hash: str) -> int:
    if not book_state:
        return 0
    if book_state.get("source_hash") == source_hash and not book_state.get("complete"):
        return int(book_state.get("next_batch_index", 0) or 0)
    return 0


def reset_book_state(state: Dict[str, Dict], book_id: str, source_hash: str) -> None:
    state["books"][book_id] = {
        "source_hash": source_hash,
        "next_batch_index": 0,
        "complete": False,
        "chunk_hashes": [],
        "point_count": 0,
        "updated_at": time.time(),
    }


def commit_batch(
    state: Dict[str, Dict],
    book_id: str,
    source_hash: str,
    batch_hashes: List[str],
    batch_index: int,
) -> int:
    book_state = state["books"].setdefault(
        book_id,
        {
            "source_hash": source_hash,
            "next_batch_index": 0,
            "complete": False,
            "chunk_hashes": [],
            "point_count": 0,
            "updated_at": time.time(),
        },
    )

    seen = set(book_state.get("chunk_hashes", []))
    inserted = 0
    for chunk_hash in batch_hashes:
        if chunk_hash in seen:
            continue
        chunk_state = state["chunks"].setdefault(
            chunk_hash,
            {
                "point_id": point_id_for_hash(chunk_hash),
                "ref_count": 0,
                "first_seen_at": time.time(),
            },
        )
        chunk_state["ref_count"] = int(chunk_state.get("ref_count", 0)) + 1
        seen.add(chunk_hash)
        inserted += 1

    book_state["source_hash"] = source_hash
    book_state["chunk_hashes"] = list(seen)
    book_state["next_batch_index"] = batch_index + 1
    book_state["complete"] = False
    book_state["point_count"] = len(seen)
    book_state["updated_at"] = time.time()
    return inserted


def finish_book(state: Dict[str, Dict], book_id: str) -> None:
    if book_id not in state["books"]:
        return
    state["books"][book_id]["complete"] = True
    state["books"][book_id]["updated_at"] = time.time()


def bootstrap_state_from_collection(client: QdrantClient) -> Dict[str, Dict]:
    state = _empty_state()
    duplicate_ids: List[str] = []
    offset = None
    now = time.time()

    while True:
        points, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=None,
            limit=256,
            with_payload=True,
            with_vectors=False,
            offset=offset,
        )
        if not points:
            break

        for point in points:
            payload = point.payload or {}
            book_id = payload.get("book_id") or payload.get("filename") or "unknown"
            text = payload.get("text", "")
            chunk_identity = {
                "text": text,
                "metadata": {
                    "book_id": book_id,
                    "page_start": payload.get("page_start", ""),
                    "page_end": payload.get("page_end", ""),
                    "chunk_idx": payload.get("chunk_idx", ""),
                },
            }
            chunk_key = chunk_hash(chunk_identity)
            point_id = str(point.id)

            book_state = state["books"].setdefault(
                book_id,
                {
                    "source_hash": "",
                    "next_batch_index": 0,
                    "complete": False,
                    "chunk_hashes": [],
                    "point_count": 0,
                    "updated_at": now,
                },
            )
            if chunk_key not in book_state["chunk_hashes"]:
                book_state["chunk_hashes"].append(chunk_key)
                book_state["point_count"] = len(book_state["chunk_hashes"])

            if chunk_key not in state["chunks"]:
                state["chunks"][chunk_key] = {
                    "point_id": point_id,
                    "ref_count": 1,
                    "first_seen_at": now,
                }
            else:
                state["chunks"][chunk_key]["ref_count"] = int(state["chunks"][chunk_key]["ref_count"]) + 1
                duplicate_ids.append(point_id)

    if duplicate_ids:
        delete_point_ids(client, duplicate_ids)

    return state


def build_points(chunks: List[Dict], embeddings: List[List[float]]) -> List[PointStruct]:
    points: List[PointStruct] = []
    for chunk, vec in zip(chunks, embeddings):
        text = chunk["text"]
        chunk_hash_value = chunk_hash(chunk)
        metadata = chunk["metadata"]
        points.append(
            PointStruct(
                id=point_id_for_hash(chunk_hash_value),
                vector=vec,
                payload={"text": text, **metadata},
            )
        )
    return points


def chunk_hashes(chunks: List[Dict]) -> List[str]:
    return [chunk_hash(chunk) for chunk in chunks]
