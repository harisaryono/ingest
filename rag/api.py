from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict
import json
import os

from config import JSON_DIR, QDRANT_PATH, COLLECTION_NAME
from retriever import retrieve
from generator import generate, extract_sources
from ingest_common import resolve_index_json_path

app = FastAPI(title="RAG Buku Islam", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _load_index() -> Dict:
    index_path = os.path.join(JSON_DIR, "_index.json")
    with open(index_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _book_entry(book_id: str) -> Dict | None:
    for record in _load_index().get("files", []):
        record_book_id = os.path.splitext(record["filename"])[0]
        if record_book_id == book_id:
            return record
    return None


def _book_json_path(record: Dict) -> str:
    return resolve_index_json_path(record, JSON_DIR)


class AskRequest(BaseModel):
    query: str
    top_k: int = 5
    language: str = "id"
    strict: bool = True
    mode: str = "auto"


class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 5
    language: str = "id"


class AskResponse(BaseModel):
    answer: str
    backend_used: str
    mode: str
    sources: List[Dict]


class SourceInfo(BaseModel):
    title: str
    page_start: int
    page_end: int
    book_id: str
    filename: str


@app.get("/health")
def health():
    from qdrant_client import QdrantClient
    try:
        c = QdrantClient(path=QDRANT_PATH)
        info = c.get_collection(COLLECTION_NAME)
        qdrant_status = f"OK ({info.points_count} points)"
    except Exception as e:
        qdrant_status = f"ERROR: {e}"

    return {
        "status": "ok",
        "qdrant": qdrant_status,
    }


@app.get("/stats")
def stats():
    idx = _load_index()
    from qdrant_client import QdrantClient
    try:
        c = QdrantClient(path=QDRANT_PATH)
        info = c.get_collection(COLLECTION_NAME)
        point_count = info.points_count
    except Exception:
        point_count = 0

    return {
        "total_books": idx["total_files"],
        "total_pages": sum(r["total_pages"] for r in idx["files"]),
        "total_points_indexed": point_count,
        "languages": idx["languages"],
    }


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    context_chunks = retrieve(
        query=req.query,
        top_k=req.top_k,
        language=req.language,
    )

    answer, backend_used, mode_used, _chunks = generate(
        query=req.query,
        context_chunks=context_chunks,
        strict=req.strict,
        mode=req.mode,
    )

    sources = extract_sources(answer, context_chunks)

    return AskResponse(
        answer=answer,
        backend_used=backend_used,
        mode=mode_used,
        sources=sources,
    )


@app.get("/search")
def search(q: str, top_k: int = 5, language: str = "id"):
    results = retrieve(query=q, top_k=top_k, language=language)
    return {
        "query": q,
        "results": [
            {
                "text": r["text"],
                "score": r["score"],
                "payload": r["payload"],
            }
            for r in results
        ],
    }


@app.get("/books")
def list_books():
    return [{
        "book_id": os.path.splitext(r["filename"])[0],
        "filename": r["filename"],
        "json_filename": os.path.basename(resolve_index_json_path(r, JSON_DIR)),
        "title": r["title"],
        "language": r["language"],
        "total_pages": r["total_pages"],
    } for r in _load_index()["files"]]


@app.get("/books/{book_id}")
def book_detail(book_id: str):
    record = _book_entry(book_id)
    if not record:
        return JSONResponse({"error": "not found"}, status_code=404)

    json_path = _book_json_path(record)
    if not os.path.exists(json_path):
        return JSONResponse({"error": "not found"}, status_code=404)

    with open(json_path, "r", encoding="utf-8") as f:
        book = json.load(f)

    return {
        "book_id": book_id,
        "filename": book["filename"],
        "json_filename": os.path.basename(json_path),
        "title": book["title"],
        "language": book["language"],
        "total_pages": book["total_pages"],
        "pages": [p["page"] for p in book["pages"]],
    }


@app.get("/books/{book_id}/pages/{page_num}")
def page_content(book_id: str, page_num: int):
    record = _book_entry(book_id)
    if not record:
        return JSONResponse({"error": "not found"}, status_code=404)

    json_path = _book_json_path(record)
    if not os.path.exists(json_path):
        return JSONResponse({"error": "not found"}, status_code=404)

    with open(json_path, "r", encoding="utf-8") as f:
        book = json.load(f)

    for p in book["pages"]:
        if p["page"] == page_num:
            return {
                "book_id": book_id,
                "title": book["title"],
                "page": p["page"],
                "content": p["content"],
            }

    return JSONResponse({"error": "page not found"}, status_code=404)


@app.post("/debug/retrieve")
def debug_retrieve(req: RetrieveRequest):
    results = retrieve(query=req.query, top_k=req.top_k, language=req.language)
    return {
        "query": req.query,
        "results": [
            {
                "text": r["text"],
                "score": r["score"],
                "payload": r["payload"],
            }
            for r in results
        ],
    }


STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
if os.path.exists(STATIC_DIR):
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
