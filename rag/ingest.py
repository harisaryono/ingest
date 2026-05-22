#!/usr/bin/env python3
"""RAG Ingest - Qdrant local mode."""

import json
import os
import time
from typing import List

from qdrant_client import QdrantClient

from config import COLLECTION_NAME, INGEST_BATCH_SIZE, INGEST_STATE_PATH, JSON_DIR, QDRANT_PATH
from chunker import chunk_book
from embeddings import embed_texts
from ingest_common import (
    build_points,
    chunk_hashes,
    bootstrap_state_from_collection,
    ensure_collection,
    finish_book,
    get_book_state,
    hash_file,
    iter_batches,
    load_state,
    release_book,
    reset_book_state,
    resume_batch_index,
    resolve_index_json_path,
    save_state,
    should_skip_book,
    commit_batch,
)


def log(msg: str) -> None:
    print(msg, flush=True)


def parse_languages() -> List[str]:
    raw = os.environ.get("INGEST_LANGUAGES", "id").strip()
    if not raw:
        return ["id"]
    langs = [part.strip().lower() for part in raw.split(",") if part.strip()]
    return langs or ["id"]


def main() -> None:
    selected_languages = parse_languages()
    log("=" * 50)
    log(f"RAG Ingestion - Qdrant ({', '.join(selected_languages)})")
    log("=" * 50)

    index_path = os.path.join(JSON_DIR, "_index.json")
    with open(index_path, "r", encoding="utf-8") as f:
        index_data = json.load(f)

    files = [f for f in index_data["files"] if f["language"] in selected_languages]
    limit_env = os.environ.get("INGEST_LIMIT_BOOKS", "").strip()
    if limit_env:
        try:
            limit = max(1, int(limit_env))
            files = files[:limit]
        except ValueError:
            log(f"Ignoring invalid INGEST_LIMIT_BOOKS={limit_env!r}")
    force_refresh = os.environ.get("INGEST_FORCE_REFRESH", "").strip().lower() in {"1", "true", "yes", "on"}
    total_files = len(files)
    log(f"Total books in scope: {total_files}")

    client = QdrantClient(path=QDRANT_PATH)
    collection_created = ensure_collection(client)
    skip_bootstrap = os.environ.get("INGEST_SKIP_BOOTSTRAP", "").strip().lower() in {"1", "true", "yes", "on"}
    if collection_created:
        log(f"Created collection '{COLLECTION_NAME}' (dim=768, cosine)")
        state = {"books": {}, "chunks": {}}
        save_state(state, INGEST_STATE_PATH)
    else:
        log(f"Using existing collection '{COLLECTION_NAME}'")

    if not collection_created:
        state = load_state(INGEST_STATE_PATH)
        if not skip_bootstrap and not state["books"] and not state["chunks"]:
            log("Bootstrapping ingest state from existing Qdrant collection.")
            state = bootstrap_state_from_collection(client)
            save_state(state, INGEST_STATE_PATH)
        elif skip_bootstrap and not state["books"] and not state["chunks"]:
            log("Skipping bootstrap; starting from empty ingest state.")

    if state["books"]:
        log(f"Loaded ingest state for {len(state['books'])} books.")
    else:
        log("No ingest state found; books will be refreshed as needed.")

    total_points_added = 0
    skipped_books = 0
    failed_books: List[str] = []
    overall_start = time.time()

    for idx, book_info in enumerate(files, 1):
        filename = book_info["filename"]
        book_id = book_info.get("book_id") or os.path.splitext(filename)[0]
        json_path = resolve_index_json_path(book_info, JSON_DIR)
        book_start = time.time()

        if not os.path.exists(json_path):
            log(f"[{idx:03d}/{total_files}] SKIP   missing source {json_path}")
            continue

        source_hash = hash_file(json_path)
        book_state = get_book_state(state, book_id)

        if not force_refresh and should_skip_book(book_state, source_hash):
            skipped_books += 1
            log(f"[{idx:03d}/{total_files}] SKIP   unchanged {filename}")
            continue

        if force_refresh and book_state:
            log(f"[{idx:03d}/{total_files}] REFRESH {book_info['title'][:50]} (forced)")
            try:
                release_book(state, client, book_id)
                save_state(state, INGEST_STATE_PATH)
            except Exception as e:
                log(f"[{idx:03d}/{total_files}] ERROR  release: {filename} - {e}")
                failed_books.append(filename)
                continue
            reset_book_state(state, book_id, source_hash)
            book_state = get_book_state(state, book_id)
        elif book_state and book_state.get("source_hash") != source_hash:
            log(f"[{idx:03d}/{total_files}] REFRESH {book_info['title'][:50]} (source changed)")
            try:
                release_book(state, client, book_id)
                save_state(state, INGEST_STATE_PATH)
            except Exception as e:
                log(f"[{idx:03d}/{total_files}] ERROR  release: {filename} - {e}")
                failed_books.append(filename)
                continue
            reset_book_state(state, book_id, source_hash)
            book_state = get_book_state(state, book_id)
        elif not book_state:
            log(f"[{idx:03d}/{total_files}] NEW     {book_info['title'][:50]}")
            reset_book_state(state, book_id, source_hash)
            book_state = get_book_state(state, book_id)
        else:
            log(f"[{idx:03d}/{total_files}] RESUME  {book_info['title'][:50]}")

        try:
            chunk_start = time.time()
            chunks = chunk_book(json_path)
            chunk_t = time.time() - chunk_start
        except Exception as e:
            log(f"[{idx:03d}/{total_files}] ERROR  chunk: {filename} - {e}")
            failed_books.append(filename)
            continue

        if not chunks:
            finish_book(state, book_id)
            save_state(state, INGEST_STATE_PATH)
            elapsed = time.time() - book_start
            log(
                f"[{idx:03d}/{total_files}] EMPTY  chunk={0.0:.1f}s "
                f"total={elapsed:.1f}s"
            )
            continue

        start_batch = resume_batch_index(book_state, source_hash)
        batch_iter = list(iter_batches(chunks, INGEST_BATCH_SIZE))
        total_batches = len(batch_iter)

        book_points_added = 0
        embed_t = 0.0
        upsert_t = 0.0

        try:
            for batch_index, batch in enumerate(batch_iter):
                if batch_index < start_batch:
                    continue

                batch_hashes = []
                batch_unique = []
                batch_seen = set()
                for chunk, chunk_hash in zip(batch, chunk_hashes(batch)):
                    if chunk_hash in batch_seen:
                        continue
                    batch_seen.add(chunk_hash)
                    batch_hashes.append(chunk_hash)
                    batch_unique.append(chunk)

                book_seen = set(book_state.get("chunk_hashes", []))
                new_chunks = []
                for chunk, chunk_hash in zip(batch_unique, batch_hashes):
                    if chunk_hash in book_seen:
                        continue
                    if chunk_hash in state["chunks"]:
                        continue
                    new_chunks.append(chunk)

                if new_chunks:
                    texts = [c["text"] for c in new_chunks]
                    embed_start = time.time()
                    embeddings = embed_texts(texts)
                    embed_t += time.time() - embed_start

                    points = build_points(new_chunks, embeddings)
                    upsert_start = time.time()
                    client.upsert(collection_name=COLLECTION_NAME, points=points)
                    upsert_t += time.time() - upsert_start
                    book_points_added += len(new_chunks)

                commit_batch(state, book_id, source_hash, batch_hashes, batch_index)
                save_state(state, INGEST_STATE_PATH)

                log(
                    f"[{idx:03d}/{total_files}] BATCH  {batch_index + 1}/{total_batches} "
                    f"new={len(new_chunks)} existing={len(batch_hashes) - len(new_chunks)}"
                )

        except Exception as e:
            log(f"[{idx:03d}/{total_files}] ERROR  ingest: {filename} - {e}")
            failed_books.append(filename)
            continue

        finish_book(state, book_id)
        save_state(state, INGEST_STATE_PATH)

        total_points_added += book_points_added
        elapsed = time.time() - book_start
        elapsed_total = time.time() - overall_start
        log(
            f"[{idx:03d}/{total_files}] OK     chunk={chunk_t:.1f}s "
            f"embed={embed_t:.1f}s upsert={upsert_t:.1f}s "
            f"new={book_points_added} total={elapsed:.1f}s "
            f"({elapsed_total/60:.0f}min total)"
        )

    overall_elapsed = time.time() - overall_start
    log(
        f"\nDONE in {overall_elapsed/60:.1f} min - "
        f"{total_points_added} new points, {skipped_books} skipped, {len(failed_books)} failed"
    )


if __name__ == "__main__":
    main()
