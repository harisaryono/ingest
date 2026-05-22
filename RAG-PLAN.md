# RAG Plan - Buku Islam JSON to Search/Q&A

## Ringkasan

Repo ini sekarang berisi pipeline RAG lokal untuk koleksi buku Islam yang sudah dikonversi ke JSON.

Status data saat ini:
- `315` buku
- `7068` halaman total
- sekitar `36.4 MB` data JSON sumber
- bahasa terindeks: `id`, `en`, dan `ru`

Tujuan sistem:
- user mencari atau bertanya dalam bahasa Indonesia
- sistem menampilkan cuplikan relevan dan, bila perlu, jawaban ringkas berbasis referensi
- hasil harus bisa ditelusuri ke buku dan halaman sumber

## Layout Repo

```
./
├── convert_to_json.py
├── ../DATABASE/json_output/   # runtime data hasil konversi, di luar repo
├── ../DATABASE/qdrant_db/     # runtime data vector store lokal, di luar repo
├── rag/
│   ├── api.py
│   ├── chunker.py
│   ├── config.py
│   ├── dedupe_qdrant_storage.py
│   ├── embeddings.py
│   ├── generator.py
│   ├── ingest.py
│   ├── ingest_common.py
│   ├── ingest_id.py
│   ├── retriever.py
│   ├── run_api.sh
│   ├── run_ingest.sh
│   └── static/index.html
└── RAG-PLAN.md
```

Catatan:
- `../DATABASE/json_output/` dan `../DATABASE/qdrant_db/` sengaja tidak dipush ke git
- file kerja yang dipush adalah kode, script, dan dokumentasi
- `rag/run_api.sh` adalah launcher lokal untuk menyajikan UI search di `http://127.0.0.1:8000`

## Arsitektur Runtime

```
[User] -> FastAPI -> Retriever -> Context -> Generator -> Answer/Sources
                     ↘ /search langsung menampilkan cuplikan
```

Komponen runtime:
- Embedding: `nomic-embed-text` via Ollama
- Vector store: Qdrant lokal dengan path repo-relative
- Generator lokal: `qwen3:4b` via Ollama
- Generator besar: Lease Coordinator untuk model `gpt-oss-120b`
- Frontend: HTML + vanilla JS

## Keputusan Arsitektural

| Komponen | Pilihan saat ini | Catatan |
|----------|------------------|---------|
| Embedding | `nomic-embed-text` via Ollama | dipakai untuk dokumen dan query |
| Vector store | Qdrant lokal | disimpan di `../DATABASE/qdrant_db` |
| Retrieval | Dense retrieval + rerank ringan | ada query expansion untuk kata kunci penting |
| Generator lokal | `qwen3:4b` via Ollama | fallback cepat |
| Generator besar | Lease Coordinator | untuk pertanyaan yang butuh kualitas lebih tinggi |
| API backend | FastAPI | endpoint `/search`, `/ask`, `/books`, `/stats`, `/health` |
| UI | Search-only page | fokus ke hasil pencarian dan cuplikan |
| Bahasa | Default `id`, opsi `all` | metadata bahasa dipakai untuk filter |

## Konfigurasi Inti

File konfigurasi: [`rag/config.py`](/media/harry/DATA120B/GIT/INGEST/rag/config.py)

Konstanta yang dipakai sekarang:

```python
OLLAMA_BASE = "http://127.0.0.1:11434"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen3:4b"

GENERATION_BACKEND = "auto"
LEASE_COORDINATOR_URL = "http://127.0.0.1:9000/chat/completions"
LEASE_MODEL = "gpt-oss-120b"

QDRANT_PATH = "<repo>/../DATABASE/qdrant_db"
INGEST_STATE_PATH = "<repo>/../DATABASE/qdrant_db/ingest_state.json"
JSON_DIR = "<repo>/../DATABASE/json_output"
DATABASE_DIR = "<repo>/../DATABASE"
COLLECTION_NAME = "buku_islam"
VECTOR_DIM = 768

DEFAULT_TOP_K = 5
RETRIEVAL_CANDIDATES = 20
RETRIEVAL_CANDIDATES_PER_QUERY = 12

CHUNK_MAX_CHARS = 500
CHUNK_OVERLAP = 50
CHUNK_MIN_CHARS = 80

EMBED_BATCH_SIZE = 64
INGEST_BATCH_SIZE = 128
EMBED_RETRY_COUNT = 3
EMBED_TIMEOUT = 60
```

## Chunking

File: [`rag/chunker.py`](/media/harry/DATA120B/GIT/INGEST/rag/chunker.py)

Strategi yang dipakai:
- page dijadikan kandidat chunk
- page yang terlalu panjang dipecah dengan overlap
- page pendek yang noise di-skip
- metadata buku dan halaman dibawa ke tiap chunk

Metadata chunk yang disimpan:

```python
{
    "book_id": "id-tata-cara-praktis-wudu",
    "filename": "id-tata-cara-praktis-wudu.txt",
    "title": "Tata Cara Praktis Wudhu",
    "language": "id",
    "page_start": 3,
    "page_end": 4,
    "chunk_idx": 0
}
```

## Embedding

File: [`rag/embeddings.py`](/media/harry/DATA120B/GIT/INGEST/rag/embeddings.py)

Fungsi utama:
- `embed_texts(texts)`
- `embed_query(query)`

Implementasi memakai Ollama embed API:

```http
POST /api/embed
{"model": "nomic-embed-text", "input": [...]}
```

## Ingestion

File: [`rag/ingest.py`](/media/harry/DATA120B/GIT/INGEST/rag/ingest.py)

Perilaku ingest saat ini:
- membaca JSON dari `../DATABASE/json_output/`
- chunking per buku
- embedding batch
- upsert ke Qdrant lokal
- state disimpan di `../DATABASE/qdrant_db/ingest_state.json`
- ingest bersifat idempotent
- rerun tidak menambah data ganda untuk chunk yang sama
- jika file berubah, data lama untuk buku itu dibersihkan lalu diinsert ulang
- progress disimpan per batch agar proses bisa resume

Komponen bantu:
- [`rag/ingest_common.py`](/media/harry/DATA120B/GIT/INGEST/rag/ingest_common.py)
- [`rag/ingest_id.py`](/media/harry/DATA120B/GIT/INGEST/rag/ingest_id.py)
- [`rag/dedupe_qdrant_storage.py`](/media/harry/DATA120B/GIT/INGEST/rag/dedupe_qdrant_storage.py)

## Retrieval

File: [`rag/retriever.py`](/media/harry/DATA120B/GIT/INGEST/rag/retriever.py)

Retrieval sekarang bukan ChromaDB. Yang dipakai:
- Qdrant local client
- query embedding
- beberapa query variant untuk expansion
- rerank ringan berbasis:
  - kecocokan judul
  - cakupan konsep query
  - penalti untuk chunk terlalu pendek
  - penalti noise

Alur:
1. embed query
2. ambil kandidat dari Qdrant
3. rerank hasil
4. return top-k

Endpoint `/search` memakai retrieval ini tanpa LLM, sehingga aman untuk tampilkan cuplikan.

## Generator

File: [`rag/generator.py`](/media/harry/DATA120B/GIT/INGEST/rag/generator.py)

Mode yang tersedia:
- `local` -> Ollama `qwen3:4b`
- `large` -> Lease Coordinator
- `auto` -> pilih otomatis
- `search_only` -> tampilkan raw chunks tanpa generasi jawaban

Mode `strict` membatasi jawaban hanya dari referensi yang diambil dari retrieval.

## API

File: [`rag/api.py`](/media/harry/DATA120B/GIT/INGEST/rag/api.py)

Endpoint yang aktif:

| Endpoint | Method | Fungsi |
|----------|--------|--------|
| `/health` | GET | cek service + Qdrant |
| `/stats` | GET | jumlah buku, halaman, point, bahasa |
| `/search` | GET | search chunk + skor + payload |
| `/ask` | POST | retrieval + generasi jawaban |
| `/books` | GET | daftar buku |
| `/books/{book_id}` | GET | metadata buku |
| `/books/{book_id}/pages/{page_num}` | GET | isi halaman tertentu |
| `/debug/retrieve` | POST | raw retrieval untuk debugging |

Static site dimount di root sehingga [`rag/static/index.html`](/media/harry/DATA120B/GIT/INGEST/rag/static/index.html) bisa dibuka langsung dari server FastAPI.

## Frontend

File: [`rag/static/index.html`](/media/harry/DATA120B/GIT/INGEST/rag/static/index.html)

UI saat ini:
- search-only
- input query
- pilihan bahasa
- pilihan top-k
- daftar hasil berisi score, title, book_id, halaman, filename, dan cuplikan

UI ini memang sengaja sederhana untuk fokus ke kualitas retrieval.

## Tahap Kerja

1. Ingest data JSON ke Qdrant lokal
2. Pastikan rerun aman dan tidak duplikatif
3. Evaluasi kualitas retrieval dengan query nyata
4. Tambah tuning query expansion dan rerank bila perlu
5. Pakai `/ask` bila ingin jawaban generatif berbasis referensi
6. Pertahankan `../DATABASE/json_output/` dan `../DATABASE/qdrant_db/` sebagai runtime data lokal di luar git

## Catatan Operasional

- `../DATABASE/json_output/` adalah sumber data yang dipakai ingest
- `../DATABASE/qdrant_db/` adalah state lokal vector store
- keduanya tidak perlu dipush ke GitHub
- perubahan yang dipush hanya kode, script, dan dokumentasi
- dedupe chunk dibatasi per identitas buku/halaman/chunk agar sitasi tetap aman
