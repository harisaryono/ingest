# Rencana RAG — Buku Islam JSON → Q&A dengan Referensi

## Ringkasan

Membangun sistem RAG (Retrieval-Augmented Generation) di atas 315 buku Islam (~7068 halaman, 42MB JSON) yang sudah dikonversi.   
User bertanya dalam bahasa Indonesia, sistem menjawab ringkas **hanya dari referensi** dengan sumber buku + halaman.

---

## Arsitektur

```
[User] → FastAPI → Retriever (ChromaDB where language=X → top-20 → rerank → top-5) → Context → Generator → Answer + Sources
                          ↕                                                                    ↕
                   Vector Store (ChromaDB)                                             ┌──────────────────┐
                          ↕                                                           │  Mode: local      │ → qwen3:4b (Ollama)
                  Embedding: nomic-embed-text via Ollama                              │  Mode: large/auto  │ → Lease Coordinator
                                                                                      └──────────────────┘
```

---

## Keputusan Arsitektural

| Komponen | Pilihan | Alasan |
|----------|---------|--------|
| Embedding | `nomic-embed-text` via Ollama | gratis, multilingual, ringan (~274MB) |
| Vector Store | ChromaDB | persistent, zero-config, Python native |
| Reranker | Sederhana (skor + keyword + bonus judul) | sampai budget untuk reranker model |
| LLM Generator — **local** | `qwen3:4b` via Ollama | sudah terinstall, support Indonesia, untuk query ringan |
| LLM Generator — **large** | Model besar via **Lease Coordinator** | kualitas tinggi untuk pertanyaan serius/agama; API key dikelola lease coordinator |
| Mode Generator | **Strict RAG** — hanya rangkum konteks | cegah halusinasi, tidak pakai pengetahuan model |
| API Backend | FastAPI | user sudah familiar |
| Frontend | HTML form sederhana dulu (`static/index.html`) | fokus ke kualitas retrieval dulu |
| Filter Bahasa | **Default `id`**, opsi `all` | via parameter request |

---

## Langkah Implementasi

### 1. Persiapan Environment

```bash
python3 -m venv ~/venv/rag-buku
source ~/venv/rag-buku/bin/activate
pip install chromadb fastapi uvicorn requests numpy
```

Pull embedding model:
```bash
ollama pull nomic-embed-text   # ~274MB
```

### 2. Konfigurasi (`config.py`)

```python
OLLAMA_BASE = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen3:4b"

# Lease Coordinator (mode large/auto)
GENERATION_BACKEND = "auto"     # "ollama" | "lease" | "auto"
LEASE_COORDINATOR_URL = "http://127.0.0.1:9000/chat/completions"
LEASE_MODEL = "gpt-oss-120b"

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "buku_islam"
JSON_DIR = "./json_output"
DEFAULT_TOP_K = 5
RETRIEVAL_CANDIDATES = 20
CHUNK_MAX_CHARS = 2000
CHUNK_OVERLAP = 100
CHUNK_MIN_CHARS = 100
```

### 3. Chunker (`chunker.py`)

Input: file JSON buku → output: list chunk.

**Strategi chunking:**
1. Tiap page jadi calon chunk
2. Page > `CHUNK_MAX_CHARS` (2000) chars → split dengan overlap `CHUNK_OVERLAP` (100)
3. Page < `CHUNK_MIN_CHARS` (100) chars:
   - Jika berupa cover/daftar isi/noise → **skip**
   - Jika berupa judul bab (pendek, all caps, atau didahului angka romawi) → **prepend** ke page sesudahnya
   - Jika bukan judul bab → **append** ke page sebelumnya
4. Skip page yang hanya berisi metadata/cover (deteksi: judul saja, daftar isi HYPERLINK, dll.)

**Metadata per chunk:**
```python
{
    "book_id": "id-tata-cara-praktis-wudu",    # ID stabil untuk URL/filter
    "filename": "id-tata-cara-praktis-wudu.txt",
    "title": "Tata Cara Praktis Wudhu",
    "language": "id",
    "page_start": 3,
    "page_end": 4,    # atau 5 jika merge beberapa page
    "chunk_idx": 0
}
```

`book_id` = stem filename tanpa ekstensi. Lebih stabil dari filename, dipakai untuk URL dan filter per buku.

Contoh referensi output:
- 1 halaman: `[Nama Buku, hlm. 5]`
- Merge: `[Nama Buku, hlm. 3–4]`

### 4. Embedding (`embeddings.py`)

Fungsi `embed_texts(texts: list[str]) → list[list[float]]`:

```python
POST http://localhost:11434/api/embed
Body: {"model": "nomic-embed-text", "input": texts}
```

Support batch embedding untuk efisiensi.

### 5. Ingestion Pipeline (`ingest.py`)

1. Load `_index.json` → daftar semua file
2. Untuk setiap buku:
   - Baca JSON book → chunking via `chunker.py`
   - Embed chunks via `embeddings.py`
   - Upsert ke ChromaDB:
     - `documents` = teks chunk
     - `metadatas` = `{book_id, filename, title, language, page_start, page_end, chunk_idx}`
     - `ids` = `{book_id}_p{page_start:04d}_c{chunk_idx:03d}`
       (Pakai `book_id` bukan filename — hindari karakter aneh/spasi/slash)
3. Simpan ChromaDB persistent di `./chroma_db/`

### 6. Retrieval + Rerank (`retriever.py`)

Fungsi `retrieve(query, top_k=5, language="id")`:

```
1. Embed query → Ollama nomic-embed-text
2. ChromaDB similarity search dengan filter metadata:
   if language != "all":
     where = {"language": language}
   ambil RETRIEVAL_CANDIDATES (20) candidates
3. Rerank sederhana:
   - Base: normalize_score(chromadb_score)
     ChromaDB bisa mengembalikan "distance" atau "similarity".
     Fungsi aman: base = 1 / (1 + d) jika d adalah distance; langsung pakai jika sudah similarity.
   - +0.15 jika judul buku mengandung salah satu keyword query
   - -0.10 jika chunk < 50 chars (terlalu pendek)
   - -0.05 jika chunk mengandung HYPERLINK (noise)
4. Sort by final score → ambil top_k
5. Return: [{text, score, metadata}]
```

Tambahan: `normalize_score()` memastikan semakin tinggi = semakin relevan.

### 7. Generator (`generator.py`)

Fungsi `generate(query, context_chunks, strict=True, mode="auto")`:

**Arsitektur dua tingkat:**

```
generate() → pilih backend:
  "local"  → generate_local()  → Ollama qwen3:4b
  "large"  → generate_remote() → Lease Coordinator (model besar)
  "auto"   → pilih otomatis berdasarkan query (lihat logika di bawah)
```

**Abstraksi backend:**

```python
def generate(query, context_chunks, strict=True, mode="auto"):
    prompt = build_prompt(query, context_chunks, strict)

    if mode == "auto":
        mode = select_mode(query, strict)

    if mode == "local":
        return generate_local(prompt)
    elif mode == "large":
        return generate_remote(prompt, LEASE_MODEL)
    elif mode == "search_only":
        return format_chunks_only(context_chunks)  # tanpa LLM
```

**Logika `auto`:**

```text
Jika mode="search_only" →
  tampilkan raw chunks, tanpa LLM

Jika query pendek (< 50 chars) DAN top_k <= 5 DAN strict=false →
  pakai local (qwen3:4b)

Jika strict=true DAN query mengandung kata kunci hukum/hadis/khilaf/akidah →
  pakai lease coordinator (model besar)

Jika local gagal (timeout/error) →
  fallback ke lease coordinator

Jika lease coordinator gagal →
  fallback ke local
```

**Fungsi `generate_local(prompt)`:**

```python
POST http://localhost:11434/api/generate
Body: {"model": "qwen3:4b", "prompt": prompt, "stream": false}
```

**Fungsi `generate_remote(prompt, model)`:**

Kirim ke Lease Coordinator sebagai OpenAI-compatible chat completion:

```python
POST {LEASE_COORDINATOR_URL}  # e.g. http://127.0.0.1:9000/chat/completions
Body: {
    "model": LEASE_MODEL,         # "gpt-oss-120b"
    "messages": [
        {"role": "user", "content": prompt}
    ],
    "temperature": 0.1
}
```

Lease coordinator yang memilih provider, API key, dan menangani TTL/fencing/retry.  
Aplikasi RAG tidak menyimpan API key sama sekali.

**Prompt (strict):**

```
Anda adalah asisten RAG. Jawaban wajib hanya berdasarkan INFORMASI REFERENSI di bawah.

ATURAN WAJIB:
1. Jangan memakai pengetahuan di luar referensi.
2. Jangan menambah ayat, hadis, nama ulama, atau hukum jika tidak muncul dalam referensi.
3. Jangan menyimpulkan status hukum seperti wajib, sunnah, haram, makruh, bid'ah, sah, atau batal kecuali istilah itu disebut jelas dalam referensi.
4. Jika referensi tidak cukup menjawab, jawab:
   "Tidak ditemukan secara cukup dalam buku referensi."
5. Jawab ringkas maksimal 3 paragraf.
6. Setelah setiap poin penting, cantumkan rujukan: [Nama Buku, hlm. X–Y].
7. Jika ada perbedaan antar referensi, sebutkan perbedaannya secara netral.

INFORMASI REFERENSI:
[Buku: {title}, Halaman {page_start}–{page_end}]
{chunk_text}

[Buku: {title}, Halaman {page_start}–{page_end}]
{chunk_text}
...

PERTANYAAN: {query}

JAWABAN:
```

Jika `strict=False` (untuk eksplorasi), prompt dilembutkan sedikit tapi tetap wajib referensi.

### 8. API (`api.py`)

FastAPI endpoints:

| Endpoint | Method | Parameter | Response |
|----------|--------|-----------|----------|
| `/ask` | POST | `{"query", "top_k":5, "language":"id", "strict":true, "mode":"auto"}` | `{answer, sources, backend_used, mode}` |
| `/search` | GET | `?q=&top_k=5&language=id` | `{results: [{text, score, metadata}]}` — **mode aman tanpa LLM**, untuk pertanyaan sensitif |
| `/books` | GET | - | daftar semua buku dari index |
| `/books/{book_id}` | GET | - | metadata + halaman buku |
| `/books/{book_id}/pages/{n}` | GET | - | konten halaman spesifik |
| `/debug/retrieve` | POST | `{"query", "top_k":5, "language":"id"}` | raw chunks + skor (debug retrieval) |
| `/stats` | GET | - | `{total_books, total_pages, total_chunks, languages}` |
| `/health` | GET | - | status service + ChromaDB + Lease Coordinator |

**Parameter `mode` pada `/ask`:**

```text
"local"       → paksa qwen3:4b lokal
"large"       → paksa lease coordinator (model besar)
"auto"        → sistem memilih (default)
"search_only" → tanpa LLM, tampilkan raw chunks
```

### 9. Frontend (`static/index.html`)

HTML satu halaman dengan vanilla JS:
- Form input: textbox pertanyaan + dropdown language (id/all) + checkbox strict
- Tombol "Tanya"
- Output area: jawaban + daftar sumber (buku, halaman) yang bisa diklik → buka `/books/{book_id}/pages/{n}`
- Bagian debug: toggle untuk lihat raw chunks retrieval

---

## Alur Query Contoh

**Request:**
```json
{
  "query": "tata cara wudhu",
  "top_k": 5,
  "language": "id",
  "strict": true,
  "mode": "auto"
}
```

**Proses:**
1. Embed query → ChromaDB cari top-20 dengan `where={"language": "id"}`
2. Rerank (normalize_score + bonus judul + penalty noise) → top-5
3. Prompt ke Qwen3:4b → generate jawaban
4. Parse output → answer + sources

**Response:**
```json
{
  "answer": "Tata cara wudhu: 1) Niat, 2) Membasuh muka, 3) Membasuh tangan sampai siku, 4) Mengusap kepala, 5) Membasuh kaki sampai mata kaki. [Tata Cara Praktis Wudhu, hlm. 1–2] Disunnahkan berkumur dan istinsyaq. [Ringkasan Tata Cara Salat, hlm. 3]",
  "backend_used": "ollama",
  "mode": "local",
  "sources": [
    {"title": "Tata Cara Praktis Wudhu", "page_start": 1, "page_end": 2, "book_id": "id-tata-cara-praktis-wudu", "filename": "id-tata-cara-praktis-wudu.txt"},
    {"title": "Ringkasan Tata Cara Salat", "page_start": 3, "page_end": 3, "book_id": "id-ringkasan-tata-cara-salat", "filename": "id-ringkasan-tata-cara-salat.txt"}
  ]
}
```

---

## Struktur Direktori Final

```
./
├── json_output/              # hasil konversi sebelumnya (done)
│   ├── _index.json
│   ├── _covers/
│   └── _empty/
├── qdrant_db/                # database vektor lokal Qdrant
├── rag/
│   ├── config.py             # konstanta & konfigurasi
│   ├── chunker.py            # page → chunk + metadata
│   ├── embeddings.py         # Ollama embed API wrapper
│   ├── ingest.py             # indexing pipeline
│   ├── retriever.py          # search + rerank
│   ├── generator.py          # prompt builder + LLM call
│   ├── api.py                # FastAPI server
│   ├── static/
│   │   └── index.html        # frontend sederhana
│   └── README.md             # dokumentasi RAG
├── convert_to_json.py        # converter sebelumnya (done)
└── README.md
```

---

## Catatan & Pertimbangan

- **Anti-halusinasi**: prompt strict, filter chunk pendek/noise, rerank dengan penalty
- **Multi-bahasa**: embedding nomic-embed-text support EN/ID/AR, filter language via metadata
- **Debug-first**: endpoint `/debug/retrieve` untuk troubleshooting retrieval sebelum LLM
- **Scaling**: ~7000 chunks, embed sekali ~5-10 menit, retrieval <100ms
- **Upgrade path**: chunker.py → ganti chunk strategy; embeddings.py → ganti ke model lain; retriever.py → tambah reranker sungguhan; generator.py → ganti LLM ke API eksternal atau tambah provider di Lease Coordinator
- **Update buku**: jalankan `ingest.py` ulang (ChromaDB upsert berdasarkan ID)
- **Keamanan API key**: API key model besar tidak disimpan di aplikasi RAG; seluruh manajemen key, TTL, fencing, retry, dan rotasi provider ditangani Lease Coordinator
- **Fallback**: jika lease coordinator gagal/tidak terjangkau, generator otomatis fallback ke Ollama lokal; jika Ollama lokal gagal, fallback ke lease coordinator

---

## Tahap Pengembangan

1. **Tahap 1** (prioritas): `config.py` + `chunker.py` + `embeddings.py` + `ingest.py` → indexing berhasil
2. **Tahap 2**: `retriever.py` + `generator.py` → RAG pipeline selesai, test via CLI
3. **Tahap 3**: `api.py` + `static/index.html` → web interface
4. **Tahap 4** (jika perlu): evaluasi kualitas retrieval, tuning chunking & rerank
