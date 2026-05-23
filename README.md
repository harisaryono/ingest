# INGEST

Pipeline RAG lokal untuk korpus buku Islam:

- konversi sumber ke JSON
- quality gate dan review status
- dedupe konten
- ingest ke Qdrant lokal
- search dan QA lewat FastAPI

## Prasyarat

- Python 3.11+
- `ollama` jalan di lokal
- Qdrant lokal via file store
- source data tersedia di luar repo

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Konfigurasi

Copy `.env.example` ke `.env` lalu sesuaikan bila perlu.

Environment yang paling penting:

- `DATABASE_DIR`
- `OLLAMA_BASE`
- `LEASE_COORDINATOR_URL`
- `HADITH_JSON_SOURCE_DIR`
- `INGEST_LANGUAGES`
- `INGEST_LIMIT_BOOKS`
- `INGEST_FORCE_REFRESH`
- `EMBED_BATCH_SIZE`
- `INGEST_BATCH_SIZE`

## Struktur Data Runtime

Data runtime disimpan di luar repo:

- `DATABASE/json_output/`
- `DATABASE/qdrant_db/`

## Alur Kerja

### 1. Konversi sumber ke JSON

Jika perlu, jalankan converter yang sesuai sumber data.

### 2. Import korpus Islamhouse

```bash
python3 import_islamhouse.py \
  --input-dir /media/harry/DATA250/Islamhouse \
  --recursive
```

File yang mencurigakan tetap masuk JSON, tapi diberi status:

- `review_status=pending_review`
- `review_required=true`
- `ingest_ready=false`

Supaya bisa diputuskan manual atau via lease coordinator.

### 2b. Konversi referensi hadits lokal

Jika Anda punya sumber `AhmedBaset/hadith-json`, jalankan converter ini untuk
menyusun referensi hadits lokal yang dipakai marker replacement:

```bash
python3 rag/import_hadith_json.py \
  --source-dir /tmp/hadith-json
```

Hasilnya akan ditulis ke:

- `DATABASE/reference_data/hadith/by_book/`
- `DATABASE/reference_data/hadith/index.json`

Format ini dipakai oleh marker seperti:

- `[[FIX_HADITH bukhari:1]]`
- `[[FIX_HADITH muslim:1]]`

### 3. Approve / reject item review

```bash
python3 rag/review_book.py \
  --json-path islamhouse/nama_buku.json \
  --status approved_manual \
  --reviewed-by manual
```

Status yang didukung:

- `approved_manual`
- `approved_lease`
- `rejected`
- `pending_review`

### 4. Rebuild / ingest Qdrant

```bash
bash rag/run_ingest.sh
```

Atau langsung:

```bash
python3 rag/ingest.py
```

### 5. Jalankan API

```bash
bash rag/run_api.sh
```

API tersedia di:

- `http://127.0.0.1:8000`

### 6. Evaluasi retrieval

Pastikan API sudah hidup dulu:

```bash
bash rag/run_api.sh
```

Lalu jalankan evaluasi:

```bash
python3 rag/evaluate_retrieval.py --queries eval/queries.jsonl
```

Report evaluasi akan ditulis ke:

- `reports/eval-YYYYMMDDTHHMMSSZ.json`
- `reports/eval-YYYYMMDDTHHMMSSZ.jsonl`

Metrik utama yang dihitung:

- `Recall@1`
- `Recall@3`
- `Recall@5`
- `Recall@10`
- `MRR@10`
- `avg concept coverage`

## Endpoint API

- `GET /health`
- `GET /stats`
- `GET /search`
- `POST /ask`
- `GET /books`
- `GET /books/{book_id}`
- `GET /books/{book_id}/pages/{page_num}`
- `POST /debug/retrieve`

## Catatan Operasional

- `pending_review` tidak diingest ke Qdrant
- dedupe konten berbasis isi, bukan nama file
- sitasi chunk membawa `book_id` dan info halaman
- hasil evaluasi disimpan di `reports/` dan tidak dipush
