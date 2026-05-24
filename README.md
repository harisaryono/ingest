# INGEST

Pipeline RAG lokal untuk korpus buku Islam.

Repo ini sekarang tidak lagi sekadar search-only. Bentuknya adalah:

- konversi sumber ke JSON per buku
- review dan edit JSON dari web
- quality gate dan dedupe konten
- metadata operasional di SQLite
- ingest ke Qdrant lokal
- search dan Q&A lewat FastAPI

## Gambaran Singkat

Prinsip penyimpanan data sekarang:

- **JSON** tetap menjadi sumber isi buku per buku
- **SQLite** dipakai untuk metadata operasional, status review, hash, status per halaman, dan aksi review
- **Qdrant** menyimpan vektor untuk retrieval
- **reference_data** dipakai untuk referensi Qur'an dan hadits lokal

Jadi data dibagi menjadi:

1. isi buku, tetap di JSON
2. status dan indeks cepat, di SQLite
3. embedding dan retrieval, di Qdrant

## Lokasi Data Runtime

Data runtime disimpan di luar repo:

- `../DATABASE/json_output/`
- `../DATABASE/qdrant_db/`
- `../DATABASE/review_metadata.sqlite`
- `../DATABASE/reference_data/`
- `../DATABASE/arabic_blocks/`

Catatan:
- runtime data ini tidak dipush ke GitHub
- repo menyimpan kode, script, dan dokumentasi

## Isi `json_output`

Folder ini berisi JSON buku hasil konversi.

Yang umum ada:

- `json_output/_index.json`
- `json_output/_content_index.json`
- `json_output/_duplicates.jsonl`
- `json_output/_quality_issues.jsonl`
- `json_output/<source>/*.json`

Status buku penting yang sekarang dipakai:

- `conversion_status`
- `quality_status`
- `review_status`
- `ingest_ready`
- `source_type`
- `document_type`
- `json_path`

## SQLite Metadata

File SQLite metadata:

- `../DATABASE/review_metadata.sqlite`

Isi utamanya:

- `books`
- `pages`
- `book_families`
- `review_actions`

Fungsi SQLite ini:

- lookup cepat daftar buku
- status review per buku
- status review per halaman
- hash dan kualitas per halaman
- status `first_pass`, `ocr_needed`, `ocr_done`
- catatan tindakan review

SQLite ini adalah indeks operasional. Isi buku tetap di JSON.

## Alur Kerja

### 1. Import / Konversi Sumber

Contoh import Islamhouse:

```bash
python3 import_islamhouse.py \
  --input-dir /media/harry/DATA250/Islamhouse \
  --recursive
```

Atau source lain seperti:

- `/media/harry/DATA250/IH/`
- `/media/harry/DATA250/BOOKS/AGAMA/`

Perilaku utama importer:

- file diurutkan kecil dulu, besar belakangan
- source hash dicek dulu agar file yang sudah pernah diproses bisa di-skip
- duplikasi isi tetap dipetakan ke family canonical
- file yang bermasalah tetap masuk audit JSONL
- item yang belum lolos review ditandai `pending_review`

### 2. Pipeline PDF

Untuk PDF tahap 1:

- `pdftotext`
- fallback `pdfminer`
- analisis kualitas per halaman
- OCR ditunda per halaman yang bermasalah

Tujuannya:

- cepat di tahap awal
- tidak menjalankan OCR ke seluruh halaman
- OCR hanya dipanggil untuk halaman yang memang ditandai bermasalah

### 3. Review dari Web

UI utama sekarang ada di:

- `/library`
- `/books/{book_id}/pages/{page_num}/review`
- `/books/{book_id}/edit`
- `/books/{book_id}/raw`

Fitur review yang tersedia:

- daftar buku dengan paging
- filter bahasa dan status
- reader halaman
- editor JSON per halaman
- editor JSON penuh per buku
- review manual `approved_manual`, `approved_lease`, `rejected`, `pending_review`
- review sumber PDF asli sebagai gambar halaman
- tab `Marker`, `Hadits`, `Arabic`, `Repair`, `Metadata`

### 4. Qur'an dan Hadits Referensi

Referensi lokal dipakai untuk marker replacement.

Qur'an:

- sumber Arab Uthmani lokal
- terjemahan Indonesia dan English lokal
- lookup offline-only

Hadits:

- dataset lokal dari `AhmedBaset/hadith-json`
- format lokal dipakai untuk lookup dan marker replacement
- Dorar.net dipakai sebagai kandidat pencarian, bukan overwrite final otomatis

### 5. Ingest ke Qdrant

Setelah buku lolos review:

```bash
bash rag/run_ingest.sh
```

Atau langsung:

```bash
python3 rag/ingest.py
```

Ingest:

- membaca JSON buku
- chunking
- embedding batch
- upsert ke Qdrant lokal
- state resume di `qdrant_db/ingest_state.json`

## Format File yang Dikonversi

Importer utama sekarang menangani:

- `.txt`
- `.htm`
- `.html`
- `.doc`
- `.docx`
- `.pdf`
- `.epub`
- `.ibooks`

Catatan:

- `.htm` dan `.html` diperlakukan sebagai `html`
- `.epub` dan `.ibooks` dibaca sebagai paket HTML/XHTML di dalam arsip
- file di luar daftar ini biasanya di-skip

## UI dan Endpoint

Endpoint utama:

- `GET /health`
- `GET /stats`
- `GET /search`
- `POST /ask`
- `GET /books`
- `GET /books/{book_id}`
- `GET /books/{book_id}/pages/{page_num}`
- `GET /books/{book_id}/pages/{page_num}/view`
- `GET /books/{book_id}/pages/{page_num}/review`
- `GET /sources/{book_id}/pages/{page_num}/image`
- `POST /debug/retrieve`

Endpoint review tambahan:

- `POST /books/{book_id}/edit`
- `POST /books/{book_id}/pages/{page_num}/edit`
- `POST /books/{book_id}/pages/{page_num}/review`
- `POST /books/{book_id}/pages/{page_num}/apply-markers`
- `POST /books/{book_id}/pages/{page_num}/detect-arabic-blocks`

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Konfigurasi

Copy `.env.example` ke `.env` lalu sesuaikan bila perlu.

Environment penting:

- `DATABASE_DIR`
- `OLLAMA_BASE`
- `LEASE_COORDINATOR_URL`
- `REFERENCE_DATA_DIR`
- `QURAN_REFERENCE_PATH`
- `QURAN_REFERENCE_SOURCE_PATH`
- `QURAN_TRANSLATION_PATH`
- `QURAN_TRANSLATION_EN_PATH`
- `HADITH_REFERENCE_DIR`
- `ARABIC_REVIEW_DIR`
- `TESSERACT_BIN`
- `INGEST_LANGUAGES`
- `INGEST_LIMIT_BOOKS`
- `INGEST_FORCE_REFRESH`

## Evaluasi Retrieval

Jalankan API dulu:

```bash
bash rag/run_api.sh
```

Lalu evaluasi:

```bash
python3 rag/evaluate_retrieval.py --queries eval/queries.jsonl
```

Output evaluasi ditulis ke:

- `reports/eval-YYYYMMDDTHHMMSSZ.json`
- `reports/eval-YYYYMMDDTHHMMSSZ.jsonl`

Metrik utama:

- `Recall@1`
- `Recall@3`
- `Recall@5`
- `Recall@10`
- `MRR@10`
- `avg concept coverage`

## Perintah Umum

Import Islamhouse:

```bash
bash rag/run_import_islamhouse.sh
```

Run API:

```bash
bash rag/run_api.sh
```

Run ingest Qdrant:

```bash
bash rag/run_ingest.sh
```

Backfill metadata SQLite:

```bash
/home/harry/venv/rag-buku/bin/python rag/backfill_metadata_sqlite.py --batch-size 25
```

Konversi hadits lokal:

```bash
python3 rag/import_hadith_json.py --source-dir /tmp/hadith-json
```

## Catatan Operasional

- `pending_review` tidak diingest ke Qdrant
- dedupe konten berbasis isi, bukan nama file
- sitasi chunk membawa `book_id` dan info halaman
- `review_metadata.sqlite` dipakai untuk status per buku dan per halaman
- `json_output/` dan `qdrant_db/` tetap runtime data, bukan source code
- `reports/` dipakai untuk hasil evaluasi dan tidak dipush

