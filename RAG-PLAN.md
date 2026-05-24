# RAG Plan - Buku Islam JSON to Search/Q&A

## Ringkasan

Repo ini sekarang adalah pipeline lokal untuk:

- konversi sumber buku ke JSON
- review dan edit JSON dari web
- dedupe konten antar format
- metadata operasional di SQLite
- ingest ke Qdrant lokal
- search dan Q&A berbasis referensi

Prinsip data:

- **JSON** = isi buku
- **SQLite** = indeks/status/review metadata
- **Qdrant** = embedding dan retrieval

## Status Arsitektur Saat Ini

### Data content

Isi buku tetap disimpan sebagai JSON per buku di:

- `../DATABASE/json_output/`

### Metadata operasional

Status buku, status halaman, hash, review action, dan status OCR disimpan di:

- `../DATABASE/review_metadata.sqlite`

### Retrieval

Retrieval memakai Qdrant lokal plus lexical cache. Hasil bisa dibaca di web atau dipakai sebagai konteks untuk generator.

### Reference data

Repo juga menyimpan referensi lokal untuk:

- Qur'an
- hadits
- blok Arab hasil review

## Layout Repo

```
./
├── README.md
├── .env.example
├── requirements.txt
├── eval/
│   └── queries.jsonl
├── rag/
│   ├── api.py
│   ├── arabic_blocks.py
│   ├── backfill_metadata_sqlite.py
│   ├── chunker.py
│   ├── config.py
│   ├── dedupe_qdrant_storage.py
│   ├── embeddings.py
│   ├── generator.py
│   ├── ingest.py
│   ├── ingest_common.py
│   ├── ingest_id.py
│   ├── import_hadith_json.py
│   ├── metadata_store.py
│   ├── reference_replace.py
│   ├── retriever.py
│   ├── review_book.py
│   ├── evaluate_retrieval.py
│   ├── run_api.sh
│   ├── run_ingest.sh
│   └── static/index.html
└── RAG-PLAN.md
```

Catatan:

- runtime data ada di `../DATABASE/`
- runtime data tidak dipush ke GitHub
- repo ini menyimpan kode, template, dan dokumentasi saja

## Keputusan Arsitektural

| Komponen | Pilihan saat ini | Catatan |
|----------|------------------|---------|
| Isi buku | JSON per buku | sumber utama untuk teks |
| Metadata | SQLite | status review, hash, page quality, OCR flags |
| Vector store | Qdrant lokal | untuk embedding retrieval |
| Retrieval | Dense + lexical hybrid | ada rerank dan score komponen |
| Generator lokal | `qwen3:4b` via Ollama | fallback cepat |
| Generator besar | Lease Coordinator | untuk jawaban yang butuh kualitas tinggi |
| Review UI | FastAPI + HTML | daftar buku, editor, review halaman |
| Reference data | Qur'an + hadits lokal | marker replacement offline-first |

## Alur Kerja Utama

### 1. Konversi sumber ke JSON

Sumber masuk dari folder lokal seperti:

- `Islamhouse/`
- `IH/`
- `BOOKS/AGAMA/`

Importer:

- mengurutkan file kecil dulu
- hash source file sebelum ekstraksi
- melewati file yang sudah ada di cache atau family cache
- mencatat duplicate dan quality issue
- menyimpan hasil ke JSON corpus

### 2. Review dan edit

UI review yang aktif:

- `/library`
- `/books/{book_id}/pages/{page_num}/review`
- `/books/{book_id}/edit`
- `/books/{book_id}/raw`

Fitur review:

- edit JSON per halaman
- edit JSON penuh
- review halaman
- review buku
- preview sumber PDF asli sebagai image
- marker replacement Qur'an dan hadits
- pencarian hadits lokal dan Dorar
- deteksi blok Arab untuk OCR per blok

### 3. Metadata SQLite

`review_metadata.sqlite` dipakai untuk:

- `books`
- `pages`
- `book_families`
- `review_actions`

Gunanya:

- status `pending_review`, `approved_manual`, `approved_lease`
- `conversion_status`, `quality_status`, `ingest_ready`
- `page_quality_status`, `page_ocr_needed`
- `ocr_attempted`, `ocr_done`
- catatan review

### 4. Ingest ke Qdrant

Setelah lolos review:

```bash
bash rag/run_ingest.sh
```

Atau:

```bash
python3 rag/ingest.py
```

Ingest:

- membaca JSON aktif
- chunking per buku dan per halaman
- embedding batch
- upsert ke Qdrant
- menyimpan state resume

### 5. Retrieval dan Q&A

Pipeline retrieval:

1. embed query
2. ambil kandidat dari Qdrant
3. ambil kandidat lexical bila perlu
4. rerank dengan score komponen
5. tampilkan cuplikan atau kirim ke generator

Generator:

- lokal untuk kasus ringan
- Lease Coordinator untuk jawaban yang butuh kualitas lebih tinggi

## PDF Pipeline

Untuk PDF tahap 1:

1. `pdftotext`
2. fallback `pdfminer`
3. analisis kualitas per halaman
4. tandai halaman yang perlu OCR
5. OCR dilakukan per halaman yang memang bermasalah

Tujuan:

- cepat di tahap awal
- jangan OCR semua halaman
- OCR hanya pada halaman yang ditandai perlu

## Hadits dan Qur'an

### Qur'an

- sumber lokal Uthmani
- terjemahan Indonesia dan English lokal
- offline-first

### Hadits

- dataset lokal dari `AhmedBaset/hadith-json`
- lookup lokal diprioritaskan
- Dorar.net dipakai sebagai kandidat pencarian
- marker replacement tidak auto-overwrite final tanpa review

## Status Target

Yang dianggap penting untuk kualitas repo ini:

- bisa diinstal ulang di mesin lain
- bisa diaudit
- bisa ditelusuri ke sumber
- bisa gagal aman saat referensi tidak cukup
- tidak perlu OCR atau LLM mahal untuk semua kasus

## Tahap Lanjutan

1. Lengkapi metadata SQLite untuk family duplicate dan review history
2. Pertahankan JSON sebagai source of truth untuk isi buku
3. Gunakan SQLite untuk status dan audit operasional
4. Pertahankan retrieval hybrid
5. Kecilkan tampilan UI supaya review harian tetap cepat

