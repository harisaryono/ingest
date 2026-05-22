#!/bin/bash
# RAG Ingestion Launcher
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
LOGFILE="$SCRIPT_DIR/ingest.log"
echo "=== RAG Ingestion started $(date) ===" > "$LOGFILE"
/home/harry/venv/rag-buku/bin/python -u "$SCRIPT_DIR/ingest.py" >> "$LOGFILE" 2>&1
echo "=== RAG Ingestion finished $(date) ===" >> "$LOGFILE"
