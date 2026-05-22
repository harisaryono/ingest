#!/bin/bash
# RAG API Launcher
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

exec /home/harry/venv/rag-buku/bin/python -m uvicorn api:app --host 127.0.0.1 --port 8000
