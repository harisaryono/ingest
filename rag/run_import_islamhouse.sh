#!/bin/bash
# Islamhouse import launcher with file logging.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

INPUT_DIR="${IMPORT_INPUT_DIR:-/media/harry/DATA250/BOOKS/AGAMA}"
SOURCE_LABEL="${IMPORT_SOURCE_LABEL:-books_agama}"
LOGFILE="${IMPORT_ISLAMHOUSE_LOG_FILE:-/media/harry/DATA120B/DATABASE/import_islamhouse.log}"

exec /home/harry/venv/rag-buku/bin/python -u "$SCRIPT_DIR/../import_islamhouse.py" \
  --input-dir "$INPUT_DIR" \
  --source-label "$SOURCE_LABEL" \
  --log-file "$LOGFILE" \
  "$@"
