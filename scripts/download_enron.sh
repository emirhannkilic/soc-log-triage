#!/bin/bash
# Downloads and extracts the CMU Enron Email Dataset (single large archive).
set -euo pipefail

if ! command -v aria2c >/dev/null 2>&1; then
  echo "Error: 'aria2c' not found, install it first." >&2
  exit 1
fi

URL="https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_ROOT/data"
DEST_DIR="$DATA_DIR/enron"
ARCHIVE="$DEST_DIR/enron_mail_20150507.tar.gz"

mkdir -p "$DEST_DIR"

aria2c -x16 -s16 -c -d "$DEST_DIR" "$URL"

echo "Extracting archive..."
tar -xzf "$ARCHIVE" -C "$DEST_DIR"

echo "Done: $DEST_DIR"
