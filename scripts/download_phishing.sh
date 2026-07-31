#!/bin/bash
# Shallow-clones the rf-peixoto/phishing_pot repo (raw .eml phishing emails).
set -euo pipefail

if ! command -v git >/dev/null 2>&1; then
  echo "Error: 'git' not found, install it first." >&2
  exit 1
fi

REPO_URL="https://github.com/rf-peixoto/phishing_pot.git"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_ROOT/data"
DEST_DIR="$DATA_DIR/phishing_pot"

mkdir -p "$DATA_DIR"

if [[ -d "$DEST_DIR/.git" ]]; then
  echo "Already exists, updating: $DEST_DIR"
  git -C "$DEST_DIR" pull --depth=1
else
  git clone --depth=1 "$REPO_URL" "$DEST_DIR"
fi

echo "Done: $DEST_DIR"
