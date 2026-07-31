#!/bin/bash
# Downloads the Seneca model from Hugging Face.
# Usage:
#   ./download_model.sh gguf   -> Q4_K_M GGUF for quick validation (4.68GB)
#   ./download_model.sh bf16   -> BF16 source as mlx_lm.convert input (~16GB)
set -euo pipefail

MODE="${1:-}"
if [[ "$MODE" != "gguf" && "$MODE" != "bf16" ]]; then
  echo "Usage: $0 gguf|bf16" >&2
  exit 1
fi

for cmd in aria2c python3; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Error: '$cmd' not found, install it first." >&2
    exit 1
  fi
done
if ! python3 -c "import huggingface_hub" >/dev/null 2>&1; then
  echo "Error: 'huggingface_hub' python package not found (pip install huggingface_hub)." >&2
  exit 1
fi

if [[ "$MODE" == "gguf" ]]; then
  REPO_ID="AlicanKiraz0/SenecaLLM_x_Qwen2.5-7B-CyberSecurity-Q4_K_M-GGUF"
else
  REPO_ID="AlicanKiraz0/Seneca-Cybersecurity-LLM_x_Qwen2.5-7B-CyberSecurity"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="$PROJECT_ROOT/models"
REPO_NAME="${REPO_ID#*/}"
DEST_DIR="$MODELS_DIR/$REPO_NAME"
URL_FILE="$SCRIPT_DIR/urls_${REPO_NAME}.txt"

mkdir -p "$DEST_DIR"

REPO_ID="$REPO_ID" URL_FILE="$URL_FILE" python3 -c "
import os
from huggingface_hub import HfApi

repo_id = os.environ['REPO_ID']
url_file = os.environ['URL_FILE']

api = HfApi()
files = api.list_repo_files(repo_id=repo_id)

with open(url_file, 'w') as f:
    for file in files:
        url = f'https://huggingface.co/{repo_id}/resolve/main/{file}'
        f.write(f'{url}\n out={file}\n')
"

cd "$DEST_DIR"
aria2c -i "$URL_FILE" -j 4 -s 16 -x 16 -c
rm "$URL_FILE"

echo "Done: $DEST_DIR"
