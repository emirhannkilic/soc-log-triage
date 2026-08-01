#!/bin/bash
# Downloads the v3 teacher model (Qwen3.5-9B, MLX 4-bit, already-quantized —
# no mlx_lm.convert step needed) from Hugging Face.
#
# Note: this is a native multimodal model, loaded with mlx_vlm (not mlx_lm)
# even though we only ever pass it text. See CLAUDE.md "Kilitlenen Kararlar".
set -euo pipefail

REPO_ID="mlx-community/Qwen3.5-9B-MLX-4bit"

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
# --async-dns=false + --disable-ipv6: aria2c's own async DNS resolver can
# fail to resolve some HF Xet CDN hosts (us.aws.cdn.hf.co) even when the
# system resolver handles them fine — falling back to system getaddrinfo
# over IPv4 avoids that.
# -j 2 -s 4 -x 4 (lower than the model script's -j4 -s16 -x16): HF's Xet CDN
# returns 403 on some connections when too many parallel requests hit the
# same presigned URL at once; fewer concurrent connections avoids that.
# --retry-wait + --max-tries: back off and retry on transient 403s instead
# of giving up on the first one.
aria2c -i "$URL_FILE" -j 2 -s 4 -x 4 -c --async-dns=false --disable-ipv6=true \
  --retry-wait=5 --max-tries=10
rm "$URL_FILE"

echo "Done: $DEST_DIR"
