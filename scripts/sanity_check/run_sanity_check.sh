#!/bin/bash
# Section 3 sanity check: feeds sample SOC log JSON entries to the GGUF model
# with the persona system prompt so the output can be reviewed manually.
set -euo pipefail

if ! command -v llama-simple >/dev/null 2>&1; then
  echo "Error: 'llama-simple' not found (brew install llama.cpp)." >&2
  exit 1
fi
if ! command -v jq >/dev/null 2>&1; then
  echo "Error: 'jq' not found (brew install jq)." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
MODEL_PATH="$PROJECT_ROOT/models/SenecaLLM_x_Qwen2.5-7B-CyberSecurity-Q4_K_M-GGUF/senecallm_x_qwen2.5-7b-cybersecurity-q4_k_m.gguf"
SYSTEM_PROMPT_FILE="$SCRIPT_DIR/system_prompt.txt"
SAMPLES_FILE="$SCRIPT_DIR/samples.json"
OUTPUT_FILE="$SCRIPT_DIR/results.txt"

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "Error: model not found: $MODEL_PATH" >&2
  echo "Run 'scripts/download_model.sh gguf' first." >&2
  exit 1
fi

SYSTEM_PROMPT="$(cat "$SYSTEM_PROMPT_FILE")"
COUNT=$(jq 'length' "$SAMPLES_FILE")

: > "$OUTPUT_FILE"

for ((i = 0; i < COUNT; i++)); do
  SAMPLE=$(jq -c ".[$i]" "$SAMPLES_FILE")
  echo "=== Sample $((i + 1))/$COUNT ===" | tee -a "$OUTPUT_FILE"
  echo "Input: $SAMPLE" | tee -a "$OUTPUT_FILE"
  echo "--- Model output ---" | tee -a "$OUTPUT_FILE"

  PROMPT="<|im_start|>system
${SYSTEM_PROMPT}<|im_end|>
<|im_start|>user
${SAMPLE}<|im_end|>
<|im_start|>assistant
"

  llama-simple -m "$MODEL_PATH" -n 256 "$PROMPT" \
    </dev/null 2>>"$SCRIPT_DIR/errors.log" | tee -a "$OUTPUT_FILE" || true

  echo -e "\n" | tee -a "$OUTPUT_FILE"
done

echo "Done. Full output: $OUTPUT_FILE"
