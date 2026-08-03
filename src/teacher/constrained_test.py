"""
Ad-hoc test: constrained JSON decoding via mlx_vlm's built-in llguidance
logits processor, as an alternative to the current parse+retry approach
in src/teacher/generate.py's _extract_json()/generate_one().

llguidance forces the model to only emit tokens that keep the output
inside the schema's grammar — a malformed JSON output becomes
structurally impossible instead of merely "hopefully avoided by the
prompt". This does NOT replace the verdict_mismatch sanity check (schema
validity and factual correctness are separate concerns) — it only
targets invalid_json/schema_error drops.

Run standalone against a couple of holdout candidates to compare against
the existing free-form approach before deciding whether to fold it into
the real pipeline. NOT wired into generate.py — this is a side experiment.

Usage:
    python3 src/teacher/constrained_test.py --indices 2,14
"""
import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from mlx_vlm import generate, load  # noqa: E402
from mlx_vlm.prompt_utils import apply_chat_template  # noqa: E402
from mlx_vlm.structured import build_json_schema_logits_processor  # noqa: E402

from schemas.facts import EmailFacts  # noqa: E402
from schemas.report import Report  # noqa: E402
from src.rules.engine import evaluate, load_rules  # noqa: E402
from src.teacher.few_shot_examples import (  # noqa: E402
    FEW_SHOT_GUVENILIR,
    FEW_SHOT_MUHTEMEL,
    FEW_SHOT_PHISHING,
)
from src.teacher.prompts import build_messages  # noqa: E402

MODEL_PATH = PROJECT_ROOT / "models" / "Qwen3.5-9B-MLX-4bit"
CANDIDATES_PATH = PROJECT_ROOT / "data" / "holdout" / "candidates.jsonl"
METADATA_KEYS = ("source_label", "_eml_path", "is_spam_not_phishing", "spam_reason")
FEW_SHOT_REPORTS = {1: FEW_SHOT_PHISHING, 8: FEW_SHOT_MUHTEMEL, 20: FEW_SHOT_GUVENILIR}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--indices", type=str, required=True,
                         help="comma-separated 1-indexed holdout candidates")
    args = parser.parse_args()

    indices = [int(x) for x in args.indices.split(",")]

    rules = load_rules()
    with open(CANDIDATES_PATH, encoding="utf-8") as f:
        candidates = [json.loads(line) for line in f]

    few_shot = []
    for idx in sorted(FEW_SHOT_REPORTS):
        cand = candidates[idx - 1]
        facts_dict = {k: v for k, v in cand.items() if k not in METADATA_KEYS}
        facts = EmailFacts(**facts_dict)
        verdict = evaluate(facts.flat_signals(), rules)
        few_shot.append((facts, verdict, FEW_SHOT_REPORTS[idx]))

    print(f"Loading model from {MODEL_PATH} ...", file=sys.stderr)
    model, processor = load(str(MODEL_PATH))
    config = model.config
    print("Model loaded.", file=sys.stderr)

    schema = Report.model_json_schema()
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    logits_processor = build_json_schema_logits_processor(tokenizer, schema)
    print("Built JSON schema logits processor.", file=sys.stderr)

    for idx in indices:
        cand = candidates[idx - 1]
        facts_dict = {k: v for k, v in cand.items() if k not in METADATA_KEYS}
        facts = EmailFacts(**facts_dict)
        verdict = evaluate(facts.flat_signals(), rules)

        messages = build_messages(facts, verdict, few_shot=few_shot)
        prompt = apply_chat_template(processor, config, messages, num_images=0)

        t0 = time.time()
        result = generate(
            model,
            processor,
            prompt,
            image=None,
            max_tokens=1200,
            temperature=0,
            verbose=False,
            logits_processors=[logits_processor],
        )
        elapsed = time.time() - t0
        raw = result.text if hasattr(result, "text") else str(result)

        print(f"\n=== candidate {idx} ({elapsed:.1f}s) ===")
        try:
            parsed = json.loads(raw)
            report = Report(**parsed)
            match = "MATCH" if report.risk_seviyesi == verdict.verdict else "MISMATCH"
            print(f"valid JSON: YES | schema valid: YES | verdict {match} "
                  f"(model={report.risk_seviyesi!r}, rules={verdict.verdict!r})")
        except json.JSONDecodeError as e:
            print(f"valid JSON: NO — {e}")
        except Exception as e:
            print(f"schema valid: NO — {e}")

        print("--- raw output (first 500 chars) ---")
        print(raw[:500])


if __name__ == "__main__":
    main()
