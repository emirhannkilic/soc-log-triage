"""
v3 plan Adım 6 — teacher smoke test. Runs the teacher model
(mlx-community/Qwen3.5-9B-MLX-4bit, loaded via mlx_vlm — NOT mlx_lm, see
CLAUDE.md "Kilitlenen Kararlar") over a handful of hold-out emails,
text-only (image=None), and checks whether its JSON output matches
schemas/report.py and whether risk_seviyesi matches the rule engine's
verdict verbatim (plan §6.2's sanity check: mismatch -> drop the sample).

This is Adım 6, the smoke test — NOT Adım 7 (the full 2500-example
generation run, checkpointed/resumable). This script is meant to be run
by hand against ~20 examples and its output read by a human (CLAUDE.md
"Ağır/Uzun Süren Script Çalıştırmaları": the user runs model inference
themselves rather than this running unattended in the background).

Usage:
    python3 src/teacher/generate.py --limit 20
    python3 src/teacher/generate.py --limit 3   # quick sanity check first
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from mlx_vlm import generate, load  # noqa: E402
from mlx_vlm.prompt_utils import apply_chat_template  # noqa: E402

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
OUT_PATH = PROJECT_ROOT / "data" / "training" / "teacher_smoke_test.jsonl"

METADATA_KEYS = ("source_label", "_eml_path", "is_spam_not_phishing", "spam_reason")

# 1-indexed positions in data/holdout/review.md / candidates.jsonl used as
# hand-written few-shot examples (src/teacher/few_shot_examples.py) — these
# are EXCLUDED from the smoke test set so the model isn't tested on the
# same examples it's shown as worked answers.
FEW_SHOT_INDICES = {1, 8, 20}

# Few-shot pairing: (candidate index, EmailFacts field overrides not
# needed — we re-derive facts+verdict from candidates.jsonl at runtime)
_FEW_SHOT_REPORTS = {1: FEW_SHOT_PHISHING, 8: FEW_SHOT_MUHTEMEL, 20: FEW_SHOT_GUVENILIR}

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _load_candidates() -> list[dict]:
    with open(CANDIDATES_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _facts_and_verdict(cand: dict, rules: dict):
    facts_dict = {k: v for k, v in cand.items() if k not in METADATA_KEYS}
    facts = EmailFacts(**facts_dict)
    verdict = evaluate(facts.flat_signals(), rules)
    return facts, verdict


def _extract_json(raw_text: str) -> dict | None:
    """Model is instructed to return ONLY JSON, but strips fences/prose
    defensively rather than assuming perfect compliance."""
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[len("json"):]
    match = _JSON_BLOCK_RE.search(text)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _call_model(model, processor, config, messages: list[dict]) -> str:
    prompt = apply_chat_template(processor, config, messages, num_images=0)
    result = generate(
        model,
        processor,
        prompt,
        image=None,
        max_tokens=1200,
        temperature=0,
        verbose=False,
    )
    return result.text if hasattr(result, "text") else str(result)


def generate_one(model, processor, config, facts: EmailFacts, verdict, few_shot: list) -> dict:
    """Returns a result dict with either a validated Report or a failure
    reason. One retry on JSON/schema failure (plan §6.3), per-attempt raw
    text logged so the human can read AND diagnose a rejected sample."""
    messages = build_messages(facts, verdict, few_shot=few_shot)

    for attempt in (1, 2):
        raw = _call_model(model, processor, config, messages)
        parsed = _extract_json(raw)

        if parsed is None:
            if attempt == 2:
                return {"status": "dropped", "reason": "invalid_json", "raw": raw}
            continue

        try:
            report = Report(**parsed)
        except Exception as e:
            if attempt == 2:
                return {"status": "dropped", "reason": f"schema_error: {e}", "raw": raw}
            continue

        if report.risk_seviyesi != verdict.verdict:
            if attempt == 2:
                return {
                    "status": "dropped",
                    "reason": (
                        f"verdict_mismatch: model said {report.risk_seviyesi!r}, "
                        f"rule engine said {verdict.verdict!r}"
                    ),
                    "raw": raw,
                }
            continue

        return {"status": "ok", "report": report.model_dump(), "raw": raw, "attempt": attempt}

    return {"status": "dropped", "reason": "unreachable"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=20,
                         help="how many non-few-shot candidates to run (default 20)")
    parser.add_argument("--indices", type=str, default=None,
                         help="comma-separated 1-indexed candidate numbers to run instead "
                              "of the first --limit (e.g. --indices 14,23), for re-testing "
                              "specific cases after a prompt/param change")
    args = parser.parse_args()

    rules = load_rules()
    candidates = _load_candidates()

    # few-shot examples: rebuild (facts, verdict) for the 3 fixed indices,
    # pair with their hand-written Report
    few_shot = []
    for idx in sorted(FEW_SHOT_INDICES):
        facts, verdict = _facts_and_verdict(candidates[idx - 1], rules)
        few_shot.append((facts, verdict, _FEW_SHOT_REPORTS[idx]))

    if args.indices:
        smoke_test_indices = [int(x) for x in args.indices.split(",")]
    else:
        smoke_test_indices = [i for i in range(1, len(candidates) + 1) if i not in FEW_SHOT_INDICES]
        smoke_test_indices = smoke_test_indices[: args.limit]

    out_path = OUT_PATH
    if args.indices:
        out_path = OUT_PATH.with_name("teacher_smoke_test_retest.jsonl")

    print(f"Loading model from {MODEL_PATH} ...", file=sys.stderr)
    model, processor = load(str(MODEL_PATH))
    config = model.config
    print("Model loaded.", file=sys.stderr)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = []
    ok, dropped = 0, 0

    with open(out_path, "w", encoding="utf-8") as out_f:
        for n, idx in enumerate(smoke_test_indices, start=1):
            cand = candidates[idx - 1]
            facts, verdict = _facts_and_verdict(cand, rules)

            t0 = time.time()
            result = generate_one(model, processor, config, facts, verdict, few_shot)
            elapsed = time.time() - t0

            record = {
                "candidate_index": idx,
                "eml_path": cand["_eml_path"],
                "verdict": verdict.verdict,
                "score": verdict.score,
                "elapsed_sec": round(elapsed, 1),
                **result,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()
            results.append(record)

            if result["status"] == "ok":
                ok += 1
                print(f"[{n}/{len(smoke_test_indices)}] candidate {idx}: OK "
                      f"({elapsed:.1f}s, attempt {result['attempt']})", file=sys.stderr)
            else:
                dropped += 1
                print(f"[{n}/{len(smoke_test_indices)}] candidate {idx}: DROPPED "
                      f"({result['reason']})", file=sys.stderr)

    print(f"\nDone: {ok} ok, {dropped} dropped, out of {len(smoke_test_indices)}", file=sys.stderr)
    print(f"Output: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
