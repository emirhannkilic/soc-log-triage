"""
v3 plan Adım 7 — full teacher generation, adapted for a demo-scale run.

Plan v3 §6.4 originally called for all 2500 examples (phishing_facts.jsonl
+ gmail_facts.jsonl), checkpointed in 250-example chunks with 10-minute
pauses between chunks, estimating ~24h total. That estimate didn't hold:
Adım 6's smoke test measured ~161s/example (including few-shot prompt
processing) on the user's fanless Mac Air, which puts 2500 examples at
~112h — not feasible for a demo with a same-day deadline.

**Decision (2026-08-02, CLAUDE.md "Kilitlenen Kararlar"): generate a
random SAMPLE_SIZE=230 subset instead of the full 2500.** Rationale is in
CLAUDE.md — LoRA only trains a small adapter on top of a frozen base
model, so it needs far fewer examples than full fine-tuning would; the
behavior being taught here (write in this JSON schema, in this Turkish
style, grounded in given facts) is a narrow, repetitive pattern, not new
domain knowledge (the actual security reasoning lives in the rule engine,
not the teacher).

Key differences from src/teacher/generate.py (the Adım 6 smoke test
script):
- Draws from data/processed/phishing_facts.jsonl + gmail_facts.jsonl (the
  full 2500-example pool), NOT data/holdout/candidates.jsonl.
- EXCLUDES every email whose _eml_path appears in the holdout (by path,
  not just by count) — the holdout is the one honest measurement source
  (CLAUDE.md) and must never leak into training data.
- Checkpointed and resumable: writes one line per result immediately,
  tracks completed _eml_paths in a checkpoint file, and --resume skips
  anything already done — safe to Ctrl-C and restart.
- No expectation a human reads the raw JSONL directly (unlike Adım 6);
  progress goes to stderr for a human running it unattended overnight.

Usage:
    python3 src/teacher/generate_training_data.py --resume
    python3 src/teacher/generate_training_data.py --sample-size 230 --seed 7
"""
import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from mlx_vlm import load  # noqa: E402

from schemas.facts import EmailFacts  # noqa: E402
from src.rules.engine import evaluate, load_rules  # noqa: E402
from src.teacher.few_shot_examples import (  # noqa: E402
    FEW_SHOT_GUVENILIR,
    FEW_SHOT_PHISHING,
)
from src.teacher.generate import generate_one  # noqa: E402

MODEL_PATH = PROJECT_ROOT / "models" / "Qwen3.5-9B-MLX-4bit"
PHISHING_FACTS_PATH = PROJECT_ROOT / "data" / "processed" / "phishing_facts.jsonl"
GMAIL_FACTS_PATH = PROJECT_ROOT / "data" / "processed" / "gmail_facts.jsonl"
# phishing_facts.jsonl/gmail_facts.jsonl (parse_and_anonymize.py's output)
# don't carry an _eml_path field themselves — the .eml path only exists in
# the *_sample.jsonl files (Adım 1's output) at the same line position,
# same convention scripts/select_holdout.py uses (load_facts_with_path).
PHISHING_SAMPLE_PATH = PROJECT_ROOT / "data" / "processed" / "phishing_sample.jsonl"
GMAIL_SAMPLE_PATH = PROJECT_ROOT / "data" / "processed" / "gmail_sample.jsonl"
HOLDOUT_PATH = PROJECT_ROOT / "data" / "holdout" / "candidates.jsonl"

OUT_PATH = PROJECT_ROOT / "data" / "training" / "pairs.jsonl"
CHECKPOINT_PATH = PROJECT_ROOT / "data" / "training" / "pairs_checkpoint.json"

DEFAULT_SAMPLE_SIZE = 230
DEFAULT_SEED = 7

# candidate index (1-indexed, data/holdout/review.md order) used as
# hand-written few-shot examples in src/teacher/generate.py's FEW_SHOT_INDICES
_FEW_SHOT_HOLDOUT_INDICES = {1: FEW_SHOT_PHISHING, 20: FEW_SHOT_GUVENILIR}


def _load_facts_pool() -> list[dict]:
    """Attaches _eml_path to each facts record by zipping facts.jsonl with
    the corresponding sample.jsonl (same line order, see module docstring
    and scripts/select_holdout.py's load_facts_with_path)."""
    pool = []
    for facts_path, sample_path in (
        (PHISHING_FACTS_PATH, PHISHING_SAMPLE_PATH),
        (GMAIL_FACTS_PATH, GMAIL_SAMPLE_PATH),
    ):
        with open(facts_path, encoding="utf-8") as f:
            facts = [json.loads(line) for line in f if line.strip()]
        with open(sample_path, encoding="utf-8") as f:
            paths = [json.loads(line)["path"] for line in f if line.strip()]
        if len(facts) != len(paths):
            raise ValueError(
                f"{facts_path} has {len(facts)} rows but {sample_path} has "
                f"{len(paths)} — can't zip by position safely"
            )
        for row, p in zip(facts, paths):
            row["_eml_path"] = p
        pool.extend(facts)
    return pool


def _holdout_paths() -> set[str]:
    with open(HOLDOUT_PATH, encoding="utf-8") as f:
        return {json.loads(line)["_eml_path"] for line in f if line.strip()}


def _load_checkpoint() -> set[str]:
    if not CHECKPOINT_PATH.exists():
        return set()
    return set(json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8")))


def _save_checkpoint(done_paths: set[str]) -> None:
    CHECKPOINT_PATH.write_text(
        json.dumps(sorted(done_paths), ensure_ascii=False), encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE,
                         help=f"how many examples to generate (default {DEFAULT_SAMPLE_SIZE}, "
                              "see CLAUDE.md for why this isn't 2500)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                         help="random seed for sampling the pool (default matches "
                              "the project's other sampling scripts)")
    parser.add_argument("--resume", action="store_true",
                         help="skip examples already recorded in the checkpoint file")
    args = parser.parse_args()

    import random

    rules = load_rules()
    pool = _load_facts_pool()
    holdout_paths = _holdout_paths()

    before = len(pool)
    pool = [r for r in pool if r.get("_eml_path") not in holdout_paths]
    print(f"Pool: {before} total, {before - len(pool)} excluded as holdout, "
          f"{len(pool)} eligible", file=sys.stderr)

    rng = random.Random(args.seed)
    sample = pool[:]
    rng.shuffle(sample)
    sample = sample[: args.sample_size]

    done_paths = _load_checkpoint() if args.resume else set()
    if done_paths:
        print(f"Resuming: {len(done_paths)} already done", file=sys.stderr)
        sample = [r for r in sample if r.get("_eml_path") not in done_paths]

    if not sample:
        print("Nothing to do — all requested examples already completed.", file=sys.stderr)
        return

    print(f"Generating {len(sample)} examples ...", file=sys.stderr)

    # few-shot examples: built from the holdout (never from the training pool)
    with open(HOLDOUT_PATH, encoding="utf-8") as f:
        holdout_rows = [json.loads(line) for line in f]
    few_shot = []
    for idx, report in _FEW_SHOT_HOLDOUT_INDICES.items():
        cand = holdout_rows[idx - 1]
        facts_dict = {k: v for k, v in cand.items()
                      if k not in ("source_label", "_eml_path", "is_spam_not_phishing", "spam_reason")}
        facts = EmailFacts(**facts_dict)
        verdict = evaluate(facts.flat_signals(), rules)
        few_shot.append((facts, verdict, report))

    print(f"Loading model from {MODEL_PATH} ...", file=sys.stderr)
    model, processor = load(str(MODEL_PATH))
    config = model.config
    print("Model loaded.", file=sys.stderr)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if (args.resume and OUT_PATH.exists()) else "w"

    ok, dropped = 0, 0
    t_start = time.time()

    with open(OUT_PATH, mode, encoding="utf-8") as out_f:
        for n, row in enumerate(sample, start=1):
            facts_dict = {k: v for k, v in row.items() if k != "_eml_path"}
            facts = EmailFacts(**facts_dict)
            verdict = evaluate(facts.flat_signals(), rules)

            t0 = time.time()
            result = generate_one(model, processor, config, facts, verdict, few_shot)
            elapsed = time.time() - t0

            record = {
                "eml_path": row.get("_eml_path"),
                "verdict": verdict.verdict,
                "score": verdict.score,
                "elapsed_sec": round(elapsed, 1),
                **result,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()

            done_paths.add(row.get("_eml_path"))
            _save_checkpoint(done_paths)

            if result["status"] == "ok":
                ok += 1
            else:
                dropped += 1

            total_elapsed = time.time() - t_start
            avg = total_elapsed / n
            remaining = (len(sample) - n) * avg
            print(f"[{n}/{len(sample)}] {result['status'].upper()} "
                  f"({elapsed:.1f}s) — ok={ok} dropped={dropped} — "
                  f"ETA {remaining / 60:.0f}min", file=sys.stderr)

    print(f"\nDone: {ok} ok, {dropped} dropped, out of {len(sample)}", file=sys.stderr)
    print(f"Output: {OUT_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()
