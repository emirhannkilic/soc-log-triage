"""
Draws a fresh, untouched dev set for calibrating Rule Engine v2 (CLAUDE.md
"Rule Engine v2 — Aile Bazlı Skorlama", locked 2026-08-06 after Codex's
review). This set has never been used for anything — not the hold-out, not
LoRA training, not the prompt-review sample, not either of the two
shadow-classify dev sets (whole-email classifiers and the abandoned NLI
attempt both drew from seed=173).

WHY A NEW SET, NOT THE OLD 80-MAIL SHADOW-CLASSIFY DEV SET
    CLAUDE.md's locked rule: calibration happens on a separate dev set,
    measurement happens on the hold-out — and a dev set that has already
    been read/measured against multiple times (shadow_classify_eval.py's
    80 mails were scored against 3 whole-email classifiers plus an NLI
    pass) risks the same "spent" problem the hold-out itself is protected
    from. This draw is independent: new seed, and explicitly excludes
    every path any prior draw already touched.

WHAT IT EXCLUDES
    - Hold-out (data/holdout/candidates.jsonl) — seed=7
    - LoRA training pairs (data/training/pairs.jsonl)
    - The prompt-review sample (data/prompt_review/sample.jsonl) — seed=97
    - The shadow-classify dev set (data/shadow_eval/results.jsonl) — seed=173,
      already used for both the whole-email classifiers and the NLI attempt
    Same readability/spam filters as select_holdout.py on the phishing
    side (audit-flagged spam, spam-infrastructure senders, known spam
    templates, garbled/too-short bodies all excluded).

WHAT IT DOES NOT DO
    Touch config/rules.yaml or run the rule engine's decision anywhere
    except to STRATIFY the draw (same principle as
    expand_holdout_legitimate.py's _difficulty_bucket — sampling across
    the engine's CURRENT output isn't circular, because the resulting
    emails still need separate ground-truth judgement; it just ensures
    the set contains cases across the score range, not only the easy
    ones).

Usage:
    python3 scripts/select_rule_engine_v2_devset.py --count 100
    python3 scripts/select_rule_engine_v2_devset.py --count 100 --dry-run
"""
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from select_holdout import (  # noqa: E402
    HAND_VERIFIED_SPAM_PATHS,
    PROCESSED_DIR,
    SPAM_INFRASTRUCTURE_DOMAINS,
    _NO_SIGNAL_REASON,
    _SPAM_TEMPLATE_RE,
    _has_usable_body,
    attach_spam_audit,
    load_facts_with_path,
    stratified_pick,
)

HOLDOUT_PATH = PROJECT_ROOT / "data" / "holdout" / "candidates.jsonl"
TRAINING_PAIRS_PATH = PROJECT_ROOT / "data" / "training" / "pairs.jsonl"
PROMPT_REVIEW_PATH = PROJECT_ROOT / "data" / "prompt_review" / "sample.jsonl"
SHADOW_EVAL_PATH = PROJECT_ROOT / "data" / "shadow_eval" / "results.jsonl"
OUT_PATH = PROJECT_ROOT / "data" / "rule_engine_v2_devset" / "candidates.jsonl"

# Not 7 (select_holdout.py), 41 (expand_holdout_legitimate.py), 97
# (sample_prompt_review_set.py), or 173 (shadow_classify_eval.py) —
# independent draw.
DEFAULT_SEED = 229

_METADATA_KEYS = ("_eml_path", "source_label", "is_spam_not_phishing", "spam_reason")


def _excluded_paths() -> set[str]:
    excluded = set()
    for path in (HOLDOUT_PATH, TRAINING_PAIRS_PATH, PROMPT_REVIEW_PATH, SHADOW_EVAL_PATH):
        if not path.is_file():
            continue
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            d = json.loads(line)
            p = d.get("_eml_path") or d.get("eml_path")
            if p:
                excluded.add(p)
    return excluded


def _usable_phishing_pool(records: list[dict]) -> list[dict]:
    return [
        r for r in records
        if r["_eml_path"] not in HAND_VERIFIED_SPAM_PATHS
        and not r.get("is_spam_not_phishing")
        and r["from_domain"] not in SPAM_INFRASTRUCTURE_DOMAINS
        and not _SPAM_TEMPLATE_RE.search(r.get("body_text") or "")
        and _has_usable_body(r.get("body_text"))
        and not (r.get("spam_reason") or "").startswith(_NO_SIGNAL_REASON)
    ]


def _verdict_bucket(r: dict, rules: dict) -> str:
    from schemas.facts import EmailFacts
    from src.rules.engine import evaluate

    facts = EmailFacts(**{k: v for k, v in r.items() if k not in _METADATA_KEYS})
    return evaluate(facts.flat_signals(), rules).verdict


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--count", type=int, default=100,
                     help="toplam mail sayısı, sınıflar arası eşit bölünür (varsayılan 100)")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--dry-run", action="store_true",
                     help="dosyaya yazma, sadece ne seçileceğini göster")
    args = ap.parse_args()

    if args.count % 2 != 0:
        raise SystemExit("--count çift olmalı (iki sınıf arasında eşit bölünüyor)")
    half = args.count // 2

    excluded = _excluded_paths()
    print(f"Hold-out + LoRA training + prompt-review + shadow-eval'den "
          f"hariç tutulan: {len(excluded)} mail", file=sys.stderr)

    from src.rules.engine import load_rules
    rules = load_rules()

    phishing = load_facts_with_path(
        PROCESSED_DIR / "phishing_facts.jsonl", PROCESSED_DIR / "phishing_sample.jsonl")
    attach_spam_audit(phishing)
    phishing_pool = [r for r in _usable_phishing_pool(phishing)
                      if r["_eml_path"] not in excluded]

    legitimate = load_facts_with_path(
        PROCESSED_DIR / "gmail_facts.jsonl", PROCESSED_DIR / "gmail_sample.jsonl")
    legitimate_pool = [r for r in legitimate if r["_eml_path"] not in excluded]

    print(f"Uygun havuzlar: {len(phishing_pool)} phishing, "
          f"{len(legitimate_pool)} legitimate", file=sys.stderr)
    if len(phishing_pool) < half or len(legitimate_pool) < half:
        raise SystemExit(f"havuzda yeterli mail yok (--count {args.count} için "
                          f"her sınıftan {half} gerekiyor)")

    import random
    rng = random.Random(args.seed)
    phishing_picked = stratified_pick(
        phishing_pool, lambda r: _verdict_bucket(r, rules), half, rng)
    legitimate_picked = stratified_pick(
        legitimate_pool, lambda r: _verdict_bucket(r, rules), half, rng)

    picked = [{"source_label": "phishing", **r} for r in phishing_picked] + \
             [{"source_label": "legitimate", **r} for r in legitimate_picked]

    from collections import Counter
    buckets = Counter(_verdict_bucket(r, rules) for r in picked)
    print(f"\nSeçilen {len(picked)} mail — mevcut (v1/additive) rule engine'in "
          f"kararına göre:", file=sys.stderr)
    for verdict, n in buckets.most_common():
        print(f"  {verdict:<20} {n:>3}", file=sys.stderr)

    if args.dry_run:
        print("\n--- DRY RUN, dosya değiştirilmedi ---", file=sys.stderr)
        for r in picked:
            print(f"  [{r['source_label']:<11}] {r['_eml_path']}", file=sys.stderr)
        return

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for r in picked:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n{len(picked)} mail -> {OUT_PATH}", file=sys.stderr)
    print("\nBu set HAM — ground truth etiketi YOK. Rule Engine v2'nin "
          "kalibrasyonu (CLAUDE.md 'GEÇİŞ ATOMİK OLMALI' sırası) bu setin "
          "mevcut v1 kararlarını referans/karşılaştırma noktası olarak "
          "kullanabilir; phishing tarafının gerçek doğruluğu için elle "
          "etiketleme (data/holdout/review.md'nin yaptığı gibi) ayrı bir "
          "iş, burada yapılmadı.", file=sys.stderr)


if __name__ == "__main__":
    main()
