"""
Draws a fixed, reproducible dev set and runs all three shadow-mode
phishing classifier backends (src/classifier/phishing.py) against it,
scoring agreement with the rule engine's verdict — NOT ground truth
accuracy.

WHY "AGREEMENT WITH THE RULE ENGINE", NOT "ACCURACY"
    The phishing side of the data pool needs real hand labelling to trust
    (phishing_pot is an estimated ~43% plain commercial spam per
    scripts/audit_spam_vs_phishing.py, source_label is not reliable —
    same reason data/holdout/review.md exists at all). This script does
    not hand-label anything; it answers Codex's actual question instead
    (PROGRESS.md "shadow mode" section): "does the model catch things the
    rule engine misses, or just echo the same signals less precisely?"
    That is measured as agreement/disagreement with the rule engine's
    Phishing/Muhtemel Phishing/Güvenilir verdict, broken out by language
    — not a single blended "accuracy" number (same principle as the rule
    engine's own precision/recall/abstention_rate split, CLAUDE.md
    "Kalibrasyon metriği").

WHY NOT THE HOLD-OUT
    CLAUDE.md's locked rule: calibration happens on a separate dev set,
    measurement happens on the hold-out. Deciding whether/how a shadow
    classifier's score should ever be fused into the rule engine is a
    calibration decision — pulling from the hold-out to make it would
    spend the hold-out exactly the way "%0 yanlış-pozitif" got spent.

WHAT IT EXCLUDES
    - Hold-out (data/holdout/candidates.jsonl)
    - LoRA training pairs (data/training/pairs.jsonl)
    - The prompt-review sample (data/prompt_review/sample.jsonl) — already
      manually read by the user for a different purpose (report prose
      quality), reusing it here would not be an independent draw
    Same readability/spam filters as select_holdout.py / sample_prompt_review_set.py
    on the phishing side (audit-flagged spam, spam-infrastructure senders,
    known spam templates, garbled/too-short bodies all excluded — none of
    those are judgable by a classifier either).

WHAT IT DOES NOT DO
    Feed anything back into the rule engine or config/rules.yaml. Output
    is a per-mail comparison table plus a summary — a human decides what,
    if anything, happens next (PROGRESS.md's three open options: pick one
    backend, try an ensemble, or drop the shadow-classify effort).

Usage:
    python3 scripts/shadow_classify_eval.py --count 80
    python3 scripts/shadow_classify_eval.py --count 80 --dry-run
    python3 scripts/shadow_classify_eval.py --count 20 --backends aamoshdahal,cybersectony
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
OUT_PATH = PROJECT_ROOT / "data" / "shadow_eval" / "results.jsonl"

# Neither select_holdout.py's SEED (7), expand_holdout_legitimate.py's (41),
# nor sample_prompt_review_set.py's (97) — independent draw.
DEFAULT_SEED = 173

_METADATA_KEYS = ("_eml_path", "source_label", "is_spam_not_phishing", "spam_reason")


def _excluded_paths() -> set[str]:
    excluded = set()
    for path in (HOLDOUT_PATH, TRAINING_PAIRS_PATH, PROMPT_REVIEW_PATH):
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
    ap.add_argument("--count", type=int, default=80,
                     help="toplam mail sayısı, sınıflar arası eşit bölünür (varsayılan 80)")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--backends", default="ealvaradob,cybersectony,aamoshdahal",
                     help="virgülle ayrılmış backend listesi (src/classifier/phishing.py'nin BACKENDS'i)")
    ap.add_argument("--dry-run", action="store_true",
                     help="sadece hangi maillerin seçileceğini göster, model çalıştırma")
    args = ap.parse_args()

    if args.count % 2 != 0:
        raise SystemExit("--count çift olmalı (iki sınıf arasında eşit bölünüyor)")
    half = args.count // 2
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]

    excluded = _excluded_paths()
    print(f"Hold-out + LoRA training + prompt-review'den hariç tutulan: "
          f"{len(excluded)} mail", file=sys.stderr)

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
    print(f"\nSeçilen {len(picked)} mail — rule engine'in ŞU ANKİ kararına göre:",
          file=sys.stderr)
    for verdict, n in buckets.most_common():
        print(f"  {verdict:<20} {n:>3}", file=sys.stderr)

    if args.dry_run:
        print("\n--- DRY RUN, model çalıştırılmadı ---", file=sys.stderr)
        for r in picked:
            print(f"  [{r['source_label']:<11}] {r['_eml_path']}", file=sys.stderr)
        return

    from schemas.facts import EmailFacts
    from src.classifier.phishing import classify
    from src.rules.engine import evaluate

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results = []
    print(f"\n{len(backends)} backend x {len(picked)} mail çalıştırılıyor "
          f"(bu birkaç dakika sürebilir) ...", file=sys.stderr)
    for i, r in enumerate(picked, 1):
        facts = EmailFacts(**{k: v for k, v in r.items() if k not in _METADATA_KEYS})
        verdict = evaluate(facts.flat_signals(), rules)
        row = {
            "eml_path": r["_eml_path"],
            "source_label": r["source_label"],
            "rule_verdict": verdict.verdict,
            "rule_score": verdict.score,
            "language": facts.language,
        }
        for backend in backends:
            result = classify(facts.subject or "", facts.body_text, backend=backend)
            row[f"{backend}_usable"] = result.usable
            row[f"{backend}_prob"] = result.phishing_probability
            row[f"{backend}_translated"] = result.translated
        results.append(row)
        print(f"  [{i}/{len(picked)}] {r['_eml_path']}", file=sys.stderr)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\n{len(results)} sonuç -> {OUT_PATH}", file=sys.stderr)
    print("Özet için: python3 scripts/shadow_classify_summary.py", file=sys.stderr)


if __name__ == "__main__":
    main()
