"""
Draws a fixed, reproducible sample of emails for manually reviewing LLM
report QUALITY (e.g. whether sonuc_ve_gerekce/genel_degerlendirme still
duplicate each other after a prompt change) — not for measuring accuracy.

WHY NOT THE HOLD-OUT
    CLAUDE.md's locked rule: calibration happens on a separate dev set,
    measurement happens on the hold-out ("ayar dev sette, ölçüm
    hold-out'ta"). Judging report prose quality is a form of calibration —
    the prompt will likely be adjusted again based on what this sample
    shows — so pulling from the hold-out would spend it the same way
    "%0 yanlış-pozitif" got spent (CLAUDE.md, "Kalibrasyon hold-out
    üzerinde YAPILMAZ"). This script never touches data/holdout/
    candidates.jsonl, only reads it to exclude overlap.

WHAT IT DOES
    Stratified pick across both classes (phishing_facts.jsonl,
    gmail_facts.jsonl), excluding:
      - anything already in the hold-out (data/holdout/candidates.jsonl)
      - anything already used for LoRA training (data/training/pairs.jsonl)
      - phishing records scripts/select_holdout.py's own filters would
        reject anyway (audit-flagged spam, spam-infrastructure senders,
        known spam templates, unusably short/garbled bodies) — reusing
        those filters here for the same reason select_holdout.py has
        them: an unreadable or actually-spam record can't be judged for
        report quality either.
    Stratifies legitimate picks the same way expand_holdout_legitimate.py
    does (by the rule engine's CURRENT verdict bucket), so the sample
    isn't all easy "score 0, obviously fine" mail — the prose duplication
    bug showed up on multi-signal cases, so the sample needs some.

WHAT IT DOES NOT DO
    Run the LLM. Output is just a list of .eml paths (plus one-line
    context) and the exact demo.py commands to run — CLAUDE.md: heavy/
    model-running commands are run by the user, not from here.

Usage:
    python3 scripts/sample_prompt_review_set.py --count 24
    python3 scripts/sample_prompt_review_set.py --count 24 --seed 101
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from select_holdout import (  # noqa: E402
    HAND_VERIFIED_SPAM_PATHS,
    MAX_PLAUSIBLE_WORD_CHARS,
    MIN_USABLE_BODY_CHARS,
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
# Deliberately NOT under data/holdout/ — this sample is not part of the
# hold-out (see module docstring) and living in that directory would blur
# a distinction CLAUDE.md treats as load-bearing.
OUT_PATH = PROJECT_ROOT / "data" / "prompt_review" / "sample.jsonl"

# Neither select_holdout.py's SEED (7) nor expand_holdout_legitimate.py's
# SEED (41) — this draw must not replay either of those, and its own
# default is overridable via --seed so a second, independent draw is a
# one-flag rerun rather than a code edit.
DEFAULT_SEED = 97


def _excluded_paths() -> set[str]:
    excluded = set()
    if HOLDOUT_PATH.is_file():
        excluded |= {json.loads(line)["_eml_path"]
                     for line in open(HOLDOUT_PATH, encoding="utf-8") if line.strip()}
    if TRAINING_PAIRS_PATH.is_file():
        for line in open(TRAINING_PAIRS_PATH, encoding="utf-8"):
            if not line.strip():
                continue
            d = json.loads(line)
            p = d.get("eml_path") or d.get("_eml_path")
            if p:
                excluded.add(p)
    return excluded


def _usable_phishing_pool(records: list[dict]) -> list[dict]:
    """Same readability/spam filters select_holdout.py applies — a record
    unreadable or confirmed-spam there is equally useless for judging
    report prose here."""
    return [
        r for r in records
        if r["_eml_path"] not in HAND_VERIFIED_SPAM_PATHS
        and not r.get("is_spam_not_phishing")
        and r["from_domain"] not in SPAM_INFRASTRUCTURE_DOMAINS
        and not _SPAM_TEMPLATE_RE.search(r.get("body_text") or "")
        and _has_usable_body(r.get("body_text"))
        and not (r.get("spam_reason") or "").startswith(_NO_SIGNAL_REASON)
    ]


_METADATA_KEYS = ("_eml_path", "source_label", "is_spam_not_phishing", "spam_reason")


def _difficulty_bucket(r: dict, rules: dict) -> str:
    from schemas.facts import EmailFacts
    from src.rules.engine import evaluate

    facts = EmailFacts(**{k: v for k, v in r.items() if k not in _METADATA_KEYS})
    return evaluate(facts.flat_signals(), rules).verdict


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=24,
                     help="toplam mail sayısı, sınıflar arası eşit bölünür (varsayılan 24)")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--dry-run", action="store_true",
                     help="dosyaya yazma, sadece ne seçileceğini göster")
    args = ap.parse_args()

    if args.count % 2 != 0:
        raise SystemExit("--count çift olmalı (iki sınıf arasında eşit bölünüyor)")
    half = args.count // 2

    excluded = _excluded_paths()
    print(f"Hold-out + LoRA training'den hariç tutulan: {len(excluded)} mail",
          file=sys.stderr)

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

    rng = random.Random(args.seed)
    # Stratify both sides on the rule engine's CURRENT verdict bucket, same
    # reasoning as expand_holdout_legitimate.py's _difficulty_bucket: a
    # sample of only easy score-0/score-9 cases wouldn't exercise the
    # multi-signal path where the prose-duplication bug actually showed up.
    phishing_picked = stratified_pick(
        phishing_pool, lambda r: _difficulty_bucket(r, rules), half, rng)
    legitimate_picked = stratified_pick(
        legitimate_pool, lambda r: _difficulty_bucket(r, rules), half, rng)

    picked = [{"source_label": "phishing", **r} for r in phishing_picked] + \
             [{"source_label": "legitimate", **r} for r in legitimate_picked]

    from collections import Counter
    buckets = Counter(_difficulty_bucket(r, rules) for r in picked)
    print(f"\nSeçilen {len(picked)} mail — rule engine'in ŞU ANKİ kararına göre:",
          file=sys.stderr)
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

    print("\nÇalıştırma komutları (kullanıcı tarafından, CLAUDE.md kuralı):",
          file=sys.stderr)
    for r in picked:
        eml = PROJECT_ROOT / r["_eml_path"]
        print(f"  python3 src/demo.py {eml} --open", file=sys.stderr)


if __name__ == "__main__":
    main()
