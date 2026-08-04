"""
Expands the hold-out's LEGITIMATE side only, appending to the existing set
rather than redrawing it.

WHY ONLY THE LEGITIMATE SIDE
    The weakest claim in the project's numbers is "false-positive rate
    0.0%". Measured on 15 legitimate emails, that says very little: the
    Wilson 95% upper bound sits near 20%, so the true rate could plausibly
    be one in five. Going to 65 legitimate examples pulls that upper bound
    down to roughly 5%.

    The phishing side is left alone because expanding it needs real hand
    labelling of adversarial samples (the phishing_pot pool is an estimated
    ~43% plain commercial spam, so source_label cannot be trusted). The
    legitimate side is the account owner's own mailbox, where labelling is
    cheap and reliable.

WHY A SEPARATE SCRIPT INSTEAD OF RAISING LEGITIMATE_COUNT
    select_holdout.py opens candidates.jsonl with "w" and redraws both
    classes. Raising LEGITIMATE_COUNT there would resample the legitimate
    side, replacing the 15 emails that were already hand-labelled — the
    hold-out's entire value comes from those labels (CLAUDE.md: the
    hold-out is locked after Adım 4). This script only ever APPENDS, and
    refuses to touch anything already present.

WHAT IT GUARANTEES
    - Existing candidates are preserved byte-for-byte, in order.
    - New picks cannot collide with the existing hold-out or with the LoRA
      training set (both are excluded by .eml path).
    - Picks are stratified the same way select_holdout.py does it
      (brand-like display name vs. personal), so the added emails exercise
      the same rule-engine paths.
    - A different RNG seed from select_holdout.py's, so this draw is
      independent of the original one rather than replaying it.

Usage:
    python3 scripts/expand_holdout_legitimate.py --count 50
    python3 scripts/expand_holdout_legitimate.py --count 50 --dry-run
"""
import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from select_holdout import (  # noqa: E402
    PROCESSED_DIR,
    load_facts_with_path,
    stratified_pick,
)

CANDIDATES_PATH = PROJECT_ROOT / "data" / "holdout" / "candidates.jsonl"
TRAINING_PAIRS_PATH = PROJECT_ROOT / "data" / "training" / "pairs.jsonl"

# Deliberately not select_holdout.py's SEED (7): reusing it would make the
# first picks replay the original draw, so the "new" emails would partly be
# ones already excluded as duplicates.
SEED = 41


def _existing_paths() -> tuple[list[dict], set[str]]:
    if not CANDIDATES_PATH.is_file():
        raise SystemExit(f"hold-out bulunamadı: {CANDIDATES_PATH}\n"
                         "Önce scripts/select_holdout.py çalıştırılmalı.")
    records = [json.loads(line) for line in open(CANDIDATES_PATH, encoding="utf-8")
               if line.strip()]
    return records, {r["_eml_path"] for r in records}


def _training_paths() -> set[str]:
    """Paths used for LoRA training, so the hold-out never overlaps them."""
    if not TRAINING_PAIRS_PATH.is_file():
        return set()
    paths = set()
    for line in open(TRAINING_PAIRS_PATH, encoding="utf-8"):
        if not line.strip():
            continue
        d = json.loads(line)
        p = d.get("eml_path") or d.get("_eml_path")
        if p:
            paths.add(p)
    return paths


def _is_brand_like(r: dict) -> bool:
    """select_holdout.py's stratification key, kept for reference.

    Useless on this pool: 995 of the 1000 Gmail records come back
    brand-like, and the remaining 5 only qualify because their display
    name is empty — there is no person-to-person correspondence in the
    mailbox sample to stratify against. Stratifying on it would draw a
    single bucket, which is what the first run of this script did.
    """
    name = (r.get("display_name") or "").lower()
    return bool(name) and not (len(name.split()) == 2 and name.istitle())


def _difficulty_bucket(r: dict, rules: dict) -> str:
    """Stratify on what the rule engine currently says about the email.

    This is the axis that decides whether the expanded hold-out can
    measure anything. The false-positive rate is the metric being
    tightened, and a false positive can only come from a legitimate email
    the engine scores highly — so the set has to contain some. In this
    pool the engine already calls 5.9% of legitimate mail Phishing and
    another 24.3% Muhtemel Phishing; drawing only from the easy 69.8%
    would keep the measured rate at 0% by construction and prove nothing.

    Note this deliberately samples across the engine's CURRENT output. It
    is not circular: the resulting emails still get labelled by hand, and a
    label of "legitimate" on an email the engine calls Phishing is exactly
    the false positive the metric is meant to catch.
    """
    from schemas.facts import EmailFacts
    from src.rules.engine import evaluate

    facts = EmailFacts(**{k: v for k, v in r.items() if k != "_eml_path"})
    return evaluate(facts.flat_signals(), rules).verdict


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=50,
                    help="eklenecek legitimate mail sayısı (varsayılan 50)")
    ap.add_argument("--dry-run", action="store_true",
                    help="dosyaya yazma, sadece ne seçileceğini göster")
    args = ap.parse_args()

    existing, existing_paths = _existing_paths()
    n_legit = sum(1 for r in existing if r.get("source_label") == "legitimate")
    n_phish = sum(1 for r in existing if r.get("source_label") == "phishing")
    print(f"Mevcut hold-out: {n_phish} phishing + {n_legit} legitimate "
          f"= {len(existing)} mail", file=sys.stderr)

    training_paths = _training_paths()
    if training_paths:
        print(f"LoRA training'de kullanılan {len(training_paths)} mail hariç "
              f"tutulacak", file=sys.stderr)

    legitimate = load_facts_with_path(
        PROCESSED_DIR / "gmail_facts.jsonl",
        PROCESSED_DIR / "gmail_sample.jsonl",
    )
    pool = [r for r in legitimate
            if r["_eml_path"] not in existing_paths
            and r["_eml_path"] not in training_paths]
    print(f"Uygun havuz: {len(pool)} mail ({len(legitimate)} toplam "
          f"gmail kaydından)", file=sys.stderr)

    if len(pool) < args.count:
        raise SystemExit(f"havuzda yeterli mail yok: {len(pool)} < {args.count}")

    from src.rules.engine import load_rules
    rules = load_rules()
    picked = stratified_pick(
        pool, lambda r: _difficulty_bucket(r, rules), args.count, random.Random(SEED)
    )

    # Paranoia: stratified_pick draws from `pool`, which is already filtered,
    # but a duplicate here would silently corrupt the hold-out.
    picked_paths = {r["_eml_path"] for r in picked}
    assert not (picked_paths & existing_paths), "seçim mevcut hold-out ile çakıştı"
    assert not (picked_paths & training_paths), "seçim training seti ile çakıştı"
    assert len(picked_paths) == len(picked), "seçimde tekrar eden mail var"

    from collections import Counter
    buckets = Counter(_difficulty_bucket(r, rules) for r in picked)
    print(f"\nSeçilen {len(picked)} mail — rule engine'in ŞU ANKİ kararına göre:",
          file=sys.stderr)
    for verdict, n in buckets.most_common():
        print(f"  {verdict:<20} {n:>3}", file=sys.stderr)
    print("  (Bu bir etiket DEĞİL. 'Phishing' çıkanlar elle 'legitimate'",
          file=sys.stderr)
    print("   etiketlenirse, ölçülmek istenen yanlış-pozitif tam olarak odur.)",
          file=sys.stderr)

    if args.dry_run:
        print("\n--- DRY RUN, dosya değiştirilmedi ---", file=sys.stderr)
        for r in picked[:10]:
            print(f"  {r['_eml_path']}", file=sys.stderr)
            print(f"      from={r.get('from_domain')}  "
                  f"subject={(r.get('subject') or '')[:60]!r}", file=sys.stderr)
        if len(picked) > 10:
            print(f"  … ve {len(picked) - 10} tane daha", file=sys.stderr)
        return

    # Append-only. Existing lines are never rewritten, so the hand-labelled
    # records keep their exact bytes.
    with open(CANDIDATES_PATH, "a", encoding="utf-8") as f:
        for r in picked:
            f.write(json.dumps({"source_label": "legitimate", **r},
                               ensure_ascii=False) + "\n")

    total = len(existing) + len(picked)
    print(f"\n{len(picked)} legitimate mail eklendi -> {CANDIDATES_PATH}",
          file=sys.stderr)
    print(f"Hold-out artık: {n_phish} phishing + {n_legit + len(picked)} "
          f"legitimate = {total} mail", file=sys.stderr)
    print("\nSONRAKİ ADIM: yeni maillerin etiketlenmesi gerekiyor.", file=sys.stderr)
    print("  python3 scripts/render_holdout_review.py", file=sys.stderr)


if __name__ == "__main__":
    main()
