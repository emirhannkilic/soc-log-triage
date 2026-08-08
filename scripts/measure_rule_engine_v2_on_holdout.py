"""
Measures Rule Engine v2 (src/rules/engine_v2.py) against the hold-out
(data/holdout/review.md's 80 hand-labeled emails) — the final step of
CLAUDE.md's locked v1→v2 migration sequence: "ayar dev sette, ölçüm
hold-out'ta." Calibration/tuning happened entirely on
data/rule_engine_v2_devset (adım 4-8); this is the FIRST time v2 is run
against the hold-out at all.

WHY THIS DOESN'T RE-OPEN THE HOLD-OUT
    CLAUDE.md's rule is "kalibrasyon hold-out üzerinde YAPILMAZ" — this
    script only READS review.md's existing ground truth and RUNS the
    already-frozen v2 engine against it. No threshold, weight, or signal
    is touched here or after seeing this script's output; if the numbers
    are unfavorable, the fix is a NEW dev-set round, not adjusting v2
    against what this script reports.

Ground truth comes straight from data/holdout/review.md (binary:
phishing/legitimate, T8's locked design — see that file's own docstring
for why hold-out labeling deliberately never used a third "spam" bucket
the way the dev set's did).

Usage:
    python3 scripts/measure_rule_engine_v2_on_holdout.py
    python3 scripts/measure_rule_engine_v2_on_holdout.py --show-divergences
"""
import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.rules.engine import evaluate as evaluate_v1  # noqa: E402
from src.rules.engine import load_rules as load_rules_v1  # noqa: E402
from src.rules.engine_v2 import evaluate_v2, load_rules as load_rules_v2  # noqa: E402
from src.parser.parse import parse_eml  # noqa: E402

REVIEW_PATH = PROJECT_ROOT / "data" / "holdout" / "review.md"


def _load_ground_truth() -> list[tuple[str, str]]:
    """Returns [(eml_path, "phishing"|"legitimate"), ...] parsed straight
    from review.md — the hold-out's one authoritative label source."""
    text = REVIEW_PATH.read_text(encoding="utf-8")
    blocks = re.findall(
        r"## Candidate \d+.*?eml path: `([^`]+)`.*?"
        r"GROUND TRUTH \(phishing / legitimate\):\s*(\w+)",
        text, re.DOTALL,
    )
    if not blocks:
        raise SystemExit(f"HATA: {REVIEW_PATH}'de hiç etiketlenmiş kayıt bulunamadı.")
    return [(path, label.lower()) for path, label in blocks]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--show-divergences", action="store_true",
                     help="v1/v2'nin farklı karar verdiği her maili detaylı göster")
    args = ap.parse_args()

    ground_truth = _load_ground_truth()
    rules_v1 = load_rules_v1()
    rules_v2 = load_rules_v2()

    results = []
    for path, label in ground_truth:
        facts = parse_eml(path)
        v1 = evaluate_v1(facts.flat_signals(), rules_v1)
        v2 = evaluate_v2(facts, rules_v2)
        results.append({
            "eml_path": path,
            "ground_truth": label,
            "v1_verdict": v1.verdict,
            "v1_score": v1.score,
            "v2_verdict": v2.verdict,
            "v2_total": v2.total,
            "v2_families": {f: fs.score for f, fs in v2.families.items()},
            "v2_critical": v2.critical_matches,
        })

    def is_positive(row, key):
        return row[key] in ("Phishing", "Muhtemel Phishing")

    def is_strict_positive(row, key):
        return row[key] == "Phishing"

    def fp_rate(results, key, positive_fn):
        legit = [r for r in results if r["ground_truth"] == "legitimate"]
        flagged = [r for r in legit if positive_fn(r, key)]
        return len(flagged), len(legit)

    def recall(results, key, positive_fn):
        phish = [r for r in results if r["ground_truth"] == "phishing"]
        flagged = [r for r in phish if positive_fn(r, key)]
        return len(flagged), len(phish)

    v1_fp, legit_n = fp_rate(results, "v1_verdict", is_strict_positive)
    v2_fp, _ = fp_rate(results, "v2_verdict", is_strict_positive)
    v1_recall, phish_n = recall(results, "v1_verdict", is_positive)
    v2_recall, _ = recall(results, "v2_verdict", is_positive)
    v1_fp_wide, _ = fp_rate(results, "v1_verdict", is_positive)
    v2_fp_wide, _ = fp_rate(results, "v2_verdict", is_positive)

    def abstention_rate(results, key):
        subset = [r for r in results]
        band = [r for r in subset if r[key] == "Muhtemel Phishing"]
        return len(band), len(subset)

    v1_abst, total_n = abstention_rate(results, "v1_verdict")
    v2_abst, _ = abstention_rate(results, "v2_verdict")

    print(f"Hold-out: {total_n} mail ({phish_n} phishing, {legit_n} legitimate, "
          f"data/holdout/review.md'nin elle etiketlediği ground truth)\n")
    print("CLAUDE.md'nin kendi metrik tanımı: precision (Phishing eşiği, "
          "STRICT — Muhtemel Phishing dahil değil), recall (Phishing+Muhtemel, "
          "abstention bandı dahil), abstention_rate (Muhtemel Phishing oranı, "
          "TÜM örneklemde).\n")
    print(f"v1 (additive, mevcut production baseline):")
    print(f"  recall (geniş, Phishing+Muhtemel): {v1_recall}/{phish_n} = {v1_recall/phish_n:.1%}")
    print(f"  false_positive_rate (STRICT, sadece Phishing): {v1_fp}/{legit_n} = {v1_fp/legit_n:.1%}")
    print(f"  flagged-oranı (geniş, Phishing+Muhtemel): {v1_fp_wide}/{legit_n} = {v1_fp_wide/legit_n:.1%}")
    print(f"  abstention_rate: {v1_abst}/{total_n} = {v1_abst/total_n:.1%}")
    print()
    print(f"v2 (family, YENİ — İLK KEZ hold-out'ta ölçülüyor):")
    print(f"  recall (geniş, Phishing+Muhtemel): {v2_recall}/{phish_n} = {v2_recall/phish_n:.1%}")
    print(f"  false_positive_rate (STRICT, sadece Phishing): {v2_fp}/{legit_n} = {v2_fp/legit_n:.1%}")
    print(f"  flagged-oranı (geniş, Phishing+Muhtemel): {v2_fp_wide}/{legit_n} = {v2_fp_wide/legit_n:.1%}")
    print(f"  abstention_rate: {v2_abst}/{total_n} = {v2_abst/total_n:.1%}")

    divergences = [r for r in results if r["v1_verdict"] != r["v2_verdict"]]
    print(f"\n{len(divergences)}/{total_n} mailde v1/v2 verdict FARKLI.")

    if args.show_divergences:
        print()
        for r in divergences:
            print(f"[{r['ground_truth']:<11}] {r['eml_path']}")
            print(f"    v1: {r['v1_verdict']:<18} (score {r['v1_score']})")
            print(f"    v2: {r['v2_verdict']:<18} (total {r['v2_total']}, "
                  f"families={r['v2_families']}, critical={r['v2_critical']})")


if __name__ == "__main__":
    main()
