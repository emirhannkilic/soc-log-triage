"""
Same v1/v2 comparison as scripts/compare_rule_engine_v1_v2.py, but scored
against data/rule_engine_v2_devset/ground_truth.json (hand-labeled, see
scripts/render_devset_review.py) instead of source_label.

WHY A SEPARATE SCRIPT, NOT AN EDIT TO compare_rule_engine_v1_v2.py
    That script's source_label-based numbers are already the ones written
    into CLAUDE.md's "Rule Engine v2 — Aile Bazlı Skorlama" section —
    keeping it unedited preserves that measurement's provenance. This
    script produces the SUCCESSOR number (hand-verified ground truth on
    the phishing half) as a distinct, separately-dated artifact rather
    than silently mutating what an earlier commit's numbers were computed
    from.

WHAT GROUND TRUTH ADDS OVER source_label
    audit_spam_vs_phishing.py estimates ~43% of phishing_pot is plain
    commercial spam, not phishing with phishing mechanics — source_label
    doesn't distinguish that. ground_truth.json is Codex's per-email
    read (phishing / spam / unclear), reviewed and approved by the user,
    for the dev set's 50 phishing-labeled candidates (see
    scripts/render_devset_review.py's docstring for why the legitimate
    half wasn't re-labeled: same Gmail-Takeout trust basis the hold-out
    already relies on for its own legitimate half).

RECALL DENOMINATOR CHANGES
    source_label's recall denominator was "50 phishing-labeled emails,"
    which silently included every spam email that happened to sit in the
    phishing_pot corpus. This script's recall denominator is "42 emails a
    human confirmed carry actual phishing mechanics" — spam and unclear
    candidates are excluded from the recall calculation entirely (they
    are not phishing, so whether the rule engine flags them isn't a
    recall question) but are still reported so their own flagged rate is
    visible (a spam email getting flagged as Phishing/Muhtemel Phishing
    isn't a false positive against a "legitimate" ground truth—the
    legitimate half never included spam—but it's still worth knowing).

Usage:
    python3 scripts/compare_rule_engine_v1_v2_ground_truth.py
    python3 scripts/compare_rule_engine_v1_v2_ground_truth.py --show-divergences
"""
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.rules.engine import evaluate as evaluate_v1  # noqa: E402
from src.rules.engine import load_rules as load_rules_v1  # noqa: E402
from src.rules.engine_v2 import evaluate_v2, load_rules as load_rules_v2  # noqa: E402
from src.parser.parse import parse_eml  # noqa: E402

DEVSET_PATH = PROJECT_ROOT / "data" / "rule_engine_v2_devset" / "candidates.jsonl"
GROUND_TRUTH_PATH = PROJECT_ROOT / "data" / "rule_engine_v2_devset" / "ground_truth.json"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--show-divergences", action="store_true",
                     help="v1/v2'nin farklı karar verdiği her maili detaylı göster")
    args = ap.parse_args()

    if not DEVSET_PATH.is_file():
        raise SystemExit(f"HATA: {DEVSET_PATH} yok.")
    if not GROUND_TRUTH_PATH.is_file():
        raise SystemExit(
            f"HATA: {GROUND_TRUTH_PATH} yok — önce scripts/render_devset_review.py "
            f"ile üretilen review.md elle etiketlenmeli."
        )

    ground_truth = json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))
    rules_v1 = load_rules_v1()
    rules_v2 = load_rules_v2()
    rows = [json.loads(line) for line in open(DEVSET_PATH, encoding="utf-8") if line.strip()]

    results = []
    for r in rows:
        # Legitimate half has no hand-verified re-label — source_label is
        # the ground truth for it (same basis the hold-out's legitimate
        # half already relies on). Phishing half uses ground_truth.json.
        if r["source_label"] == "legitimate":
            gt = "legitimate"
        else:
            gt = ground_truth.get(r["_eml_path"])
            if gt is None:
                raise SystemExit(
                    f"HATA: {r['_eml_path']} ground_truth.json'da yok — "
                    f"dev set candidates.jsonl ile ground_truth.json senkron değil."
                )

        facts = parse_eml(r["_eml_path"])
        v1 = evaluate_v1(facts.flat_signals(), rules_v1)
        v2 = evaluate_v2(facts, rules_v2)
        results.append({
            "eml_path": r["_eml_path"],
            "ground_truth": gt,
            "v1_verdict": v1.verdict,
            "v1_score": v1.score,
            "v2_verdict": v2.verdict,
            "v2_total": v2.total,
            "v2_families": {f: fs.score for f, fs in v2.families.items()},
            "v2_critical": v2.critical_matches,
        })

    def is_positive(row, key):
        return row[key] in ("Phishing", "Muhtemel Phishing")

    def fp_rate(results, key):
        legit = [r for r in results if r["ground_truth"] == "legitimate"]
        flagged = [r for r in legit if is_positive(r, key)]
        return len(flagged), len(legit)

    def recall(results, key):
        phish = [r for r in results if r["ground_truth"] == "phishing"]
        flagged = [r for r in phish if is_positive(r, key)]
        return len(flagged), len(phish)

    def flagged_rate(results, key, label):
        subset = [r for r in results if r["ground_truth"] == label]
        if not subset:
            return 0, 0
        flagged = [r for r in subset if is_positive(r, key)]
        return len(flagged), len(subset)

    v1_fp, legit_n = fp_rate(results, "v1_verdict")
    v2_fp, _ = fp_rate(results, "v2_verdict")
    v1_recall, phish_n = recall(results, "v1_verdict")
    v2_recall, _ = recall(results, "v2_verdict")
    spam_n = sum(1 for r in results if r["ground_truth"] == "spam")
    unclear_n = sum(1 for r in results if r["ground_truth"] == "unclear")

    print(f"{len(results)} mail — {phish_n} phishing (hand-verified), "
          f"{spam_n} spam (hand-verified, excluded from recall), "
          f"{unclear_n} unclear (excluded from recall), "
          f"{legit_n} legitimate (source_label)\n")
    print(f"v1 (additive):  recall {v1_recall}/{phish_n} = {v1_recall/phish_n:.1%}  |  "
          f"flagged-as-positive rate on legitimate: {v1_fp}/{legit_n} = {v1_fp/legit_n:.1%}")
    print(f"v2 (family):    recall {v2_recall}/{phish_n} = {v2_recall/phish_n:.1%}  |  "
          f"flagged-as-positive rate on legitimate: {v2_fp}/{legit_n} = {v2_fp/legit_n:.1%}")

    if spam_n:
        v1_spam_flagged, _ = flagged_rate(results, "v1_verdict", "spam")
        v2_spam_flagged, _ = flagged_rate(results, "v2_verdict", "spam")
        print(f"\n(informational, not part of recall/FP) spam flagged-as-positive rate: "
              f"v1 {v1_spam_flagged}/{spam_n} = {v1_spam_flagged/spam_n:.1%}  |  "
              f"v2 {v2_spam_flagged}/{spam_n} = {v2_spam_flagged/spam_n:.1%}")

    print()
    print("NOT: 'flagged-as-positive' = Phishing YA DA Muhtemel Phishing "
          "(abstention bandı dahil). Recall paydası artık source_label "
          "DEĞİL, elle doğrulanmış phishing sayısı (42/50) — audit'in "
          "tahmin ettiği spam oranı (8/50) burada gerçek etiketle "
          "doğrulandı ve recall/FP hesabından çıkarıldı.")

    divergences = [r for r in results if r["v1_verdict"] != r["v2_verdict"]]
    print(f"\n{len(divergences)}/{len(results)} mailde v1/v2 verdict FARKLI.")

    if args.show_divergences:
        print()
        for r in divergences:
            print(f"[{r['ground_truth']:<11}] {r['eml_path']}")
            print(f"    v1: {r['v1_verdict']:<18} (score {r['v1_score']})")
            print(f"    v2: {r['v2_verdict']:<18} (total {r['v2_total']}, "
                  f"families={r['v2_families']}, critical={r['v2_critical']})")


if __name__ == "__main__":
    main()
