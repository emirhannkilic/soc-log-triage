"""
Runs both src/rules/engine.py (v1) and src/rules/engine_v2.py (v2) on
data/rule_engine_v2_devset/candidates.jsonl and reports where they
diverge — CLAUDE.md's locked "GEÇİŞ ATOMİK OLMALI" sequence, the
measurement step after the family formula + auth guard + 3 FP fixes
land together.

WHY AGREEMENT WITH source_label, NOT A CLAIMED ACCURACY NUMBER
    This dev set has no hand-verified ground truth (see
    scripts/select_rule_engine_v2_devset.py's docstring) — source_label
    is the corpus's own label, same caveat as every other shadow-mode
    measurement in this project (phishing_pot is an estimated ~43%
    plain commercial spam). This script reports v1 vs v2 agreement with
    source_label side by side, and flags every DIVERGENCE for manual
    review — the divergences are the actually useful output, since they
    show exactly which emails the family formula/auth guard/FP fixes
    changed the verdict on and in which direction.

Usage:
    python3 scripts/compare_rule_engine_v1_v2.py
    python3 scripts/compare_rule_engine_v1_v2.py --show-divergences
"""
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from schemas.facts import EmailFacts  # noqa: E402
from src.rules.engine import evaluate as evaluate_v1  # noqa: E402
from src.rules.engine import load_rules as load_rules_v1  # noqa: E402
from src.rules.engine_v2 import evaluate_v2, load_rules as load_rules_v2  # noqa: E402
from src.parser.parse import parse_eml  # noqa: E402

DEVSET_PATH = PROJECT_ROOT / "data" / "rule_engine_v2_devset" / "candidates.jsonl"
_METADATA_KEYS = ("_eml_path", "source_label", "is_spam_not_phishing", "spam_reason")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--show-divergences", action="store_true",
                     help="v1/v2'nin farklı karar verdiği her maili detaylı göster")
    args = ap.parse_args()

    if not DEVSET_PATH.is_file():
        raise SystemExit(f"HATA: {DEVSET_PATH} yok — önce "
                         f"scripts/select_rule_engine_v2_devset.py çalıştırın.")

    rules_v1 = load_rules_v1()
    rules_v2 = load_rules_v2()
    rows = [json.loads(line) for line in open(DEVSET_PATH, encoding="utf-8") if line.strip()]

    results = []
    for r in rows:
        # Re-parse from the raw .eml rather than reconstructing EmailFacts
        # from the JSONL row — the JSONL was written before
        # has_large_hidden_text existed (schemas/facts.py change), so a
        # direct EmailFacts(**row) would fail validation the same way
        # the compare script's first draft did.
        facts = parse_eml(r["_eml_path"])
        v1 = evaluate_v1(facts.flat_signals(), rules_v1)
        v2 = evaluate_v2(facts, rules_v2)
        results.append({
            "eml_path": r["_eml_path"],
            "source_label": r["source_label"],
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
        legit = [r for r in results if r["source_label"] == "legitimate"]
        flagged = [r for r in legit if is_positive(r, key)]
        return len(flagged), len(legit)

    def recall(results, key):
        phish = [r for r in results if r["source_label"] == "phishing"]
        flagged = [r for r in phish if is_positive(r, key)]
        return len(flagged), len(phish)

    v1_fp, legit_n = fp_rate(results, "v1_verdict")
    v2_fp, _ = fp_rate(results, "v2_verdict")
    v1_recall, phish_n = recall(results, "v1_verdict")
    v2_recall, _ = recall(results, "v2_verdict")

    print(f"{len(results)} mail ({phish_n} phishing, {legit_n} legitimate, source_label'e göre)\n")
    print(f"v1 (additive):  recall {v1_recall}/{phish_n} = {v1_recall/phish_n:.1%}  |  "
          f"flagged-as-positive rate on legitimate: {v1_fp}/{legit_n} = {v1_fp/legit_n:.1%}")
    print(f"v2 (family):    recall {v2_recall}/{phish_n} = {v2_recall/phish_n:.1%}  |  "
          f"flagged-as-positive rate on legitimate: {v2_fp}/{legit_n} = {v2_fp/legit_n:.1%}")
    print()
    print("NOT: 'flagged-as-positive' = Phishing YA DA Muhtemel Phishing "
          "(abstention bandı dahil) — v1'in kendi Adım 4 metriğindeki "
          "false_positive_rate SADECE Phishing'i sayıyordu, burada ikisi "
          "birlikte gösteriliyor çünkü v1/v2'nin orta bant tanımı farklı "
          "olabilir; ayrıştırılmış rakamlar --show-divergences ile "
          "görülebilir.")

    divergences = [r for r in results if r["v1_verdict"] != r["v2_verdict"]]
    print(f"\n{len(divergences)}/{len(results)} mailde v1/v2 verdict FARKLI.")

    if args.show_divergences:
        print()
        for r in divergences:
            print(f"[{r['source_label']:<11}] {r['eml_path']}")
            print(f"    v1: {r['v1_verdict']:<18} (score {r['v1_score']})")
            print(f"    v2: {r['v2_verdict']:<18} (total {r['v2_total']}, "
                  f"families={r['v2_families']}, critical={r['v2_critical']})")


if __name__ == "__main__":
    main()
