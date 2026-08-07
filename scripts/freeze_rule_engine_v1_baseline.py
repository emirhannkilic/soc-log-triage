"""
Freezes the current (v1/additive) rule engine's output on the Rule
Engine v2 dev set (scripts/select_rule_engine_v2_devset.py) — step 1 of
CLAUDE.md's locked "GEÇİŞ ATOMİK OLMALI" sequence: "mevcut rule engine
sonuçlarını baseline olarak dondur."

This is a snapshot, not a measurement — v1's own accuracy against ground
truth isn't being claimed here (the dev set has no hand labels, see
select_rule_engine_v2_devset.py's docstring). The point is narrower: once
Rule Engine v2 lands, this file is the fixed point of comparison for
"did this specific email's verdict/score change, and in which direction"
— without it, a v1-vs-v2 diff would require re-running v1 from a
possibly-already-modified config/rules.yaml or src/rules/engine.py.

Usage:
    python3 scripts/freeze_rule_engine_v1_baseline.py
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from schemas.facts import EmailFacts  # noqa: E402
from src.rules.engine import load_rules, evaluate  # noqa: E402

DEVSET_PATH = PROJECT_ROOT / "data" / "rule_engine_v2_devset" / "candidates.jsonl"
OUT_PATH = PROJECT_ROOT / "data" / "rule_engine_v2_devset" / "v1_baseline.jsonl"
_METADATA_KEYS = ("_eml_path", "source_label", "is_spam_not_phishing", "spam_reason")


def main() -> None:
    if not DEVSET_PATH.is_file():
        raise SystemExit(f"HATA: {DEVSET_PATH} yok — önce "
                         f"scripts/select_rule_engine_v2_devset.py çalıştırın.")

    rules = load_rules()
    rows = [json.loads(line) for line in open(DEVSET_PATH, encoding="utf-8") if line.strip()]

    results = []
    for r in rows:
        facts = EmailFacts(**{k: v for k, v in r.items() if k not in _METADATA_KEYS})
        verdict = evaluate(facts.flat_signals(), rules)
        results.append({
            "eml_path": r["_eml_path"],
            "source_label": r["source_label"],
            "v1_verdict": verdict.verdict,
            "v1_score": verdict.score,
            "v1_matches": [m.signal for m in verdict.matches],
        })

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    from collections import Counter
    buckets = Counter(r["v1_verdict"] for r in results)
    print(f"{len(results)} mail donduruldu -> {OUT_PATH}", file=sys.stderr)
    for verdict, n in buckets.most_common():
        print(f"  {verdict:<20} {n:>3}", file=sys.stderr)


if __name__ == "__main__":
    main()
