"""
Summarizes scripts/shadow_classify_eval.py's output
(data/shadow_eval/results.jsonl) into per-backend, per-language agreement
metrics against the rule engine's verdict.

METRICS, reported separately per backend (never blended, same principle
as CLAUDE.md's rule-engine metrics):
  - On rule_verdict == "Phishing" or "Muhtemel Phishing" rows: recall =
    fraction where the backend's phishing_probability >= --threshold
    ("did the model also flag this").
  - On rule_verdict == "Güvenilir" rows: false_positive_rate = fraction
    where phishing_probability >= --threshold ("model flagged something
    the rule engine cleared").
  - skip_rate: fraction where the backend never produced a usable score
    (language unsupported/low-confidence/too short).
Each split further broken out by language ("tr" vs "en" vs other) per
Codex's own warning that a single blended number hides exactly the
language gap this whole mechanism exists to measure.

--ensemble adds two synthetic "backends" computed from the three real
ones' already-collected probabilities (no re-running models):
  - ensemble_mean: arithmetic mean of the three phishing_probability
    values (rows where any backend was skipped are excluded — averaging
    in a missing value would silently treat "unusable" as "0% phishing")
  - ensemble_majority: flagged if >=2 of the 3 backends individually
    cross --threshold
Tried on 2026-08-06 because all three individual backends showed
40-87% false_positive_rate on the same 80-mail dev set (PROGRESS.md) —
each backend's errors landed in different places (aamoshdahal good on
Turkish/bad on English, the other two the reverse), so a combination
might cancel out what no single backend does alone. This is exploratory,
not a decision to fuse into the rule engine — see PROGRESS.md's three
open options.

Usage:
    python3 scripts/shadow_classify_summary.py
    python3 scripts/shadow_classify_summary.py --threshold 0.5
    python3 scripts/shadow_classify_summary.py --ensemble
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = PROJECT_ROOT / "data" / "shadow_eval" / "results.jsonl"


def _add_ensemble_columns(rows: list[dict], backends: list[str], threshold: float) -> None:
    """Mutates each row in place, adding ensemble_mean_prob/_usable and
    ensemble_majority_prob/_usable so the rest of the script can treat
    them exactly like any other backend."""
    for row in rows:
        usable = [row[f"{b}_usable"] for b in backends]
        if not all(usable):
            row["ensemble_mean_usable"] = False
            row["ensemble_mean_prob"] = None
            row["ensemble_majority_usable"] = False
            row["ensemble_majority_prob"] = None
            continue

        probs = [row[f"{b}_prob"] for b in backends]
        mean_prob = sum(probs) / len(probs)
        row["ensemble_mean_usable"] = True
        row["ensemble_mean_prob"] = mean_prob

        votes = sum(1 for p in probs if p >= threshold)
        # Represented as 1.0/0.0 rather than a real probability so it
        # reuses the same ">= threshold" flagging logic as every other
        # backend below — a majority vote is inherently a 0/1 decision at
        # whatever threshold each individual backend already used.
        row["ensemble_majority_usable"] = True
        row["ensemble_majority_prob"] = 1.0 if votes >= 2 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--threshold", type=float, default=0.5,
                     help="phishing_probability >= bu değer -> model 'phishing' diyor sayılır")
    ap.add_argument("--ensemble", action="store_true",
                     help="ensemble_mean ve ensemble_majority'yi de hesaplayıp raporla")
    args = ap.parse_args()

    if not RESULTS_PATH.is_file():
        raise SystemExit(f"HATA: {RESULTS_PATH} yok — önce "
                         f"scripts/shadow_classify_eval.py çalıştırın.")

    rows = [json.loads(line) for line in open(RESULTS_PATH, encoding="utf-8") if line.strip()]
    backends = sorted({k[:-5] for row in rows for k in row if k.endswith("_prob")})

    if args.ensemble:
        _add_ensemble_columns(rows, backends, args.threshold)
        backends = backends + ["ensemble_mean", "ensemble_majority"]

    print(f"{len(rows)} mail, {len(backends)} backend, eşik={args.threshold}\n")

    for backend in backends:
        print(f"=== {backend} ===")
        prob_key, usable_key = f"{backend}_prob", f"{backend}_usable"

        by_lang_flagged = defaultdict(lambda: defaultdict(int))
        by_lang_total = defaultdict(lambda: defaultdict(int))
        by_lang_skipped = defaultdict(int)
        by_lang_all = defaultdict(int)

        for row in rows:
            lang = row["language"] or "und"
            verdict = row["rule_verdict"]
            by_lang_all[lang] += 1
            if not row[usable_key]:
                by_lang_skipped[lang] += 1
                continue
            flagged = row[prob_key] >= args.threshold
            by_lang_total[lang][verdict] += 1
            if flagged:
                by_lang_flagged[lang][verdict] += 1

        for lang in sorted(by_lang_all):
            total = by_lang_all[lang]
            skipped = by_lang_skipped[lang]
            skip_rate = skipped / total if total else 0.0
            print(f"  dil={lang} (n={total}, skip_rate={skip_rate:.1%})")

            positive_verdicts = ("Phishing", "Muhtemel Phishing")
            pos_total = sum(by_lang_total[lang].get(v, 0) for v in positive_verdicts)
            pos_flagged = sum(by_lang_flagged[lang].get(v, 0) for v in positive_verdicts)
            if pos_total:
                print(f"    recall (Phishing+Muhtemel üzerinde): "
                      f"{pos_flagged}/{pos_total} = {pos_flagged/pos_total:.1%}")

            neg_total = by_lang_total[lang].get("Güvenilir", 0)
            neg_flagged = by_lang_flagged[lang].get("Güvenilir", 0)
            if neg_total:
                print(f"    false_positive_rate (Güvenilir üzerinde): "
                      f"{neg_flagged}/{neg_total} = {neg_flagged/neg_total:.1%}")
        print()


if __name__ == "__main__":
    main()
