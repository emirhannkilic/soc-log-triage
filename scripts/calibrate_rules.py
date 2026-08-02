"""
Runs the rule engine (src/rules/engine.py) against the 30 hand-labeled
hold-out emails (data/holdout/review.md) and reports precision, recall,
and abstention_rate — never a single accuracy number (CLAUDE.md "Rule
engine eşikleri" kalibrasyon metriği kararı).

Ground truth is parsed from review.md (the locked, hand-labeled source),
not from candidates.jsonl's source_label — source_label is what the
sampler assumed going in, review.md is what a human actually confirmed
after reading the raw .eml. They happen to match 30/30 (T6 already
selected unambiguous candidates), but review.md is the source of truth.

review.md and candidates.jsonl share candidate order (Candidate N <->
line N), asserted below rather than assumed.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from schemas.facts import EmailFacts
from src.rules.engine import evaluate, load_rules

REVIEW_PATH = Path("data/holdout/review.md")
CANDIDATES_PATH = Path("data/holdout/candidates.jsonl")

FACTS_ONLY_KEYS_TO_DROP = ("source_label", "_eml_path", "is_spam_not_phishing", "spam_reason")


def parse_ground_truth(path: Path) -> list[tuple[str, str]]:
    """Returns [(eml_path, label), ...] in candidate order."""
    text = path.read_text(encoding="utf-8")
    blocks = re.split(r"^## Candidate \d+\s*$", text, flags=re.MULTILINE)[1:]
    results = []
    for block in blocks:
        path_m = re.search(r"- eml path: `([^`]+)`", block)
        label_m = re.search(r"GROUND TRUTH \(phishing / legitimate\):\s*(\w+)", block)
        if not path_m or not label_m:
            raise ValueError(f"Could not parse block:\n{block[:200]}")
        label = label_m.group(1).strip()
        if label not in ("phishing", "legitimate"):
            raise ValueError(f"Unfilled or invalid ground truth: {label!r}")
        results.append((path_m.group(1), label))
    return results


def load_candidates(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main():
    ground_truth = parse_ground_truth(REVIEW_PATH)
    candidates = load_candidates(CANDIDATES_PATH)

    assert len(ground_truth) == len(candidates) == 30, (
        f"Expected 30/30, got {len(ground_truth)} ground truth, {len(candidates)} candidates"
    )
    for i, ((gt_path, _), cand) in enumerate(zip(ground_truth, candidates)):
        assert gt_path == cand["_eml_path"], (
            f"Order mismatch at index {i}: review.md says {gt_path!r}, "
            f"candidates.jsonl says {cand['_eml_path']!r}"
        )

    rules = load_rules()

    rows = []
    for (_, label), cand in zip(ground_truth, candidates):
        facts_dict = {k: v for k, v in cand.items() if k not in FACTS_ONLY_KEYS_TO_DROP}
        facts = EmailFacts(**facts_dict)
        signals = facts.flat_signals()
        result = evaluate(signals, rules)
        rows.append((label, result))

    n_phishing = sum(1 for label, _ in rows if label == "phishing")
    n_legit = sum(1 for label, _ in rows if label == "legitimate")

    # recall: of the true phishing emails, how many did the engine flag as Phishing (>= phishing threshold)
    phishing_caught = sum(
        1 for label, r in rows if label == "phishing" and r.verdict == "Phishing"
    )
    # false positive rate: of the true legitimate emails, how many did the engine flag as Phishing
    legit_flagged_phishing = sum(
        1 for label, r in rows if label == "legitimate" and r.verdict == "Phishing"
    )
    # abstention: fraction landing in the middle band (Muhtemel Phishing), regardless of true label
    abstained = sum(1 for _, r in rows if r.verdict == "Muhtemel Phishing")

    recall = phishing_caught / n_phishing if n_phishing else float("nan")
    false_positive_rate = legit_flagged_phishing / n_legit if n_legit else float("nan")
    abstention_rate = abstained / len(rows)

    print(f"n = {len(rows)} (phishing={n_phishing}, legitimate={n_legit})")
    print()
    print(f"recall (phishing caught at >= {rules['thresholds']['phishing']}):"
          f" {phishing_caught}/{n_phishing} = {recall:.1%}")
    print(f"false_positive_rate (legitimate flagged as Phishing):"
          f" {legit_flagged_phishing}/{n_legit} = {false_positive_rate:.1%}")
    print(f"abstention_rate (landed in Muhtemel Phishing, either class):"
          f" {abstained}/{len(rows)} = {abstention_rate:.1%}")
    print()
    print("Per-email detail:")
    print(f"{'label':<12} {'verdict':<20} {'score':>6}  eml_path")
    for (gt_path, label), (_, r) in zip(ground_truth, rows):
        flag = "  " if (label == "phishing") == (r.verdict == "Phishing") else " !" if r.verdict != "Muhtemel Phishing" else " ?"
        print(f"{label:<12} {r.verdict:<20} {r.score:>6}{flag}  {gt_path}")


if __name__ == "__main__":
    main()
