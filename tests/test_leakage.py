"""
v3 holdout-fix-tasks.md T1: leakage regression test.

For every boolean/nullable fact field, checks whether the field's value
perfectly separates source_label (phishing vs legitimate) across the
30-candidate holdout. A field that's 100% predictive on its own is a strong
signal that something is LEAKING (usually a parser bug that behaves
differently on the two corpora, like Compat32 handing back a Header object
only on malformed phishing_pot headers) rather than a genuine phishing
signal — real signals should be strong but not perfect, and a perfect
single-field split means the rule engine would just learn that shortcut
instead of doing real multi-signal reasoning.

Run after every T-task fix (T1-T8) per holdout-fix-tasks.md's closing order.
Run with: python3 tests/test_leakage.py
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CANDIDATES_PATH = PROJECT_ROOT / "data" / "holdout" / "candidates.jsonl"

# Structural fields that are never a leak signal by construction (identity/
# free-text/list fields, not per-message classification facts).
SKIP_FIELDS = {"source_label", "_eml_path", "urls", "attachments",
                "subject", "date", "body_text", "display_name",
                "urgency_keywords", "is_spam_not_phishing", "spam_reason"}
# is_spam_not_phishing/spam_reason (holdout-fix-tasks.md T6) are holdout
# selection metadata, not parser facts — legitimate records don't have
# them at all, and select_holdout.py's phishing pool is now filtered to
# is_spam_not_phishing=False by construction, so they trivially "leak"
# without indicating any parser bug.

# Fields where perfect separation is a KNOWN, ACCEPTED strong signal rather
# than a suspected bug (e.g. spf_result/dkim_result genuinely correlating
# with phishing is expected and fine). Everything else — including
# from_domain, which is exactly what caught the real T1 bug — is checked.
# Do not add a field here without a documented reason; this list is a
# deliberate exception list, not a place to silence inconvenient findings.
KNOWN_ACCEPTED_SIGNALS: set[str] = set()


MIN_OCCURRENCES_TO_FLAG = 2
# ^ After anonymization, every real domain gets its own unique alias, so a
# high-cardinality field like from_domain naturally has each value appear
# in exactly one record — that's an artifact of anonymization, not a leak.
# A value is only meaningful as a leak signal if it RECURS across multiple
# records (e.g. None appearing in 6 different phishing records and 0
# legitimate ones) — a repeated value that always lands on one label is
# what actually indicates the parser behaves differently on the two corpora.


def check_field_separation(records: list[dict]) -> list[tuple[str, dict]]:
    """Returns (field_name, {value: set_of_labels}) for every field that
    has at least one recurring value which perfectly separates phishing
    from legitimate."""
    field_values: dict[str, dict] = defaultdict(lambda: defaultdict(set))
    field_value_counts: dict[str, dict] = defaultdict(lambda: defaultdict(int))

    for r in records:
        label = r["source_label"]
        for key, value in r.items():
            if key in SKIP_FIELDS or key in KNOWN_ACCEPTED_SIGNALS:
                continue
            if isinstance(value, (list, dict)):
                continue
            field_values[key][value].add(label)
            field_value_counts[key][value] += 1

    leaks = []
    for field, value_to_labels in field_values.items():
        recurring = {
            value: labels for value, labels in value_to_labels.items()
            if field_value_counts[field][value] >= MIN_OCCURRENCES_TO_FLAG
        }
        if not recurring:
            continue
        # A field leaks if every RECURRING value it takes maps to only ONE
        # label (not necessarily the same label across values) — e.g. None
        # always means phishing AND non-None always means legitimate.
        if all(len(labels) == 1 for labels in recurring.values()):
            value_to_labels = recurring
            leaks.append((field, dict(value_to_labels)))

    return leaks


def main() -> None:
    if not CANDIDATES_PATH.exists():
        print(f"SKIP: {CANDIDATES_PATH} not found")
        return True

    records = [json.loads(line) for line in open(CANDIDATES_PATH) if line.strip()]
    if not records:
        print("SKIP: no records")
        return True

    leaks = check_field_separation(records)

    if leaks:
        print(f"LEAKAGE DETECTED in {len(leaks)} field(s):\n")
        for field, value_to_labels in leaks:
            print(f"  {field}:")
            for value, labels in value_to_labels.items():
                print(f"    {value!r} -> always {labels}")
        return False

    print(f"OK: no perfectly-separating fields found across {len(records)} records")
    return True


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
