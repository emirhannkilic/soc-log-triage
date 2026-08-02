"""
v3 holdout-fix-tasks.md T4: scans anonymized JSONL output for the things
that must never survive anonymization — the account holder's own email
address, their own name (via anonymize.py's literal
_OWN_NAME_VARIANTS_RE), and any name still sitting in a structured
salutation (via anonymize.py's _SALUTATION_RE) that anonymize_facts()
should have already replaced with an alias.

This deliberately does NOT do a global substring search for every name
ever recorded in the alias map. That was tried and produced ~150 false
positives on the Gmail corpus: _SALUTATION_RE first mis-captured generic
mass-mail greetings ("Dear Customer,", "Hi Friend,", "Dear Valued
Recipient,") as if they were real names, filing "Customer"/"Friend"/
"There"/"Community" etc. into the alias map — and even after
_NAME_PATTERN was tightened to exclude those, a short real name (e.g.
"Emir") or a name that's also a common word/substring is fundamentally
unsafe to grep for globally, since it will coincidentally appear in
unrelated text (a brand's display_name, an unrelated sentence) that has
nothing to do with the salutation it was originally redacted from. The
thing that actually matters — did anonymize_facts() fail to redact a name
it should have — is fully captured by re-running _SALUTATION_RE against
the *output*: if it still matches, redaction didn't happen.

Domains, IPs, and third-party email addresses are NOT checked here — per
the current anonymize.py scope, those are deliberately left real (genuine
training signal, not personal data). See anonymize.py's module docstring
for the reasoning.

Only meant to be run against data/processed/gmail_facts.jsonl and
data/holdout/* — phishing_pot is a public dataset, running this against
phishing_facts.jsonl would just flag phishing_pot's own (irrelevant)
sender/domain data as false "leaks".

Usage: python3 scripts/check_anonymization.py data/processed/gmail_facts.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from anonymize import OWN_EMAIL, _OWN_NAME_VARIANTS_RE, _SALUTATION_RE  # noqa: E402


def check_text(text: str, source: str) -> list[str]:
    issues = []
    if OWN_EMAIL.lower() in text.lower():
        issues.append(f"{source}: account holder's own email address leaked")
    if _OWN_NAME_VARIANTS_RE.search(text):
        issues.append(f"{source}: account holder's own name leaked")
    m = _SALUTATION_RE.search(text)
    if m:
        issues.append(f"{source}: unredacted salutation name: {m.group(0)!r}")
    return issues


def check_jsonl_file(path: Path) -> list[str]:
    issues = []
    with open(path) as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            text_blob = json.dumps(record, ensure_ascii=False)
            issues.extend(check_text(text_blob, f"{path.name}:{i}"))
    return issues


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path)
    args = parser.parse_args()

    all_issues = []
    for path in args.files:
        if not path.exists():
            print(f"SKIP: {path} not found")
            continue
        all_issues.extend(check_jsonl_file(path))

    if all_issues:
        print(f"FAILED: {len(all_issues)} anonymization leak(s) found:\n")
        for issue in all_issues[:50]:
            print(f"  {issue}")
        if len(all_issues) > 50:
            print(f"  ... and {len(all_issues) - 50} more")
        sys.exit(1)

    print(f"OK: no anonymization leaks found across {len(args.files)} file(s)")


if __name__ == "__main__":
    main()
