"""
v3 plan Adim 2+3 end to end: parses every .eml referenced in a sample JSONL
(data/processed/phishing_sample.jsonl or gmail_sample.jsonl) into EmailFacts,
anonymizes each record (scripts/anonymize.py), and writes the result to
data/processed/<name>_facts.jsonl.

This is the actual bridge between "we have sampled .eml paths" (Adim 1) and
"we have anonymized facts ready for the rule engine / teacher generation"
(Adim 4+). Files that fail to parse are logged and skipped rather than
aborting the whole run.
"""
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT.parent))
sys.path.insert(0, str(PROJECT_ROOT))

from anonymize import MAP_PATH, AliasMap, anonymize_facts  # noqa: E402
from src.parser.parse import parse_eml  # noqa: E402

PROCESSED_DIR = PROJECT_ROOT.parent / "data" / "processed"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sample_jsonl", type=Path,
                         help="e.g. data/processed/phishing_sample.jsonl")
    parser.add_argument("--label", required=True,
                         help="output name prefix, e.g. 'phishing' or 'gmail'")
    args = parser.parse_args()

    project_root = PROJECT_ROOT.parent
    paths = [json.loads(line)["path"] for line in open(args.sample_jsonl) if line.strip()]
    print(f"Parsing {len(paths)} files from {args.sample_jsonl} ...")

    alias_map = AliasMap(MAP_PATH)
    written, failed = 0, 0
    out_path = PROCESSED_DIR / f"{args.label}_facts.jsonl"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as out_f:
        for rel_path in paths:
            try:
                facts = parse_eml(project_root / rel_path)
                anonymized = anonymize_facts(facts.model_dump(), alias_map)
                out_f.write(json.dumps(anonymized, ensure_ascii=False) + "\n")
                written += 1
            except Exception as e:
                print(f"  FAILED: {rel_path}: {type(e).__name__}: {e}")
                failed += 1

    alias_map.save()
    print(f"\nWrote {written} records to {out_path} ({failed} failed)")
    print(f"Alias map: {len(alias_map.domains)} domains, {len(alias_map.emails)} emails, "
          f"{len(alias_map.filenames)} filenames, {len(alias_map.ips)} IPs -> {MAP_PATH}")


if __name__ == "__main__":
    main()
