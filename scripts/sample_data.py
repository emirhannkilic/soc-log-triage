"""
Stratified sampling for the phishing_pot corpus (v3 plan, Adim 1).

Draws a proportional sample from data/phishing_pot/email/*.eml, stratified by
(year-month, has_authentication_results) so a single phishing campaign or
sending infrastructure can't dominate the training set. Plain random sampling
would risk over-representing whichever burst of the ~8600 files happens to be
largest.

Output: data/processed/phishing_sample.jsonl — one line per selected file,
containing the file path and the two stratification keys, so downstream
steps (parser, teacher generation) can locate the original .eml.
"""
import argparse
import email
import json
import random
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PHISHING_DIR = PROJECT_ROOT / "data" / "phishing_pot" / "email"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

MONTH_RE_YEARS = [str(y) for y in range(2015, 2031)]


def extract_year_month(date_header: str) -> str:
    """Best-effort year-month key from an RFC 2822 Date header; falls back
    to 'unknown' rather than raising, since malformed dates are common in
    phishing samples and shouldn't crash sampling."""
    if not date_header:
        return "unknown"
    for year in MONTH_RE_YEARS:
        idx = date_header.find(year)
        if idx == -1:
            continue
        # crude month extraction: look for a 3-letter month name near the year
        for month in ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                      "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]:
            if month in date_header:
                return f"{year}-{month}"
        return year
    return "unknown"


def build_strata(files: list[Path]) -> dict[tuple[str, bool], list[Path]]:
    strata: dict[tuple[str, bool], list[Path]] = defaultdict(list)
    for path in files:
        try:
            with open(path, "rb") as f:
                msg = email.message_from_binary_file(f)
        except Exception:
            continue

        year_month = extract_year_month(str(msg.get("Date", "")))
        has_auth = bool(msg.get("Authentication-Results"))
        strata[(year_month, has_auth)].append(path)
    return strata


def proportional_sample(
    strata: dict[tuple[str, bool], list[Path]], target_count: int, seed: int
) -> list[Path]:
    rng = random.Random(seed)
    total = sum(len(v) for v in strata.values())
    if total == 0:
        return []

    selected: list[Path] = []
    remainders: list[tuple[float, tuple[str, bool]]] = []

    for key, paths in strata.items():
        exact_share = target_count * len(paths) / total
        take = int(exact_share)
        take = min(take, len(paths))
        chosen = rng.sample(paths, take)
        selected.extend(chosen)
        remainders.append((exact_share - take, key))

    # Distribute leftover slots (from integer truncation) to the strata with
    # the largest fractional remainder first, largest-remainder method.
    shortfall = target_count - len(selected)
    if shortfall > 0:
        remainders.sort(key=lambda r: r[0], reverse=True)
        already_taken = {key: set(strata[key]).intersection(selected) for key in strata}
        for _, key in remainders:
            if shortfall <= 0:
                break
            remaining_in_stratum = [p for p in strata[key] if p not in already_taken[key]]
            if not remaining_in_stratum:
                continue
            pick = rng.choice(remaining_in_stratum)
            selected.append(pick)
            already_taken[key].add(pick)
            shortfall -= 1

    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=1500,
                         help="target sample size (default 1500)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    files = sorted(PHISHING_DIR.glob("*.eml"))
    print(f"Found {len(files)} phishing_pot files")

    print("Building strata (year-month x has_authentication_results) ...")
    strata = build_strata(files)
    print(f"  {len(strata)} strata found")
    for key in sorted(strata.keys(), key=lambda k: (k[0], k[1])):
        print(f"    {key}: {len(strata[key])} files")

    selected = proportional_sample(strata, args.count, args.seed)
    print(f"\nSelected {len(selected)} files (target was {args.count})")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "phishing_sample.jsonl"
    with open(out_path, "w") as f:
        for path in selected:
            f.write(json.dumps({"path": str(path.relative_to(PROJECT_ROOT))},
                                ensure_ascii=False) + "\n")

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
