"""
v3 plan, Adim 1 (negatif sinif): converts the Gmail Takeout mbox export
(data/raw/gmail/Takeout/Posta/*.mbox) into individual .eml files, then draws
a stratified 1000-message sample analogous to sample_data.py's phishing_pot
sampling — stratified by (year-month, has_authentication_results) so no
single period dominates.

Also reports a rough language distribution (Turkish vs other) over the full
mbox corpus, per CLAUDE.md "Riskler": if Turkish share is under 30%, this is
a signal to sample additional bank/university/e-commerce notification mail
before finalizing (see plan v3 section 11).

Output:
  data/raw/gmail/eml/*.eml      — every message, individually
  data/processed/gmail_sample.jsonl  — selected 1000-message sample (paths)
"""
import argparse
import email
import email.policy
import json
import mailbox
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GMAIL_DIR = PROJECT_ROOT / "data" / "raw" / "gmail"
MBOX_DIR = GMAIL_DIR / "Takeout" / "Posta"
EML_DIR = GMAIL_DIR / "eml"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

MBOX_FILES = ["Gelen Kutusu.mbox", "Gönderilenler.mbox"]

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
YEARS = [str(y) for y in range(2015, 2031)]

# Rough, cheap heuristic — not a real language detector. Counts Turkish
# diacritics and a handful of very common Turkish function words/particles
# that rarely appear in English text. Good enough for a "is this roughly
# Turkish or not" split at sampling time, not for anything downstream.
TURKISH_CHARS = set("çğıöşüÇĞİÖŞÜ")
TURKISH_WORDS = {
    "ve", "bir", "bu", "için", "ile", "de", "da", "çok", "ama", "gibi",
    "var", "yok", "merhaba", "selam", "teşekkür", "iyi", "günler",
}


def is_turkish(text: str) -> bool:
    if any(c in TURKISH_CHARS for c in text):
        return True
    words = set(re.findall(r"[a-zçğıöşü]+", text.lower()))
    hits = len(words & TURKISH_WORDS)
    return hits >= 2


def extract_year_month(date_header: str) -> str:
    if not date_header:
        return "unknown"
    for year in YEARS:
        if year not in date_header:
            continue
        for month in MONTH_NAMES:
            if month in date_header:
                return f"{year}-{month}"
        return year
    return "unknown"


def mbox_to_eml_files() -> list[Path]:
    EML_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for mbox_name in MBOX_FILES:
        mbox_path = MBOX_DIR / mbox_name
        if not mbox_path.exists():
            print(f"  WARNING: {mbox_path} not found, skipping")
            continue

        label = "sent" if "nderilen" in mbox_name else "inbox"
        print(f"Reading {mbox_name} ...")
        mbox = mailbox.mbox(str(mbox_path))
        count = 0
        for i, msg in enumerate(mbox):
            out_path = EML_DIR / f"{label}-{i}.eml"
            with open(out_path, "wb") as f:
                f.write(msg.as_bytes())
            written.append(out_path)
            count += 1
        print(f"  {count} messages -> {EML_DIR}")
    return written


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
        take = min(int(exact_share), len(paths))
        chosen = rng.sample(paths, take)
        selected.extend(chosen)
        remainders.append((exact_share - take, key))

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


def report_language_distribution(files: list[Path]) -> None:
    sample_size = min(500, len(files))
    sample = random.Random(42).sample(files, sample_size)
    counts = Counter()
    for path in sample:
        try:
            with open(path, "rb") as f:
                msg = email.message_from_binary_file(f, policy=email.policy.default)
            body = msg.get_body(preferencelist=("plain", "html"))
            text = body.get_content() if body else ""
            subject = str(msg.get("Subject", ""))
        except Exception:
            continue
        counts["turkish" if is_turkish(subject + " " + text[:500]) else "other"] += 1

    total = sum(counts.values())
    if total == 0:
        print("Language check: no messages sampled")
        return
    tr_share = counts["turkish"] / total
    print(f"Language distribution (heuristic, {total} sampled): "
          f"Turkish={counts['turkish']} ({tr_share:.0%}), other={counts['other']}")
    if tr_share < 0.30:
        print("  WARNING: Turkish share under 30% — per CLAUDE.md 'Riskler', "
              "consider sampling additional bank/university/e-commerce mail.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=1000,
                         help="target sample size (default 1000)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("Converting mbox files to individual .eml files ...")
    files = mbox_to_eml_files()
    print(f"\nTotal messages: {len(files)}")

    print("\nChecking language distribution ...")
    report_language_distribution(files)

    print("\nBuilding strata (year-month x has_authentication_results) ...")
    strata = build_strata(files)
    print(f"  {len(strata)} strata found")

    selected = proportional_sample(strata, args.count, args.seed)
    print(f"\nSelected {len(selected)} files (target was {args.count})")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "gmail_sample.jsonl"
    with open(out_path, "w") as f:
        for path in selected:
            f.write(json.dumps({"path": str(path.relative_to(PROJECT_ROOT))},
                                ensure_ascii=False) + "\n")

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
