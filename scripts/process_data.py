"""
Parses phishing_pot (.eml, modern headers) and the CMU Enron maildir into a
common schema, filters out Enron bulk/automated mail so the legitimate class
stays limited to real person-to-person correspondence, balances Enron down to
roughly the phishing_pot volume, and writes two JSONL files
(phishing.jsonl, legitimate.jsonl) under data/processed/.

Raw headers are KEPT (not stripped) — the target system prompt requires
SPF/DKIM/DMARC and Received-chain analysis, so the header block is part of
the model input. See CLAUDE.md "System Prompt" for why this overturned the
earlier header-normalize decision.

See CLAUDE.md for why the Enron sender/recipient filter exists.
"""
import argparse
import email
import json
import random
import re
from email import policy
from html.parser import HTMLParser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PHISHING_DIR = PROJECT_ROOT / "data" / "phishing_pot" / "email"
ENRON_DIR = PROJECT_ROOT / "data" / "enron" / "maildir"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

URL_RE = re.compile(r'https?://[^\s"\'<>)]+')

BULK_SENDER_PATTERNS = [
    "listserv",
    "noreply",
    "no-reply",
    "newsletter",
    "mailer-daemon",
    "postmaster",
    "notification",
    "majordomo",
]


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.chunks: list[str] = []

    def handle_data(self, data: str) -> None:
        self.chunks.append(data)

    def text(self) -> str:
        return "".join(self.chunks)


def html_to_text(html: str) -> str:
    extractor = _TextExtractor()
    extractor.feed(html)
    text = extractor.text()
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def extract_urls(body: str) -> list[str]:
    return URL_RE.findall(body)


def get_body_raw(msg: email.message.EmailMessage) -> tuple[str, bool]:
    """Returns (content, is_html)."""
    body = msg.get_body(preferencelist=("plain", "html"))
    if body is None:
        return "", False
    try:
        content = body.get_content()
    except Exception:
        return "", False
    is_html = body.get_content_type() == "text/html"
    return content, is_html


def get_header_block(raw_msg: email.message.Message) -> str:
    # raw_msg is parsed with Compat32 (no strict address/date parsing), so
    # malformed headers (common in phishing samples) don't raise.
    return "\n".join(f"{key}: {value}" for key, value in raw_msg.items())


def parse_eml_file(path: Path) -> dict | None:
    try:
        with open(path, "rb") as f:
            raw_bytes = f.read()
        # Compat32 (default parser) reads headers as plain strings without
        # the strict address-object parsing that policy.default does, which
        # crashes on malformed From/To headers (frequent in phishing samples,
        # e.g. group-syntax addresses that trip email.headerregistry).
        raw_msg = email.message_from_bytes(raw_bytes)
        # policy.default is still used for the body, since MIME/multipart
        # handling (get_body) is what we actually need from it.
        msg = email.message_from_bytes(raw_bytes, policy=policy.default)
    except Exception:
        return None

    sender = str(raw_msg.get("From", "")).strip()
    subject = str(raw_msg.get("Subject", "")).strip()
    headers = get_header_block(raw_msg)

    try:
        raw_body, is_html = get_body_raw(msg)
    except Exception:
        raw_body, is_html = "", False

    urls = extract_urls(raw_body)
    body = html_to_text(raw_body) if is_html else raw_body.strip()

    if not sender and not subject and not body:
        return None

    return {
        "sender": sender,
        "subject": subject,
        "headers": headers,
        "body": body,
        "urls": urls,
    }


def is_bulk_sender(address: str) -> bool:
    lowered = address.lower()
    return any(pattern in lowered for pattern in BULK_SENDER_PATTERNS)


def load_phishing_samples() -> list[dict]:
    samples = []
    for path in sorted(PHISHING_DIR.glob("*.eml")):
        parsed = parse_eml_file(path)
        if parsed is not None:
            samples.append(parsed)
    return samples


def load_enron_samples() -> list[dict]:
    samples = []
    for path in ENRON_DIR.rglob("*"):
        if not path.is_file():
            continue

        parsed = parse_eml_file(path)
        if parsed is None:
            continue

        sender = parsed["sender"]
        if not sender or is_bulk_sender(sender):
            continue

        samples.append(parsed)
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ratio",
        type=float,
        default=1.5,
        help="legitimate:phishing volume ratio to sample down to (default 1.5)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    print(f"Loading phishing_pot samples from {PHISHING_DIR} ...")
    phishing_samples = load_phishing_samples()
    print(f"  {len(phishing_samples)} phishing samples parsed")

    print(f"Loading Enron samples from {ENRON_DIR} ...")
    enron_samples = load_enron_samples()
    print(f"  {len(enron_samples)} legitimate samples parsed after bulk-sender filter")

    target_legitimate_count = min(
        len(enron_samples), round(len(phishing_samples) * args.ratio)
    )
    legitimate_samples = random.sample(enron_samples, target_legitimate_count)
    print(f"  downsampled to {len(legitimate_samples)} legitimate samples "
          f"(ratio {args.ratio}:1 vs phishing)")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    phishing_out = OUTPUT_DIR / "phishing.jsonl"
    with open(phishing_out, "w") as f:
        for sample in phishing_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    legitimate_out = OUTPUT_DIR / "legitimate.jsonl"
    with open(legitimate_out, "w") as f:
        for sample in legitimate_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"Wrote {phishing_out}")
    print(f"Wrote {legitimate_out}")


if __name__ == "__main__":
    main()
