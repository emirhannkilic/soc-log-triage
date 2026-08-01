"""
v3 plan Adim 3: anonymizes EmailFacts before they leave the local machine
(e.g. before teacher generation, Adim 6-7). Real domains and email addresses
are replaced with consistent aliases — the same real value always maps to
the same alias, so header consistency signals (domain X matches domain Y)
survive anonymization. See CLAUDE.md "Kilitlenen Kararlar" for why the
mapping must be consistent rather than random.

Anonymized: domain fields (from_domain, return_path_domain, reply_to_domain,
message_id_domain, dkim_domain, urls[].href_domain/anchor_text_domain),
urls[].url, attachments[].filename, first_received_ip.

NOT anonymized: outcome/boolean fields (spf_result, *_mismatch,
*_matches_from, has_html_form, etc.) — they carry no identifying
information. display_name is also left untouched: it's usually a brand or
sender-chosen name rather than sensitive personal data, and the rule
engine's display_name_brand_mismatch signal (plan section 4.2) reads its
actual content — anonymizing it would silently break that check.
subject/body_text: only email addresses inside them (found via regex) are
replaced; person names are NOT detected or redacted — reliable NER isn't
available here, and attempting a heuristic name-scrub risks both false
positives (redacting ordinary words) and false negatives (a false sense of
privacy). See PROGRESS.md for the decision record.

The alias map is stored in data/raw/gmail/anonymization_map.json (gitignored,
never leaves the local machine) so re-running this script — or anonymizing
more samples later — produces the same aliases for values already seen.
"""
import argparse
import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MAP_PATH = PROJECT_ROOT / "data" / "raw" / "gmail" / "anonymization_map.json"

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")

# A handful of common brand/institution domains that must stay real for the
# rule engine's brand-mismatch signal (display_name_brand_mismatch, section
# 4.2) to keep working — anonymizing e.g. "paypal.com" to "domain-042.test"
# would make every legitimate PayPal email look like a display-name spoof.
_PRESERVE_DOMAINS = {
    "paypal.com", "apple.com", "microsoft.com", "google.com", "amazon.com",
    "netflix.com", "vakifbank.com.tr", "garanti.com.tr", "isbank.com.tr",
    "akbank.com", "ziraatbank.com.tr", "yapikredi.com.tr", "dhl.com",
    "fedex.com", "ups.com", "gmail.com",
}


class AliasMap:
    def __init__(self, path: Path):
        self.path = path
        self.domains: dict[str, str] = {}
        self.emails: dict[str, str] = {}
        self.filenames: dict[str, str] = {}
        self.ips: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        with open(self.path) as f:
            data = json.load(f)
        self.domains = data.get("domains", {})
        self.emails = data.get("emails", {})
        self.filenames = data.get("filenames", {})
        self.ips = data.get("ips", {})

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump({
                "domains": self.domains,
                "emails": self.emails,
                "filenames": self.filenames,
                "ips": self.ips,
            }, f, ensure_ascii=False, indent=2)

    def domain(self, real: str | None) -> str | None:
        if real is None:
            return None
        if real.lower() in _PRESERVE_DOMAINS:
            return real
        if real not in self.domains:
            self.domains[real] = f"domain-{len(self.domains):04d}.test"
        return self.domains[real]

    def email(self, real: str) -> str:
        local, _, domain = real.partition("@")
        if domain.lower() in _PRESERVE_DOMAINS:
            return real
        if real not in self.emails:
            self.emails[real] = f"user-{len(self.emails):04d}@{self.domain(domain)}"
        return self.emails[real]

    def filename(self, real: str) -> str:
        ext = real.rsplit(".", 1)[-1] if "." in real else ""
        if real not in self.filenames:
            base = f"file-{len(self.filenames):04d}"
            self.filenames[real] = f"{base}.{ext}" if ext else base
        return self.filenames[real]

    def ip(self, real: str) -> str:
        if real not in self.ips:
            # Keep it looking like an IP (some downstream code/regex may
            # expect IP-shaped strings) but from a documented test range.
            n = len(self.ips)
            self.ips[real] = f"203.0.113.{n % 256}"
        return self.ips[real]


def anonymize_email_addresses_in_text(text: str, alias_map: AliasMap) -> str:
    def replace(match: re.Match) -> str:
        return alias_map.email(match.group(0))
    return EMAIL_RE.sub(replace, text)


def anonymize_url(url: str, alias_map: AliasMap) -> str:
    m = re.match(r"(https?://)([^/\s]+)(.*)", url)
    if not m:
        return url
    scheme, netloc, rest = m.groups()
    userinfo, _, host_port = netloc.rpartition("@")
    host, _, port = host_port.partition(":")
    aliased_host = alias_map.domain(host.lower()) or host
    new_netloc = aliased_host + (f":{port}" if port else "")
    if userinfo:
        new_netloc = f"{userinfo}@{new_netloc}"
    return f"{scheme}{new_netloc}{rest}"


def anonymize_facts(facts: dict, alias_map: AliasMap) -> dict:
    facts = dict(facts)  # shallow copy, don't mutate caller's dict

    for field in ("from_domain", "return_path_domain", "reply_to_domain",
                  "message_id_domain", "dkim_domain"):
        facts[field] = alias_map.domain(facts.get(field))

    # display_name is deliberately left untouched — it's usually a brand or
    # sender-chosen display name, not sensitive on its own, and the rule
    # engine's display_name_brand_mismatch signal (plan section 4.2) reads
    # its actual content. Anonymizing it would silently break that check.

    if facts.get("first_received_ip"):
        facts["first_received_ip"] = alias_map.ip(facts["first_received_ip"])

    if facts.get("subject"):
        facts["subject"] = anonymize_email_addresses_in_text(facts["subject"], alias_map)
    if facts.get("body_text"):
        facts["body_text"] = anonymize_email_addresses_in_text(facts["body_text"], alias_map)

    facts["urls"] = [
        {
            **u,
            "url": anonymize_url(u["url"], alias_map),
            "href_domain": alias_map.domain(u.get("href_domain")),
            "anchor_text_domain": alias_map.domain(u.get("anchor_text_domain")),
        }
        for u in facts.get("urls", [])
    ]

    facts["attachments"] = [
        {**a, "filename": alias_map.filename(a["filename"])}
        for a in facts.get("attachments", [])
    ]

    return facts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSONL file of facts dicts (one per line)")
    parser.add_argument("output", type=Path, help="destination JSONL for anonymized facts")
    args = parser.parse_args()

    alias_map = AliasMap(MAP_PATH)

    count = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.input) as inp, open(args.output, "w") as out:
        for line in inp:
            line = line.strip()
            if not line:
                continue
            facts = json.loads(line)
            anonymized = anonymize_facts(facts, alias_map)
            out.write(json.dumps(anonymized, ensure_ascii=False) + "\n")
            count += 1

    alias_map.save()
    print(f"Anonymized {count} records -> {args.output}")
    print(f"Alias map ({len(alias_map.domains)} domains, {len(alias_map.emails)} emails, "
          f"{len(alias_map.filenames)} filenames, {len(alias_map.ips)} IPs) saved to {MAP_PATH}")


if __name__ == "__main__":
    main()
