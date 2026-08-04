"""
v3 plan Adim 3: anonymizes the user's personal identity out of EmailFacts
before they leave the local machine (e.g. before teacher generation, Adim
6-7). This data feeds LoRA fine-tuning (Adim 9) — real domain structure,
sender addresses, and IPs are genuine training signal, so they are kept
real. Only the account holder's own identity (their name, their own email
address) is redacted, since that's the one thing in this corpus that's
truly personal rather than already-public sender/brand/infrastructure
information.

holdout-fix-tasks.md T4 originally flagged this module for anonymizing
domains/URLs inconsistently (facts said domain-3141.test, body_text said
netflix.com) and asked to extend coverage. Working through it surfaced a
bigger question: this data is training input, not a public release — an
LLM being fine-tuned on "domain-3141.test" instead of "netflix.com" learns
nothing about real domain structure, which actively hurts training quality
for no privacy benefit (sender/brand domains are already public). So this
module's scope narrowed instead of widened: domain anonymization was
REMOVED entirely (all domains, IPs, and third-party email addresses are
now left real), and what's left is redaction of exactly the account
holder's own identity — their name and their own email address (matched
literally, not "any email that happens to be on gmail.com" — someone
else's Gmail address is not the user's personal data).

Redacted: the account holder's own email address (OWN_EMAIL below,
wherever it appears — headers, body_text, subject, URLs) and their own
name, via two complementary matchers: a literal list of known name
variants (_OWN_NAME_VARIANTS_RE — catches leaks in any context, e.g. a
marketing subject line that drops the recipient's first name mid-
sentence, which isn't a salutation at
all) and a structured-salutation pattern ("Sayın X Y,", "Dear X Y,") that
also catches THIRD PARTIES' names when the account holder's own mailbox
addresses someone else by name (e.g. "Sayın <third party>," in an order
confirmation) — a real person's name is personal data regardless of whose
mailbox it appears in. Attachment filenames are still anonymized — unlike
domains, a filename like "fatura_ahmet_yilmaz.pdf" can itself carry a real
person's name.

NOT redacted (deliberately, as of this revision): all domains (from_domain,
return_path_domain, urls[].href_domain, etc.), all IP addresses
(first_received_ip), and any email address that isn't the account holder's
own — these are real training signal, not personal data. display_name was
never redacted (see the rule engine's display_name_brand_mismatch signal,
plan section 4.2). General third-party names in free text are still NOT
detected — that remains out of scope, same reasoning as before (no
reliable NER, false positive/negative risk).

The alias map (data/raw/gmail/anonymization_map.json, gitignored) tracks
name aliases only now, so the same real name always maps to the same
alias across records.

Run scripts/check_anonymization.py after this to verify the account
holder's identity doesn't survive — that script is the actual guarantee,
not this docstring.
"""
import argparse
import json
import os
import re
import urllib.parse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MAP_PATH = PROJECT_ROOT / "data" / "raw" / "gmail" / "anonymization_map.json"

OWN_EMAIL_ALIAS = "user@example.test"
OWN_NAME_ALIAS = "Ad Soyad 0"

# The account holder's own identity is READ FROM THE ENVIRONMENT, never
# hardcoded — this file is committed to a public repository, and baking the
# maintainer's real name and address into it would leak exactly the personal
# data the rest of this module exists to remove.
#
# Configure via a local, gitignored `.env.anonymize` (see
# `.env.anonymize.example`) or plain environment variables:
#
#   ANONYMIZE_OWN_EMAIL="you@example.com"
#   ANONYMIZE_OWN_NAME_VARIANTS="Firstname Lastname|Firstname|Nickname Lastname"
#
# The variants list is a LITERAL alternation, not general NER. Unlike
# _SALUTATION_RE (which matches any name in a structured "Sayın X,"
# greeting), it exists because the salutation pattern alone missed real
# leaks — e.g. a marketing subject line that drops the recipient's first
# name mid-sentence, which isn't a salutation at all. Matching a small,
# known, literal set of the account holder's own name variants carries
# essentially no false-positive risk and catches leaks in contexts a
# structural pattern can't anticipate.
#
# IMPORTANT — do NOT list a bare surname on its own. Surnames are shared by
# unrelated third parties (an order confirmation addressed to a relative,
# say), and matching one alone would redact a different real person's name
# as if it were the account holder's. List only the first name and
# multi-word combinations that include it.


def _load_local_env(path: Path) -> None:
    """Populate os.environ from a simple KEY=VALUE file, if present.

    Deliberately minimal (no python-dotenv dependency): blank lines and
    `#` comments are skipped, surrounding quotes are stripped, and existing
    environment variables always win so an explicit export can override the
    file.
    """
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_local_env(PROJECT_ROOT / ".env.anonymize")

OWN_EMAIL = os.environ.get("ANONYMIZE_OWN_EMAIL", "").strip()

_OWN_NAME_VARIANTS = [
    variant.strip()
    for variant in os.environ.get("ANONYMIZE_OWN_NAME_VARIANTS", "").split("|")
    if variant.strip()
]

# Matches nothing when unconfigured. `(?!)` is a never-satisfiable negative
# lookahead, so an unset environment degrades to "redact no personal name"
# rather than to an empty alternation like `\b()\b`, which would match at
# every word boundary and corrupt the whole corpus.
_OWN_NAME_VARIANTS_RE = re.compile(
    r"\b(" + "|".join(re.escape(v).replace(r"\ ", r"\s+") for v in _OWN_NAME_VARIANTS) + r")\b"
    if _OWN_NAME_VARIANTS
    else r"(?!)"
)

# Structured salutation patterns — deliberately narrow (not general NER).
# Matches "Sayın Ad Soyad," / "Dear Ad Soyad," / "Hi Ad," etc., where a
# capitalized 1-3 word run follows a known salutation marker. This catches
# the concrete leak found in holdout-fix-tasks.md T4 (a third party named
# in an order-confirmation salutation, and the mailbox owner greeted by
# first name in a WiFi registration) without
# attempting to find names anywhere in free-flowing prose, which is what
# the earlier "no NER available" decision in PROGRESS.md was about.
#
# Mass-mail salutations frequently address the recipient generically
# ("Dear Customer,", "Hi Friend,", "Dear Valued Recipient,") rather than by
# name. Without excluding these, the generic word gets filed into the
# alias map as if it were a real person's name — and since
# check_anonymization.py flags any alias-map name reappearing anywhere in
# the corpus, a short common word like "Friend" or "There" then produces
# false-positive "leaks" wherever it happens to appear in unrelated text
# (found empirically: "Friend of the House Jude Bellingham" in an LV
# newsletter, "There are only a few days left..." in a GoDaddy email).
# _GENERIC_SALUTATION_NOUNS excludes the concrete false positives found in
# the Gmail corpus; it is not an exhaustive dictionary of non-names.
_GENERIC_SALUTATION_NOUNS = (
    r"Customer|User|Friend|Friends|Reader|Client|Member|Shopper|Beloved|"
    r"Beneficiary|Recipient|There|Community|Owner|Developers|Notice|Paid|"
    r"Valued|Dear|My|Important"
)
_NAME_PATTERN = (
    rf"(?!(?:{_GENERIC_SALUTATION_NOUNS})\b)"
    r"[A-ZÇĞİÖŞÜ][a-zçğıöşü']+(?:\s+[A-ZÇĞİÖŞÜ][a-zçğıöşü']+){0,2}"
)
_SALUTATION_RE = re.compile(
    rf"\b(Sayın|Dear|Hi|Hello|Merhaba)\s+({_NAME_PATTERN})\s*[,:]",
)


class AliasMap:
    def __init__(self, path: Path):
        self.path = path
        self.filenames: dict[str, str] = {}
        self.names: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        with open(self.path) as f:
            data = json.load(f)
        self.filenames = data.get("filenames", {})
        self.names = data.get("names", {})

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump({
                "filenames": self.filenames,
                "names": self.names,
            }, f, ensure_ascii=False, indent=2)

    def filename(self, real: str) -> str:
        ext = real.rsplit(".", 1)[-1] if "." in real else ""
        if real not in self.filenames:
            base = f"file-{len(self.filenames):04d}"
            self.filenames[real] = f"{base}.{ext}" if ext else base
        return self.filenames[real]

    def name(self, real: str) -> str:
        # Same real name always maps to the same alias, consistent across
        # records — matters if the same person is addressed more than once
        # in the corpus.
        if real not in self.names:
            self.names[real] = f"Ad Soyad {len(self.names) + 1}"
        return self.names[real]


def redact_own_email_in_text(text: str) -> str:
    """Only the account holder's own address is replaced — a third party's
    email address appearing in the same text is left real (it's not the
    user's personal data, and it's genuine training signal)."""
    if not OWN_EMAIL:
        return text
    return re.sub(re.escape(OWN_EMAIL), OWN_EMAIL_ALIAS, text, flags=re.IGNORECASE)


def _contains_own_identity(decoded: str) -> bool:
    # The OWN_EMAIL guard is load-bearing: an unconfigured (empty) address
    # would make `"" in decoded` true for every string, so every URL would
    # be reported as containing the account holder's identity.
    return (
        bool(OWN_EMAIL) and OWN_EMAIL.lower() in decoded.lower()
    ) or bool(_OWN_NAME_VARIANTS_RE.search(decoded))


def redact_own_identity_in_url(url: str) -> str:
    """Marketing/tracking links (order confirmations, review-request
    pings, WiFi portals) routinely embed the recipient's email and/or name
    as URL-encoded query params, sometimes double-encoded by a tracking
    redirector wrapping the original link. redact_own_email_in_text and
    redact_own_name_in_text only match literal substrings, so an encoded
    occurrence (a `name=` param that only reveals the recipient after two
    rounds of percent-decoding) survives them untouched —
    found empirically via check_anonymization.py on the Gmail corpus.
    Surgically patching a match back into a multiply-encoded string is
    fragile, and these tracking URLs carry no phishing-detection signal
    worth preserving, so on a hit the entire URL is replaced with a
    placeholder rather than partially redacted."""
    decoded = url
    for _ in range(3):  # bounded: real redirectors nest at most 1-2 deep
        new = urllib.parse.unquote(decoded)
        if new == decoded:
            break
        decoded = new
    if _contains_own_identity(decoded):
        return "https://redacted.example.test/own-identity-in-url"
    return url


def redact_own_name_in_text(text: str) -> str:
    """Replaces known literal variants of the account holder's own name —
    see _OWN_NAME_VARIANTS_RE for why this exists alongside (not instead
    of) the structured-salutation matcher."""
    return _OWN_NAME_VARIANTS_RE.sub(OWN_NAME_ALIAS, text)


def anonymize_salutation_names_in_text(text: str, alias_map: AliasMap) -> str:
    """Replaces person names appearing in structured salutations ("Sayın X
    Y,", "Dear X Y,", "Hi X,") — see _SALUTATION_RE for scope/rationale.
    Note this catches ANY name in a salutation, not just the account
    holder's — a message addressed to a third party ("Sayın <third party>")
    still has that name redacted, since a real person's name is personal
    data regardless of whose mailbox the message came from."""
    def replace(match: re.Match) -> str:
        marker, name = match.group(1), match.group(2)
        alias = alias_map.name(name)
        trailing = match.group(0)[-1]  # preserve the , or : that was matched
        return f"{marker} {alias}{trailing}"
    return _SALUTATION_RE.sub(replace, text)


def anonymize_facts(facts: dict, alias_map: AliasMap) -> dict:
    facts = dict(facts)  # shallow copy, don't mutate caller's dict

    if facts.get("subject"):
        subject = redact_own_email_in_text(facts["subject"])
        subject = redact_own_name_in_text(subject)
        facts["subject"] = anonymize_salutation_names_in_text(subject, alias_map)
    if facts.get("body_text"):
        body = redact_own_email_in_text(facts["body_text"])
        body = redact_own_name_in_text(body)
        facts["body_text"] = anonymize_salutation_names_in_text(body, alias_map)

    facts["urls"] = [
        {**u, "url": redact_own_identity_in_url(redact_own_email_in_text(u["url"]))}
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
    print(f"Alias map ({len(alias_map.filenames)} filenames, "
          f"{len(alias_map.names)} names) saved to {MAP_PATH}")


if __name__ == "__main__":
    main()
