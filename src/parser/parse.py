"""
Top-level entry point: .eml file -> EmailFacts. Combines headers.py,
urls.py, attachments.py, body.py per v3 plan section 4.
"""
import email
from email import policy
from email.header import decode_header
from pathlib import Path

from src.parser.attachments import extract_attachment_facts
from src.parser.body import extract_body_facts, get_body
from src.parser.headers import (
    parse_address_facts,
    parse_authentication_results,
    parse_routing_facts,
)
from src.parser.urls import extract_url_facts
from schemas.facts import AttachmentFacts, EmailFacts, UrlFacts


def _decode_mime_header(raw_value) -> str | None:
    """Decodes RFC 2047 encoded-words (=?UTF-8?B?...?=) in a header value,
    e.g. Subject. Compat32 sometimes hands back an email.header.Header
    object instead of a plain str for malformed headers, so this coerces
    to str first. Falls back to that raw string on any decode failure —
    better to show something than to raise on a malformed header."""
    if raw_value is None:
        return None
    raw_str = str(raw_value)
    try:
        parts = decode_header(raw_str)
        decoded = "".join(
            chunk.decode(charset or "utf-8", errors="replace")
            if isinstance(chunk, bytes) else chunk
            for chunk, charset in parts
        )
        return decoded
    except Exception:
        return raw_str


def parse_eml(path: Path) -> EmailFacts:
    with open(path, "rb") as f:
        raw_bytes = f.read()

    # Compat32 (default, no-args parser) for header reads — policy.default's
    # strict address-object parsing crashes on malformed From/To headers,
    # which are common in phishing samples. See PROGRESS.md "v2'den Kalan
    # Referans Bilgiler" for the original AttributeError this avoids.
    raw_msg = email.message_from_bytes(raw_bytes)
    # policy.default for the body/MIME side, since get_body() needs it.
    msg = email.message_from_bytes(raw_bytes, policy=policy.default)

    address_facts = parse_address_facts(raw_msg)
    from_domain = address_facts["from_domain"]

    auth_facts = parse_authentication_results(raw_msg, from_domain)
    routing_facts = parse_routing_facts(raw_msg, from_domain)
    attachment_facts = extract_attachment_facts(raw_msg)

    try:
        raw_body, is_html = get_body(msg)
    except Exception:
        raw_body, is_html = "", False

    body_facts = extract_body_facts(raw_body, is_html)
    url_facts = extract_url_facts(raw_body, is_html)

    subject = _decode_mime_header(raw_msg.get("Subject"))
    date_raw = raw_msg.get("Date")
    date = str(date_raw) if date_raw is not None else None

    return EmailFacts(
        **auth_facts,
        **address_facts,
        **routing_facts,
        urls=[UrlFacts(**f) for f in url_facts],
        attachments=[AttachmentFacts(**f) for f in attachment_facts],
        **body_facts,
        subject=subject,
        date=date,
    )
