"""
Top-level entry point: .eml file -> EmailFacts. Combines headers.py,
urls.py, attachments.py, body.py per v3 plan section 4.
"""
import email
import re
from email import policy
from pathlib import Path

from src.parser.attachments import extract_attachment_facts
from src.parser.body import extract_body_facts, get_body
from src.parser.headers import (
    _decode_rfc2047,
    parse_address_facts,
    parse_authentication_results,
    parse_routing_facts,
)
from src.parser.urls import extract_url_facts
from schemas.facts import AttachmentFacts, EmailFacts, UrlFacts


# Single implementation lives in headers.py — display-name decoding needs it
# too, and headers.py cannot import from here (this module already imports
# that one, so the reverse direction would be a cycle).
_decode_mime_header = _decode_rfc2047

# Headers whose display-name text matters for downstream logic (brand
# matching, künye rendering, the report prompt). Compat32 decodes a header
# containing raw (non-RFC-2047-encoded) multi-byte UTF-8 — e.g. a literal
# "'" (U+2019) sitting unencoded in "Men's Wellness Today" — as Latin-1,
# replacing every non-ASCII byte with U+FFFD. That loss is irreversible
# once str(header) has been called: _decode_rfc2047 cannot recover bytes
# that were never valid UTF-8 to begin with, because by then they aren't
# there anymore. Found on a real sample ("Men's Wellness Today" rendered as
# "Men��s Wellness Today" throughout the report).
_REHEADER_FIELDS = ("From", "To", "Cc", "Reply-To", "Subject")


def _reheader_utf8(raw_msg: email.message.Message, raw_bytes: bytes) -> None:
    """Re-decode select headers from the raw header block as UTF-8,
    overwriting Compat32's lossy Latin-1 decoding in place.

    Only touches a header if Compat32's version contains U+FFFD — a header
    that decoded cleanly is left as Compat32 produced it, since Compat32
    still handles RFC 2047 encoded-words and folded lines correctly; this
    only patches the one case it gets wrong (unencoded raw UTF-8 bytes).
    """
    header_block = raw_bytes.split(b"\r\n\r\n", 1)[0].split(b"\n\n", 1)[0]
    for field in _REHEADER_FIELDS:
        current = raw_msg.get(field)
        if current is None or "�" not in str(current):
            continue
        # Match the header and any RFC 5322 folded continuation lines.
        m = re.search(
            rf"(?im)^{re.escape(field)}:[ \t]*(.*(?:\r?\n[ \t]+.*)*)".encode(),
            header_block,
        )
        if not m:
            continue
        decoded = m.group(1).decode("utf-8", errors="replace").strip()
        if "�" not in decoded:
            del raw_msg[field]
            raw_msg[field] = decoded


def parse_eml(path: Path) -> EmailFacts:
    with open(path, "rb") as f:
        raw_bytes = f.read()

    # Compat32 (default, no-args parser) for header reads — policy.default's
    # strict address-object parsing crashes on malformed From/To headers,
    # which are common in phishing samples. See PROGRESS.md "v2'den Kalan
    # Referans Bilgiler" for the original AttributeError this avoids.
    raw_msg = email.message_from_bytes(raw_bytes)
    _reheader_utf8(raw_msg, raw_bytes)
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

    url_facts = extract_url_facts(raw_body, is_html)
    # credential_request (T3) needs to know whether the message offers an
    # action channel (external link / attachment) alongside a request verb
    # + target object — url_facts/attachment_facts must be computed first.
    has_action_channel = bool(url_facts) or bool(attachment_facts)
    body_facts = extract_body_facts(
        raw_body,
        is_html,
        has_action_channel=has_action_channel,
        has_attachments=bool(attachment_facts),
    )

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
