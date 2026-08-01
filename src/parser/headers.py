"""
Parses Authentication-Results, address headers, and Message-ID/routing facts
from an email.message.Message. See v3 plan section 4.1-4.3 and
schemas/facts.py.

Real-world Authentication-Results headers are messy: Gmail emits multiple
dkim= results on one header (one per signing domain), folded across several
lines. This module extracts the overall spf=/dmarc= verdict (there's only
ever one of each) and, for dkim=, prefers the result whose header.i domain
matches the From domain — that's the signature that actually matters for
"does this message's claimed sender check out," not just any passing
signature (a phishing email can carry a valid DKIM signature for an
unrelated domain it doesn't claim to be).
"""
import re
from email.message import Message
from email.utils import getaddresses, parseaddr

_AUTH_RESULT_RE = re.compile(r"\b(spf|dkim|dmarc)=([a-zA-Z0-9_-]+)")
_DKIM_DOMAIN_RE = re.compile(r"dkim=([a-zA-Z0-9_-]+)[^;]*?header\.[id]=@?([\w.-]+)")


def _domain_of(address: str) -> str | None:
    _, addr = parseaddr(address)
    if not addr or "@" not in addr:
        return None
    return addr.rsplit("@", 1)[-1].lower()


def parse_authentication_results(msg: Message, from_domain: str | None) -> dict:
    """Returns spf_result, dkim_result, dmarc_result, dkim_domain,
    dkim_domain_matches_from."""
    header = msg.get("Authentication-Results", "")
    header = re.sub(r"\s+", " ", header)

    spf_result = dmarc_result = None
    for keyword, value in _AUTH_RESULT_RE.findall(header):
        if keyword == "spf" and spf_result is None:
            spf_result = value.lower()
        elif keyword == "dmarc" and dmarc_result is None:
            dmarc_result = value.lower()

    dkim_matches = _DKIM_DOMAIN_RE.findall(header)
    dkim_result = None
    dkim_domain = None
    if dkim_matches:
        # Prefer the dkim= entry whose signing domain matches From, since
        # that's the one that actually authenticates the claimed sender.
        matching = [(r, d) for r, d in dkim_matches
                    if from_domain and d.lower() == from_domain]
        chosen = matching[0] if matching else dkim_matches[0]
        dkim_result, dkim_domain = chosen[0].lower(), chosen[1].lower()

    dkim_domain_matches_from = (
        None if dkim_domain is None or from_domain is None
        else dkim_domain == from_domain
    )

    return {
        "spf_result": spf_result,
        "dkim_result": dkim_result,
        "dkim_domain": dkim_domain,
        "dmarc_result": dmarc_result,
        "dkim_domain_matches_from": dkim_domain_matches_from,
    }


# Common brand names that phishing display-name spoofing tends to impersonate.
# Not exhaustive — a heuristic signal for the rule engine, not a verdict.
_BRAND_NAMES = [
    "paypal", "apple", "microsoft", "google", "amazon", "netflix",
    "vakifbank", "vakıfbank", "garanti", "isbank", "iş bankası", "akbank",
    "ziraat", "yapikredi", "yapı kredi", "dhl", "fedex", "ups",
]


def parse_address_facts(msg: Message) -> dict:
    """Returns from_domain, return_path_domain, reply_to_domain,
    return_path_mismatch, reply_to_mismatch, display_name,
    display_name_has_email, display_name_brand_mismatch."""
    from_header = msg.get("From", "")
    display_name, from_addr = parseaddr(from_header)
    from_domain = _domain_of(from_addr) if from_addr else None

    return_path_domain = _domain_of(msg.get("Return-Path", ""))
    reply_to_domain = _domain_of(msg.get("Reply-To", ""))

    return_path_mismatch = bool(
        from_domain and return_path_domain and from_domain != return_path_domain
    )
    reply_to_mismatch = bool(
        from_domain and reply_to_domain and from_domain != reply_to_domain
    )

    display_name_has_email = bool(display_name and "@" in display_name)

    display_name_lower = (display_name or "").lower()
    display_name_brand_mismatch = False
    for brand in _BRAND_NAMES:
        if brand in display_name_lower:
            if from_domain is None or brand.replace(" ", "") not in from_domain.replace("-", ""):
                display_name_brand_mismatch = True
            break

    return {
        "from_domain": from_domain,
        "return_path_domain": return_path_domain,
        "reply_to_domain": reply_to_domain,
        "return_path_mismatch": return_path_mismatch,
        "reply_to_mismatch": reply_to_mismatch,
        "display_name": display_name or None,
        "display_name_has_email": display_name_has_email,
        "display_name_brand_mismatch": display_name_brand_mismatch,
    }


def parse_routing_facts(msg: Message, from_domain: str | None) -> dict:
    """Returns message_id_domain, message_id_domain_matches_from,
    received_hop_count, first_received_ip."""
    message_id = msg.get("Message-ID", "") or msg.get("Message-Id", "")
    message_id_domain = None
    m = re.search(r"@([\w.-]+)>?\s*$", message_id.strip())
    if m:
        message_id_domain = m.group(1).lower()

    message_id_domain_matches_from = (
        None if message_id_domain is None or from_domain is None
        else message_id_domain == from_domain
    )

    received_headers = msg.get_all("Received", [])
    received_hop_count = len(received_headers)

    first_received_ip = None
    if received_headers:
        # Received headers are prepended by each hop, so the LAST one in
        # the list is closest to the original sender.
        m = re.search(r"\[?(\d{1,3}(?:\.\d{1,3}){3})\]?", received_headers[-1])
        if m:
            first_received_ip = m.group(1)

    return {
        "message_id_domain": message_id_domain,
        "message_id_domain_matches_from": message_id_domain_matches_from,
        "received_hop_count": received_hop_count,
        "first_received_ip": first_received_ip,
    }
