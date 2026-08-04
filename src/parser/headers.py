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
from email.header import Header, decode_header
from email.message import Message
from email.utils import getaddresses, parseaddr


def _decode_rfc2047(raw_value) -> str | None:
    """Decode RFC 2047 encoded-words (=?UTF-8?B?...?=) in a header value.

    Compat32 sometimes hands back an email.header.Header instead of a plain
    str for malformed headers, so coerce first. Falls back to the raw string
    on any decode failure — showing something beats raising on a malformed
    header, and phishing samples are full of malformed headers.

    Lives here rather than in parse.py because headers.py must not import
    from parse.py (parse.py already imports this module; the reverse would
    be a cycle). parse.py re-exports it for subject decoding.
    """
    if raw_value is None:
        return None
    raw_str = str(raw_value)
    try:
        return "".join(
            chunk.decode(charset or "utf-8", errors="replace")
            if isinstance(chunk, bytes) else chunk
            for chunk, charset in decode_header(raw_str)
        )
    except Exception:
        return raw_str

_AUTH_RESULT_RE = re.compile(r"\b(spf|dkim|dmarc)=([a-zA-Z0-9_-]+)")
_DKIM_DOMAIN_RE = re.compile(r"dkim=([a-zA-Z0-9_-]+)[^;]*?header\.[id]=@?([\w.-]+)")

# Last-resort fallback for addresses parseaddr can't handle — e.g. malformed
# multi-name group syntax like '"Grüner Georg", jehd <service@x.com>', which
# parseaddr silently returns ('', '') on even after str() coercion of a
# Header object. This just grabs the last thing that looks like an email
# address in the string, which is enough to recover a domain even when we
# can't parse the display name correctly.
_EMAIL_FALLBACK_RE = re.compile(r"([\w.+-]+@[\w-]+(?:\.[\w-]+)+)")


def _coerce_header_str(value) -> str:
    """Compat32 sometimes hands back an email.header.Header object instead
    of a plain str for malformed headers (RFC 2047 encoded-words that don't
    decode cleanly, non-ASCII bytes in odd places). str()-ing it is
    necessary before any regex/parseaddr call, or those silently fail /
    crash instead of working on the degraded-but-recoverable text. See
    holdout-fix-tasks.md T1 — this was the root cause of from_domain=None
    correlating with phishing samples (Header objects only appeared on
    phishing_pot's messier headers, not Gmail's)."""
    if isinstance(value, Header):
        return str(value)
    return value or ""


def _domain_of(address) -> str | None:
    address = _coerce_header_str(address)
    if not address:
        return None

    _, addr = parseaddr(address)
    if addr and "@" in addr:
        return addr.rsplit("@", 1)[-1].lower()

    # parseaddr failed (returns ('', '') on some malformed multi-name /
    # group-syntax addresses even on a clean str) — fall back to a direct
    # regex scan rather than giving up, since the address is often still
    # visibly present in the string.
    m = _EMAIL_FALLBACK_RE.search(address)
    if m:
        return m.group(1).rsplit("@", 1)[-1].lower()
    return None


def parse_authentication_results(msg: Message, from_domain: str | None) -> dict:
    """Returns spf_result, dkim_result, dmarc_result, dkim_domain,
    dkim_domain_matches_from."""
    header = _coerce_header_str(msg.get("Authentication-Results"))
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
    "facebook", "instagram", "whatsapp", "coinbase", "binance", "trust wallet",
    "bradesco", "itau", "itaú", "santander", "correios", "chronopost",
    "mercado pago", "mercadopago", "icloud", "google storage",
    # Turkish e-commerce, cargo and telecom brands. The corpus is TR/EN mixed
    # and the banks were already here, but the retail and shipping brands most
    # often spoofed at Turkish recipients were missing — a real sample spoofing
    # "Hepsiburada İletişim" from acwild.eu matched nothing.
    "hepsiburada", "trendyol", "n11", "gittigidiyor", "sahibinden",
    "getir", "yemeksepeti", "migros", "a101", "bim",
    "ptt", "aras kargo", "yurtiçi kargo", "yurtici kargo", "mng kargo",
    "sürat kargo", "surat kargo",
    "turkcell", "vodafone", "türk telekom", "turk telekom",
    "e-devlet", "edevlet", "turkiye.gov.tr",
]


_RECEIVED_FOR_RE = re.compile(r"\bfor\s+<?([\w.+-]+@[\w.-]+)>?", re.IGNORECASE)


def _from_domain_with_source(msg: Message) -> tuple[str | None, str | None]:
    """Tries From -> Sender -> Return-Path -> the 'for=' address in the
    oldest Received header, in that order, and returns (domain, source) —
    source records which one actually worked so a None result downstream
    means "genuinely no sender address found," not "we only tried one
    header and gave up." See holdout-fix-tasks.md T1."""
    from_header = _coerce_header_str(msg.get("From", ""))
    domain = _domain_of(from_header)
    if domain:
        return domain, "From"

    sender_header = _coerce_header_str(msg.get("Sender", ""))
    domain = _domain_of(sender_header)
    if domain:
        return domain, "Sender"

    return_path_header = _coerce_header_str(msg.get("Return-Path", ""))
    domain = _domain_of(return_path_header)
    if domain:
        return domain, "Return-Path"

    received_headers = msg.get_all("Received", [])
    if received_headers:
        # oldest hop (closest to origin) is last in the list
        oldest = _coerce_header_str(received_headers[-1])
        m = _RECEIVED_FOR_RE.search(oldest)
        if m:
            domain = _domain_of(m.group(1))
            if domain:
                return domain, "Received-for"

    return None, None


def parse_address_facts(msg: Message) -> dict:
    """Returns from_domain, from_source, return_path_domain, reply_to_domain,
    return_path_mismatch, reply_to_mismatch, display_name,
    display_name_has_email, display_name_brand_mismatch."""
    from_header = _coerce_header_str(msg.get("From", ""))
    display_name, _ = parseaddr(from_header)
    if not display_name and from_header:
        # parseaddr couldn't extract a display name either (same malformed
        # multi-name case as the address itself) — leave it as None rather
        # than a misleading empty string.
        display_name = None
    else:
        # RFC 2047 decode. parseaddr returns the display name verbatim, so a
        # non-ASCII one arrives still encoded ("=?UTF-8?Q?Hepsiburada...?=").
        # Brand matching below is a substring test, which that encoded form
        # silently defeats: a spoofed Turkish brand name would sit right
        # there in the header and never match. Found on a real sample whose
        # display name read "Hepsiburada İletişim" while From was acwild.eu —
        # scored 2 (Güvenilir) instead of firing display_name_brand_mismatch.
        display_name = _decode_rfc2047(display_name)

    from_domain, from_source = _from_domain_with_source(msg)

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
        "from_source": from_source,
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
    message_id = _coerce_header_str(msg.get("Message-ID") or msg.get("Message-Id"))
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
        oldest_received = _coerce_header_str(received_headers[-1])
        m = re.search(r"\[?(\d{1,3}(?:\.\d{1,3}){3})\]?", oldest_received)
        if m:
            first_received_ip = m.group(1)

    return {
        "message_id_domain": message_id_domain,
        "message_id_domain_matches_from": message_id_domain_matches_from,
        "received_hop_count": received_hop_count,
        "first_received_ip": first_received_ip,
    }
