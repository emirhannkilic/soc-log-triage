"""
Extracts URLs from an HTML or plain-text email body and computes per-URL
facts (v3 plan section 4.4).

For HTML bodies, uses BeautifulSoup to find <a href> tags so anchor text can
be compared against the href target — a plain regex over the body text can't
tell what text a link is displayed as. Plain-text bodies fall back to a
regex scan (no anchor text concept there).
"""
import re
from urllib.parse import urlparse

from bs4 import BeautifulSoup

URL_RE = re.compile(r'https?://[^\s"\'<>)]+')

# Common shortener domains — not exhaustive, a heuristic signal.
_SHORTENER_DOMAINS = {
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd", "buff.ly",
    "rebrand.ly", "shorturl.at", "cutt.ly", "tiny.cc", "rb.gy",
}


def _domain_of_url(url: str) -> str | None:
    try:
        netloc = urlparse(url).netloc
    except ValueError:
        return None
    if not netloc:
        return None
    # strip userinfo and port
    netloc = netloc.split("@")[-1].split(":")[0]
    return netloc.lower() or None


def _is_ip_based(domain: str | None) -> bool:
    if not domain:
        return False
    return bool(re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}", domain))


def _has_punycode(domain: str | None) -> bool:
    if not domain:
        return False
    return any(label.startswith("xn--") for label in domain.split("."))


def _has_redirect_param(url: str) -> bool:
    # Looks for another http(s):// URL embedded in the query string —
    # common in open-redirect / tracking-link phishing patterns.
    query = url.split("?", 1)[1] if "?" in url else ""
    return "http%3a" in query.lower() or "http://" in query.lower() or "https://" in query.lower()


def _anchor_text_domain(anchor_text: str) -> str | None:
    """If the visible link text itself looks like a domain/URL (the classic
    "text says paypal.com but links elsewhere" pattern), extract it."""
    text = anchor_text.strip()
    m = re.search(r"([\w-]+\.[a-z]{2,})(?:/\S*)?$", text, re.IGNORECASE)
    return m.group(1).lower() if m else None


def _build_url_fact(url: str, anchor_text: str | None) -> dict:
    href_domain = _domain_of_url(url)
    anchor_domain = _anchor_text_domain(anchor_text) if anchor_text else None
    text_href_mismatch = bool(
        anchor_domain and href_domain and anchor_domain != href_domain
    )
    return {
        "url": url,
        "href_domain": href_domain,
        "anchor_text_domain": anchor_domain,
        "text_href_mismatch": text_href_mismatch,
        "is_ip_based": _is_ip_based(href_domain),
        "is_shortener": href_domain in _SHORTENER_DOMAINS if href_domain else False,
        "has_punycode": _has_punycode(href_domain),
        "redirect_param": _has_redirect_param(url),
    }


def extract_url_facts(body_text: str, is_html: bool) -> list[dict]:
    seen: set[str] = set()
    facts: list[dict] = []

    if is_html:
        soup = BeautifulSoup(body_text, "html.parser")
        for tag in soup.find_all("a", href=True):
            href = tag["href"]
            if not href.lower().startswith(("http://", "https://")):
                continue
            if href in seen:
                continue
            seen.add(href)
            facts.append(_build_url_fact(href, tag.get_text()))

    # Also scan raw text for URLs not caught as <a href> (plain-text bodies,
    # or URLs mentioned outside of anchor tags in HTML bodies).
    for url in URL_RE.findall(body_text):
        if url in seen:
            continue
        seen.add(url)
        facts.append(_build_url_fact(url, None))

    return facts
