"""
Registrable-domain lookup via a real Public Suffix List (tldextract), not
the "last two labels" heuristic that both src/parser/headers.py's
_registrable_domain and src/rules/engine_v2.py's _root_domain used to
implement. That heuristic gets multi-label TLDs wrong (co.uk, com.tr):
"mail.example.co.uk" would come back as "example.co.uk" (right, by luck)
but "co.uk" itself as root domain for "foo.co.uk" is wrong, and the
heuristic can't tell.

Single source of truth so headers.py and engine_v2.py can't drift into
different registrable-domain answers for the same string.

suffix_list_urls=() pins tldextract to its bundled snapshot and disables
the live PSL fetch it does by default on first use — this parser must stay
deterministic and offline (CLAUDE.md: no network calls in the deterministic
layer). The bundled snapshot ages over time (new TLDs won't be recognized)
but that's an acceptable, silent-degradation-only risk for a demo project;
a stale entry only means a very new TLD falls back to being treated as-is,
it doesn't produce a wrong split for TLDs the snapshot does know.
"""
import tldextract

_extract = tldextract.TLDExtract(suffix_list_urls=())


def registrable_domain(domain: str | None) -> str | None:
    """The organization-owned part of a domain: "mail.example.co.uk" ->
    "example.co.uk", "email.uber.com" -> "uber.com". Returns the input
    lowercased if the PSL doesn't recognize the suffix at all (e.g. a bare
    IP or a brand-new TLD the bundled snapshot doesn't know) rather than
    None — callers compare this against another registrable_domain()
    result, and two unresolved-but-identical inputs should still compare
    equal.

    Returns None when the input IS itself a bare public suffix with no
    organization label in front of it ("gov.tr", "co.uk", "com") — tldextract
    reports this as an empty `domain` with a non-empty `suffix`. This isn't
    a real registrable domain (nobody's mailbox is @gov.tr), so it must not
    silently compare equal to another truncated suffix. Found on a real
    Turkish e-Devlet email (inbox-1913.eml): its link anchor text read the
    visually-shortened "gov.tr" while href was the real
    "www.turkiye.gov.tr" — under the old "last two labels" heuristic both
    reduced to "gov.tr" and matched, hiding the fact that "gov.tr" was never
    a real domain to begin with. Returning None here makes callers that
    require both sides non-None (e.g. url mismatch checks) correctly treat
    a bare-suffix anchor text as "can't compare" rather than a false match."""
    if not domain:
        return None
    domain = domain.lower()
    result = _extract(domain)
    if result.domain and result.suffix:
        return f"{result.domain}.{result.suffix}"
    if result.suffix and not result.domain:
        return None
    return domain


def same_organization(a: str | None, b: str | None) -> bool:
    """Same domain, or one a subdomain of the same registrable domain as the
    other (mailer.netflix.com / netflix.com) — the pattern large senders use
    for bulk-mail infrastructure. Exact-string comparison flagged this as
    third-party spoofing on a genuine Netflix email (dkim_domain=netflix.com,
    from_domain=mailer.netflix.com, both real Netflix domains) — see
    PROGRESS.md "inbox-9945.eml" for the false positive this produced."""
    if not a or not b:
        return False
    a, b = a.lower(), b.lower()
    return a == b or registrable_domain(a) == registrable_domain(b)
