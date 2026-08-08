"""
Builds PhishingDecisionContext (schemas/decision.py) from EmailFacts
(PHISHING_ROUTING_PLAN.md step 9). This is the ONLY place EmailFacts is
read for decision-policy purposes — src/decision/phishing_policy.py
itself never sees EmailFacts, only this narrow, already-extracted
context. See schemas/decision.py's module docstring for why the
boundary is drawn here rather than passing EmailFacts straight through.

has_external_url USES same_organization(), NOT "any URL exists"
    See schemas/decision.py's "WHY 'external' MEANS
    same_organization()-FALSE" for the real over-trigger this replaced
    (a self-referential link, e.g. a real password-reset email linking
    back to the sender's own domain, used to count as "has a URL" and
    could upgrade a verdict on no real evidence). same_organization()
    is the SAME src/parser/psl.py helper src/rules/engine_v2.py and
    src/parser/headers.py already use — imported here rather than
    reimplemented, so this can't drift into a different, weaker
    same-organization definition of its own.

    A URL with href_domain=None (parser couldn't resolve an href at
    all) is treated as NOT external — same_organization(None, x)
    returns False (no evidence either way), and blindly negating that
    into "external" would wrongly treat "unknown" as "different
    organization," which is a stronger claim than the parser actually
    supports. This is checked explicitly below rather than relying on
    same_organization()'s own None handling, since that function
    returning False for "can't tell" and this one needing False for
    "can't tell" too are only coincidentally the same value — negating
    blindly would make that coincidence load-bearing.
"""
from schemas.decision import PhishingDecisionContext
from schemas.facts import EmailFacts
from src.parser.psl import same_organization


def build_context(facts: EmailFacts) -> PhishingDecisionContext:
    has_external_url = any(
        u.href_domain is not None and not same_organization(u.href_domain, facts.from_domain)
        for u in facts.urls
    )
    return PhishingDecisionContext(
        has_external_url=has_external_url,
        url_count=len(facts.urls),
        url_ids=[u.url for u in facts.urls],
        parser_credential_request=facts.credential_request,
    )
