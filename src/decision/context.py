"""
Builds PhishingDecisionContext (schemas/decision.py) from EmailFacts
(PHISHING_ROUTING_PLAN.md step 9). This is the ONLY place EmailFacts is
read for decision-policy purposes — src/decision/phishing_policy.py
itself never sees EmailFacts, only this narrow, already-extracted
context. See schemas/decision.py's module docstring for why the
boundary is drawn here rather than passing EmailFacts straight through.
"""
from schemas.decision import PhishingDecisionContext
from schemas.facts import EmailFacts


def build_context(facts: EmailFacts) -> PhishingDecisionContext:
    return PhishingDecisionContext(
        has_url=len(facts.urls) > 0,
        url_count=len(facts.urls),
        url_ids=[u.url for u in facts.urls],
        parser_credential_request=facts.credential_request,
    )
