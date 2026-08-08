"""Unit tests for src/decision/context.py (PHISHING_ROUTING_PLAN.md
step 9). Confirms build_context() extracts exactly the narrow fields
PhishingDecisionContext needs from EmailFacts — no other EmailFacts
data should leak through.

has_external_url tests (below) are a follow-up fix: the field was
originally has_url ("any URL exists at all"), which over-triggered on
a purely self-referential link (e.g. a real password-reset link
pointing back at the sender's own domain) — same_organization() is now
required to be False for at least one URL before this counts as
"external". See schemas/decision.py's module docstring."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from schemas.facts import EmailFacts, UrlFacts  # noqa: E402
from src.decision.context import build_context  # noqa: E402

BASE_FACTS_KWARGS = dict(
    spf_result="pass",
    dkim_result="pass",
    dmarc_result="pass",
    dkim_domain="example.com",
    dkim_domain_matches_from=True,
    spf_mailfrom_domain="example.com",
    spf_aligned=True,
    from_domain="example.com",
    from_source="From",
    return_path_domain="example.com",
    reply_to_domain="example.com",
    return_path_mismatch=False,
    reply_to_mismatch=False,
    display_name="Example Co",
    display_name_has_email=False,
    display_name_brand_mismatch=False,
    message_id_domain="example.com",
    message_id_domain_matches_from=True,
    received_hop_count=2,
    first_received_ip="1.2.3.4",
    urls=[],
    attachments=[],
    has_html_form=False,
    form_action_domain=None,
    has_hidden_text=False,
    has_large_hidden_text=False,
    image_only_body=False,
    urgency_keywords=[],
    credential_request=False,
    claims_attachment=False,
    has_advance_fee_fraud_language=False,
    has_fake_reward_claim_language=False,
    subject="Hello",
    date="Mon, 1 Jan 2024 00:00:00 +0000",
    body_text="hello",
    language="en",
)


def facts(**overrides) -> EmailFacts:
    kwargs = dict(BASE_FACTS_KWARGS)
    kwargs.update(overrides)
    return EmailFacts(**kwargs)


def _url(url="http://external.example/x", href_domain="external.example") -> UrlFacts:
    return UrlFacts(
        url=url, href_domain=href_domain, anchor_text_domain=href_domain,
        text_href_mismatch=False, is_ip_based=False, is_shortener=False,
        has_punycode=False, redirect_param=False,
    )


def test_no_urls_gives_has_external_url_false():
    context = build_context(facts())
    assert context.has_external_url is False
    assert context.url_count == 0
    assert context.url_ids == []


def test_external_urls_present_gives_has_external_url_true_with_correct_count():
    """BASE_FACTS_KWARGS's from_domain is "example.com" — these URLs
    point at "external.example", a different organization."""
    f = facts(urls=[_url("http://a.example", "a.example"), _url("http://b.example", "b.example")])
    context = build_context(f)
    assert context.has_external_url is True
    assert context.url_count == 2
    assert set(context.url_ids) == {"http://a.example", "http://b.example"}


def test_same_organization_url_does_not_count_as_external():
    """Regression test for the real over-trigger has_url used to have:
    a self-referential link (from_domain and href_domain the same
    organization — e.g. a real password-reset email linking back to
    the sender's own domain) is not evidence of anything suspicious and
    must not count as "external"."""
    f = facts(from_domain="bank.com",
              urls=[_url("https://bank.com/reset", "bank.com")])
    context = build_context(f)
    assert context.has_external_url is False
    assert context.url_count == 1


def test_subdomain_of_same_organization_does_not_count_as_external():
    """same_organization() treats a subdomain of the sender's own
    registrable domain (mailer.bank.com vs bank.com) as the SAME
    organization — the bulk-mail-infrastructure pattern
    src/parser/psl.py's same_organization() already documents."""
    f = facts(from_domain="mailer.bank.com",
              urls=[_url("https://bank.com/reset", "bank.com")])
    context = build_context(f)
    assert context.has_external_url is False


def test_different_organization_url_counts_as_external():
    """The actual attack shape this field exists to detect: a link
    pointing at a DIFFERENT organization than the sender claims to be."""
    f = facts(from_domain="bank.com",
              urls=[_url("https://phishy-lookalike.example/login",
                         "phishy-lookalike.example")])
    context = build_context(f)
    assert context.has_external_url is True


def test_mix_of_same_and_different_organization_urls_counts_as_external():
    """Only ONE external URL is needed to trip has_external_url — a mix
    of a same-org link (e.g. an unsubscribe link) and an external one
    (e.g. a tracking/phishing link) must still count."""
    f = facts(from_domain="bank.com", urls=[
        _url("https://bank.com/unsubscribe", "bank.com"),
        _url("https://phishy.example/login", "phishy.example"),
    ])
    context = build_context(f)
    assert context.has_external_url is True


def test_url_with_no_href_domain_does_not_count_as_external():
    """href_domain=None (parser couldn't resolve an href at all) must
    not be treated as external — same_organization() already returns
    False for a None argument on either side, so this is not a
    suspicious link, just an unresolvable one."""
    f = facts(urls=[_url("http://example.com/x", href_domain=None)])
    context = build_context(f)
    assert context.has_external_url is False


def test_parser_credential_request_carried_through_as_provenance_only():
    f = facts(credential_request=True)
    context = build_context(f)
    assert context.parser_credential_request is True


def test_parser_credential_request_false_when_absent():
    f = facts(credential_request=False)
    context = build_context(f)
    assert context.parser_credential_request is False


if __name__ == "__main__":
    import traceback

    tests = [(name, obj) for name, obj in list(globals().items())
              if name.startswith("test_") and callable(obj)]
    passed, failed = 0, 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS: {name}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL: {name}: {e}")
            failed += 1
        except Exception:
            print(f"ERROR: {name}")
            traceback.print_exc()
            failed += 1

    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
