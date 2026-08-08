"""Unit tests for src/decision/context.py (PHISHING_ROUTING_PLAN.md
step 9). Confirms build_context() extracts exactly the narrow fields
PhishingDecisionContext needs from EmailFacts — no other EmailFacts
data should leak through."""
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


def _url(url="http://example.com/x") -> UrlFacts:
    return UrlFacts(
        url=url, href_domain="example.com", anchor_text_domain="example.com",
        text_href_mismatch=False, is_ip_based=False, is_shortener=False,
        has_punycode=False, redirect_param=False,
    )


def test_no_urls_gives_has_url_false():
    context = build_context(facts())
    assert context.has_url is False
    assert context.url_count == 0
    assert context.url_ids == []


def test_urls_present_gives_has_url_true_with_correct_count():
    f = facts(urls=[_url("http://a.example"), _url("http://b.example")])
    context = build_context(f)
    assert context.has_url is True
    assert context.url_count == 2
    assert set(context.url_ids) == {"http://a.example", "http://b.example"}


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
