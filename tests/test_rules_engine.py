"""Unit tests for src/rules/engine.py (v3 Adım 4)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rules.engine import evaluate, load_rules

RULES = load_rules()

BASE_SIGNALS = {
    "spf_result": "pass",
    "dkim_result": "pass",
    "dmarc_result": "pass",
    "dkim_domain": "example.com",
    "dkim_domain_matches_from": True,
    "from_domain": "example.com",
    "from_source": "From",
    "return_path_domain": "example.com",
    "reply_to_domain": "example.com",
    "return_path_mismatch": False,
    "reply_to_mismatch": False,
    "display_name": "Example Co",
    "display_name_has_email": False,
    "display_name_brand_mismatch": False,
    "message_id_domain": "example.com",
    "message_id_domain_matches_from": True,
    "received_hop_count": 2,
    "first_received_ip": "1.2.3.4",
    "has_html_form": False,
    "has_hidden_text": False,
    "image_only_body": False,
    "credential_request": False,
    "claims_attachment": False,
    "subject": "Hello",
    "date": "Mon, 1 Jan 2024 00:00:00 +0000",
    "body_text": "hello",
    "language": "en",
    "url_count": 0,
    "url_text_href_mismatch_count": 0,
    "url_ip_based_count": 0,
    "url_shortener_count": 0,
    "url_punycode_count": 0,
    "url_redirect_param_count": 0,
    "attachment_count": 0,
    "attachment_risky_type_count": 0,
    "attachment_double_extension_count": 0,
    "attachment_is_archive_count": 0,
    "urgency_keyword_count": 0,
}


def signals(**overrides):
    s = dict(BASE_SIGNALS)
    s.update(overrides)
    return s


def test_clean_email_is_guvenilir():
    result = evaluate(signals(), RULES)
    assert result.verdict == "Güvenilir", result.verdict
    assert result.score < RULES["thresholds"]["suspicious"]


def test_all_auth_pass_consistent_gives_negative_bonus():
    result = evaluate(signals(), RULES)
    assert any(m.signal == "all_auth_pass_and_consistent" for m in result.matches)


def test_spf_fail_fires():
    result = evaluate(signals(spf_result="fail"), RULES)
    assert any(m.signal == "spf_or_dmarc_fail" for m in result.matches)


def test_dkim_missing_and_domain_mismatch_fires():
    result = evaluate(
        signals(dkim_result="none", dkim_domain_matches_from=False), RULES
    )
    assert any(
        m.signal == "dkim_missing_or_fail_domain_mismatch" for m in result.matches
    )


def test_dkim_missing_but_domain_matches_does_not_fire():
    result = evaluate(
        signals(dkim_result="none", dkim_domain_matches_from=True), RULES
    )
    assert not any(
        m.signal == "dkim_missing_or_fail_domain_mismatch" for m in result.matches
    )


def test_credential_request_needs_external_link():
    without_link = evaluate(signals(credential_request=True, url_count=0), RULES)
    with_link = evaluate(signals(credential_request=True, url_count=1), RULES)
    assert not any(
        m.signal == "credential_request_with_external_link" for m in without_link.matches
    )
    assert any(
        m.signal == "credential_request_with_external_link" for m in with_link.matches
    )


def test_from_domain_no_tld_fires():
    result = evaluate(signals(from_domain="randomhostname"), RULES)
    assert any(m.signal == "from_domain_no_tld" for m in result.matches)


def test_from_domain_with_tld_does_not_fire():
    result = evaluate(signals(from_domain="example.com"), RULES)
    assert not any(m.signal == "from_domain_no_tld" for m in result.matches)


def test_claims_attachment_but_empty_fires():
    result = evaluate(
        signals(claims_attachment=True, attachment_count=0), RULES
    )
    assert any(m.signal == "claims_attachment_but_empty" for m in result.matches)


def test_is_archive_needs_credential_request():
    without_cred = evaluate(
        signals(attachment_is_archive_count=1, credential_request=False), RULES
    )
    with_cred = evaluate(
        signals(attachment_is_archive_count=1, credential_request=True), RULES
    )
    assert not any(
        m.signal == "is_archive_with_credential_request" for m in without_cred.matches
    )
    assert any(
        m.signal == "is_archive_with_credential_request" for m in with_cred.matches
    )


def test_high_score_email_is_phishing():
    s = signals(
        spf_result="fail",
        dmarc_result="fail",
        dkim_result="none",
        dkim_domain_matches_from=False,
        display_name_brand_mismatch=True,
        url_text_href_mismatch_count=1,
        url_ip_based_count=1,
        attachment_double_extension_count=1,
        url_count=1,
    )
    result = evaluate(s, RULES)
    assert result.verdict == "Phishing", (result.verdict, result.score)


def test_thresholds_are_ordered():
    t = RULES["thresholds"]
    assert t["suspicious"] < t["phishing"]


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
