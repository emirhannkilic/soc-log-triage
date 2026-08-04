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


def test_dkim_pass_but_wrong_domain_fires():
    """A VALID signature from the wrong domain is third-party spoofing: the
    attacker signs with a domain they control while From claims another.
    The missing/failing-DKIM rule does not cover it, so this slipped through
    entirely — found on a phishing sample that scored 2 (Güvenilir) with
    DKIM signed by ladelanoagency.com for a From of jwgmedia.com."""
    result = evaluate(
        signals(dkim_result="pass", dkim_domain="other.com",
                dkim_domain_matches_from=False), RULES)
    assert any(m.signal == "dkim_pass_but_domain_mismatch" for m in result.matches)


def test_dkim_pass_matching_domain_does_not_fire_spoof_signal():
    result = evaluate(signals(dkim_result="pass", dkim_domain_matches_from=True), RULES)
    assert not any(m.signal == "dkim_pass_but_domain_mismatch" for m in result.matches)


def test_auth_bonus_survives_return_path_mismatch():
    """An ESP routes bounces through its own domain, so return_path_mismatch
    is the normal state of legitimate bulk mail — it must not cancel the
    all-auth-pass bonus. 16 of 17 measured false positives were caused by
    exactly that."""
    result = evaluate(signals(return_path_mismatch=True,
                              return_path_domain="bounces.esp.example"), RULES)
    assert any(m.signal == "all_auth_pass_and_consistent" for m in result.matches)


def test_auth_bonus_requires_matching_dkim_domain():
    """The bonus rests on DKIM proving the sender's own domain. Without that
    match there is no evidence of identity and no bonus."""
    result = evaluate(signals(dkim_domain_matches_from=False), RULES)
    assert not any(m.signal == "all_auth_pass_and_consistent" for m in result.matches)


def test_credential_request_needs_external_link():
    without_link = evaluate(signals(credential_request=True, url_count=0), RULES)
    with_link = evaluate(signals(credential_request=True, url_count=1), RULES)
    assert not any(
        m.signal == "credential_request_with_external_link" for m in without_link.matches
    )
    assert any(
        m.signal == "credential_request_with_external_link" for m in with_link.matches
    )


def test_reply_to_free_mail_fires():
    """A corporate sender redirecting replies to consumer webmail is the
    shape of 419 fraud and BEC. Found on a real sample sent from
    firat.edu.tr with SPF and DKIM passing, no URLs and no attachments —
    every technical signal clean — but Reply-To on gmail.com. It scored 1
    (Güvenilir)."""
    result = evaluate(signals(from_domain="firat.edu.tr",
                              reply_to_domain="gmail.com",
                              reply_to_mismatch=True), RULES)
    assert any(m.signal == "reply_to_free_mail" for m in result.matches)


def test_reply_to_free_mail_needs_corporate_from():
    """A free-mail sender replying to free mail is ordinary personal
    correspondence, not the pattern this signal describes."""
    result = evaluate(signals(from_domain="gmail.com",
                              reply_to_domain="yahoo.com",
                              reply_to_mismatch=True), RULES)
    assert not any(m.signal == "reply_to_free_mail" for m in result.matches)


def test_reply_to_corporate_domain_does_not_fire():
    """The raw mismatch is too noisy to score (40% of phishing but 25% of
    legitimate mail); only the free-mailbox variant qualifies."""
    result = evaluate(signals(from_domain="sender.com",
                              reply_to_domain="replies.sender-crm.com",
                              reply_to_mismatch=True), RULES)
    assert not any(m.signal == "reply_to_free_mail" for m in result.matches)


def test_from_domain_no_tld_fires():
    result = evaluate(signals(from_domain="randomhostname"), RULES)
    assert any(m.signal == "from_domain_no_tld" for m in result.matches)


def test_from_domain_with_tld_does_not_fire():
    result = evaluate(signals(from_domain="example.com"), RULES)
    assert not any(m.signal == "from_domain_no_tld" for m in result.matches)


def test_url_shortener_fires():
    """The parser has extracted is_shortener since Adım 2, but the rule
    engine had no matching signal until 2026-08-04 — a real phishing sample
    hid its target behind rebrand.ly and scored nothing for it."""
    result = evaluate(signals(url_count=1, url_shortener_count=1), RULES)
    assert any(m.signal == "url_shortener" for m in result.matches)


def test_url_shortener_does_not_fire_without_shortener():
    result = evaluate(signals(url_count=1, url_shortener_count=0), RULES)
    assert not any(m.signal == "url_shortener" for m in result.matches)


def test_has_html_form_fires():
    """A <form> in the body means credentials are being collected inside the
    mail client itself — legitimate senders link to their own site instead."""
    result = evaluate(signals(has_html_form=True), RULES)
    assert any(m.signal == "has_html_form" for m in result.matches)


def test_display_name_has_email_fires():
    """The inbox shows the display name, not the address. Making the display
    name itself an address ("destek@banka.com" <x@evil.ru>) forges the only
    thing the recipient sees."""
    result = evaluate(signals(display_name_has_email=True), RULES)
    assert any(m.signal == "display_name_has_email" for m in result.matches)


def test_url_redirect_param_fires():
    result = evaluate(signals(url_count=1, url_redirect_param_count=1), RULES)
    assert any(m.signal == "url_redirect_param" for m in result.matches)


def test_new_signals_do_not_fire_on_clean_email():
    """None of the three signals added on 2026-08-04 may fire on a clean
    baseline — they were added precisely because they never triggered in the
    hold-out, so a false positive here would be a coding error."""
    result = evaluate(signals(), RULES)
    fired = {m.signal for m in result.matches}
    assert "has_html_form" not in fired
    assert "display_name_has_email" not in fired
    assert "url_redirect_param" not in fired


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
