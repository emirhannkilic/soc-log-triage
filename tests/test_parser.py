"""
Unit tests for src/parser/*, v3 plan section 4 / Adim 2.

Run with: python3 -m pytest tests/test_parser.py -v
(or, without pytest installed: python3 tests/test_parser.py)
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.parser.attachments import extract_attachment_facts
from src.parser.body import detect_language, extract_body_facts
from src.parser.headers import parse_address_facts, parse_authentication_results, parse_routing_facts
from src.parser.parse import parse_eml
from src.parser.urls import extract_url_facts


# --- headers.py --------------------------------------------------------

def test_authentication_results_all_pass():
    import email
    msg = email.message_from_string(
        "From: alerts@bank.com\n"
        "Authentication-Results: mx.google.com; spf=pass smtp.mailfrom=bank.com; "
        "dkim=pass header.i=@bank.com; dmarc=pass\n\n"
    )
    facts = parse_authentication_results(msg, "bank.com")
    assert facts["spf_result"] == "pass"
    assert facts["dkim_result"] == "pass"
    assert facts["dmarc_result"] == "pass"
    assert facts["dkim_domain"] == "bank.com"
    assert facts["dkim_domain_matches_from"] is True


def test_authentication_results_missing_header_is_none():
    import email
    msg = email.message_from_string("From: a@b.com\n\n")
    facts = parse_authentication_results(msg, "b.com")
    assert facts["spf_result"] is None
    assert facts["dkim_result"] is None
    assert facts["dmarc_result"] is None


def test_authentication_results_prefers_dkim_matching_from():
    """Gmail-style multi-dkim header: should pick the entry whose signing
    domain matches From, not just the first dkim= in the header."""
    import email
    msg = email.message_from_string(
        "From: newsletter@retailer.com\n"
        "Authentication-Results: mx.google.com;\n"
        " dkim=pass header.i=@sendgrid.net header.s=x;\n"
        " dkim=pass header.i=@retailer.com header.s=y;\n"
        " spf=pass\n\n"
    )
    facts = parse_authentication_results(msg, "retailer.com")
    assert facts["dkim_domain"] == "retailer.com"
    assert facts["dkim_domain_matches_from"] is True


def test_address_facts_return_path_mismatch():
    import email
    msg = email.message_from_string(
        "From: \"PayPal Support\" <alerts@paypal.com>\n"
        "Return-Path: <bounce@totally-different.ru>\n\n"
    )
    facts = parse_address_facts(msg)
    assert facts["from_domain"] == "paypal.com"
    assert facts["return_path_domain"] == "totally-different.ru"
    assert facts["return_path_mismatch"] is True
    assert facts["display_name"] == "PayPal Support"


def test_address_facts_no_mismatch_when_domains_match():
    import email
    msg = email.message_from_string(
        "From: alice@company.com\n"
        "Return-Path: <alice@company.com>\n\n"
    )
    facts = parse_address_facts(msg)
    assert facts["return_path_mismatch"] is False


def test_address_facts_brand_display_name_mismatch():
    import email
    msg = email.message_from_string(
        "From: \"VakifBank Guvenlik\" <no-reply@random-mailer123.xyz>\n\n"
    )
    facts = parse_address_facts(msg)
    assert facts["display_name_brand_mismatch"] is True


def test_routing_facts_message_id_domain():
    import email
    msg = email.message_from_string(
        "From: a@example.com\n"
        "Message-ID: <abc123@example.com>\n\n"
    )
    facts = parse_routing_facts(msg, "example.com")
    assert facts["message_id_domain"] == "example.com"
    assert facts["message_id_domain_matches_from"] is True


def test_routing_facts_hop_count():
    import email
    msg = email.message_from_string(
        "From: a@example.com\n"
        "Received: from a by b; Mon, 1 Jan 2024 00:00:00 +0000\n"
        "Received: from c by d; Mon, 1 Jan 2024 00:00:01 +0000\n\n"
    )
    facts = parse_routing_facts(msg, "example.com")
    assert facts["received_hop_count"] == 2


# --- urls.py -------------------------------------------------------------

def test_url_text_href_mismatch():
    # The anchor text itself is also a URL-looking string, so it's picked
    # up a second time by the plain-text regex scan (deliberate: a URL
    # mentioned as visible text is still a URL worth flagging). Only the
    # <a href> entry carries anchor-text context.
    html = '<a href="http://evil-phish.ru/x">https://paypal.com/login</a>'
    facts = extract_url_facts(html, is_html=True)
    assert len(facts) == 2
    href_entry = next(f for f in facts if f["url"] == "http://evil-phish.ru/x")
    assert href_entry["href_domain"] == "evil-phish.ru"
    assert href_entry["anchor_text_domain"] == "paypal.com"
    assert href_entry["text_href_mismatch"] is True


def test_url_no_mismatch_when_same_domain():
    html = '<a href="https://paypal.com/login">paypal.com/login</a>'
    facts = extract_url_facts(html, is_html=True)
    assert facts[0]["text_href_mismatch"] is False


def test_url_ip_based():
    facts = extract_url_facts("http://192.168.1.1/login", is_html=False)
    assert facts[0]["is_ip_based"] is True


def test_url_shortener():
    facts = extract_url_facts("https://bit.ly/abc123", is_html=False)
    assert facts[0]["is_shortener"] is True


def test_url_punycode():
    facts = extract_url_facts("https://xn--pypal-4ve.com/login", is_html=False)
    assert facts[0]["has_punycode"] is True


def test_url_plain_text_no_anchor():
    facts = extract_url_facts("Visit https://example.com for details", is_html=False)
    assert facts[0]["anchor_text_domain"] is None
    assert facts[0]["text_href_mismatch"] is False


# --- attachments.py --------------------------------------------------------

def test_double_extension_detected():
    from src.parser.attachments import _has_double_extension
    assert _has_double_extension("fatura.pdf.exe") is True
    assert _has_double_extension("fatura.pdf") is False
    assert _has_double_extension("archive.tar.gz") is False  # gz not risky


def test_risky_extension():
    from src.parser.attachments import _extension_of
    assert _extension_of("invoice.exe") == "exe"
    assert _extension_of("noext") is None


# --- body.py --------------------------------------------------------

def test_detect_language_turkish():
    assert detect_language("Merhaba, hesabınızı doğrulayın lütfen") == "tr"


def test_detect_language_english():
    assert detect_language("Please verify your account immediately") == "en"


def test_urgency_keywords_detected():
    facts = extract_body_facts("Your account will be suspended. Click here to verify your account now.", is_html=False)
    assert "suspended" in facts["urgency_keywords"]
    assert "verify your account" in facts["urgency_keywords"]


def test_credential_request_detected():
    facts = extract_body_facts("Lütfen şifrenizi ve kart numaranızı giriniz.", is_html=False)
    assert facts["credential_request"] is True


def test_no_credential_request_in_normal_text():
    facts = extract_body_facts("Yarın toplantımız var, saat 10da görüşelim.", is_html=False)
    assert facts["credential_request"] is False


def test_html_form_detected():
    html = "<html><body><form action='http://evil.com'><input type='password'></form></body></html>"
    facts = extract_body_facts(html, is_html=True)
    assert facts["has_html_form"] is True


def test_hidden_text_detected():
    html = '<div style="display:none">hidden tracking text</div><p>Visible text</p>'
    facts = extract_body_facts(html, is_html=True)
    assert facts["has_hidden_text"] is True


# --- parse.py (integration, real corpus files) --------------------------

def test_parse_eml_full_corpus_no_crashes():
    """Every sampled file (phishing_pot + Gmail, 2500 total) must parse
    without raising — this is the Adim 2 acceptance gate before moving to
    Adim 3."""
    import json

    processed_dir = PROJECT_ROOT / "data" / "processed"
    phishing_file = processed_dir / "phishing_sample.jsonl"
    gmail_file = processed_dir / "gmail_sample.jsonl"

    if not phishing_file.exists() or not gmail_file.exists():
        print("SKIP: sample files not found, run sample_data.py / prepare_gmail_data.py first")
        return

    paths = [json.loads(line)["path"] for line in open(phishing_file)]
    paths += [json.loads(line)["path"] for line in open(gmail_file)]

    errors = []
    for p in paths:
        try:
            parse_eml(PROJECT_ROOT / p)
        except Exception as e:
            errors.append((p, e))

    assert not errors, f"{len(errors)} files failed to parse: {errors[:5]}"


def test_parse_eml_known_phishing_sample():
    """sample-2.eml is a known phishing_pot example with spf=none,
    dkim=fail, dmarc=fail, and a From/DKIM domain mismatch — spot-check
    that the full pipeline gets these right end to end."""
    path = PROJECT_ROOT / "data" / "phishing_pot" / "email" / "sample-2.eml"
    if not path.exists():
        print("SKIP: sample-2.eml not found")
        return
    facts = parse_eml(path)
    assert facts.spf_result == "none"
    assert facts.dkim_result == "fail"
    assert facts.dmarc_result == "fail"
    assert facts.from_domain == "digitalmashreq.mg.tdi.tc"


if __name__ == "__main__":
    # Minimal runner for when pytest isn't installed — collects and runs
    # every test_* function in this module, prints a pass/fail summary.
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
