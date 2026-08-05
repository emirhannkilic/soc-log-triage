"""
Unit tests for src/parser/*, v3 plan section 4 / Adim 2.

Run with: python3 -m pytest tests/test_parser.py -v
(or, without pytest installed: python3 tests/test_parser.py)
"""
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.parser.attachments import extract_attachment_facts
from src.parser.body import (
    detect_language,
    extract_body_facts,
    strip_gateway_banner,
)
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
        "From: \"PayPal Security\" <no-reply@random-mailer123.xyz>\n\n"
    )
    facts = parse_address_facts(msg)
    assert facts["display_name_brand_mismatch"] is True


def test_rfc2047_encoded_display_name_is_decoded():
    """A non-ASCII display name arrives RFC 2047 encoded, and brand matching
    is a substring test — so leaving it encoded silently defeats the whole
    signal. Regression test for a real sample that spoofed
    'Hepsiburada İletişim' from acwild.eu and scored Güvenilir (2) because
    the display name was still '=?UTF-8?Q?Hepsiburada_...?=' at match time."""
    import email
    msg = email.message_from_string(
        "From: =?UTF-8?Q?Hepsiburada_=C4=B0leti=C5=9Fim?= <destek@acwild.eu>\n\n"
    )
    facts = parse_address_facts(msg)
    assert facts["display_name"] == "Hepsiburada İletişim"
    assert facts["display_name_brand_mismatch"] is True


def test_encoded_display_name_matching_own_domain_is_not_flagged():
    """The decode must not turn into a false positive: an encoded display
    name whose brand DOES match the sending domain is legitimate."""
    import email
    msg = email.message_from_string(
        "From: =?UTF-8?Q?Hepsiburada_=C4=B0leti=C5=9Fim?= "
        "<bilgi@hepsiburada.com>\n\n"
    )
    facts = parse_address_facts(msg)
    assert facts["display_name_brand_mismatch"] is False


def test_unencoded_raw_utf8_display_name_is_not_mangled():
    """Compat32 decodes a header as Latin-1 by default. A display name that
    contains raw, UN-encoded multi-byte UTF-8 (no RFC 2047 =?...?= wrapper —
    e.g. a literal U+2019 apostrophe typed directly into the header) survives
    RFC 2047 decoding fine (there's nothing to decode) but comes out of
    Compat32 with every non-ASCII byte replaced by U+FFFD, irreversibly.
    Regression test for a real sample: 'Men's Wellness Today' (with a
    right-single-quote apostrophe) rendered as 'Men��s Wellness
    Today' throughout the report. Needs a real file on disk because the
    fix re-reads the raw header bytes from the file, not from the
    already-mangled Message object."""
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "sample.eml"
        p.write_bytes((
            "From: Men’s Wellness Today <deals@example.test>\r\n"
            "To: user@example.test\r\n"
            "Subject: Test\r\n"
            "\r\n"
            "body\r\n"
        ).encode("utf-8"))

        facts = parse_eml(p)
        assert facts.display_name == "Men’s Wellness Today"
        assert "�" not in facts.display_name


def test_header_that_decodes_cleanly_is_left_alone():
    """A plain ASCII header must not be touched by the UTF-8 re-decode
    path — only headers Compat32 actually mangled (containing U+FFFD) are
    re-read."""
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "sample.eml"
        p.write_bytes(
            b"From: Plain Sender <sender@example.test>\r\n"
            b"To: user@example.test\r\n"
            b"Subject: Test\r\n"
            b"\r\n"
            b"body\r\n"
        )

        facts = parse_eml(p)
        assert facts.display_name == "Plain Sender"


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


def test_double_extension_detected_for_archive():
    """holdout-fix-tasks.md T5: 'invoice.pdf.zip' disguising a payload
    behind a fake document extension is the same disguise pattern as
    'invoice.pdf.exe' — double_extension must also fire for archives."""
    from src.parser.attachments import _has_double_extension
    assert _has_double_extension("invoice.pdf.zip") is True


def test_risky_extension():
    from src.parser.attachments import _extension_of
    assert _extension_of("invoice.exe") == "exe"
    assert _extension_of("noext") is None


def test_archive_extension_is_not_risky_type():
    """holdout-fix-tasks.md T5: archives get their own is_archive signal,
    separate from risky_type — a .zip order confirmation attachment isn't
    inherently malicious the way a .exe is."""
    import email
    msg = email.message_from_string(
        "Content-Type: multipart/mixed; boundary=B\n\n"
        "--B\n"
        "Content-Type: application/zip\n"
        "Content-Disposition: attachment; filename=\"invoice.zip\"\n\n"
        "fake payload\n"
        "--B--\n"
    )
    facts = extract_attachment_facts(msg)
    assert len(facts) == 1
    assert facts[0]["is_archive"] is True
    assert facts[0]["risky_type"] is False


def test_inline_image_not_counted_as_attachment():
    """holdout-fix-tasks.md T5 spot-check finding: inline signature/logo
    images (Content-Disposition: inline) with a filename were being
    counted as attachments, inflating attachment-risk facts with pure
    noise — an embedded email-signature logo isn't a security signal."""
    import email
    msg = email.message_from_string(
        "Content-Type: multipart/mixed; boundary=B\n\n"
        "--B\n"
        "Content-Type: image/png\n"
        "Content-Disposition: inline; filename=\"image001.png\"\n\n"
        "fake image bytes\n"
        "--B\n"
        "Content-Type: application/pdf\n"
        "Content-Disposition: attachment; filename=\"invoice.pdf\"\n\n"
        "fake pdf bytes\n"
        "--B--\n"
    )
    facts = extract_attachment_facts(msg)
    filenames = [f["filename"] for f in facts]
    assert "image001.png" not in filenames
    assert "invoice.pdf" in filenames


# --- body.py --------------------------------------------------------

def test_detect_language_turkish():
    assert detect_language("Merhaba, hesabınızı doğrulayın lütfen") == "tr"


def test_detect_language_english():
    assert detect_language("Please verify your account immediately") == "en"


def test_urgency_keywords_detected():
    facts = extract_body_facts("Your account will be suspended. Click here to verify your account now.", is_html=False)
    matched = {m["keyword"] for m in facts["urgency_keywords"]}
    assert "suspended" in matched
    assert "verify your account" in matched
    # each match must carry surrounding context, not just the bare keyword
    # (case-insensitive: matching is case-insensitive, e.g. "Click here")
    for m in facts["urgency_keywords"]:
        assert m["keyword"].lower() in m["context"].lower()


def test_urgency_keyword_substring_false_positive_avoided():
    """holdout-fix-tasks.md T2: 'acil' must not match inside French
    'facilement' — this was a real bug that made unrelated German/French
    spam pick up Turkish urgency keywords."""
    facts = extract_body_facts("Ceci est facilement compris par tous.", is_html=False)
    matched = {m["keyword"] for m in facts["urgency_keywords"]}
    assert "acil" not in matched


def test_urgency_keyword_turkish_suffix_matches():
    """'hemen' should match even with following punctuation/words (word
    boundary at the START only, per body.py's design), and Turkish 'acil'
    with a legitimate suffix like 'acilen' should still be caught."""
    facts = extract_body_facts("Hemen tıklayın, hesabınızı doğrulayın!", is_html=False)
    matched = {m["keyword"] for m in facts["urgency_keywords"]}
    assert "hemen" in matched


def test_credential_request_needs_action_channel():
    """holdout-fix-tasks.md T3: verb + target object alone (no link/
    attachment/form) is NOT enough to flag credential_request — the same
    text without an action channel should be False, but True once one is
    present. This is what distinguishes an actual phishing ask from prose
    that happens to mention a password."""
    text = "Lütfen şifrenizi giriniz ve hesabınızı doğrulayınız."
    without_channel = extract_body_facts(text, is_html=False, has_action_channel=False)
    assert without_channel["credential_request"] is False

    with_channel = extract_body_facts(text, is_html=False, has_action_channel=True)
    assert with_channel["credential_request"] is True


def test_credential_request_security_notice_not_flagged():
    """holdout-fix-tasks.md T3 regression: a message that mentions
    'password' only as a security notice ("we will never ask for your
    password") must NOT be flagged — there's a target object but no
    request verb directed at the reader."""
    text = ("Firmamız hiçbir zaman kullanıcı adı, şifre veya kişisel "
            "bilgilerinizi e-posta ile istememektedir.")
    facts = extract_body_facts(text, is_html=False, has_action_channel=True)
    assert facts["credential_request"] is False


def test_credential_request_relogin_phishing_detected():
    """holdout-fix-tasks.md T3 regression: 'kindly re-login with the
    attachment' + mailbox + an action channel must be flagged — this was
    the false negative the old word-list-only check missed."""
    text = ("Your mailbox cloud capacity is at 97%. kindly re-login with "
            "the attachment to ensure that your mailbox does not reach "
            "full capacity.")
    facts = extract_body_facts(text, is_html=False, has_action_channel=True)
    assert facts["credential_request"] is True


def test_no_credential_request_in_normal_text():
    facts = extract_body_facts("Yarın toplantımız var, saat 10da görüşelim.",
                                is_html=False, has_action_channel=True)
    assert facts["credential_request"] is False


def test_claims_attachment_when_none_exists():
    """holdout-fix-tasks.md T5, candidate 15: 'Attached Re-login' promises
    an attachment but the message has none (it's actually a link) —
    promising a nonexistent file is itself a signal."""
    text = ("Your mailbox cloud capacity is at 97%. kindly re-login with "
            "the attachment to ensure continued access.")
    facts = extract_body_facts(text, is_html=False, has_attachments=False)
    assert facts["claims_attachment"] is True


def test_no_claims_attachment_when_attachment_present():
    text = "Please find the invoice attached for your records."
    facts = extract_body_facts(text, is_html=False, has_attachments=True)
    assert facts["claims_attachment"] is False


def test_no_claims_attachment_when_not_mentioned():
    facts = extract_body_facts("Yarın toplantımız var, saat 10da görüşelim.",
                                is_html=False, has_attachments=False)
    assert facts["claims_attachment"] is False


def test_claims_attachment_turkish():
    facts = extract_body_facts("Faturanız ekte yer almaktadır.",
                                is_html=False, has_attachments=False)
    assert facts["claims_attachment"] is True


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


def test_gateway_banner_is_stripped():
    """Corporate gateways append a fixed external-sender banner to every
    inbound message. It is text the defender added, not part of the email
    under analysis — and on a real sample it was 59% of the body, which the
    LLM then mined for fake findings ("E-posta, 'Gönderici adı ve e-posta
    adresini doğrulayınız' gibi bir uyarı mesajı içeriyor")."""
    body = (
        "Merhaba, siparişiniz hazırlanıyor.\n\n"
        "HARİCİ E-POSTA BİLGİLENDİRMESİ\n"
        "Bu ileti kurum dışından gönderilmiştir.\n"
        "• Gönderici adı ve e-posta adresini doğrulayınız.\n"
        "• Alan adını (domain) dikkatle kontrol ediniz.\n"
    )
    cleaned, stripped = strip_gateway_banner(body)
    assert stripped is True
    assert "doğrulayınız" not in cleaned
    assert "siparişiniz hazırlanıyor" in cleaned


def test_english_gateway_banner_is_stripped():
    body = "Hello, your invoice is attached.\n\nCAUTION: This e-mail originated outside the organisation."
    cleaned, stripped = strip_gateway_banner(body)
    assert stripped is True
    assert "your invoice is attached" in cleaned
    assert "CAUTION" not in cleaned


def test_ordinary_body_is_not_stripped():
    """The match is anchored on the banner heading, so an ordinary email
    mentioning security vocabulary must survive untouched."""
    body = "Güvenlik ayarlarınızı kontrol etmek için hesabınıza giriş yapın."
    cleaned, stripped = strip_gateway_banner(body)
    assert stripped is False
    assert cleaned == body


def test_banner_does_not_reach_body_signals():
    """End-to-end: the banner must be gone before urgency/credential
    patterns and body_text are computed, not merely gone from the report."""
    body = (
        "Merhaba, kaydınız tamamlandı.\n\n"
        "HARİCİ E-POSTA BİLGİLENDİRMESİ\n"
        "Şüpheli bir durumda hemen bilgi güvenliği ekibine bildiriniz.\n"
    )
    facts = extract_body_facts(body, is_html=False)
    assert "HARİCİ" not in facts["body_text"]
    assert "bildiriniz" not in facts["body_text"]
    assert "kaydınız tamamlandı" in facts["body_text"]
