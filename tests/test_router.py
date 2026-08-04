"""Unit tests for src/router.py — the input router in front of the
phishing pipeline.

Run with: python3 tests/test_router.py
"""
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.router import Route, looks_like_raw_email, route  # noqa: E402

RAW_EMAIL = (
    "Return-Path: <bounce@mailer.example.test>\n"
    "Received: from mx.example.test by mail.example.test; "
    "Mon, 1 Jan 2024 00:00:00 +0000\n"
    "From: \"Destek\" <destek@example.test>\n"
    "To: user@example.test\n"
    "Subject: Siparişiniz hakkında\n"
    "Date: Mon, 1 Jan 2024 00:00:00 +0000\n"
    "Message-ID: <abc@example.test>\n"
    "\n"
    "Merhaba, siparişiniz hazırlanıyor.\n"
)


def test_eml_file_routes_to_phishing():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "mail.eml"
        p.write_text(RAW_EMAIL, encoding="utf-8")
        d = route(file_path=p)
        assert d.route is Route.PHISHING
        assert d.matched_rule == "file_extension"
        assert d.eml_path == p


def test_pasted_raw_email_routes_to_phishing():
    """The case that matters most in practice: an analyst pastes the message
    into a chat box instead of attaching a file."""
    d = route(text=RAW_EMAIL)
    assert d.route is Route.PHISHING
    assert d.matched_rule == "pasted_email"
    assert d.raw_email == RAW_EMAIL


def test_plain_question_is_unsupported():
    """Only the phishing pipeline exists, and it needs an actual email — so
    a general question must be refused with an explanation, not routed to a
    persona that was never built."""
    d = route(text="Bu mail sahte mi acaba, ne dersin?")
    assert d.route is Route.UNSUPPORTED
    assert d.matched_rule == "text_not_email"


def test_single_header_mention_is_not_an_email():
    """One header name is not enough. 'From:' appears in ordinary prose and
    in quoted replies; requiring several distinct headers is what separates
    a pasted message from text that merely mentions one."""
    text = "From: the security team we received a phishing warning today"
    assert looks_like_raw_email(text) is False
    assert route(text=text).route is Route.UNSUPPORTED


def test_two_headers_still_not_enough():
    text = "Subject: acil\nFrom: patron\nlütfen bu ödemeyi bugün yap"
    assert looks_like_raw_email(text) is False


def test_email_content_without_eml_extension_is_detected():
    """The extension is a shortcut, the content is the authority — a raw
    message saved as .txt is still a message."""
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "forwarded.txt"
        p.write_text(RAW_EMAIL, encoding="utf-8")
        d = route(file_path=p)
        assert d.route is Route.PHISHING
        assert d.matched_rule == "file_content"


def test_non_email_file_is_unsupported():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "notes.md"
        p.write_text("# Toplantı notları\n\nBugün phishing konuştuk.\n",
                     encoding="utf-8")
        d = route(file_path=p)
        assert d.route is Route.UNSUPPORTED
        assert d.matched_rule == "file_not_email"


def test_missing_file_is_reported():
    d = route(file_path=Path("/tmp/definitely-does-not-exist-12345.eml"))
    assert d.route is Route.UNSUPPORTED
    assert d.matched_rule == "file_missing"


def test_empty_input_is_reported():
    assert route().matched_rule == "empty_input"
    assert route(text="   ").route is Route.UNSUPPORTED


def test_unsupported_decisions_always_explain_why():
    """An unsupported route without a reason would leave the user guessing
    what to do differently."""
    for d in (route(text="merhaba"), route(), route(file_path=Path("/nope.eml"))):
        assert d.route is Route.UNSUPPORTED
        assert d.reason.strip(), "gerekçe boş olmamalı"


if __name__ == "__main__":
    import traceback

    tests = [(n, o) for n, o in list(globals().items())
             if n.startswith("test_") and callable(o)]
    passed = failed = 0
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
