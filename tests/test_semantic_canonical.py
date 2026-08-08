"""Unit tests for src/semantic/canonical.py (PHISHING_ROUTING_PLAN.md
step 8). Regression coverage for a real bug: facts.body_text's CRLF
line endings silently became LF somewhere in the semantic-eval labeling
round-trip (a text-mode file read), causing two hand-labeled
ground-truth findings to fail grounding against the raw, still-CRLF
body_text. See PROGRESS.md and src/semantic/canonical.py's module
docstring for the full story."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.semantic.canonical import canonicalize_body  # noqa: E402


def test_crlf_becomes_lf():
    assert canonicalize_body("a\r\nb\r\nc") == "a\nb\nc"


def test_lone_cr_becomes_lf():
    """Old Mac-style line endings, or a stray \\r not part of a CRLF
    pair — both must normalize to \\n, not be left as \\r or turned
    into \\r\\n."""
    assert canonicalize_body("a\rb\rc") == "a\nb\nc"


def test_already_lf_is_unchanged():
    assert canonicalize_body("a\nb\nc") == "a\nb\nc"


def test_mixed_line_endings_all_normalize():
    """A body could plausibly mix CRLF and lone-CR (different mail
    clients/relays) — every variant must collapse to \\n."""
    assert canonicalize_body("a\r\nb\rc\nd") == "a\nb\nc\nd"


def test_idempotent_on_already_lf():
    """canonicalize_body(canonicalize_body(x)) == canonicalize_body(x) —
    calling it twice (e.g. once when rendering a worksheet, again when
    later re-parsing the same .eml) must be a no-op the second time."""
    body = "Sayın müşterimiz,\r\nhesabınız\raskıya\n\nalınacaktır."
    once = canonicalize_body(body)
    twice = canonicalize_body(once)
    assert once == twice


def test_no_line_endings_at_all_is_unchanged():
    assert canonicalize_body("no newlines here") == "no newlines here"


def test_empty_string_is_unchanged():
    assert canonicalize_body("") == ""


def test_invisible_unicode_is_preserved():
    """canonicalize_body only touches line endings — U+034F (combining
    grapheme joiner, the real Gmail tracking artifact that made offset
    counting hard for the model, see src/semantic/validate.py's tests)
    and other invisible/zero-width characters must survive unchanged."""
    invisible = "͏" * 10
    body = f"{invisible}\r\nMerhaba\r\ndünya"
    result = canonicalize_body(body)
    assert result == f"{invisible}\nMerhaba\ndünya"
    assert result.count("͏") == 10


def test_does_not_collapse_or_strip_whitespace():
    """Only \\r\\n / \\r -> \\n — no stripping, no whitespace collapsing.
    Multiple blank lines and leading/trailing spaces must survive."""
    body = "line1\r\n\r\n\r\nline2   \r\n   line3"
    result = canonicalize_body(body)
    assert result == "line1\n\n\nline2   \n   line3"


# --- regression: the two real candidates that failed before the fix ----

def test_regression_candidate_2_facebook_team_quote_now_grounds():
    """The exact real-world failure: evidence quoted with LF ("\\n")
    against a body whose actual bytes are CRLF ("\\r\\n") must be found
    once canonicalize_body() is applied to BOTH sides."""
    raw_body = "Yes, me\r\nThanks,\r\nThe Facebook Team\r\nThis message was sent to..."
    evidence = "Thanks,\nThe Facebook Team"
    canonical = canonicalize_body(raw_body)
    assert evidence in canonical
    # And the raw (uncanonicalized) body must NOT contain it verbatim —
    # confirming this test actually exercises the fix, not a no-op.
    assert evidence not in raw_body


def test_regression_candidate_2_mailto_quote_now_grounds():
    raw_body = (
        "...it's really you.\r\n"
        "<mailto:acc-info@esp-accessacc.com?subject=Send+Statement%20x@hotmail.com>\r\n"
        "Report the user\r\n"
        "<mailto:acc-info@esp-accessacc.com?subject=Yes+me%20x@hotmail.com>"
    )
    evidence = (
        "<mailto:acc-info@esp-accessacc.com?subject=Send+Statement%20x@hotmail.com>\n"
        "Report the user"
    )
    canonical = canonicalize_body(raw_body)
    assert evidence in canonical
    assert evidence not in raw_body


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
