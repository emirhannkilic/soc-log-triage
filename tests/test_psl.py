"""
Unit tests for src/parser/psl.py — the real Public Suffix List (tldextract)
lookup that replaced the "last two labels" heuristic previously duplicated
in src/parser/headers.py and src/rules/engine_v2.py (Rule Engine v2 adım 6,
CLAUDE.md, 2026-08-07).

Run with: python3 tests/test_psl.py
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.parser.psl import registrable_domain, same_organization


def test_simple_domain_is_itself():
    assert registrable_domain("uber.com") == "uber.com"


def test_subdomain_reduces_to_registrable_domain():
    assert registrable_domain("email.uber.com") == "uber.com"


def test_multi_part_tld_co_uk():
    """The old "last two labels" heuristic got this wrong: it would return
    "co.uk" for "mail.example.co.uk" (a public suffix, not an organization's
    domain)."""
    assert registrable_domain("mail.example.co.uk") == "example.co.uk"


def test_multi_part_tld_gov_tr():
    assert registrable_domain("www.turkiye.gov.tr") == "turkiye.gov.tr"


def test_bare_public_suffix_returns_none():
    """A bare suffix like "gov.tr" or "co.uk" has no organization label in
    front of it — it isn't a real registrable domain and must not compare
    equal to another truncated suffix. Found on a real e-Devlet email
    (inbox-1913.eml) whose link anchor text was visually shortened to
    "gov.tr" while href was "www.turkiye.gov.tr" — the old heuristic
    reduced both to "gov.tr" and called it a match, masking that "gov.tr"
    was never a real domain."""
    assert registrable_domain("gov.tr") is None
    assert registrable_domain("co.uk") is None


def test_none_input_returns_none():
    assert registrable_domain(None) is None


def test_unrecognized_tld_falls_back_to_input():
    assert registrable_domain("host.totallyfaketld") == "host.totallyfaketld"


def test_same_organization_true_for_subdomain():
    assert same_organization("mailer.netflix.com", "netflix.com") is True


def test_same_organization_false_for_different_orgs():
    assert same_organization("evil.tld", "bank.com") is False


def test_same_organization_false_when_either_side_bare_suffix():
    assert same_organization("gov.tr", "www.turkiye.gov.tr") is False


def test_same_organization_false_on_none():
    assert same_organization(None, "example.com") is False
    assert same_organization("example.com", None) is False


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
