"""
Unit tests for scripts/anonymize.py, v3 plan Adim 3 (revised scope per
holdout-fix-tasks.md T4: only the account holder's identity is redacted;
domains/IPs/third-party emails are left real as genuine training signal).

Run with: python3 tests/test_anonymize.py
"""
import os
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# Pin a FICTIONAL identity before importing anonymize, which reads these at
# import time. Two reasons this is set here rather than relying on the
# developer's own `.env.anonymize`:
#   1. No real personal data belongs in a committed test file.
#   2. The suite must assert on fixed strings to be meaningful; reading
#      whoever's identity happens to be configured locally would make the
#      tests pass or fail depending on the machine.
# `_load_local_env` never overwrites an already-set variable, so these win.
TEST_OWN_EMAIL = "test.owner@example.test"
TEST_OWN_FIRST = "Testcan"
TEST_OWN_SURNAME = "Ornek"
os.environ["ANONYMIZE_OWN_EMAIL"] = TEST_OWN_EMAIL
os.environ["ANONYMIZE_OWN_NAME_VARIANTS"] = (
    f"{TEST_OWN_FIRST} {TEST_OWN_SURNAME}|{TEST_OWN_FIRST}"
)

# A different, unrelated person who happens to share the owner's surname —
# used to prove the bare surname is NOT matched as the owner's identity.
THIRD_PARTY_NAME = f"Baskabir {TEST_OWN_SURNAME}"

from anonymize import (
    OWN_EMAIL,
    OWN_EMAIL_ALIAS,
    OWN_NAME_ALIAS,
    AliasMap,
    anonymize_facts,
    anonymize_salutation_names_in_text,
    redact_own_email_in_text,
    redact_own_name_in_text,
)


def _blank_facts(**overrides) -> dict:
    facts = {
        "subject": None, "body_text": "",
        "urls": [], "attachments": [],
    }
    facts.update(overrides)
    return facts


def test_own_email_is_redacted():
    text = f"Reach me at {OWN_EMAIL} for details."
    result = redact_own_email_in_text(text)
    assert OWN_EMAIL not in result
    assert OWN_EMAIL_ALIAS in result


def test_third_party_email_is_not_redacted():
    """A different person's email address is NOT the account holder's
    personal data — it must survive, since it's real training signal
    (e.g. a sender address, or a third party mentioned in the body)."""
    text = "Contact support at someone-else@example.com for help."
    result = redact_own_email_in_text(text)
    assert "someone-else@example.com" in result


def test_own_email_redaction_case_insensitive():
    text = f"Account: {OWN_EMAIL.upper()}"
    result = redact_own_email_in_text(text)
    assert OWN_EMAIL.upper() not in result
    assert OWN_EMAIL_ALIAS in result


def test_own_name_redacted_outside_salutation():
    """The literal name-variant matcher must catch leaks the structural
    salutation pattern can't anticipate, e.g. a marketing subject line
    that addresses the user by first name without any salutation marker
    ("Sende <name>" in a Turkish e-commerce push, not "Dear <name>,")."""
    text = f"Yeni sezon indirimleri seni bekliyor, Sende {TEST_OWN_FIRST} olabilirsin!"
    result = redact_own_name_in_text(text)
    assert TEST_OWN_FIRST not in result
    assert OWN_NAME_ALIAS in result


def test_own_full_name_redacted():
    text = f"{TEST_OWN_FIRST} {TEST_OWN_SURNAME} adına oluşturulan hesap onaylandı."
    result = redact_own_name_in_text(text)
    assert TEST_OWN_FIRST not in result
    assert TEST_OWN_SURNAME not in result


def test_third_party_sharing_surname_not_matched_by_own_name_regex():
    """A surname is commonly shared by unrelated third parties, so the bare
    surname is deliberately excluded from _OWN_NAME_VARIANTS_RE — otherwise
    a different real person sharing it would be misredacted as if they were
    the account holder. (Handling third-party names is the salutation
    matcher's job, not this one's.)"""
    text = f"Sayın {THIRD_PARTY_NAME}, siparişiniz kargoya verildi."
    result = redact_own_name_in_text(text)
    assert THIRD_PARTY_NAME in result


def test_salutation_name_is_consistent():
    with tempfile.TemporaryDirectory() as tmp:
        alias_map = AliasMap(Path(tmp) / "map.json")
        r1 = anonymize_salutation_names_in_text(
            f"Sayın {THIRD_PARTY_NAME}, siparişiniz alındı.", alias_map)
        r2 = anonymize_salutation_names_in_text(
            f"Merhaba {THIRD_PARTY_NAME}, tekrar hoş geldiniz.", alias_map)
        # extract the alias from each and confirm they match
        alias1 = r1.split("Sayın ")[1].split(",")[0]
        alias2 = r2.split("Merhaba ")[1].split(",")[0]
        assert alias1 == alias2, "same real name must map to the same alias"


def test_salutation_name_removed_from_output():
    with tempfile.TemporaryDirectory() as tmp:
        alias_map = AliasMap(Path(tmp) / "map.json")
        result = anonymize_salutation_names_in_text(
            f"Hi {TEST_OWN_FIRST}, thanks for registering.", alias_map)
        assert TEST_OWN_FIRST not in result


def test_map_persists_across_instances():
    with tempfile.TemporaryDirectory() as tmp:
        map_path = Path(tmp) / "map.json"
        m1 = AliasMap(map_path)
        alias = m1.name(THIRD_PARTY_NAME)
        m1.save()

        m2 = AliasMap(map_path)
        assert m2.name(THIRD_PARTY_NAME) == alias, \
            "re-loading the map must reproduce the same alias for a known name"


def test_attachment_filename_extension_preserved():
    with tempfile.TemporaryDirectory() as tmp:
        alias_map = AliasMap(Path(tmp) / "map.json")
        facts = _blank_facts(attachments=[
            {"filename": "invoice_ahmet_yilmaz.exe", "mime_type": "application/x-msdownload",
             "size": 1024, "double_extension": False, "risky_type": True}
        ])
        result = anonymize_facts(facts, alias_map)
        assert result["attachments"][0]["filename"].endswith(".exe")
        assert "ahmet" not in result["attachments"][0]["filename"].lower()


def test_domains_are_left_real():
    """Regression test for the T4 scope revision: domains are NOT
    anonymized anymore (unlike the earlier version of this module) — real
    domain structure is genuine training signal for LoRA fine-tuning."""
    with tempfile.TemporaryDirectory() as tmp:
        alias_map = AliasMap(Path(tmp) / "map.json")
        facts = _blank_facts(
            urls=[{"url": "https://evil-phish.ru/login", "href_domain": "evil-phish.ru",
                   "anchor_text_domain": None, "text_href_mismatch": False,
                   "is_ip_based": False, "is_shortener": False, "has_punycode": False,
                   "redirect_param": False}],
        )
        result = anonymize_facts(facts, alias_map)
        assert result["urls"][0]["url"] == "https://evil-phish.ru/login"
        assert result["urls"][0]["href_domain"] == "evil-phish.ru"


def test_third_party_email_in_url_survives():
    with tempfile.TemporaryDirectory() as tmp:
        alias_map = AliasMap(Path(tmp) / "map.json")
        facts = _blank_facts(
            urls=[{"url": "https://mailer.example.com/unsubscribe?email=other-person@gmail.com",
                   "href_domain": "mailer.example.com", "anchor_text_domain": None,
                   "text_href_mismatch": False, "is_ip_based": False, "is_shortener": False,
                   "has_punycode": False, "redirect_param": False}],
        )
        result = anonymize_facts(facts, alias_map)
        assert "other-person@gmail.com" in result["urls"][0]["url"]


def test_own_identity_url_encoded_in_url_is_redacted():
    """Regression test: tracking/review-request links found in the Gmail
    corpus embed the recipient's email and name as (sometimes doubly)
    URL-encoded query params — e.g. a `name=` value that only reveals the
    recipient after being percent-decoded twice. redact_own_email_in_text /
    redact_own_name_in_text only match literal substrings and miss this, so
    anonymize_facts must also catch it at the URL level."""
    with tempfile.TemporaryDirectory() as tmp:
        alias_map = AliasMap(Path(tmp) / "map.json")
        local_part = TEST_OWN_EMAIL.split("@")[0]
        # Doubly-encoded: '%2540' decodes to '%40', which decodes to '@';
        # '%2B' decodes to '+' (a space in query-string terms).
        encoded_email = TEST_OWN_EMAIL.replace("@", "%2540")
        encoded_url = (
            "https://mailtrack.example.test/CL0/https:%2F%2Freviews.example.test"
            f"%2Femails%2Freviews%2Fnew%3Femail={encoded_email}"
            f"%26name={TEST_OWN_FIRST}%2B{TEST_OWN_SURNAME}"
        )
        facts = _blank_facts(
            urls=[{"url": encoded_url,
                   "href_domain": "mailtrack.example.test", "anchor_text_domain": None,
                   "text_href_mismatch": False, "is_ip_based": False, "is_shortener": False,
                   "has_punycode": False, "redirect_param": False}],
        )
        result = anonymize_facts(facts, alias_map)
        assert local_part not in result["urls"][0]["url"]
        assert TEST_OWN_FIRST not in result["urls"][0]["url"]


def test_own_email_in_url_is_redacted():
    with tempfile.TemporaryDirectory() as tmp:
        alias_map = AliasMap(Path(tmp) / "map.json")
        facts = _blank_facts(
            urls=[{"url": f"https://mailer.example.com/unsubscribe?email={OWN_EMAIL}",
                   "href_domain": "mailer.example.com", "anchor_text_domain": None,
                   "text_href_mismatch": False, "is_ip_based": False, "is_shortener": False,
                   "has_punycode": False, "redirect_param": False}],
        )
        result = anonymize_facts(facts, alias_map)
        assert OWN_EMAIL not in result["urls"][0]["url"]


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
