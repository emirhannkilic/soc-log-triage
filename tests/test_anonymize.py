"""
Unit tests for scripts/anonymize.py, v3 plan Adim 3.

Run with: python3 tests/test_anonymize.py
"""
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from anonymize import AliasMap, anonymize_facts, anonymize_url


def _blank_facts(**overrides) -> dict:
    facts = {
        "from_domain": None, "return_path_domain": None, "reply_to_domain": None,
        "message_id_domain": None, "dkim_domain": None,
        "display_name": None, "first_received_ip": None,
        "subject": None, "body_text": "",
        "urls": [], "attachments": [],
    }
    facts.update(overrides)
    return facts


def test_domain_anonymization_is_consistent():
    with tempfile.TemporaryDirectory() as tmp:
        alias_map = AliasMap(Path(tmp) / "map.json")
        a1 = alias_map.domain("evil-phish.ru")
        a2 = alias_map.domain("evil-phish.ru")
        assert a1 == a2, "same real domain must map to the same alias"


def test_domain_anonymization_differs_across_domains():
    with tempfile.TemporaryDirectory() as tmp:
        alias_map = AliasMap(Path(tmp) / "map.json")
        a1 = alias_map.domain("evil-phish.ru")
        a2 = alias_map.domain("another-domain.com")
        assert a1 != a2


def test_preserved_domain_stays_real():
    with tempfile.TemporaryDirectory() as tmp:
        alias_map = AliasMap(Path(tmp) / "map.json")
        assert alias_map.domain("paypal.com") == "paypal.com"
        assert alias_map.domain("gmail.com") == "gmail.com"


def test_display_name_untouched():
    with tempfile.TemporaryDirectory() as tmp:
        alias_map = AliasMap(Path(tmp) / "map.json")
        facts = _blank_facts(display_name="PayPal Support", from_domain="paypal.com")
        result = anonymize_facts(facts, alias_map)
        assert result["display_name"] == "PayPal Support"


def test_map_persists_across_instances():
    with tempfile.TemporaryDirectory() as tmp:
        map_path = Path(tmp) / "map.json"
        m1 = AliasMap(map_path)
        alias = m1.domain("secret-corp.com")
        m1.save()

        m2 = AliasMap(map_path)
        assert m2.domain("secret-corp.com") == alias, \
            "re-loading the map must reproduce the same alias for a known domain"


def test_email_address_in_body_text_is_replaced():
    with tempfile.TemporaryDirectory() as tmp:
        alias_map = AliasMap(Path(tmp) / "map.json")
        facts = _blank_facts(body_text="Reach me at emir@example.com for details.")
        result = anonymize_facts(facts, alias_map)
        assert "emir@example.com" not in result["body_text"]
        assert "@" in result["body_text"]  # still email-shaped


def test_url_domain_is_anonymized_but_path_preserved():
    with tempfile.TemporaryDirectory() as tmp:
        alias_map = AliasMap(Path(tmp) / "map.json")
        anon = anonymize_url("https://evil-phish.ru/login?user=victim", alias_map)
        assert "evil-phish.ru" not in anon
        assert "/login?user=victim" in anon


def test_url_preserved_domain_kept_real():
    with tempfile.TemporaryDirectory() as tmp:
        alias_map = AliasMap(Path(tmp) / "map.json")
        anon = anonymize_url("https://paypal.com/login", alias_map)
        assert anon == "https://paypal.com/login"


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


def test_outcome_fields_pass_through_unchanged():
    """spf_result etc. aren't part of anonymize_facts's job — this test
    documents that fields outside its scope survive a round-trip."""
    with tempfile.TemporaryDirectory() as tmp:
        alias_map = AliasMap(Path(tmp) / "map.json")
        facts = _blank_facts()
        facts["spf_result"] = "fail"
        facts["urgency_keywords"] = ["urgent", "acil"]
        result = anonymize_facts(facts, alias_map)
        assert result["spf_result"] == "fail"
        assert result["urgency_keywords"] == ["urgent", "acil"]


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
