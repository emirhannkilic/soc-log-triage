"""Lightweight (no model call) validation for
tests/fixtures/hybrid_credential_upgrade.eml — a synthetic, PII-free
.eml built for the upgrade smoke-test task list (item 3, "Güvenilir ->
Muhtemel Phishing yükseltmesi yapan mailde iki çağrılı test").

This file does NOT run the model — it only confirms the fixture has
the properties that scenario actually depends on, so a future real
scripts/smoke_test_hybrid.py run against it is testing the real
upgrade path rather than something else entirely:
    1. fast mode (deterministic parser + rule engine v1 only) already
       classifies this email as "Güvenilir" — the upgrade the smoke
       test is meant to observe must come from the semantic layer, not
       from the rule engine already deciding Phishing/Muhtemel Phishing
       on its own.
    2. EmailFacts.credential_request is True — the deterministic
       body-text heuristic (src/parser/body.py's verb+target+window
       check) already recognizes a credential-request pattern here,
       independent of whatever the semantic extractor separately finds
       in the body.
    3. build_context() computes has_external_url=True — the email's
       one URL (portal.sirket-b-hizmet.test) is a DIFFERENT
       organization than the From domain (sirket-a.test), the exact
       has_external_url shape src/decision/phishing_policy.py's
       credential_request-upgrade rule requires (see
       tests/test_decision_context.py's same-organization regression
       tests for why "any URL" was not this condition).

If a real semantic extractor run against this fixture reports a
CREDENTIAL_REQUEST finding, decide() will upgrade Güvenilir to
Muhtemel Phishing via DECISION_PATH_CREDENTIAL_REQUEST_URL_UPGRADE —
that real-model behavior is exactly what the smoke test (run by the
user, not this suite) is meant to observe. This file only guarantees
the STARTING conditions are correct; it makes no claim about what the
model itself will find."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.decision.context import build_context  # noqa: E402
from src.parser.parse import parse_eml  # noqa: E402
from src.workflows.phishing import analyze_phishing  # noqa: E402

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "hybrid_credential_upgrade.eml"


def test_fixture_file_exists():
    assert FIXTURE_PATH.is_file()


def test_fast_mode_classifies_as_guvenilir():
    result = analyze_phishing(FIXTURE_PATH, mode="fast")
    assert result.rule_assessment.rule_verdict == "Güvenilir"
    assert result.report.risk_seviyesi == "Güvenilir"


def test_parser_detects_credential_request():
    facts = parse_eml(FIXTURE_PATH)
    assert facts.credential_request is True


def test_context_reports_external_url():
    facts = parse_eml(FIXTURE_PATH)
    context = build_context(facts)
    assert context.has_external_url is True
    assert context.url_count == 1


def test_url_points_at_a_different_organization_than_from_domain():
    """Confirms the fixture exercises the REAL has_external_url
    condition (different organization), not merely "any URL exists" —
    same_organization()-false is the actual requirement, see
    schemas/decision.py."""
    facts = parse_eml(FIXTURE_PATH)
    assert facts.from_domain == "sirket-a.test"
    assert len(facts.urls) == 1
    assert facts.urls[0].href_domain == "portal.sirket-b-hizmet.test"


def test_auth_headers_are_all_pass_and_aligned():
    """The fixture's SPF/DKIM/DMARC are all pass and DKIM-aligned with
    From — this is deliberate: the fast-mode Güvenilir verdict must
    come from a genuinely clean header set, not from an auth failure
    that happens to net out to a low score. This isolates the upgrade
    path to the semantic layer + has_external_url combination the
    policy actually tests, not a coincidental score cancellation."""
    facts = parse_eml(FIXTURE_PATH)
    assert facts.spf_result == "pass"
    assert facts.dkim_result == "pass"
    assert facts.dmarc_result == "pass"
    assert facts.dkim_domain_matches_from is True


def test_fixture_contains_no_real_personal_data():
    """Sanity check for the fixture's own design constraint — built
    fresh with synthetic sender/recipient/domains under the .test TLD
    (RFC 2606, reserved for documentation/testing) specifically so this
    fixture can be used in smoke tests without CLAUDE.md's data/raw/
    PII concerns."""
    raw = FIXTURE_PATH.read_bytes().decode("utf-8", errors="replace")
    assert "@example.test" in raw or "alici@example.test" in raw
    assert ".test" in raw
    # No real-looking phone number or long digit run that could be a
    # phone/ID number — this fixture's body has no digits at all.
    body_only = raw.split("\n\n", 1)[-1]
    assert not any(c.isdigit() for c in body_only.split("Doğrulama sayfası:")[0])


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
