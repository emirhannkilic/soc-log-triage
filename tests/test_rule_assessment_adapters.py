"""Unit tests for src/rules/adapters.py (PHISHING_ROUTING_PLAN.md step 3).

Verifies both Verdict -> RuleAssessment and VerdictV2 -> RuleAssessment
adapters produce a schema-valid, engine-agnostic RuleAssessment. Does not
re-test engine.py or engine_v2.py's own decision logic — see
tests/test_rules_engine.py and tests/test_rules_engine_v2.py for that.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from schemas.facts import AttachmentFacts, EmailFacts, UrgencyMatch, UrlFacts
from schemas.rule_assessment import RuleAssessment
from src.rules.adapters import from_v1, from_v2
from src.rules.engine import evaluate, load_rules as load_rules_v1
from src.rules.engine_v2 import evaluate_v2, load_rules as load_rules_v2

RULES_V1 = load_rules_v1()
RULES_V2 = load_rules_v2()

BASE_FACTS_KWARGS = dict(
    spf_result="pass",
    dkim_result="pass",
    dmarc_result="pass",
    dkim_domain="example.com",
    dkim_domain_matches_from=True,
    spf_mailfrom_domain="example.com",
    spf_aligned=True,
    from_domain="example.com",
    from_source="From",
    return_path_domain="example.com",
    reply_to_domain="example.com",
    return_path_mismatch=False,
    reply_to_mismatch=False,
    display_name="Example Co",
    display_name_has_email=False,
    display_name_brand_mismatch=False,
    message_id_domain="example.com",
    message_id_domain_matches_from=True,
    received_hop_count=2,
    first_received_ip="1.2.3.4",
    urls=[],
    attachments=[],
    has_html_form=False,
    form_action_domain=None,
    has_hidden_text=False,
    has_large_hidden_text=False,
    image_only_body=False,
    urgency_keywords=[],
    credential_request=False,
    claims_attachment=False,
    has_advance_fee_fraud_language=False,
    has_fake_reward_claim_language=False,
    subject="Hello",
    date="Mon, 1 Jan 2024 00:00:00 +0000",
    body_text="hello",
    language="en",
)


def facts(**overrides) -> EmailFacts:
    kwargs = dict(BASE_FACTS_KWARGS)
    kwargs.update(overrides)
    return EmailFacts(**kwargs)


def phishing_facts() -> EmailFacts:
    return facts(
        spf_result="fail",
        dkim_result="fail",
        dkim_domain_matches_from=False,
        display_name="PayPal Support",
        display_name_brand_mismatch=True,
        from_domain="paypa1-secure.tk",
        urls=[
            UrlFacts(
                url="http://paypa1-secure.tk/login",
                href_domain="paypa1-secure.tk",
                anchor_text_domain="paypal.com",
                text_href_mismatch=True,
                is_ip_based=False,
                is_shortener=False,
                has_punycode=False,
                redirect_param=False,
            )
        ],
        credential_request=True,
    )


# --- v1 adapter -----------------------------------------------------

def test_from_v1_clean_email_is_guvenilir_assessment():
    result = evaluate(facts().flat_signals(), RULES_V1)
    assessment = from_v1(result, RULES_V1)

    assert isinstance(assessment, RuleAssessment)
    assert assessment.engine_version == "v1"
    assert assessment.rule_verdict == "Güvenilir"
    assert assessment.score == result.score
    assert assessment.total is None
    assert assessment.families == []
    assert assessment.critical_matches == []
    # A fully-aligned clean email still carries the all_auth_pass_and_consistent
    # bonus (-3) as a match in v1 — that's the negative-score mechanism, not an
    # absence of evidence.
    assert len(assessment.evidence) == 1
    assert assessment.evidence[0].signal == "all_auth_pass_and_consistent"
    assert assessment.evidence[0].weight < 0
    assert len(assessment.decision_reasons) == 1


def test_from_v1_phishing_email_carries_matches_as_evidence():
    result = evaluate(phishing_facts().flat_signals(), RULES_V1)
    assessment = from_v1(result, RULES_V1)

    assert assessment.rule_verdict == "Phishing"
    assert len(assessment.evidence) == len(result.matches)
    signal_names = {e.signal for e in assessment.evidence}
    assert "spf_or_dmarc_fail" in signal_names
    assert "display_name_brand_mismatch" in signal_names
    for e in assessment.evidence:
        assert e.description
        assert e.weight > 0


def test_from_v1_roundtrips_score_exactly():
    result = evaluate(phishing_facts().flat_signals(), RULES_V1)
    assessment = from_v1(result, RULES_V1)
    assert assessment.score == sum(m.weight for m in result.matches)


# --- v2 adapter -----------------------------------------------------

def test_from_v2_clean_email_is_guvenilir_assessment():
    result = evaluate_v2(facts(), RULES_V2)
    assessment = from_v2(result, RULES_V2)

    assert isinstance(assessment, RuleAssessment)
    assert assessment.engine_version == "v2"
    assert assessment.rule_verdict == "Güvenilir"
    assert assessment.score is None
    assert assessment.total == result.total
    assert len(assessment.families) == 4
    assert {f.family for f in assessment.families} == {"identity", "url", "content", "payload"}
    assert assessment.critical_matches == []
    assert assessment.evidence == []
    assert len(assessment.decision_reasons) == 1


def test_from_v2_phishing_email_carries_family_scores_and_evidence():
    result = evaluate_v2(phishing_facts(), RULES_V2)
    assessment = from_v2(result, RULES_V2)

    assert assessment.rule_verdict in ("Phishing", "Muhtemel Phishing")
    assert assessment.total == result.total
    families_by_name = {f.family: f.score for f in assessment.families}
    assert families_by_name["identity"] >= 1
    assert families_by_name["url"] >= 1
    assert len(assessment.evidence) > 0
    for e in assessment.evidence:
        # description/weight are config/rules.yaml's real values — the
        # same ones from_v1 would report for this signal name — not a
        # "family/subgroup" label and not the normalized strength.
        assert e.description
        assert e.family in ("identity", "url", "content", "payload")
        assert e.subgroup
        assert e.strength in (1, 2, 3)


def test_from_v2_evidence_weight_matches_config_not_normalized_strength():
    """attachment_double_extension's config/rules.yaml weight is 4, but
    _weight_to_strength() caps normalized strength at 3 (weak/moderate/
    strong). RuleEvidence.weight must report the real 4, not the capped
    3 — catches the bug where the adapter reported the normalized
    strength as `weight`."""
    f = facts(
        attachments=[
            AttachmentFacts(
                filename="invoice.pdf.scr",
                mime_type="application/octet-stream",
                size=1000,
                risky_type=False,
                double_extension=True,
                is_archive=False,
                extension_mismatch=False,
            )
        ]
    )
    result = evaluate_v2(f, RULES_V2)
    assessment = from_v2(result, RULES_V2)

    hits = [e for e in assessment.evidence if e.signal == "attachment_double_extension"]
    assert len(hits) == 1
    configured_weight = RULES_V2["signals"]["attachment_double_extension"]["weight"]
    assert configured_weight == 4
    assert hits[0].weight == 4
    assert hits[0].strength == 3  # weight 4 normalizes to strength 3 (capped)
    assert hits[0].weight != hits[0].strength


def test_from_v2_archive_signal_has_no_config_entry_but_does_not_crash():
    """attachment_is_archive has no config/rules.yaml entry (v1's
    is_archive_with_credential_request combined rule was dropped, not
    ported — see src/rules/engine_v2.py). The adapter must fall back
    gracefully instead of raising KeyError."""
    f = facts(
        attachments=[
            AttachmentFacts(
                filename="statement.zip",
                mime_type="application/zip",
                size=500,
                risky_type=False,
                double_extension=False,
                is_archive=True,
                extension_mismatch=False,
            )
        ]
    )
    result = evaluate_v2(f, RULES_V2)
    assessment = from_v2(result, RULES_V2)

    archive_hits = [e for e in assessment.evidence if e.signal == "attachment_is_archive"]
    assert len(archive_hits) == 1
    assert archive_hits[0].family == "payload"
    assert archive_hits[0].subgroup == "archive"
    assert archive_hits[0].strength == 1
    assert archive_hits[0].weight == 1  # fallback: strength, since no configured weight exists


def test_from_v2_critical_predicate_surfaces_in_both_fields():
    f = facts(
        attachments=[
            AttachmentFacts(
                filename="invoice.pdf.exe",
                mime_type="application/octet-stream",
                size=1000,
                risky_type=True,
                double_extension=True,
                is_archive=False,
                extension_mismatch=False,
            )
        ]
    )
    result = evaluate_v2(f, RULES_V2)
    assessment = from_v2(result, RULES_V2)

    assert assessment.rule_verdict == "Phishing"
    assert assessment.critical_matches == result.critical_matches
    assert any("critical predicate" in r for r in assessment.decision_reasons)


def test_from_v2_decision_reasons_nonempty_for_every_verdict_band():
    for f in (facts(), phishing_facts()):
        result = evaluate_v2(f, RULES_V2)
        assessment = from_v2(result, RULES_V2)
        assert len(assessment.decision_reasons) >= 1
        assert all(isinstance(r, str) and r for r in assessment.decision_reasons)


# --- cross-engine schema parity --------------------------------------

def test_v1_and_v2_assessments_share_identical_schema():
    v1_result = evaluate(facts().flat_signals(), RULES_V1)
    v2_result = evaluate_v2(facts(), RULES_V2)
    v1_assessment = from_v1(v1_result, RULES_V1)
    v2_assessment = from_v2(v2_result, RULES_V2)

    assert set(v1_assessment.model_dump().keys()) == set(v2_assessment.model_dump().keys())


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
