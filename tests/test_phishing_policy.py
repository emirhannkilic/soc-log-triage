"""Unit tests for src/decision/phishing_policy.py (PHISHING_ROUTING_PLAN.md
step 9). Pure deterministic logic — RuleAssessment, ValidatedSemanticFinding,
and PhishingDecisionContext are all constructed by hand, no model call
and no rule engine invocation needed to exercise the policy."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from schemas.decision import FinalDecision, PhishingDecisionContext  # noqa: E402
from schemas.rule_assessment import RuleAssessment, RuleEvidence  # noqa: E402
from schemas.semantic import SemanticFindingType, ValidatedSemanticFinding  # noqa: E402
from src.decision.phishing_policy import (  # noqa: E402
    DECISION_PATH_CREDENTIAL_REQUEST_URL_UPGRADE,
    DECISION_PATH_PAYMENT_REQUEST_COMBINATION_UPGRADE,
    DECISION_PATH_RULE_ENGINE_ONLY,
    decide,
)


def _assessment(verdict, evidence=None) -> RuleAssessment:
    return RuleAssessment(
        engine_version="v1",
        rule_verdict=verdict,
        score=5.0 if verdict != "Güvenilir" else 0.0,
        total=None,
        families=[],
        critical_matches=[],
        evidence=evidence or [],
        decision_reasons=["x"],
    )


def _finding(type_, start=0, end=10, confidence=0.9) -> ValidatedSemanticFinding:
    return ValidatedSemanticFinding(
        type=type_,
        evidence="x" * (end - start),
        start=start,
        end=end,
        model_confidence=confidence,
        explanation="x",
    )


def _context(has_url=False, url_count=0, parser_credential_request=False) -> PhishingDecisionContext:
    return PhishingDecisionContext(
        has_url=has_url,
        url_count=url_count,
        url_ids=[f"http://x{i}.example" for i in range(url_count)],
        parser_credential_request=parser_credential_request,
    )


NO_URL = _context()
WITH_URL = _context(has_url=True, url_count=1)


# --- Phishing is never touched ------------------------------------------

def test_phishing_verdict_is_preserved_regardless_of_semantic_findings():
    assessment = _assessment("Phishing", evidence=[
        RuleEvidence(signal="spf_or_dmarc_fail", description="x", weight=3),
    ])
    result = decide(assessment, [], NO_URL)
    assert isinstance(result, FinalDecision)
    assert result.rule_verdict == "Phishing"
    assert result.final_verdict == "Phishing"
    assert result.decision_path == DECISION_PATH_RULE_ENGINE_ONLY
    assert result.analyst_review_required is False


def test_phishing_verdict_preserved_even_with_no_semantic_findings_at_all():
    """A Phishing verdict from the rule engine needs no semantic
    corroboration whatsoever — this is the "never overridden downward,
    never needs upward help" case."""
    assessment = _assessment("Phishing")
    result = decide(assessment, [], NO_URL)
    assert result.final_verdict == "Phishing"


def test_phishing_verdict_carries_rule_evidence_as_contributing_ids():
    assessment = _assessment("Phishing", evidence=[
        RuleEvidence(signal="spf_or_dmarc_fail", description="x", weight=3),
        RuleEvidence(signal="display_name_brand_mismatch", description="x", weight=3),
    ])
    result = decide(assessment, [], NO_URL)
    assert set(result.contributing_rule_ids) == {"spf_or_dmarc_fail", "display_name_brand_mismatch"}
    assert result.contributing_semantic_ids == []


# --- Muhtemel Phishing stays Muhtemel Phishing --------------------------

def test_muhtemel_phishing_stays_muhtemel_phishing():
    assessment = _assessment("Muhtemel Phishing")
    result = decide(assessment, [], NO_URL)
    assert result.final_verdict == "Muhtemel Phishing"
    assert result.decision_path == DECISION_PATH_RULE_ENGINE_ONLY
    assert result.analyst_review_required is True


def test_muhtemel_phishing_not_upgraded_to_phishing_by_semantic_findings():
    """Semantic findings only ever move Güvenilir up one band — they
    never adjudicate an already-ambiguous Muhtemel Phishing verdict."""
    assessment = _assessment("Muhtemel Phishing")
    findings = [
        _finding(SemanticFindingType.CREDENTIAL_REQUEST),
        _finding(SemanticFindingType.PAYMENT_REQUEST),
        _finding(SemanticFindingType.THREAT_OR_FEAR),
    ]
    result = decide(assessment, findings, WITH_URL)
    assert result.final_verdict == "Muhtemel Phishing"


# --- Güvenilir + credential_request + URL upgrade ------------------------

def test_guvenilir_upgraded_by_credential_request_with_url():
    assessment = _assessment("Güvenilir")
    findings = [_finding(SemanticFindingType.CREDENTIAL_REQUEST, 5, 20)]
    result = decide(assessment, findings, WITH_URL)
    assert result.rule_verdict == "Güvenilir"
    assert result.final_verdict == "Muhtemel Phishing"
    assert result.decision_path == DECISION_PATH_CREDENTIAL_REQUEST_URL_UPGRADE
    assert result.contributing_semantic_ids == ["credential_request:5-20"]
    assert result.analyst_review_required is True


def test_guvenilir_not_upgraded_by_credential_request_without_url():
    """context.has_url must come from the deterministic parser, not be
    assumed — credential_request alone (no external link) doesn't
    upgrade."""
    assessment = _assessment("Güvenilir")
    findings = [_finding(SemanticFindingType.CREDENTIAL_REQUEST)]
    result = decide(assessment, findings, NO_URL)
    assert result.final_verdict == "Güvenilir"


def test_guvenilir_not_upgraded_by_url_alone_without_credential_request():
    assessment = _assessment("Güvenilir")
    findings = [_finding(SemanticFindingType.URGENCY_OR_PRESSURE)]
    result = decide(assessment, findings, WITH_URL)
    assert result.final_verdict == "Güvenilir"


# --- Güvenilir + payment_request combination upgrade ----------------------

def test_guvenilir_upgraded_by_payment_request_plus_reward_lure():
    assessment = _assessment("Güvenilir")
    findings = [
        _finding(SemanticFindingType.PAYMENT_REQUEST, 0, 10),
        _finding(SemanticFindingType.REWARD_OR_PRIZE_LURE, 20, 30),
    ]
    result = decide(assessment, findings, NO_URL)
    assert result.final_verdict == "Muhtemel Phishing"
    assert result.decision_path == DECISION_PATH_PAYMENT_REQUEST_COMBINATION_UPGRADE
    assert set(result.contributing_semantic_ids) == {
        "payment_request:0-10", "reward_or_prize_lure:20-30",
    }


def test_guvenilir_upgraded_by_payment_request_plus_threat_or_fear():
    assessment = _assessment("Güvenilir")
    findings = [
        _finding(SemanticFindingType.PAYMENT_REQUEST),
        _finding(SemanticFindingType.THREAT_OR_FEAR, 20, 30),
    ]
    result = decide(assessment, findings, NO_URL)
    assert result.final_verdict == "Muhtemel Phishing"


def test_guvenilir_upgraded_by_payment_request_plus_reply_channel_manipulation():
    assessment = _assessment("Güvenilir")
    findings = [
        _finding(SemanticFindingType.PAYMENT_REQUEST),
        _finding(SemanticFindingType.REPLY_CHANNEL_MANIPULATION, 20, 30),
    ]
    result = decide(assessment, findings, NO_URL)
    assert result.final_verdict == "Muhtemel Phishing"


def test_payment_request_alone_does_not_upgrade():
    assessment = _assessment("Güvenilir")
    findings = [_finding(SemanticFindingType.PAYMENT_REQUEST)]
    result = decide(assessment, findings, NO_URL)
    assert result.final_verdict == "Güvenilir"


def test_payment_request_alone_with_urgency_stays_guvenilir():
    """Regression guardrail for the real Papara candidate
    (data/semantic_eval, "Hemen para yatır, ödemelerin aksamasın!") —
    payment_request + urgency_or_pressure is ordinary commercial
    payment-reminder language, not the 419/fake-reward attack shape
    this rule targets. urgency_or_pressure is deliberately EXCLUDED
    from PAYMENT_REQUEST_COMBINATION_TYPES — see module docstring."""
    assessment = _assessment("Güvenilir")
    findings = [
        _finding(SemanticFindingType.PAYMENT_REQUEST, 0, 10),
        _finding(SemanticFindingType.URGENCY_OR_PRESSURE, 20, 30),
    ]
    result = decide(assessment, findings, WITH_URL)
    assert result.final_verdict == "Güvenilir"
    assert result.decision_path == DECISION_PATH_RULE_ENGINE_ONLY


def test_payment_request_alone_with_brand_impersonation_stays_guvenilir():
    """brand_impersonation is also deliberately excluded from the
    combination set — see module docstring."""
    assessment = _assessment("Güvenilir")
    findings = [
        _finding(SemanticFindingType.PAYMENT_REQUEST, 0, 10),
        _finding(SemanticFindingType.BRAND_IMPERSONATION, 20, 30),
    ]
    result = decide(assessment, findings, NO_URL)
    assert result.final_verdict == "Güvenilir"


# --- urgency_or_pressure alone never changes anything ---------------------

def test_urgency_or_pressure_alone_never_upgrades():
    assessment = _assessment("Güvenilir")
    findings = [_finding(SemanticFindingType.URGENCY_OR_PRESSURE)]
    result = decide(assessment, findings, WITH_URL)
    assert result.final_verdict == "Güvenilir"
    assert result.decision_path == DECISION_PATH_RULE_ENGINE_ONLY


def test_no_findings_at_all_stays_guvenilir():
    assessment = _assessment("Güvenilir")
    result = decide(assessment, [], WITH_URL)
    assert result.final_verdict == "Güvenilir"
    assert result.analyst_review_required is False


# --- authority_impersonation is excluded from every rule, permanently ----

def test_authority_impersonation_alone_never_upgrades():
    assessment = _assessment("Güvenilir")
    findings = [_finding(SemanticFindingType.AUTHORITY_IMPERSONATION)]
    result = decide(assessment, findings, WITH_URL)
    assert result.final_verdict == "Güvenilir"


def test_authority_impersonation_does_not_satisfy_payment_request_combination():
    """authority_impersonation must NOT count as a combination partner
    for payment_request — the original plan's "payment_request +
    authority_impersonation" rule was retired specifically because this
    type stayed the most persistent false-positive source across three
    prompt-tuning rounds (see module docstring and PROGRESS.md)."""
    assessment = _assessment("Güvenilir")
    findings = [
        _finding(SemanticFindingType.PAYMENT_REQUEST, 0, 10),
        _finding(SemanticFindingType.AUTHORITY_IMPERSONATION, 20, 30),
    ]
    result = decide(assessment, findings, NO_URL)
    assert result.final_verdict == "Güvenilir"


def test_authority_impersonation_never_appears_in_contributing_semantic_ids():
    """Even when a REAL upgrade happens for other reasons,
    authority_impersonation findings present in the same batch must
    never be listed as contributing — it's excluded from the policy
    entirely, not just from triggering on its own."""
    assessment = _assessment("Güvenilir")
    findings = [
        _finding(SemanticFindingType.CREDENTIAL_REQUEST, 0, 10),
        _finding(SemanticFindingType.AUTHORITY_IMPERSONATION, 20, 30),
    ]
    result = decide(assessment, findings, WITH_URL)
    assert result.final_verdict == "Muhtemel Phishing"
    assert result.contributing_semantic_ids == ["credential_request:0-10"]
    assert not any("authority_impersonation" in sid for sid in result.contributing_semantic_ids)


# --- model_confidence is never added to a rule score ----------------------

def test_model_confidence_never_affects_the_decision():
    """A low-confidence finding upgrades exactly the same way a
    high-confidence one does — confidence is observability-only, never
    read by any rule. See module docstring's rationale against summing
    confidence with rule score."""
    assessment = _assessment("Güvenilir")
    low_conf = [_finding(SemanticFindingType.CREDENTIAL_REQUEST, confidence=0.01)]
    high_conf = [_finding(SemanticFindingType.CREDENTIAL_REQUEST, confidence=0.99)]
    result_low = decide(assessment, low_conf, WITH_URL)
    result_high = decide(assessment, high_conf, WITH_URL)
    assert result_low.final_verdict == result_high.final_verdict == "Muhtemel Phishing"


# --- FinalDecision schema shape -------------------------------------------

def test_final_decision_extra_field_is_rejected():
    from pydantic import ValidationError
    try:
        FinalDecision(
            rule_verdict="Güvenilir", final_verdict="Güvenilir",
            decision_path=DECISION_PATH_RULE_ENGINE_ONLY,
            contributing_rule_ids=[], contributing_semantic_ids=[],
            analyst_review_required=False, extra_field="x",
        )
        raise AssertionError("expected ValidationError")
    except ValidationError:
        pass


def test_context_extra_field_is_rejected():
    from pydantic import ValidationError
    try:
        PhishingDecisionContext(
            has_url=False, url_count=0, url_ids=[],
            parser_credential_request=False, extra_field="x",
        )
        raise AssertionError("expected ValidationError")
    except ValidationError:
        pass


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
