"""Unit tests for src/web.py's /analyze endpoint (PROGRESS.md "sıradaki
teknik iş" — CLI/web'i analyze_phishing()'e bağlama). No real Qwen call
anywhere in this file — analyze_phishing() is patched to return
pre-built PhishingAnalysisResult objects, mirroring tests/test_
workflows_phishing.py's own mocking convention."""
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

import src.web as web  # noqa: E402
from schemas.decision import FinalDecision  # noqa: E402
from schemas.rule_assessment import RuleAssessment, RuleEvidence  # noqa: E402
from schemas.semantic import SemanticFindingType, ValidatedSemanticFinding  # noqa: E402
from src.decision.phishing_policy import DECISION_PATH_RULE_ENGINE_ONLY  # noqa: E402
from src.report.mechanical import build_report  # noqa: E402
from src.semantic.validate import RejectionReason, ValidatedFinding  # noqa: E402
from src.workflows.phishing import PhishingAnalysisResult  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_EML = PROJECT_ROOT / "tests" / "fixtures" / "hybrid_credential_upgrade.eml"

client = TestClient(web.app)


def _rule_assessment(verdict="Güvenilir", evidence=None) -> RuleAssessment:
    return RuleAssessment(
        engine_version="v1",
        rule_verdict=verdict,
        score=3.0 if verdict != "Güvenilir" else 0.0,
        total=None,
        families=[],
        critical_matches=[],
        evidence=evidence if evidence is not None else [],
        decision_reasons=["test"],
    )


def _fast_result() -> PhishingAnalysisResult:
    from src.parser.parse import parse_eml

    facts = parse_eml(FIXTURE_EML)
    ra = _rule_assessment("Güvenilir")
    return PhishingAnalysisResult(
        mode="fast",
        facts=facts,
        rule_assessment=ra,
        report=build_report(ra),
    )


def _hybrid_result(*, final_verdict="Muhtemel Phishing", narrative_status="completed",
                    narrative_error_code=None, report_source="mechanical_with_qwen_narrative"
                    ) -> PhishingAnalysisResult:
    from src.parser.parse import parse_eml

    facts = parse_eml(FIXTURE_EML)
    evidence = [RuleEvidence(signal="spf_or_dmarc_fail", description="SPF/DMARC fail", weight=3.0)]
    ra = _rule_assessment("Güvenilir", evidence=evidence)
    fd = FinalDecision(
        rule_verdict="Güvenilir",
        final_verdict=final_verdict,
        decision_path=DECISION_PATH_RULE_ENGINE_ONLY,
        contributing_rule_ids=["spf_or_dmarc_fail"],
        contributing_semantic_ids=[],
        analyst_review_required=final_verdict != "Güvenilir",
    )
    accepted = [
        ValidatedSemanticFinding(
            type=SemanticFindingType.CREDENTIAL_REQUEST,
            evidence="şifrenizi girin",
            start=0,
            end=10,
            model_confidence=0.9,
            explanation="doğrudan kimlik bilgisi talebi",
        )
    ]
    rejected = [
        ValidatedFinding(
            finding={"type": "urgency_or_pressure", "evidence": "hemen tıklayın"},
            accepted=False,
            rejection_reason=RejectionReason.NOT_FOUND_IN_BODY,
        )
    ]
    return PhishingAnalysisResult(
        mode="hybrid",
        facts=facts,
        rule_assessment=ra,
        report=build_report(ra, decision=fd),
        final_decision=fd,
        semantic_status="completed",
        accepted_findings=accepted,
        rejected_findings=rejected,
        report_source=report_source,
        narrative_status=narrative_status,
        narrative_error_code=narrative_error_code,
    )


def test_fast_mode_returns_expected_shape():
    with patch.object(web, "analyze_phishing", return_value=_fast_result()) as mock_analyze:
        with open(FIXTURE_EML, "rb") as f:
            resp = client.post("/analyze", data={"mode": "fast"},
                               files={"file": ("mail.eml", f, "message/rfc822")})

    assert resp.status_code == 200
    body = resp.json()
    assert body["accepted"] is True
    assert body["mode"] == "fast"
    assert body["verdict"] == "Güvenilir"
    # fast mode never touches the hybrid-only fields.
    assert body["semantic_status"] is None
    assert body["final_verdict"] is None
    assert body["decision_path"] is None
    assert body["narrative_status"] == "not_requested"
    assert "report_html" in body and body["report_html"]
    mock_analyze.assert_called_once()
    assert mock_analyze.call_args.kwargs.get("mode") == "fast"


def test_hybrid_mode_exposes_rule_verdict_and_final_verdict_separately():
    """The rule engine's raw verdict and the (possibly upgraded) final
    verdict must be two distinct fields — collapsing them would hide
    exactly the upgrade this project's decision policy exists to make
    visible."""
    with patch.object(web, "analyze_phishing", return_value=_hybrid_result()):
        with open(FIXTURE_EML, "rb") as f:
            resp = client.post("/analyze", data={"mode": "hybrid"},
                               files={"file": ("mail.eml", f, "message/rfc822")})

    assert resp.status_code == 200
    body = resp.json()
    assert body["verdict"] == "Güvenilir"
    assert body["final_verdict"] == "Muhtemel Phishing"
    assert body["decision_path"] == DECISION_PATH_RULE_ENGINE_ONLY
    assert body["analyst_review_required"] is True


def test_hybrid_mode_exposes_accepted_and_rejected_findings_separately():
    with patch.object(web, "analyze_phishing", return_value=_hybrid_result()):
        with open(FIXTURE_EML, "rb") as f:
            resp = client.post("/analyze", data={"mode": "hybrid"},
                               files={"file": ("mail.eml", f, "message/rfc822")})

    body = resp.json()
    findings = body["semantic_findings"]
    assert len(findings["accepted"]) == 1
    assert findings["accepted"][0]["type"] == "credential_request"
    assert findings["accepted"][0]["evidence"] == "şifrenizi girin"
    assert len(findings["rejected"]) == 1
    assert findings["rejected"][0]["rejection_reason"] == "not_found_in_body"


def test_hybrid_mode_narrative_completed_reports_source_and_status():
    with patch.object(web, "analyze_phishing", return_value=_hybrid_result(
        narrative_status="completed", report_source="mechanical_with_qwen_narrative",
    )):
        with open(FIXTURE_EML, "rb") as f:
            resp = client.post("/analyze", data={"mode": "hybrid"},
                               files={"file": ("mail.eml", f, "message/rfc822")})

    body = resp.json()
    assert body["report_source"] == "mechanical_with_qwen_narrative"
    assert body["narrative_status"] == "completed"
    assert body["narrative_error_code"] is None


def test_hybrid_mode_narrative_failed_fallback_reports_error_code():
    with patch.object(web, "analyze_phishing", return_value=_hybrid_result(
        narrative_status="failed_fallback", narrative_error_code="invalid_json",
        report_source="mechanical",
    )):
        with open(FIXTURE_EML, "rb") as f:
            resp = client.post("/analyze", data={"mode": "hybrid"},
                               files={"file": ("mail.eml", f, "message/rfc822")})

    body = resp.json()
    assert body["report_source"] == "mechanical"
    assert body["narrative_status"] == "failed_fallback"
    assert body["narrative_error_code"] == "invalid_json"


def test_unknown_mode_rejected():
    with open(FIXTURE_EML, "rb") as f:
        resp = client.post("/analyze", data={"mode": "llm"},
                           files={"file": ("mail.eml", f, "message/rfc822")})
    assert resp.status_code == 400


def test_analyze_phishing_exception_returns_400_not_500():
    with patch.object(web, "analyze_phishing", side_effect=RuntimeError("boom")):
        with open(FIXTURE_EML, "rb") as f:
            resp = client.post("/analyze", data={"mode": "fast"},
                               files={"file": ("mail.eml", f, "message/rfc822")})

    assert resp.status_code == 400
    assert "error" in resp.json()


def test_no_llm_route_exists_anymore():
    """Regression guard: the old Seneca+teacher 'llm' mode must not be
    reachable through this endpoint anymore — see src/web.py's module
    docstring for why. Only 'fast' and 'hybrid' are valid mode values."""
    assert not hasattr(web, "_report_via_llm")
    assert not hasattr(web, "_load_model")


if __name__ == "__main__":
    import traceback

    if not FIXTURE_EML.is_file():
        print(f"SKIP: fixture not present at {FIXTURE_EML}")
        sys.exit(0)

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
