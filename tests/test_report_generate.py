"""Unit tests for src/report/generate.py (PHISHING_ROUTING_PLAN.md
section 10.5, "Qwen çağrı 2"). No real model call anywhere in this file —
generate_report() is exercised entirely with a QwenService whose
load_fn/generate_fn are injected mocks, mirroring
tests/test_semantic_analyze.py's _service_returning/_service_raising
convention."""
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from schemas.decision import FinalDecision  # noqa: E402
from schemas.facts import EmailFacts  # noqa: E402
from schemas.rule_assessment import RuleAssessment, RuleEvidence  # noqa: E402
from src.decision.phishing_policy import DECISION_PATH_RULE_ENGINE_ONLY  # noqa: E402
from src.llm.service import LLMServiceError, QwenService  # noqa: E402
from src.report.generate import ReportGenerationError, generate_report  # noqa: E402

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
    subject="Hesabınızı doğrulayın",
    date="Mon, 1 Jan 2024 00:00:00 +0000",
    body_text="Sayın müşterimiz, lütfen şifrenizi 24 saat içinde doğrulayın.",
    language="tr",
)


def facts(**overrides) -> EmailFacts:
    kwargs = dict(BASE_FACTS_KWARGS)
    kwargs.update(overrides)
    return EmailFacts(**kwargs)


def assessment(rule_verdict="Güvenilir") -> RuleAssessment:
    return RuleAssessment(
        engine_version="v1",
        rule_verdict=rule_verdict,
        score=3.0,
        total=None,
        families=[],
        critical_matches=[],
        evidence=[RuleEvidence(signal="test_signal", description="test", weight=3.0)],
        decision_reasons=["test"],
    )


def decision(final_verdict="Güvenilir", rule_verdict="Güvenilir") -> FinalDecision:
    return FinalDecision(
        rule_verdict=rule_verdict,
        final_verdict=final_verdict,
        decision_path=DECISION_PATH_RULE_ENGINE_ONLY,
        contributing_rule_ids=["test_signal"],
        contributing_semantic_ids=[],
        analyst_review_required=final_verdict != "Güvenilir",
    )


VALID_REPORT_JSON = """{
  "risk_seviyesi": "Güvenilir",
  "sonuc_ve_gerekce": "Bu karar; kimlik doğrulama uyumsuzluğu kategorisinin değerlendirilmesine dayanır.",
  "genel_degerlendirme": "Olası senaryo: yok. Alıcıdan beklenen eylem: yok. Olası zarar: yok.",
  "teknik_bulgular": [{"baslik": "test", "aciklama": "test aciklama"}],
  "phishing_gostergeleri": [],
  "onerilen_aksiyon": "Ek bir aksiyon gerekmiyor."
}"""


def _mock_generate_result(text: str):
    m = MagicMock()
    m.text = text
    return m


def _service_returning(text: str) -> QwenService:
    return QwenService(
        load_fn=lambda path: (MagicMock(), MagicMock()),
        apply_chat_template_fn=lambda processor, config, messages, num_images: "PROMPT",
        generate_fn=lambda *a, **k: _mock_generate_result(text),
    )


def _service_raising(exc: Exception) -> QwenService:
    def _raise(*a, **k):
        raise exc

    return QwenService(
        load_fn=lambda path: (MagicMock(), MagicMock()),
        apply_chat_template_fn=lambda processor, config, messages, num_images: "PROMPT",
        generate_fn=_raise,
    )


# --- success path -----------------------------------------------------

def test_generate_report_success_returns_matching_report():
    service = _service_returning(VALID_REPORT_JSON)
    report = generate_report(facts(), assessment(), decision(), [], service=service)
    assert report.risk_seviyesi == "Güvenilir"
    assert report.onerilen_aksiyon == "Ek bir aksiyon gerekmiyor."


def test_generate_report_extracts_json_from_code_fence():
    fenced = f"```json\n{VALID_REPORT_JSON}\n```"
    service = _service_returning(fenced)
    report = generate_report(facts(), assessment(), decision(), [], service=service)
    assert report.risk_seviyesi == "Güvenilir"


# --- model_call_failed --------------------------------------------------

def test_generate_report_wraps_llm_service_error_as_model_call_failed():
    service = _service_raising(LLMServiceError("GPU timeout"))
    try:
        generate_report(facts(), assessment(), decision(), [], service=service)
        raise AssertionError("expected ReportGenerationError")
    except ReportGenerationError as e:
        assert e.code == "model_call_failed"
        assert e.__cause__ is not None


# --- invalid_json --------------------------------------------------------

def test_generate_report_raises_invalid_json_on_unparseable_output():
    service = _service_returning("this is not json at all")
    try:
        generate_report(facts(), assessment(), decision(), [], service=service)
        raise AssertionError("expected ReportGenerationError")
    except ReportGenerationError as e:
        assert e.code == "invalid_json"


def test_generate_report_raises_invalid_json_when_no_object_present():
    """No '{...}' substring anywhere in the output — extraction must
    fail cleanly, not raise or guess."""
    service = _service_returning("I refuse to produce a report for this email.")
    try:
        generate_report(facts(), assessment(), decision(), [], service=service)
        raise AssertionError("expected ReportGenerationError")
    except ReportGenerationError as e:
        assert e.code == "invalid_json"


# --- schema_invalid --------------------------------------------------------

def test_generate_report_raises_schema_invalid_on_missing_field():
    incomplete = '{"risk_seviyesi": "Güvenilir"}'
    service = _service_returning(incomplete)
    try:
        generate_report(facts(), assessment(), decision(), [], service=service)
        raise AssertionError("expected ReportGenerationError")
    except ReportGenerationError as e:
        assert e.code == "schema_invalid"


def test_generate_report_raises_schema_invalid_on_extra_field():
    import json

    payload = json.loads(VALID_REPORT_JSON)
    payload["unexpected_field"] = "x"
    service = _service_returning(json.dumps(payload))
    try:
        generate_report(facts(), assessment(), decision(), [], service=service)
        raise AssertionError("expected ReportGenerationError")
    except ReportGenerationError as e:
        assert e.code == "schema_invalid"


# --- verdict_mismatch — the "LLM does not classify" enforcement point -----

def test_generate_report_raises_verdict_mismatch_when_model_echoes_wrong_verdict():
    """The model wrote risk_seviyesi="Güvenilir" but decision.final_verdict
    is "Phishing" — the model's own verdict must never be trusted over
    the authoritative FinalDecision."""
    service = _service_returning(VALID_REPORT_JSON)  # risk_seviyesi: Güvenilir
    mismatched_decision = decision(final_verdict="Phishing", rule_verdict="Phishing")
    try:
        generate_report(facts(), assessment("Phishing"), mismatched_decision, [],
                         service=service)
        raise AssertionError("expected ReportGenerationError")
    except ReportGenerationError as e:
        assert e.code == "verdict_mismatch"


def test_generate_report_never_retries_on_any_failure():
    """No retry, no repair — a single generate() call, period. Verified
    by counting calls on a service that always fails."""
    call_count = {"n": 0}

    def counting_raise(*a, **k):
        call_count["n"] += 1
        raise LLMServiceError("boom")

    service = QwenService(
        load_fn=lambda path: (MagicMock(), MagicMock()),
        apply_chat_template_fn=lambda processor, config, messages, num_images: "PROMPT",
        generate_fn=counting_raise,
    )
    try:
        generate_report(facts(), assessment(), decision(), [], service=service)
    except ReportGenerationError:
        pass
    assert call_count["n"] == 1


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
