"""Unit tests for src/workflows/phishing.py (PHISHING_ROUTING_PLAN.md
"hybrid workflow wiring" task, extended by the "final rapor prompt ve
şemasını güncellemek" task). Covers both mode="fast" (unchanged) and
mode="hybrid" (parse -> rule engine -> semantic extraction -> decision
policy -> Qwen report generation, falling back to the mechanical report
on failure). analyze_semantic() and generate_report() are mocked
throughout — these are workflow-wiring tests, not a real Qwen3.5-9B
smoke test (that already exists separately, see src/semantic/
smoke_test.py and scripts/evaluate_semantic_extractor.py)."""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.workflows.phishing as wf  # noqa: E402
from schemas.semantic import SemanticFindingType, ValidatedSemanticFinding  # noqa: E402
from src.report.generate import ReportGenerationError  # noqa: E402
from src.semantic.analyze import SemanticExtractionError  # noqa: E402
from src.semantic.validate import (  # noqa: E402
    RejectionReason,
    ValidatedFinding,
    ValidationResult,
)
from src.workflows.phishing import PhishingAnalysisResult, analyze_phishing  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_EML = PROJECT_ROOT / "data" / "raw" / "gmail" / "eml" / "inbox-1804.eml"
# rule_verdict == "Phishing" under v1 — used for the hybrid "skipped"
# path. Picked by scanning data/raw for a real sample that already
# scores Phishing, rather than hand-crafting facts (this module always
# starts from a real .eml on disk, never a constructed EmailFacts).
PHISHING_SAMPLE_EML = PROJECT_ROOT / "data" / "raw" / "gmail" / "eml" / "inbox-9294.eml"


def _finding(type_, evidence="x" * 10, confidence=0.9) -> ValidatedSemanticFinding:
    return ValidatedSemanticFinding(
        type=type_,
        evidence=evidence,
        start=0,
        end=len(evidence),
        model_confidence=confidence,
        explanation="x",
    )


def _qwen_report(verdict: str):
    """A minimal, schema-valid Report generate_report() could plausibly
    return for the given verdict — used to mock the Qwen success path
    without a real model call."""
    from schemas.report import Report

    return Report(
        risk_seviyesi=verdict,
        sonuc_ve_gerekce="Bu karar; kimlik ve marka taklidi kategorisinin değerlendirilmesine dayanır.",
        genel_degerlendirme="Olası senaryo: test. Alıcıdan beklenen eylem: test. Olası zarar: test.",
        teknik_bulgular=[],
        phishing_gostergeleri=[],
        onerilen_aksiyon="Test.",
    )


def _mock_qwen_success():
    """generate_report() replacement that echoes decision.final_verdict —
    used by tests that only care about final_decision/report_source
    wiring, not the report TEXT (that's src/report/prompts.py's and
    src/report/generate.py's own test files' job)."""
    def fake_generate_report(facts, rule_assessment, decision, accepted_findings):
        return _qwen_report(decision.final_verdict)

    return MagicMock(side_effect=fake_generate_report)


def test_fast_mode_returns_result_with_matching_verdict():
    result = analyze_phishing(SAMPLE_EML, mode="fast")

    assert isinstance(result, PhishingAnalysisResult)
    assert result.mode == "fast"
    assert result.rule_assessment.engine_version == "v1"
    # The report model must echo the rule engine's decision — same
    # invariant demo.py/web.py enforce for the LLM path.
    assert result.report.risk_seviyesi == result.rule_assessment.rule_verdict
    # fast mode never touches the semantic/decision layer.
    assert result.final_decision is None
    assert result.semantic_status is None
    assert result.accepted_findings == []
    assert result.rejected_findings == []
    # fast mode never calls the LLM report generator either.
    assert result.report_source == "mechanical"
    assert result.llm_report_status == "not_requested"
    assert result.llm_report_error_code is None


def test_fast_mode_defaults_when_mode_omitted():
    result = analyze_phishing(SAMPLE_EML)
    assert result.mode == "fast"


def test_invalid_mode_raises_value_error():
    try:
        analyze_phishing(SAMPLE_EML, mode="slow")  # type: ignore[arg-type]
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


# --- hybrid mode: semantic skipped when rule_verdict is already Phishing ---

def test_hybrid_mode_skips_semantic_call_when_already_phishing():
    """decide() cannot upgrade a Phishing verdict any further, so
    analyze_semantic() (tens of seconds, loads a 9B model) must never be
    called at all — this is the cheap-tier-first principle CLAUDE.md
    documents for every layer of this system."""
    mock_analyze = MagicMock()
    with patch.object(wf, "analyze_semantic", mock_analyze), \
         patch.object(wf, "generate_report", _mock_qwen_success()):
        result = analyze_phishing(PHISHING_SAMPLE_EML, mode="hybrid")

    assert result.rule_assessment.rule_verdict == "Phishing"
    assert not mock_analyze.called
    assert result.semantic_status == "skipped"
    assert result.semantic_skip_reason == "rule_verdict_already_phishing"
    assert result.final_decision.final_verdict == "Phishing"
    assert result.final_decision.rule_verdict == "Phishing"
    assert result.report.risk_seviyesi == "Phishing"
    assert result.accepted_findings == []
    assert result.rejected_findings == []
    assert result.report_source == "qwen"
    assert result.llm_report_status == "completed"
    assert result.llm_report_error_code is None


# --- hybrid mode: semantic model failure never loses the deterministic verdict ---

def test_hybrid_mode_semantic_extraction_error_invalid_json_falls_back_to_rule_verdict():
    """analyze_semantic() raises SemanticExtractionError(code="invalid_json")
    on a malformed model response. The deterministic rule_verdict must
    survive regardless — the whole point of this exception type is that
    the workflow can catch it deliberately and fall back, rather than a
    bare `except Exception` (or, in an earlier version of this module, a
    bare `except SystemExit` — the wrong exception type entirely, see
    src/semantic/analyze.py's module docstring for why it was replaced)."""
    with patch.object(
        wf, "analyze_semantic",
        side_effect=SemanticExtractionError(code="invalid_json", message="bad json"),
    ), patch.object(wf, "generate_report", _mock_qwen_success()):
        result = analyze_phishing(SAMPLE_EML, mode="hybrid")

    assert result.semantic_status == "failed"
    assert result.semantic_skip_reason is None
    assert result.final_decision.final_verdict == result.rule_assessment.rule_verdict
    assert result.final_decision.decision_path == "rule_engine_only"
    assert result.report.risk_seviyesi == result.rule_assessment.rule_verdict
    assert result.accepted_findings == []
    assert result.rejected_findings == []


def test_hybrid_mode_semantic_extraction_error_model_call_failed_also_falls_back():
    """code="model_call_failed" (the underlying QwenService/LLMServiceError
    failure — e.g. a GPU/Metal timeout, PROGRESS.md) must still fall back
    to the mechanical report and keep the deterministic rule_verdict —
    but, unlike code="invalid_json", it must do so WITHOUT attempting the
    second (report) Qwen call at all. See the two dedicated tests below
    for that distinction; this test only covers the shared fallback
    behavior (final_decision/report still correct)."""
    mock_report = MagicMock()
    with patch.object(
        wf, "analyze_semantic",
        side_effect=SemanticExtractionError(code="model_call_failed", message="GPU Timeout"),
    ), patch.object(wf, "generate_report", mock_report):
        result = analyze_phishing(SAMPLE_EML, mode="hybrid")

    assert result.semantic_status == "failed"
    assert result.final_decision.final_verdict == result.rule_assessment.rule_verdict
    assert result.report.risk_seviyesi == result.rule_assessment.rule_verdict
    assert result.report_source == "mechanical"
    assert result.llm_report_status == "failed_fallback"
    assert result.llm_report_error_code == "model_call_failed"


# --- semantic_error_code decides whether the second Qwen call is attempted ---

def test_hybrid_mode_model_call_failed_skips_second_qwen_call_entirely():
    """The FIRST Qwen call (semantic extraction) failed with
    code="model_call_failed" — the underlying QwenService itself is
    broken (e.g. a GPU/Metal timeout), and per src/llm/service.py's "tek
    model, iki çağrı" design the SECOND call (report generation) would
    reuse the exact same lazily-loaded model instance within this same
    request. Retrying it would just reproduce the same infrastructure
    failure a few seconds later, so generate_report() must never even be
    called — this is the fast-fail this task added, not merely "falls
    back eventually"."""
    mock_report = MagicMock()
    with patch.object(
        wf, "analyze_semantic",
        side_effect=SemanticExtractionError(code="model_call_failed", message="GPU Timeout"),
    ), patch.object(wf, "generate_report", mock_report):
        result = analyze_phishing(SAMPLE_EML, mode="hybrid")

    assert not mock_report.called
    assert result.report_source == "mechanical"
    assert result.llm_report_status == "failed_fallback"
    assert result.llm_report_error_code == "model_call_failed"


def test_hybrid_mode_invalid_json_still_attempts_second_qwen_call():
    """The FIRST Qwen call failed with code="invalid_json" — the model
    itself responded (just with unparseable output), so the underlying
    infrastructure is known-good. Unlike "model_call_failed", this must
    NOT short-circuit: generate_report() (the second, independent Qwen
    call) is still attempted normally."""
    mock_report = _mock_qwen_success()
    with patch.object(
        wf, "analyze_semantic",
        side_effect=SemanticExtractionError(code="invalid_json", message="bad json"),
    ), patch.object(wf, "generate_report", mock_report):
        result = analyze_phishing(SAMPLE_EML, mode="hybrid")

    assert mock_report.called
    assert result.report_source == "qwen"
    assert result.llm_report_status == "completed"


def test_hybrid_mode_unexpected_exception_is_not_swallowed():
    """An exception analyze_semantic() does NOT normalize to
    SemanticExtractionError must propagate, not be silently folded into
    semantic_status="failed" — that would hide a real bug in the wiring
    itself as if it were an ordinary degraded-service case."""
    with patch.object(wf, "analyze_semantic", side_effect=RuntimeError("unexpected bug")):
        try:
            analyze_phishing(SAMPLE_EML, mode="hybrid")
            raise AssertionError("expected RuntimeError to propagate")
        except RuntimeError:
            pass


# --- hybrid mode: completed semantic run feeds the decision policy ---

def test_hybrid_mode_completed_credential_request_upgrades_with_url():
    finding = _finding(SemanticFindingType.CREDENTIAL_REQUEST)
    validation_result = ValidationResult(accepted=[finding], rejected=[])

    with patch.object(wf, "analyze_semantic", return_value=validation_result), \
         patch.object(wf, "generate_report", _mock_qwen_success()):
        result = analyze_phishing(SAMPLE_EML, mode="hybrid")

    assert result.semantic_status == "completed"
    assert result.rule_assessment.rule_verdict == "Güvenilir"
    # This sample email is expected to carry at least one URL — the
    # upgrade rule requires credential_request + context.has_url.
    assert result.final_decision.final_verdict == "Muhtemel Phishing"
    assert result.final_decision.decision_path == "credential_request_plus_url_upgrade"
    assert result.final_decision.analyst_review_required is True
    # The report must reflect the UPGRADED verdict, not the stale
    # rule_verdict — this is the invariant the user required.
    assert result.report.risk_seviyesi == result.final_decision.final_verdict
    assert result.report.risk_seviyesi == "Muhtemel Phishing"
    assert result.accepted_findings == [finding]
    assert result.report_source == "qwen"
    assert result.llm_report_status == "completed"


def test_hybrid_mode_completed_no_upgrade_condition_keeps_rule_verdict():
    """A single urgency_or_pressure finding must NOT upgrade a Güvenilir
    verdict — see src/decision/phishing_policy.py's module docstring for
    why (legitimate marketing mail routinely uses this language)."""
    finding = _finding(SemanticFindingType.URGENCY_OR_PRESSURE)
    validation_result = ValidationResult(accepted=[finding], rejected=[])

    with patch.object(wf, "analyze_semantic", return_value=validation_result), \
         patch.object(wf, "generate_report", _mock_qwen_success()):
        result = analyze_phishing(SAMPLE_EML, mode="hybrid")

    assert result.final_decision.final_verdict == "Güvenilir"
    assert result.final_decision.decision_path == "rule_engine_only"
    assert result.report.risk_seviyesi == "Güvenilir"


def test_hybrid_mode_rejected_findings_are_carried_but_not_used_by_policy():
    """Rejected candidates must be visible for audit but must never
    reach decide() — matches src/decision/phishing_policy.py's
    "semantic_findings must already be validator-accepted" contract."""
    rejected = ValidatedFinding(
        finding={"type": "credential_request", "evidence": "not in body"},
        accepted=False,
        rejection_reason=RejectionReason.NOT_FOUND_IN_BODY,
    )
    validation_result = ValidationResult(accepted=[], rejected=[rejected])

    with patch.object(wf, "analyze_semantic", return_value=validation_result), \
         patch.object(wf, "generate_report", _mock_qwen_success()):
        result = analyze_phishing(SAMPLE_EML, mode="hybrid")

    assert result.semantic_status == "completed"
    assert result.accepted_findings == []
    assert result.rejected_findings == [rejected]
    # No accepted findings -> no upgrade condition can fire.
    assert result.final_decision.decision_path == "rule_engine_only"
    assert result.final_decision.final_verdict == result.rule_assessment.rule_verdict


def test_hybrid_mode_parses_email_exactly_once():
    """facts must be reused for both the rule engine and the semantic
    extractor — re-parsing would be wasted work and a potential source
    of drift between what the rule engine and the model saw."""
    finding = _finding(SemanticFindingType.URGENCY_OR_PRESSURE)
    validation_result = ValidationResult(accepted=[finding], rejected=[])

    real_parse_eml = wf.parse_eml
    call_count = {"n": 0}

    def counting_parse_eml(path):
        call_count["n"] += 1
        return real_parse_eml(path)

    with patch.object(wf, "generate_report", _mock_qwen_success()), \
         patch.object(wf, "parse_eml", side_effect=counting_parse_eml), \
         patch.object(wf, "analyze_semantic", return_value=validation_result):
        analyze_phishing(SAMPLE_EML, mode="hybrid")

    assert call_count["n"] == 1


# --- hybrid mode: report generation, Qwen success vs. fallback ---

def test_hybrid_mode_qwen_report_success_sets_source_and_status():
    finding = _finding(SemanticFindingType.URGENCY_OR_PRESSURE)
    validation_result = ValidationResult(accepted=[finding], rejected=[])

    with patch.object(wf, "analyze_semantic", return_value=validation_result), \
         patch.object(wf, "generate_report", _mock_qwen_success()):
        result = analyze_phishing(SAMPLE_EML, mode="hybrid")

    assert result.report_source == "qwen"
    assert result.llm_report_status == "completed"
    assert result.llm_report_error_code is None
    assert result.report.risk_seviyesi == result.final_decision.final_verdict


def test_hybrid_mode_report_generation_error_falls_back_to_mechanical_report():
    """generate_report() raising ReportGenerationError must never prevent
    a report from being produced — the deterministic mechanical report
    (the same builder fast mode always uses) substitutes, with no retry
    and no attempt to repair the model's output (CLAUDE.md
    "Yapılmayacaklar"). This is the fallback contract the user required:
    report_source/llm_report_status record what happened instead of the
    caller having to infer it from report content."""
    finding = _finding(SemanticFindingType.URGENCY_OR_PRESSURE)
    validation_result = ValidationResult(accepted=[finding], rejected=[])

    with patch.object(wf, "analyze_semantic", return_value=validation_result), \
         patch.object(
             wf, "generate_report",
             side_effect=ReportGenerationError(code="invalid_json", message="bad json"),
         ):
        result = analyze_phishing(SAMPLE_EML, mode="hybrid")

    assert result.report_source == "mechanical"
    assert result.llm_report_status == "failed_fallback"
    assert result.llm_report_error_code == "invalid_json"
    # The deterministic decision must still be reflected correctly even
    # though the LLM report generator failed — the fallback report is
    # built from the SAME final_decision, not the stale rule_verdict.
    assert result.report.risk_seviyesi == result.final_decision.final_verdict


def test_hybrid_mode_report_generation_error_code_is_preserved_for_each_failure_mode():
    """Every ReportGenerationError code (model_call_failed, invalid_json,
    schema_invalid, verdict_mismatch) must surface verbatim on
    llm_report_error_code, not be collapsed to a generic flag — an
    operator inspecting failures needs to tell them apart."""
    for code in ("model_call_failed", "invalid_json", "schema_invalid", "verdict_mismatch"):
        with patch.object(wf, "analyze_semantic", return_value=ValidationResult(accepted=[], rejected=[])), \
             patch.object(
                 wf, "generate_report",
                 side_effect=ReportGenerationError(code=code, message="x"),
             ):
            result = analyze_phishing(SAMPLE_EML, mode="hybrid")

        assert result.llm_report_status == "failed_fallback"
        assert result.llm_report_error_code == code
        assert result.report_source == "mechanical"


def test_hybrid_mode_semantic_skipped_still_attempts_qwen_report():
    """Even when semantic extraction is skipped (rule_verdict already
    Phishing), generate_report() must still be attempted — the skip only
    applies to the semantic extractor, not to report generation."""
    mock_report = _mock_qwen_success()
    with patch.object(wf, "analyze_semantic", MagicMock()), \
         patch.object(wf, "generate_report", mock_report):
        result = analyze_phishing(PHISHING_SAMPLE_EML, mode="hybrid")

    assert mock_report.called
    assert result.report_source == "qwen"
    assert result.llm_report_status == "completed"


if __name__ == "__main__":
    import traceback

    if not SAMPLE_EML.is_file():
        print(f"SKIP: sample data not present at {SAMPLE_EML}")
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
