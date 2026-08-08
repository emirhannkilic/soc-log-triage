"""Unit tests for src/workflows/phishing.py (PHISHING_ROUTING_PLAN.md
"hybrid workflow wiring" task). Covers both mode="fast" (unchanged) and
mode="hybrid" (parse -> rule engine -> semantic extraction -> decision
policy -> mechanical report). analyze_semantic() is mocked throughout —
these are workflow-wiring tests, not a real Qwen3.5-9B smoke test (that
already exists separately, see src/semantic/smoke_test.py and
scripts/evaluate_semantic_extractor.py)."""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.workflows.phishing as wf  # noqa: E402
from schemas.semantic import SemanticFindingType, ValidatedSemanticFinding  # noqa: E402
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
    with patch.object(wf, "analyze_semantic", mock_analyze):
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
    ):
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
    failure — e.g. a GPU/Metal timeout, PROGRESS.md) must be caught the
    same way as code="invalid_json" — the workflow catches
    SemanticExtractionError itself, not a specific code."""
    with patch.object(
        wf, "analyze_semantic",
        side_effect=SemanticExtractionError(code="model_call_failed", message="GPU Timeout"),
    ):
        result = analyze_phishing(SAMPLE_EML, mode="hybrid")

    assert result.semantic_status == "failed"
    assert result.final_decision.final_verdict == result.rule_assessment.rule_verdict


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

    with patch.object(wf, "analyze_semantic", return_value=validation_result):
        result = analyze_phishing(SAMPLE_EML, mode="hybrid")

    assert result.semantic_status == "completed"
    assert result.rule_assessment.rule_verdict == "Güvenilir"
    # This sample email is expected to carry at least one URL — the
    # upgrade rule requires credential_request + context.has_url.
    assert result.final_decision.final_verdict == "Muhtemel Phishing"
    assert result.final_decision.decision_path == "credential_request_plus_url_upgrade"
    assert result.final_decision.analyst_review_required is True
    # The mechanical report must reflect the UPGRADED verdict, not the
    # stale rule_verdict — this is the invariant the user required.
    assert result.report.risk_seviyesi == result.final_decision.final_verdict
    assert result.report.risk_seviyesi == "Muhtemel Phishing"
    # The upgrade's rationale must be visible in the report text, not
    # just carried silently on final_decision.
    assert "kimlik bilgisi talebi" in result.report.genel_degerlendirme
    assert result.accepted_findings == [finding]


def test_hybrid_mode_completed_no_upgrade_condition_keeps_rule_verdict():
    """A single urgency_or_pressure finding must NOT upgrade a Güvenilir
    verdict — see src/decision/phishing_policy.py's module docstring for
    why (legitimate marketing mail routinely uses this language)."""
    finding = _finding(SemanticFindingType.URGENCY_OR_PRESSURE)
    validation_result = ValidationResult(accepted=[finding], rejected=[])

    with patch.object(wf, "analyze_semantic", return_value=validation_result):
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

    with patch.object(wf, "analyze_semantic", return_value=validation_result):
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

    with patch.object(wf, "parse_eml", side_effect=counting_parse_eml), \
         patch.object(wf, "analyze_semantic", return_value=validation_result):
        analyze_phishing(SAMPLE_EML, mode="hybrid")

    assert call_count["n"] == 1


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
