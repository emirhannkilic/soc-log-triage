"""Unit tests for src/demo.py's --hybrid flag (PROGRESS.md "sıradaki
teknik iş" — CLI/web'i analyze_phishing()'e bağlama). No real Qwen call
anywhere in this file — analyze_phishing() is patched, and
_print_hybrid_summary() is exercised directly against a hand-built
PhishingAnalysisResult, mirroring tests/test_workflows_phishing.py's
own mocking convention. main() itself (argparse + sys.exit) is tested
via subprocess, matching the way this CLI is actually invoked."""
import io
import subprocess
import sys
import tempfile
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.demo as demo  # noqa: E402
from schemas.decision import FinalDecision  # noqa: E402
from schemas.rule_assessment import RuleAssessment, RuleEvidence  # noqa: E402
from schemas.semantic import SemanticFindingType, ValidatedSemanticFinding  # noqa: E402
from src.decision.phishing_policy import DECISION_PATH_RULE_ENGINE_ONLY  # noqa: E402
from src.parser.parse import parse_eml  # noqa: E402
from src.report.mechanical import build_report  # noqa: E402
from src.workflows.phishing import PhishingAnalysisResult  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_EML = PROJECT_ROOT / "tests" / "fixtures" / "hybrid_credential_upgrade.eml"


def _hybrid_result() -> PhishingAnalysisResult:
    facts = parse_eml(FIXTURE_EML)
    evidence = [RuleEvidence(signal="spf_or_dmarc_fail", description="SPF/DMARC fail", weight=3.0)]
    ra = RuleAssessment(
        engine_version="v1", rule_verdict="Güvenilir", score=0.0, total=None,
        families=[], critical_matches=[], evidence=evidence, decision_reasons=["test"],
    )
    fd = FinalDecision(
        rule_verdict="Güvenilir", final_verdict="Muhtemel Phishing",
        decision_path=DECISION_PATH_RULE_ENGINE_ONLY,
        contributing_rule_ids=["spf_or_dmarc_fail"], contributing_semantic_ids=[],
        analyst_review_required=True,
    )
    accepted = [
        ValidatedSemanticFinding(
            type=SemanticFindingType.CREDENTIAL_REQUEST, evidence="şifrenizi girin",
            start=0, end=10, model_confidence=0.9, explanation="doğrudan kimlik bilgisi talebi",
        )
    ]
    return PhishingAnalysisResult(
        mode="hybrid", facts=facts, rule_assessment=ra,
        report=build_report(ra, decision=fd), final_decision=fd,
        semantic_status="completed", accepted_findings=accepted,
        report_source="mechanical_with_qwen_narrative", narrative_status="completed",
    )


def _captured_stderr(fn) -> str:
    """Runs fn() with sys.stderr redirected to an in-memory buffer and
    returns everything it wrote — this project's test runner has no
    pytest capsys fixture, so this is the plain-stdlib equivalent."""
    buf = io.StringIO()
    with redirect_stderr(buf):
        fn()
    return buf.getvalue()


def test_print_hybrid_summary_shows_rule_and_final_verdict():
    out = _captured_stderr(lambda: demo._print_hybrid_summary(_hybrid_result()))
    assert "[Rule verdict] Güvenilir" in out
    assert "[Final verdict] Muhtemel Phishing" in out
    assert DECISION_PATH_RULE_ENGINE_ONLY in out


def test_print_hybrid_summary_shows_accepted_findings():
    out = _captured_stderr(lambda: demo._print_hybrid_summary(_hybrid_result()))
    assert "credential_request" in out
    assert "şifrenizi girin" in out


def test_print_hybrid_summary_shows_narrative_status():
    out = _captured_stderr(lambda: demo._print_hybrid_summary(_hybrid_result()))
    assert "report_source=mechanical_with_qwen_narrative" in out
    assert "narrative_status=completed" in out


def test_main_calls_analyze_phishing_with_hybrid_mode():
    with tempfile.TemporaryDirectory() as td:
        out_path = Path(td) / "rapor.html"
        with patch.object(demo, "analyze_phishing", return_value=_hybrid_result()) as mock_analyze, \
             patch.object(sys, "argv",
                          ["demo.py", str(FIXTURE_EML), "--hybrid", "-o", str(out_path)]):
            _captured_stderr(demo.main)

        mock_analyze.assert_called_once()
        call_args = mock_analyze.call_args
        assert call_args.args[0] == FIXTURE_EML
        assert call_args.kwargs.get("mode") == "hybrid"
        assert out_path.is_file()


def test_hybrid_mutually_exclusive_with_no_llm():
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "src" / "demo.py"),
         str(FIXTURE_EML), "--hybrid", "--no-llm"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "birlikte kullanılamaz" in result.stderr


def test_hybrid_mutually_exclusive_with_adapter():
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "src" / "demo.py"),
         str(FIXTURE_EML), "--hybrid", "--adapter", "0000400"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "birlikte kullanılamaz" in result.stderr


def test_hybrid_flag_present_in_help():
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "src" / "demo.py"), "--help"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "--hybrid" in result.stdout


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
