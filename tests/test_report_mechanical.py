"""Unit tests for src/report/mechanical.py's build_report(), specifically
the optional `decision` parameter added for hybrid-mode wiring
(PHISHING_ROUTING_PLAN.md "hybrid workflow wiring" task). No prior test
file existed for this module — it was only exercised indirectly via
src/demo.py, src/web.py, scripts/render_holdout_reports.py, and
tests/test_workflows_phishing.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from schemas.decision import FinalDecision  # noqa: E402
from schemas.rule_assessment import RuleAssessment, RuleEvidence  # noqa: E402
from src.decision.phishing_policy import (  # noqa: E402
    DECISION_PATH_CREDENTIAL_REQUEST_URL_UPGRADE,
    DECISION_PATH_RULE_ENGINE_ONLY,
)
from src.report.mechanical import build_report  # noqa: E402


def _assessment(verdict, evidence=None) -> RuleAssessment:
    return RuleAssessment(
        engine_version="v1",
        rule_verdict=verdict,
        score=5.0 if verdict != "Güvenilir" else 0.0,
        total=None,
        families=[],
        critical_matches=[],
        evidence=evidence or [
            RuleEvidence(signal="spf_or_dmarc_fail", description="SPF/DMARC fail", weight=3),
        ],
        decision_reasons=["x"],
    )


def _decision(rule_verdict, final_verdict, decision_path, semantic_ids=None) -> FinalDecision:
    return FinalDecision(
        rule_verdict=rule_verdict,
        final_verdict=final_verdict,
        decision_path=decision_path,
        contributing_rule_ids=[],
        contributing_semantic_ids=semantic_ids or [],
        analyst_review_required=final_verdict != "Güvenilir",
    )


def test_no_decision_uses_rule_verdict_unchanged():
    """decision=None must reproduce fast mode's original behavior
    exactly — no regression for the only caller that existed before
    hybrid mode."""
    assessment = _assessment("Phishing")
    report = build_report(assessment)
    assert report.risk_seviyesi == "Phishing"


def test_decision_matching_rule_verdict_is_a_no_op():
    assessment = _assessment("Güvenilir")
    decision = _decision("Güvenilir", "Güvenilir", DECISION_PATH_RULE_ENGINE_ONLY)
    report = build_report(assessment, decision=decision)
    assert report.risk_seviyesi == "Güvenilir"
    # No upgrade happened, so no upgrade explanation should be injected.
    assert "nihai karar" not in report.genel_degerlendirme


def test_decision_upgrade_overrides_risk_seviyesi_and_texts():
    assessment = _assessment("Güvenilir", evidence=[])
    decision = _decision(
        "Güvenilir",
        "Muhtemel Phishing",
        DECISION_PATH_CREDENTIAL_REQUEST_URL_UPGRADE,
        semantic_ids=["credential_request:0-10"],
    )
    report = build_report(assessment, decision=decision)

    assert report.risk_seviyesi == "Muhtemel Phishing"
    # sonuc_ve_gerekce must reflect the UPGRADED verdict's template, not
    # the rule engine's original "Güvenilir" text.
    assert "Muhtemel" in report.sonuc_ve_gerekce
    assert report.onerilen_aksiyon != "Ek bir aksiyon gerekmiyor."
    # The upgrade rationale (why the verdict changed) must be visible in
    # the report text — not just carried silently on FinalDecision.
    assert "Güvenilir" in report.genel_degerlendirme
    assert "kimlik bilgisi talebi" in report.genel_degerlendirme


def test_decision_upgrade_keyerror_guard_covers_every_decision_path():
    """_DECISION_PATH_LABELS is keyed by phishing_policy's DECISION_PATH_*
    constants — every non-rule_engine_only path must have a label, or
    build_report raises KeyError instead of silently omitting the
    explanation. This test locks that every currently-defined upgrade
    path has a label."""
    from src.decision import phishing_policy

    upgrade_paths = [
        v for k, v in vars(phishing_policy).items()
        if k.startswith("DECISION_PATH_") and v != DECISION_PATH_RULE_ENGINE_ONLY
    ]
    assert upgrade_paths, "expected at least one upgrade decision path to exist"

    for path in upgrade_paths:
        assessment = _assessment("Güvenilir", evidence=[])
        decision = _decision("Güvenilir", "Muhtemel Phishing", path)
        report = build_report(assessment, decision=decision)
        assert report.risk_seviyesi == "Muhtemel Phishing"


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
