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
from src.report.mechanical import _SONUC_GUVENILIR, build_report  # noqa: E402


def _assessment(verdict, evidence=None) -> RuleAssessment:
    resolved_evidence = evidence if evidence is not None else [
        RuleEvidence(signal="spf_or_dmarc_fail", description="SPF/DMARC fail", weight=3),
    ]
    return RuleAssessment(
        engine_version="v1",
        rule_verdict=verdict,
        score=5.0 if verdict != "Güvenilir" else 0.0,
        total=None,
        families=[],
        critical_matches=[],
        evidence=resolved_evidence,
        decision_reasons=["x"],
    )


def _decision(
    rule_verdict,
    final_verdict,
    decision_path,
    semantic_ids=None,
    rule_ids=None,
) -> FinalDecision:
    return FinalDecision(
        rule_verdict=rule_verdict,
        final_verdict=final_verdict,
        decision_path=decision_path,
        contributing_rule_ids=rule_ids or [],
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
    # accepted_findings supplies the ONLY contributing evidence here
    # (evidence=[] — a pure semantic upgrade, matching schemas/
    # decision.py's "contributing_rule_ids can be empty" case) so
    # sonuc_ve_gerekce's category sentence is built from the semantic
    # finding alone, via src/report/categories.py's CATEGORY_FINDING_MAP.
    from schemas.semantic import SemanticFindingType, ValidatedSemanticFinding

    assessment = _assessment("Güvenilir", evidence=[])
    decision = _decision(
        "Güvenilir",
        "Muhtemel Phishing",
        DECISION_PATH_CREDENTIAL_REQUEST_URL_UPGRADE,
        semantic_ids=["credential_request:0-10"],
    )
    accepted_findings = [
        ValidatedSemanticFinding(
            type=SemanticFindingType.CREDENTIAL_REQUEST,
            evidence="şifrenizi girin",
            start=0,
            end=10,
            model_confidence=0.9,
            explanation="doğrudan kimlik bilgisi talebi",
        )
    ]
    report = build_report(assessment, decision=decision, accepted_findings=accepted_findings)

    assert report.risk_seviyesi == "Muhtemel Phishing"
    # sonuc_ve_gerekce must reflect the UPGRADED verdict's category —
    # CREDENTIAL_REQUEST maps to "kullanıcıyı işlem yapmaya yönlendirme",
    # not the rule engine's original "Güvenilir" (no-category) sentence.
    assert "kullanıcıyı işlem yapmaya yönlendirme" in report.sonuc_ve_gerekce
    assert report.sonuc_ve_gerekce != _SONUC_GUVENILIR
    assert report.onerilen_aksiyon != "Ek bir aksiyon gerekmiyor."
    assert report.onerilen_aksiyon == (
        "E-postadaki bağlantılarla veya eklerle etkileşime girmeden önce e-postayı "
        "bir SOC analistine iletin."
    )
    # The upgrade rationale (why the verdict changed) must be visible in
    # the report text — not just carried silently on FinalDecision.
    assert "Güvenilir" in report.genel_degerlendirme
    assert "kimlik bilgisi talebi" in report.genel_degerlendirme
    assert "ancak rule engine kararı" not in report.genel_degerlendirme.lower()


def test_pure_semantic_upgrade_excludes_noncontributing_rule_categories():
    """Fired rule signals may be present without contributing to a semantic upgrade."""
    from schemas.semantic import SemanticFindingType, ValidatedSemanticFinding

    assessment = _assessment(
        "Güvenilir",
        evidence=[
            RuleEvidence(
                signal="credential_request_with_external_link",
                description="Kimlik bilgisi talebi VE en az 1 dış link var",
                weight=2,
            ),
            RuleEvidence(
                signal="urgency_keywords",
                description="En az 1 aciliyet kalıbı eşleşti",
                weight=1,
            ),
            RuleEvidence(
                signal="all_auth_pass_and_consistent",
                description="SPF+DKIM+DMARC hepsi pass ve domainler uyumlu",
                weight=-3,
            ),
        ],
    )
    finding = ValidatedSemanticFinding(
        type=SemanticFindingType.CREDENTIAL_REQUEST,
        evidence="kullanıcı adınız ile mevcut parolanızı forma girin",
        start=133,
        end=183,
        model_confidence=0.98,
        explanation="Doğrulanmış kimlik bilgisi talebi",
    )
    decision = _decision(
        "Güvenilir",
        "Muhtemel Phishing",
        DECISION_PATH_CREDENTIAL_REQUEST_URL_UPGRADE,
        semantic_ids=[f"{finding.type.value}:{finding.start}-{finding.end}"],
    )

    report = build_report(
        assessment,
        decision=decision,
        accepted_findings=[finding],
    )

    assert "kullanıcıyı işlem yapmaya yönlendirme" in report.sonuc_ve_gerekce
    assert "aciliyet ve baskı" not in report.sonuc_ve_gerekce


def test_decision_upgrade_with_no_category_evidence_falls_back_to_generic_sentence():
    """A signal with no src/report/categories.py entry, AND no
    accepted_findings passed at all — the category sentence has nothing
    to map, so it must fall back to the generic sentence rather than
    rendering a broken "Bu karar; kategorilerinin ..." with no category
    name. A real evidence list with an unmapped signal is used so the
    fallback behavior remains explicit."""
    assessment = _assessment(
        "Güvenilir",
        evidence=[
            RuleEvidence(signal="some_future_unmapped_signal", description="x", weight=2),
        ],
    )
    decision = _decision(
        "Güvenilir",
        "Muhtemel Phishing",
        DECISION_PATH_CREDENTIAL_REQUEST_URL_UPGRADE,
        semantic_ids=["credential_request:0-10"],
    )
    report = build_report(assessment, decision=decision)
    assert report.sonuc_ve_gerekce == (
        "Bu karar, tespit edilen teknik ve/veya içerik göstergelerinin "
        "birlikte değerlendirilmesine dayanır."
    )
    assert report.sonuc_ve_gerekce != _SONUC_GUVENILIR


def test_guvenilir_verdict_uses_fixed_non_category_sentence():
    assessment = _assessment("Güvenilir")
    report = build_report(assessment)
    assert report.sonuc_ve_gerekce == _SONUC_GUVENILIR


def test_phishing_verdict_sonuc_ve_gerekce_uses_category_template():
    assessment = _assessment(
        "Phishing",
        evidence=[
            RuleEvidence(signal="spf_or_dmarc_fail", description="SPF/DMARC fail", weight=3),
            RuleEvidence(
                signal="display_name_brand_mismatch",
                description="marka taklidi",
                weight=3,
            ),
        ],
    )
    report = build_report(assessment)
    assert report.sonuc_ve_gerekce.startswith("Bu karar; ")
    assert "kimlik doğrulama uyumsuzluğu" in report.sonuc_ve_gerekce
    assert "kimlik ve marka taklidi" in report.sonuc_ve_gerekce
    assert "birlikte değerlendirilmesine dayanır." in report.sonuc_ve_gerekce


def test_genel_degerlendirme_always_matches_three_sentence_template():
    for verdict in ("Phishing", "Muhtemel Phishing", "Güvenilir"):
        assessment = _assessment(verdict)
        report = build_report(assessment)
        assert report.genel_degerlendirme.startswith("Olası senaryo: ")
        assert "Alıcıdan beklenen eylem: " in report.genel_degerlendirme
        assert "Olası zarar: " in report.genel_degerlendirme


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
