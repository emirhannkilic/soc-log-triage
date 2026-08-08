"""
Mechanical (no-LLM) Report builder (PHISHING_ROUTING_PLAN.md step 4
follow-up).

Moved out of scripts/render_holdout_reports.py, which originally owned
build_report() as a side effect of its actual job (rendering the 30
hold-out emails to HTML for Adım 5's schema/template smoke test — see
that script's docstring). Three call sites (src/demo.py --no-llm,
src/web.py's fast/llm-fallback path, src/workflows/phishing.py's fast
mode) depended on a one-off script module for a function that is really
core report-generation logic, and scripts/render_holdout_reports.py
still imports build_report from here to keep its own behavior unchanged.

TAKES RuleAssessment, NOT Verdict
    The whole point of RuleAssessment (schemas/rule_assessment.py) is
    that report generation shouldn't need to know whether v1 or v2 (or
    a future v3) produced the verdict — evidence and decision_reasons
    are already engine-agnostic by the time they get here. This module
    has no import of src/rules/engine.py or engine_v2.py.

OPTIONAL decision PARAMETER (PHISHING_ROUTING_PLAN.md step "hybrid
workflow wiring")
    build_report(assessment) alone still reflects assessment.rule_verdict
    — fast mode's behavior is unchanged. Hybrid mode passes the decision
    policy's FinalDecision (schemas/decision.py) as well: risk_seviyesi
    and both report texts then follow decision.final_verdict instead,
    since a semantic upgrade (Güvenilir -> Muhtemel Phishing) means
    rule_verdict no longer equals the real verdict. This keeps the
    invariant report.risk_seviyesi == effective verdict true in both
    modes, matching PHISHING_ROUTING_PLAN.md's acceptance criterion 20
    ("Rapor modeli yalnızca policy'nin ürettiği final_verdict değerini
    kullanır") — a mechanical report is still a report.
"""
from schemas.decision import FinalDecision
from schemas.report import Report, TechnicalFinding
from schemas.rule_assessment import RuleAssessment, RuleEvidence
from src.decision.phishing_policy import (
    DECISION_PATH_CREDENTIAL_REQUEST_URL_UPGRADE,
    DECISION_PATH_PAYMENT_REQUEST_COMBINATION_UPGRADE,
    DECISION_PATH_RULE_ENGINE_ONLY,
)

# Keyed by src/decision/phishing_policy.py's DECISION_PATH_* constants —
# importing them (rather than hardcoding the strings here) means a new
# decision path added there without a matching label here fails loudly
# (KeyError in build_report) instead of silently rendering no upgrade
# explanation.
_DECISION_PATH_LABELS = {
    DECISION_PATH_CREDENTIAL_REQUEST_URL_UPGRADE: (
        "rule engine kararı Güvenilir olsa da, e-posta gövdesinde doğrulanmış bir "
        "kimlik bilgisi talebi ve dış link birlikte tespit edildiği için"
    ),
    DECISION_PATH_PAYMENT_REQUEST_COMBINATION_UPGRADE: (
        "rule engine kararı Güvenilir olsa da, e-posta gövdesinde doğrulanmış bir "
        "ödeme talebi ödül/tehdit/yanıt-kanalı manipülasyonuyla birlikte tespit "
        "edildiği için"
    ),
}

_SONUC_BY_VERDICT = {
    "Phishing": (
        "Bu e-posta, rule engine tarafından {score_desc} Phishing "
        "olarak sınıflandırıldı. Aşağıdaki teknik bulgular bu kararı destekliyor."
    ),
    "Muhtemel Phishing": (
        "Bu e-posta, rule engine tarafından {score_desc} Muhtemel "
        "Phishing olarak sınıflandırıldı. Kesin bir karar için analist "
        "incelemesi gerekiyor."
    ),
    "Güvenilir": (
        "Bu e-posta, rule engine tarafından {score_desc} Güvenilir "
        "olarak sınıflandırıldı. Aşağıdaki bulgular bu değerlendirmeyi destekliyor."
    ),
}

_ONERI_BY_VERDICT = {
    "Phishing": "E-postayı silin, linklere tıklamayın, ekleri açmayın ve gönderen adresi engelleyin.",
    "Muhtemel Phishing": "E-postadaki linklere/eklere etkileşimde bulunmadan önce SOC analistine iletin.",
    "Güvenilir": "Ek bir aksiyon gerekmiyor.",
}


def _finding_from_evidence(e: RuleEvidence) -> TechnicalFinding:
    sign = "+" if e.weight >= 0 else ""
    return TechnicalFinding(
        baslik=e.signal.replace("_", " "),
        aciklama=f"{e.description} ({sign}{e.weight:g} puan)",
    )


def _score_description(assessment: RuleAssessment) -> str:
    # v1 carries a single additive score; v2 carries a family total.
    # RuleAssessment.score is None for v2, total is None for v1 — see
    # schemas/rule_assessment.py.
    if assessment.score is not None:
        return f"toplam {assessment.score:g} puanla"
    return f"toplam {assessment.total} puanla"


def build_report(assessment: RuleAssessment, decision: FinalDecision | None = None) -> Report:
    """decision=None (default): risk_seviyesi and both report texts
    follow assessment.rule_verdict — fast mode's original behavior,
    unchanged. decision given (hybrid mode): they follow
    decision.final_verdict instead — see module docstring."""
    findings = [_finding_from_evidence(e) for e in assessment.evidence]
    gostergeler = [e.description for e in assessment.evidence if e.weight > 0]

    effective_verdict = decision.final_verdict if decision is not None else assessment.rule_verdict

    score_desc = _score_description(assessment)
    genel_degerlendirme = (
        f"Toplam {len(assessment.evidence)} sinyal değerlendirildi, "
        f"{score_desc}. " + " ".join(assessment.decision_reasons)
    )
    if decision is not None and decision.decision_path != DECISION_PATH_RULE_ENGINE_ONLY:
        upgrade_reason = _DECISION_PATH_LABELS[decision.decision_path]
        genel_degerlendirme += (
            f" Rule engine kararı '{assessment.rule_verdict}' idi, ancak {upgrade_reason} "
            f"nihai karar '{decision.final_verdict}' olarak güncellendi."
        )

    return Report(
        risk_seviyesi=effective_verdict,
        sonuc_ve_gerekce=_SONUC_BY_VERDICT[effective_verdict].format(score_desc=score_desc),
        genel_degerlendirme=genel_degerlendirme,
        teknik_bulgular=findings,
        phishing_gostergeleri=gostergeler,
        onerilen_aksiyon=_ONERI_BY_VERDICT[effective_verdict],
    )
