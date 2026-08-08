"""
Mechanical (no-LLM) Report builder (PHISHING_ROUTING_PLAN.md step 4
follow-up; rearchitected under PROGRESS.md's "rapor mimarisi
değişikliği").

ALL SIX Report FIELDS ARE AUTHORED HERE, ALWAYS
    risk_seviyesi, sonuc_ve_gerekce, teknik_bulgular,
    phishing_gostergeleri, and onerilen_aksiyon are deterministic in
    BOTH fast and hybrid mode — this module produces them unconditionally,
    the same way it always has. genel_degerlendirme is ALSO fully
    deterministic here (a three-sentence template filled from generic,
    fallback text), but hybrid mode's workflow (src/workflows/phishing.py)
    may subsequently call src/report/narrative.apply_narrative() to
    replace ONLY that one field's three sentence slots with Qwen-authored
    text — see that module's docstring for the substitution contract.
    Nothing else Qwen produces ever reaches a Report; the model has no
    write access to risk_seviyesi, categories, technical findings, or the
    recommended SOC action, in either mode.

sonuc_ve_gerekce IS NOW BUILT HERE, NOT BY A MODEL
    Previously (src/report/prompts.py, removed) a live Qwen call wrote a
    one-sentence "Bu karar; X, Y ve Z kategorilerinin birlikte
    değerlendirilmesine dayanır." claim, narrowed by an ALLOWED-category
    prompt instruction and re-checked afterward
    (extract_claimed_categories()) — and a real measurement
    (data/semantic_eval/hybrid_reliability_results.json, 2026-08-08)
    found that check rejected the model's output in 9/18 development-set
    candidates (fallback_rate 0.50), ALL NINE concentrated in
    rule_verdict="Güvenilir" (9/13 Güvenilir candidates, 69%) — because
    the fixed six-category vocabulary is entirely attack-shaped
    ("kimlik ve marka taklidi", "aciliyet ve baskı", ...) and asking the
    model to justify a CLEAN verdict through an attack-category lens
    consistently pushed it to invent phrasing outside that vocabulary.

    The fix removes the model from this field's critical path entirely.
    For "Phishing"/"Muhtemel Phishing", the SAME category vocabulary
    (src/report/categories.py, unchanged) is populated mechanically from
    the SAME contributing evidence a prompt would have been built from —
    _build_sonuc_ve_gerekce() below computes it via
    categories_from_evidence() and formats the identical fixed sentence
    template a model was previously asked to write. For "Güvenilir", no
    attack-category sentence is constructed at all — _SONUC_GUVENILIR is
    a separate, fixed, non-category sentence
    ("belirleyici phishing göstergelerinin bulunmaması ve mevcut güven
    sinyallerinin birlikte değerlendirilmesine dayanır"), matching this
    module's rationale document's explicit instruction not to force
    Güvenilir into an attack-category shape.

WHY THIS TAKES accepted_findings NOW (new optional parameter)
    build_report(assessment) alone (accepted_findings omitted) still
    behaves exactly as before this rearchitecture — no findings means no
    semantic contribution to the category sentence, matching fast mode
    (which never has semantic findings) and any pre-existing caller.
    Hybrid mode passes decision and accepted_findings together so
    _build_sonuc_ve_gerekce() can resolve decision.contributing_
    semantic_ids against actual ValidatedSemanticFinding objects, the
    same "<type>:<start>-<end>" id scheme schemas/decision.py already
    documents and src/report/narrative_prompts.py (adım 3) reuses for
    the exact same lookup.

TAKES RuleAssessment, NOT Verdict
    Unchanged from before this rearchitecture — see engine-agnostic
    rationale below; no import of src/rules/engine.py or engine_v2.py.

OPTIONAL decision PARAMETER
    build_report(assessment) alone still reflects assessment.rule_verdict
    — fast mode's behavior is unchanged. Hybrid mode passes the decision
    policy's FinalDecision (schemas/decision.py) as well: risk_seviyesi
    and every text field then follow decision.final_verdict instead,
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
from schemas.semantic import ValidatedSemanticFinding
from src.decision.phishing_policy import (
    DECISION_PATH_CREDENTIAL_REQUEST_URL_UPGRADE,
    DECISION_PATH_PAYMENT_REQUEST_COMBINATION_UPGRADE,
    DECISION_PATH_RULE_ENGINE_ONLY,
)
from src.report.categories import categories_from_evidence

# Keyed by src/decision/phishing_policy.py's DECISION_PATH_* constants —
# importing them (rather than hardcoding the strings here) means a new
# decision path added there without a matching label here fails loudly
# (KeyError in build_report) instead of silently rendering no upgrade
# explanation.
_DECISION_PATH_LABELS = {
    DECISION_PATH_CREDENTIAL_REQUEST_URL_UPGRADE: (
        "e-posta gövdesinde doğrulanmış bir "
        "kimlik bilgisi talebi ve dış link birlikte tespit edildiği için"
    ),
    DECISION_PATH_PAYMENT_REQUEST_COMBINATION_UPGRADE: (
        "e-posta gövdesinde doğrulanmış bir "
        "ödeme talebi ödül/tehdit/yanıt-kanalı manipülasyonuyla birlikte tespit "
        "edildiği için"
    ),
}

# Fixed, non-category sentence for "Güvenilir" — see module docstring's
# "sonuc_ve_gerekce IS NOW BUILT HERE" section for why this is a SEPARATE
# sentence shape rather than an empty/degenerate case of the category
# template below.
_SONUC_GUVENILIR = (
    "Bu karar, belirleyici phishing göstergelerinin bulunmaması ve mevcut güven "
    "sinyallerinin birlikte değerlendirilmesine dayanır."
)

_ONERI_BY_VERDICT = {
    "Phishing": "E-postayı silin, linklere tıklamayın, ekleri açmayın ve gönderen adresi engelleyin.",
    "Muhtemel Phishing": (
        "E-postadaki bağlantılarla veya eklerle etkileşime girmeden önce e-postayı "
        "bir SOC analistine iletin."
    ),
    "Güvenilir": "Ek bir aksiyon gerekmiyor.",
}

# Fallback text for genel_degerlendirme's three narrative slots when no
# NarrativeDraft is applied (Güvenilir mode, or a failed hybrid narrative
# call falling back) — see src/report/narrative.py's apply_narrative()
# for the substitution that replaces these with Qwen-authored text.
_FALLBACK_SENARYO_BY_VERDICT = {
    "Phishing": "e-posta, kimlik avı amacıyla hazırlanmış teknik ve/veya içerik göstergeleri taşıyor",
    "Muhtemel Phishing": "e-posta bazı şüpheli göstergeler taşıyor, ancak otomatik değerlendirme tek başına kesin sonuca varamadı",
    "Güvenilir": "e-postada belirleyici bir phishing göstergesi tespit edilmedi",
}
_FALLBACK_EYLEM = "mevcut bulgulardan otomatik olarak çıkarılamadı"
_FALLBACK_ZARAR_BY_VERDICT = {
    "Phishing": "tespit edilen göstergelere bağlı olarak kimlik bilgisi/ödeme bilgisi kaybı veya zararlı içerik çalıştırma riski",
    "Muhtemel Phishing": "kesinleşmemiş göstergeler nedeniyle net bir zarar tahmini yapılamıyor, analist incelemesi gerekiyor",
    "Güvenilir": "belirgin bir risk tespit edilmedi",
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


def _contributing_evidence(
    assessment: RuleAssessment, decision: FinalDecision | None
) -> list[RuleEvidence]:
    """Return all fast-mode evidence or explicitly contributing hybrid evidence.

    In hybrid mode, an empty ``contributing_rule_ids`` list means no rule
    evidence contributed to the final decision. This is expected for a pure
    semantic upgrade and must not fall back to every fired rule signal.
    """
    if decision is None:
        return list(assessment.evidence)
    contributing_ids = set(decision.contributing_rule_ids)
    return [e for e in assessment.evidence if e.signal in contributing_ids]


def _contributing_findings(
    accepted_findings: list[ValidatedSemanticFinding] | None,
    decision: FinalDecision | None,
) -> list[ValidatedSemanticFinding]:
    if not accepted_findings or decision is None:
        return []
    contributing_ids = set(decision.contributing_semantic_ids)
    return [
        f for f in accepted_findings
        if f"{f.type.value}:{f.start}-{f.end}" in contributing_ids
    ]


def _build_sonuc_ve_gerekce(
    effective_verdict: str,
    assessment: RuleAssessment,
    decision: FinalDecision | None,
    accepted_findings: list[ValidatedSemanticFinding] | None,
) -> str:
    if effective_verdict == "Güvenilir":
        return _SONUC_GUVENILIR

    matches = _contributing_evidence(assessment, decision)
    finding_matches = _contributing_findings(accepted_findings, decision)
    categories = categories_from_evidence(
        [e.signal for e in matches],
        [f.type for f in finding_matches],
    )
    if not categories:
        # No category-mapped evidence at all (e.g. a v2-only or future
        # signal with no src/report/categories.py entry) — a category
        # sentence with an empty list would be grammatically broken, so
        # fall back to a generic, still-honest sentence instead of
        # rendering "Bu karar; kategorilerinin ...".
        return (
            "Bu karar, tespit edilen teknik ve/veya içerik göstergelerinin "
            "birlikte değerlendirilmesine dayanır."
        )
    if len(categories) == 1:
        joined = categories[0]
    else:
        joined = ", ".join(categories[:-1]) + " ve " + categories[-1]
    suffix = "sinin" if len(categories) == 1 else "lerinin"
    birlikte = "" if len(categories) == 1 else "birlikte "
    return f"Bu karar; {joined} kategori{suffix} {birlikte}değerlendirilmesine dayanır."


def _build_genel_degerlendirme(
    effective_verdict: str, assessment: RuleAssessment, decision: FinalDecision | None
) -> str:
    """Deterministic three-sentence template — identical shape to the
    one src/report/narrative.apply_narrative() fills with Qwen-authored
    text, but populated entirely with generic fallback phrasing here.
    Used as-is for Güvenilir (narrative is never requested — see
    src/workflows/phishing.py) and as the pre-narrative default for
    Phishing/Muhtemel Phishing until/unless apply_narrative() replaces
    it."""
    senaryo = _FALLBACK_SENARYO_BY_VERDICT[effective_verdict]
    eylem = _FALLBACK_EYLEM
    zarar = _FALLBACK_ZARAR_BY_VERDICT[effective_verdict]
    text = f"Olası senaryo: {senaryo}. Alıcıdan beklenen eylem: {eylem}. Olası zarar: {zarar}."

    if decision is not None and decision.decision_path != DECISION_PATH_RULE_ENGINE_ONLY:
        upgrade_reason = _DECISION_PATH_LABELS[decision.decision_path]
        text += (
            f" Rule engine kararı '{assessment.rule_verdict}' idi, ancak {upgrade_reason} "
            f"nihai karar '{decision.final_verdict}' olarak güncellendi."
        )
    return text


def build_report(
    assessment: RuleAssessment,
    decision: FinalDecision | None = None,
    accepted_findings: list[ValidatedSemanticFinding] | None = None,
) -> Report:
    """decision=None (default): risk_seviyesi and every text field follow
    assessment.rule_verdict — fast mode's original behavior, unchanged.
    decision given (hybrid mode): they follow decision.final_verdict
    instead — see module docstring.

    accepted_findings: only consulted when decision is also given: used
    to resolve decision.contributing_semantic_ids into real
    ValidatedSemanticFinding objects so sonuc_ve_gerekce's category
    sentence can include semantic-only categories (e.g.
    "kullanıcıyı işlem yapmaya yönlendirme" from a credential_request
    finding with no matching rule signal). Omitting it simply means the
    category sentence is built from rule evidence alone.
    """
    effective_verdict = decision.final_verdict if decision is not None else assessment.rule_verdict

    findings = [_finding_from_evidence(e) for e in assessment.evidence]
    gostergeler = [e.description for e in assessment.evidence if e.weight > 0]

    return Report(
        risk_seviyesi=effective_verdict,
        sonuc_ve_gerekce=_build_sonuc_ve_gerekce(
            effective_verdict, assessment, decision, accepted_findings
        ),
        genel_degerlendirme=_build_genel_degerlendirme(effective_verdict, assessment, decision),
        teknik_bulgular=findings,
        phishing_gostergeleri=gostergeler,
        onerilen_aksiyon=_ONERI_BY_VERDICT[effective_verdict],
    )
