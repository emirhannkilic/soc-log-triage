"""
Prompt construction for src/report/narrative.py's generate_narrative()
— Qwen's ENTIRE contribution to a hybrid report, narrowed to
schemas.narrative.NarrativeDraft's three fields (PROGRESS.md "rapor
mimarisi değişikliği").

NOT src/report/prompts.py (removed)
    The old module asked the model to write a full schemas.report.Report
    — risk_seviyesi, sonuc_ve_gerekce (a category-vocabulary sentence),
    teknik_bulgular, phishing_gostergeleri, and onerilen_aksiyon, all
    model-authored and re-validated after the fact. A real measurement
    (data/semantic_eval/hybrid_reliability_results.json, 2026-08-08)
    found that pipeline rejected 9/18 development-set outputs — all via
    the category-vocabulary check, all in rule_verdict="Güvenilir"
    (9/13, 69%) — because the fixed six-category vocabulary is entirely
    attack-shaped and justifying a CLEAN verdict through it consistently
    pushed the model outside the vocabulary. This module's prompt asks
    for something structurally narrower: three sentence fragments with
    no vocabulary constraint to violate, because there is no vocabulary
    — src/report/mechanical.py now owns risk_seviyesi, sonuc_ve_gerekce,
    teknik_bulgular, phishing_gostergeleri, and onerilen_aksiyon
    entirely, deterministically, in both modes. Qwen is asked for prose,
    not classification-shaped content.

WHAT THE MODEL IS TOLD IS AUTHORITATIVE
    Same as the old module: FinalDecision.final_verdict — never
    rule_assessment.rule_verdict directly — so the narrative's tone
    matches the real, possibly-upgraded verdict.

decision_path / contributing_*_ids ARE NEVER SHOWN AS RAW CODES
    Unchanged rationale from src/report/prompts.py: decision_path
    resolves through _UPGRADE_EXPLANATIONS (mirroring
    src/report/mechanical.py's own _DECISION_PATH_LABELS) into a plain
    Turkish sentence fragment; contributing_semantic_ids/contributing_
    rule_ids are used only to SELECT which evidence is shown, never
    interpolated as raw strings.

ONLY CONTRIBUTING EVIDENCE IS INCLUDED, NOT EVERYTHING
    Unchanged rationale — rule_assessment.evidence can contain fired-
    but-not-decisive signals; showing only what FinalDecision names as
    contributing keeps the model from having to guess the reason.

NO RAW BODY, NO RAW SUBJECT — DATA-MINIMIZED VIEW ONLY
    Unchanged from src/report/prompts.py's own fix after a real PII leak
    (a smoke-test run against an unanonymized raw sample copied a real
    name/phone/address into the old Report's phishing_gostergeleri — see
    that removed module's docstring for the full incident). The model
    only ever sees: contributing rule evidence descriptions, contributing
    validated semantic findings (short, TYPED, exact-quote-grounded
    excerpts), a URL summary with query/fragment stripped, attachment
    metadata, and decision_path resolved to a fixed sentence fragment.
    Since NarrativeDraft has no field for technical claims at all (no
    teknik_bulgular-equivalent), this data-minimized view exists purely
    to give the model enough CONTEXT to write a coherent scenario/action/
    harm narrative — it is not asked to restate any of it as a "finding."

PII-COPY PROHIBITION IS A SECOND LAYER, NOT THE FIX
    Unchanged rationale — SYSTEM_PROMPT still tells the model never to
    copy a name/phone/email/address into its narrative even if one
    appears in a quoted semantic finding.

NO CATEGORY VOCABULARY, NO risk_seviyesi FIELD, NO OUTPUT-SIDE CATEGORY
CHECK
    This is the structural difference from the removed module. There is
    nothing here corresponding to ALL_CATEGORIES / allowed_categories() /
    extract_claimed_categories() — NarrativeDraft has no field the model
    could use to claim a category, a risk level, or a technical finding,
    so there is no vocabulary for the model to violate and no output-side
    category enforcement needed. src/report/narrative.py's schema
    validation (NarrativeDraft(**parsed), extra="forbid") is the only
    enforcement this path needs.
"""
from schemas.decision import FinalDecision
from schemas.facts import EmailFacts
from schemas.rule_assessment import RuleAssessment
from schemas.semantic import ValidatedSemanticFinding
from src.decision.phishing_policy import (
    DECISION_PATH_CREDENTIAL_REQUEST_URL_UPGRADE,
    DECISION_PATH_PAYMENT_REQUEST_COMBINATION_UPGRADE,
    DECISION_PATH_RULE_ENGINE_ONLY,
)

# Mirrors src/report/mechanical.py's _DECISION_PATH_LABELS — same closed
# set of decision_path values, phrased for the MODEL prompt rather than
# the mechanical report's own prose. Kept as a separate dict (not
# imported from mechanical.py) because the two audiences need different
# grammar around the fragment; the KEYS are the same imported constants,
# so a new decision path missing from either dict still fails loudly.
_UPGRADE_EXPLANATIONS = {
    DECISION_PATH_CREDENTIAL_REQUEST_URL_UPGRADE: (
        "e-posta gövdesinde doğrulanmış bir kimlik bilgisi talebi ve dış link "
        "birlikte tespit edildiği için"
    ),
    DECISION_PATH_PAYMENT_REQUEST_COMBINATION_UPGRADE: (
        "e-posta gövdesinde doğrulanmış bir ödeme talebi ödül/tehdit/yanıt-kanalı "
        "manipülasyonuyla birlikte tespit edildiği için"
    ),
}

_PII_PROHIBITION = """
KİŞİSEL VERİ YASAĞI (bir bulgu alıntısında geçse bile geçerli): metninin HİÇBİR \
alanına gerçek bir kişi adı, telefon numarası, e-posta adresi ya da açık posta \
adresi KOPYALAMA. Bu tür bir bilgiden bahsetmen gerekiyorsa sadece türünü genel \
olarak anlat (ör. "bir kişi adı ve telefon numarası paylaşılmış"), değerin \
KENDİSİNİ yazma."""

SYSTEM_PROMPT_TEMPLATE = """Sen bir SOC (Security Operations Center) analistine yardımcı olan bir \
asistansın. Bir e-posta hakkında ZATEN VERİLMİŞ bir sınıflandırma kararı ve teknik bulgular var \
— senin işin sınıflandırma yapmak ya da teknik bulgu üretmek DEĞİL, sadece bu e-postanın \
muhtemel senaryosunu, alıcıdan ne istediğini ve olası zararını üç kısa cümleyle anlatmak.
{pii_prohibition}

Kurallar:
- SADECE JSON döndür, JSON dışında hiçbir açıklama/metin ekleme.
- Görevin YALNIZCA üç alanı doldurmak: "olasi_senaryo", "mailin_talep_ettigi_eylem", \
"olasi_zarar". Bunların DIŞINDA hiçbir alan üretme — risk seviyesi, kategori adı, teknik \
bulgu başlığı, SOC aksiyonu gibi hiçbir şey senin işin DEĞİL, bunlar sana ayrıca ve \
deterministik olarak veriliyor.
- Her alan TAM OLARAK bir cümle olmalı, teknik terim (domain, DKIM, SPF, DMARC, \
Return-Path, header, link sayısı, kural adı) KULLANMA — bunlar zaten raporun başka \
bir yerinde ayrıntılı olarak listeleniyor, senin cümlelerin sade, senaryo düzeyinde olmalı.
- Cevabın NİHAİ KARAR ile tutarlı olmalı. KARAR "Phishing" ise dilin net ve ciddi olmalı; \
KARAR "Muhtemel Phishing" ise temkinli/belirsiz bir dil kullan ("olabilir", "görünüyor" \
gibi); KARAR "Güvenilir" bu prompt'a hiç gelmez (o durumda bu çağrı hiç yapılmaz).
- "olasi_senaryo": Bu e-postanın hangi sosyal mühendislik senaryosunu uyguladığını bir \
cümleyle anlat (ör. "Alıcı, bankasından geldiğini iddia eden bir e-posta alıyor.").
- "mailin_talep_ettigi_eylem": E-postanın KENDİSİNİN alıcıdan ne yapmasını istediğini \
anlat (ör. "Alıcının bir bağlantıya tıklayıp giriş bilgilerini girmesi isteniyor."). Bu \
alan ASLA bir SOC/güvenlik önerisi İÇEREMEZ — o senin işin DEĞİL.
- "olasi_zarar": Alıcı bu talebi yerine getirirse ne kaybedebileceğini bir cümleyle \
anlat (ör. "Girilen kimlik bilgileri saldırgan tarafından ele geçirilebilir.").
- JSON metin alanlarının İÇİNDE çift tırnak (") KULLANMA. Bir kelimeyi vurgulamak \
gerekiyorsa tek tırnak kullan: 'böyle'.

JSON şeması:
{{
  "olasi_senaryo": "1 cümle, teknik terim yok",
  "mailin_talep_ettigi_eylem": "1 cümle, teknik terim yok, SOC önerisi YASAK",
  "olasi_zarar": "1 cümle, teknik terim yok"
}}"""


def build_system_prompt() -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(pii_prohibition=_PII_PROHIBITION)


def _contributing_evidence_block(rule_assessment: RuleAssessment, decision: FinalDecision) -> str:
    contributing_ids = set(decision.contributing_rule_ids)
    matches = [e for e in rule_assessment.evidence if e.signal in contributing_ids]
    if not matches:
        return "  (bu kararda katkı veren kural kanıtı yok)"
    return "\n".join(
        f"  {e.weight:+g}  {e.description}" for e in matches
    )


def _contributing_findings_block(
    accepted_findings: list[ValidatedSemanticFinding], decision: FinalDecision
) -> str:
    contributing_ids = set(decision.contributing_semantic_ids)
    matches = [
        f for f in accepted_findings
        if f"{f.type.value}:{f.start}-{f.end}" in contributing_ids
    ]
    if not matches:
        return "  (bu kararda katkı veren gövde bulgusu yok)"
    return "\n".join(
        f"  - [{m.type.value}] \"{m.evidence}\" — {m.explanation}" for m in matches
    )


def _upgrade_block(rule_assessment: RuleAssessment, decision: FinalDecision) -> str:
    if decision.decision_path == DECISION_PATH_RULE_ENGINE_ONLY:
        return ""
    explanation = _UPGRADE_EXPLANATIONS[decision.decision_path]
    return (
        f"\nNOT: kural motorunun kendi kararı '{rule_assessment.rule_verdict}' idi, "
        f"ancak {explanation} nihai karar '{decision.final_verdict}' olarak "
        f"güncellendi. Bu güncellemeyi senaryonda doğal bir şekilde yansıt, ham karar "
        f"kodundan (decision_path) BAHSETME."
    )


def _strip_query_and_fragment(url: str) -> str:
    """A URL's query string or fragment can carry an email address, a
    name, a session/tracking token, or other PII-shaped data the sender
    embedded for their own routing purposes — none of that is context
    this narrative needs. Kept as a plain string operation (not
    urllib.parse) because a malformed URL must still be truncated
    safely rather than raising — display-only."""
    for sep in ("?", "#"):
        idx = url.find(sep)
        if idx != -1:
            return url[:idx]
    return url


def _url_block(facts: EmailFacts) -> str:
    if not facts.urls:
        return "  (yok)"
    return "\n".join(
        f"  - {_strip_query_and_fragment(u.url)} (href_domain={u.href_domain}, "
        f"görünen_metinle_uyuşmuyor={'evet' if u.text_href_mismatch else 'hayır'})"
        for u in facts.urls
    )


def _attachment_block(facts: EmailFacts) -> str:
    if not facts.attachments:
        return "  (yok)"
    return "\n".join(
        f"  - {a.filename} ({a.mime_type or 'bilinmiyor'})"
        for a in facts.attachments
    )


def build_user_prompt(
    facts: EmailFacts,
    rule_assessment: RuleAssessment,
    decision: FinalDecision,
    accepted_findings: list[ValidatedSemanticFinding],
) -> str:
    """No raw body, no raw subject — see module docstring's "NO RAW
    BODY, NO RAW SUBJECT" section."""
    return f"""NİHAİ KARAR: {decision.final_verdict}

KATKI VEREN KURAL KANITLARI (bağlam için, teknik bulgu ÜRETME):
{_contributing_evidence_block(rule_assessment, decision)}

DOĞRULANMIŞ GÖVDE BULGULARI (gövdeden alıntıyla doğrulanmış, karara katkı veren):
{_contributing_findings_block(accepted_findings, decision)}
{_upgrade_block(rule_assessment, decision)}

BAĞLANTILAR (varsa, bağlam için — sorgu/parametre kısımları gizlilik nedeniyle \
çıkarılmıştır):
{_url_block(facts)}

EKLER (varsa, bağlam için):
{_attachment_block(facts)}"""


def build_messages(
    facts: EmailFacts,
    rule_assessment: RuleAssessment,
    decision: FinalDecision,
    accepted_findings: list[ValidatedSemanticFinding],
) -> list[dict]:
    return [
        {"role": "system", "content": build_system_prompt()},
        {
            "role": "user",
            "content": build_user_prompt(facts, rule_assessment, decision, accepted_findings),
        },
    ]
