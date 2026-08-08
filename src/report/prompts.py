"""
Live hybrid-report prompt construction (PHISHING_ROUTING_PLAN.md section
10.5, "Qwen çağrı 2: facts + rules + doğrulanmış findings + final_verdict
-> report").

NOT src/teacher/prompts.py
    That module builds the OFFLINE teacher-generation prompt (LoRA
    training corpus, src/teacher/generate_training_data.py) and is
    frozen — CLAUDE.md's locked LoRA training-data decisions apply to
    it, and it is keyed to a raw rule-engine Verdict, not the hybrid
    workflow's FinalDecision. This module is a separate, live-inference
    prompt for src/report/generate.py: same reporting JOB (facts +
    decision -> Turkish Report JSON), different and authoritative
    decision input (FinalDecision, not Verdict), and it is allowed to
    change independently of the frozen teacher prompt.

WHAT THE MODEL IS TOLD IS AUTHORITATIVE
    FinalDecision.final_verdict — never rule_assessment.rule_verdict
    directly. In fast mode there is no FinalDecision at all (this module
    is hybrid-only); in hybrid mode final_verdict is the one number a
    semantic upgrade may have moved past rule_verdict, and it is the
    only verdict this prompt ever shows the model, so there is no way
    for the model to see a stale value and rationalize the wrong one.

decision_path / contributing_*_ids ARE NEVER SHOWN AS RAW CODES
    FinalDecision.decision_path is a fixed machine token
    ("credential_request_plus_url_upgrade") and contributing_rule_ids /
    contributing_semantic_ids are signal names or "<type>:<start>-<end>"
    strings — internal bookkeeping, not Turkish prose a SOC analyst
    should read verbatim in a report. This module never interpolates
    those raw strings into the prompt. Instead it resolves decision_path
    through _UPGRADE_EXPLANATIONS (mirroring src/report/mechanical.py's
    _DECISION_PATH_LABELS — same closed vocabulary, same "a new decision
    path without a matching label fails loudly" property) into a plain
    Turkish sentence fragment the model can use directly, and it uses
    contributing_semantic_ids only to SELECT which evidence/findings are
    included below — the ids themselves are filtered out, not surfaced.

ONLY CONTRIBUTING EVIDENCE IS INCLUDED, NOT EVERYTHING
    rule_assessment.evidence can contain signals that fired but did not
    actually decide anything (e.g. every fired rule when the rule
    engine's own verdict already settled the case — see
    FinalDecision.contributing_rule_ids's docstring in schemas/
    decision.py). Passing the model the FULL evidence list when only
    some of it explains final_verdict invites exactly the failure mode
    src/teacher/prompts.py's module docstring already recorded once
    (DKIM pass narrated as a trust signal when the engine had actually
    fired dkim_pass_but_domain_mismatch for the same field) — the model
    should not have to guess which listed signal is the REASON. This
    module filters rule_assessment.evidence and accepted_findings down
    to exactly the ids FinalDecision names as contributing, before
    either ever reaches the prompt text.

    When contributing_rule_ids/contributing_semantic_ids are both empty
    (fast-mode-shaped decisions do not reach this module at all, but a
    Güvenilir FinalDecision with no upgrade can legitimately have empty
    semantic ids and a non-empty rule id list, or vice versa for a
    semantic-only upgrade) the corresponding block says so explicitly
    rather than rendering an empty list silently.
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
from src.semantic.canonical import canonicalize_body

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

SYSTEM_PROMPT = """Sen bir SOC (Security Operations Center) analistine yardımcı olan bir \
asistansın. Görevin, bir e-posta hakkında ZATEN VERİLMİŞ bir sınıflandırma kararını \
(NİHAİ KARAR olarak sana verilecek) Türkçe bir rapora dökmek — sen sınıflandırma \
YAPMIYORSUN, sadece verilen bulgulara dayanarak gerekçelendiriyorsun.

ÖNEMLİ — ANONİMLEŞTİRME UYARISI: Sana verilen GÖVDE ve KONU metninde "Ad Soyad 0", \
"Ad Soyad 14" gibi ifadeler göreceksin. Bunlar GERÇEK İÇERİK DEĞİL, sahte veri DEĞİL, \
şüpheli bir kalıp DEĞİL — bu proje, e-postanın gerçek hesap sahibinin adını otomatik \
olarak bu şekilde maskeliyor (anonimleştirme). Bunu görmezden gel: raporunda ASLA \
"anlamsız hitap", "kişiselleştirilmemiş mesaj", "isim eksik" gibi bir yorum yapma, \
bunu bir şüphe/phishing göstergesi olarak KULLANMA, ve "Ad Soyad" ifadesinden hiç \
bahsetme. Sanki o kısımda gerçek bir isim yazıyormuş gibi davran ve konuyu atla.

Diğer kurallar:
- SADECE JSON döndür, JSON dışında hiçbir açıklama/metin ekleme.
- "risk_seviyesi" alanı, sana verilen NİHAİ KARAR ile BİREBİR AYNI olmak zorunda.
- Raporundaki HER teknik iddia, sana verilen KURAL KANITLARI ve DOĞRULANMIŞ GÖVDE \
BULGULARI listelerindeki bir maddeye dayanmalı. Orada olmayan hiçbir şey uydurma \
(ör. olmayan bir header, olmayan bir URL sayısı, GÖVDE metnini okuyarak kendi \
başına çıkardığın bir gözlem).
- Raporun TONU verilen NİHAİ KARAR ile tutarlı olmak zorunda. KARAR "Phishing" ise \
metinde "güvenilir görünüyor", "zararsız", "sorun yok" gibi ifadeler KULLANMA; \
KARAR "Güvenilir" ise alarm dili kullanma. Kararı sorgulama, yumuşatma ya da \
tartışma — o karar deterministik bir karar politikasından geliyor ve senin işin \
onu AÇIKLAMAK.
- KARAR "Muhtemel Phishing" ise raporun dili temkinli/belirsiz olmalı ("kesin \
olarak doğrulanamadı ama dışlanamadı da", "tek başına yeterli kanıt değil" gibi). \
KARAR "Güvenilir" ise raporun TAMAMI net ve kararlı olmalı, şüpheci bir ekleme \
YAPMA. KARAR "Phishing" ise doğrudan ve kesin bir dille yaz.
- "KATKI VEREN KANITLAR" listesi raporunun iskeletidir: teknik bulguların o \
kanıtların karşılığı olmalı. Listede olmayan bir sinyalden bahsetme.
- "Güvenilir" kararlarda "phishing_gostergeleri" boş bir liste OLABİLİR ve genellikle \
OLMALIDIR — her maile zorla bir gösterge uydurma.
- Tüm metin alanları Türkçe olmalı.
- JSON metin alanlarının İÇİNDE çift tırnak (") KULLANMA. Bir kelimeyi \
vurgulamak ya da alıntılamak gerekiyorsa tek tırnak kullan: 'böyle'. Çift \
tırnak string'i erken kapatır ve çıktının tamamı geçersiz JSON olur.
- "teknik_bulgular" listesindeki her "aciklama" İKİ şey içermeli, tek cümlede \
değil ayrı ayrı: (1) bulgunun NE olduğu (somut veri: hangi domain, hangi URL, \
hangi header ya da gövdeden hangi alıntı), (2) bunun NEDEN bir risk/güven \
göstergesi sayıldığı.
- "onerilen_aksiyon": SEN bu e-postanın ne kadar tehlikeli olduğuna karar \
VERMİYORSUN, bu karar zaten "risk_seviyesi" ile SANA VERİLDİ. "risk_seviyesi" == \
"Phishing" ise aksiyon KESİN ve NET olmalı (ör. "E-postayı silin, hiçbir \
bağlantıya tıklamayın veya eki açmayın, gönderen adresi engelleyin."). \
"risk_seviyesi" == "Muhtemel Phishing" ise aksiyon bir SOC analistine yönlendirme \
olmalı, KESİN bir "silin"/"tıklamayın" talimatı DEĞİL (ör. "Otomatik karar \
verilemedi, bir SOC analisti gönderen ve bağlantıları manuel olarak incelemeden \
e-postayla etkileşime girilmemesi önerilir."). "risk_seviyesi" == "Güvenilir" ise \
aksiyon nötr olmalı (ör. "Ek bir aksiyon gerekmiyor.").

İÇERİK SAHİPLİĞİ KURALI — her bilgi SADECE kendisine ayrılan alanda yazılır, \
başka bir alanda TEKRAR EDİLMEZ:

- "teknik_bulgular": Ham gözlemler, domain adları, header'lar, doğrulama \
sonuçları, kural adları, gövdeden doğrulanmış alıntılar SADECE burada yazılır. \
Detaylı, madde madde.
- "sonuc_ve_gerekce": SADECE şu kalıpla, TAM OLARAK bir cümle: \
"Bu karar; [KATEGORİ], [KATEGORİ] ve [KATEGORİ] kategorilerinin birlikte \
değerlendirilmesine dayanır." Köşeli parantezleri, aşağıdaki İZİN VERİLEN \
KURAL KATEGORİLERİ listesinden (birkaçını) seçerek doldur. Domain adı, \
header adı, URL gibi HİÇBİR ham veri burada YAZILMAZ.
- "genel_degerlendirme": SADECE şu kalıpla, TAM OLARAK üç cümle: \
"Olası senaryo: [...]. Alıcıdan beklenen eylem: [...]. Olası zarar: [...]." \
Domain, DKIM, SPF, DMARC, Return-Path, header, link sayısı, kural adı gibi \
HİÇBİR teknik terim burada YAZILMAZ. Bir köşeli parantezin cevabı mevcut \
bulgulardan çıkarılamıyorsa oraya "Mevcut bulgulardan belirlenemiyor" yaz. \
"Alıcıdan beklenen eylem" SADECE e-postanın KENDİSİNİN alıcıdan ne yapmasını \
İSTEDİĞİNİ anlatır. BU ALAN ASLA bir SOC/güvenlik önerisi İÇEREMEZ — o \
"onerilen_aksiyon" alanının işi.

İZİN VERİLEN KURAL KATEGORİLERİ (sadece bu listeden seç, başka kategori \
UYDURMA):
- kimlik ve marka taklidi
- kimlik doğrulama uyumsuzluğu
- içerik gizleme
- kullanıcıyı işlem yapmaya yönlendirme
- aciliyet ve baskı
- zararlı ek veya içerik

JSON şeması:
{
  "risk_seviyesi": "Phishing" | "Muhtemel Phishing" | "Güvenilir",
  "sonuc_ve_gerekce": "TAM OLARAK 1 cümle, sabit kalıp (yukarıya bak), sadece İZİN VERİLEN KATEGORİLER",
  "genel_degerlendirme": "TAM OLARAK 3 cümle, sabit kalıp (yukarıya bak), teknik terim YASAK",
  "teknik_bulgular": [{"baslik": "...", "aciklama": "ne olduğu + neden önemli olduğu, 2 cümle"}],
  "phishing_gostergeleri": ["...", "..."],
  "onerilen_aksiyon": "1-2 cümle, risk_seviyesi ile TUTARLI (yukarıya bak) — Muhtemel Phishing'de KESİN 'silin' talimatı YASAK"
}

HATIRLATMA: "Ad Soyad N" bir anonimleştirme maskesidir, gerçek içerik değildir — \
bundan raporunda ASLA bahsetme ve bunu bir gösterge olarak kullanma."""


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
        f"güncellendi. Bu güncellemeyi 'genel_degerlendirme'de doğal bir şekilde "
        f"yansıt, ham karar kodundan (decision_path) BAHSETME."
    )


def _url_block(facts: EmailFacts) -> str:
    if not facts.urls:
        return "  (yok)"
    return "\n".join(
        f"  - {u.url} (href_domain={u.href_domain}, "
        f"görünen_metinle_uyuşmuyor={'evet' if u.text_href_mismatch else 'hayır'}, "
        f"kısaltıcı={'evet' if u.is_shortener else 'hayır'}, "
        f"IP_tabanlı={'evet' if u.is_ip_based else 'hayır'}, "
        f"punycode={'evet' if u.has_punycode else 'hayır'})"
        for u in facts.urls
    )


def _attachment_block(facts: EmailFacts) -> str:
    if not facts.attachments:
        return "  (yok)"
    return "\n".join(
        f"  - {a.filename} ({a.mime_type or 'bilinmiyor'}, "
        f"riskli_uzantı={'evet' if a.risky_type else 'hayır'}, "
        f"çift_uzantı={'evet' if a.double_extension else 'hayır'}, "
        f"arşiv={'evet' if a.is_archive else 'hayır'})"
        for a in facts.attachments
    )


def build_user_prompt(
    facts: EmailFacts,
    rule_assessment: RuleAssessment,
    decision: FinalDecision,
    accepted_findings: list[ValidatedSemanticFinding],
) -> str:
    return f"""NİHAİ KARAR: {decision.final_verdict}

KATKI VEREN KURAL KANITLARI (raporun gerekçesi BUNLAR olmalı):
{_contributing_evidence_block(rule_assessment, decision)}

DOĞRULANMIŞ GÖVDE BULGULARI (gövdeden alıntıyla doğrulanmış, karara katkı veren):
{_contributing_findings_block(accepted_findings, decision)}
{_upgrade_block(rule_assessment, decision)}

BAĞLANTILAR (varsa, teknik bulgularda ismen atıfta bulun):
{_url_block(facts)}

EKLER (varsa, teknik bulgularda ismen atıfta bulun):
{_attachment_block(facts)}

E-POSTA KONUSU: {facts.subject or "(konu yok)"}
GÖVDE (ilk 2000 karakter): {canonicalize_body(facts.body_text)[:2000]}"""


def build_messages(
    facts: EmailFacts,
    rule_assessment: RuleAssessment,
    decision: FinalDecision,
    accepted_findings: list[ValidatedSemanticFinding],
) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_user_prompt(facts, rule_assessment, decision, accepted_findings),
        },
    ]
