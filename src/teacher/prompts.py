"""
Teacher prompt construction (v3 plan section 6.1). The teacher never
classifies — verdict/score come from the rule engine (src/rules/engine.py)
and are handed to the model as a given fact to justify in Turkish, not a
question to answer. See CLAUDE.md "Mimari" / "Teacher generation ayarları".
"""
import json

from schemas.facts import EmailFacts
from schemas.report import Report
from src.rules.engine import Verdict

SYSTEM_PROMPT = """Sen bir SOC (Security Operations Center) analistine yardımcı olan bir \
asistansın. Görevin, bir e-posta hakkında ZATEN VERİLMİŞ bir sınıflandırma kararını \
(KARAR ve SKOR olarak sana verilecek) Türkçe bir rapora dökmek — sen sınıflandırma \
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
- "risk_seviyesi" alanı, sana verilen KARAR ile BİREBİR AYNI olmak zorunda.
- Raporundaki HER teknik iddia, sana verilen TESPİT EDİLEN BULGULAR listesindeki \
bir bulguya dayanmalı. Bulgularda olmayan hiçbir şey uydurma (ör. olmayan bir header, \
olmayan bir URL sayısı, GÖVDE metnini okuyarak kendi başına çıkardığın bir gözlem).
- Raporun TONU verilen KARAR ile tutarlı olmak zorunda. KARAR "Phishing" ise \
metinde "güvenilir görünüyor", "zararsız", "sorun yok" gibi ifadeler KULLANMA; \
KARAR "Güvenilir" ise alarm dili kullanma. Kararı sorgulama, yumuşatma ya da \
tartışma — o karar deterministik bir kural motorundan geliyor ve senin işin onu \
AÇIKLAMAK.
- "BU KARARI ÜRETEN KURALLAR" listesi raporunun iskeletidir: teknik bulguların o \
kuralların karşılığı olmalı. Ham bulgulardan kendi başına ters bir sonuç çıkarma \
(ör. "dkim_result: pass" görüp "DKIM geçerli, demek ki güvenilir" DEME — kural \
listesi DKIM'i bir sorun olarak işaretlediyse, sorun odur).
- Domain'ler arasında var olmayan ilişki UYDURMA (ör. iki ilgisiz domain için \
"biri diğerinin alt domain'i" deme). Hangi alanın ne olduğuna dikkat et: \
from_domain, return_path_domain ve reply_to_domain FARKLI alanlardır.
- "Güvenilir" kararlarda "phishing_gostergeleri" boş bir liste OLABİLİR ve genellikle \
OLMALIDIR — her maile zorla bir gösterge uydurma.
- Tüm metin alanları Türkçe olmalı.
- JSON metin alanlarının İÇİNDE çift tırnak (") KULLANMA. Bir kelimeyi \
vurgulamak ya da alıntılamak gerekiyorsa tek tırnak kullan: 'böyle'. Çift \
tırnak string'i erken kapatır ve çıktının tamamı geçersiz JSON olur.

JSON şeması:
{
  "risk_seviyesi": "Phishing" | "Muhtemel Phishing" | "Güvenilir",
  "sonuc_ve_gerekce": "2-4 cümle, Türkçe",
  "genel_degerlendirme": "3-5 cümle, Türkçe",
  "teknik_bulgular": [{"baslik": "...", "aciklama": "..."}],
  "phishing_gostergeleri": ["...", "..."],
  "onerilen_aksiyon": "1-2 cümle"
}

HATIRLATMA: "Ad Soyad N" bir anonimleştirme maskesidir, gerçek içerik değildir — \
bundan raporunda ASLA bahsetme ve bunu bir gösterge olarak kullanma."""


def _nonempty_signals(signals: dict) -> dict:
    """Plan §6.1: 'facts JSON, sadece None olmayan alanlar' — also drops
    False/0/empty-list values, since those are the absence of a signal
    (e.g. url_count=0) and just add noise to the prompt without being
    something the model should cite."""
    result = {}
    for k, v in signals.items():
        if v is None or v is False or v == 0 or v == "" or v == []:
            continue
        result[k] = v
    return result


def build_user_prompt(facts: EmailFacts, verdict: Verdict) -> str:
    signals = facts.flat_signals()
    findings = _nonempty_signals(signals)

    # The fired rules, not just the final score. Without them the model sees
    # raw facts and has to re-derive why the verdict is what it is — and it
    # derives it wrongly. On a real sample (sample-8611) the facts showed
    # dkim_result "pass" with dkim_domain ladelanoagency.com against a
    # from_domain of jwgmedia.com; the model read "DKIM pass" and wrote that
    # the message "güvenilir görünmektedir" inside a report headed Phishing.
    # The rule engine had in fact fired dkim_pass_but_domain_mismatch (+3)
    # for exactly that mismatch — the model just never saw it.
    #
    # This is not the model classifying: the verdict is still handed to it.
    # It is being told which evidence produced that verdict, so the prose can
    # cite the real reasons instead of inventing plausible ones.
    if verdict.matches:
        rules_block = "\n".join(
            f"  {m.weight:+d}  {m.description}" for m in verdict.matches)
    else:
        rules_block = "  (hiçbir kural tetiklenmedi)"

    return f"""KARAR: {verdict.verdict}
SKOR: {verdict.score}

BU KARARI ÜRETEN KURALLAR (raporun gerekçesi BUNLAR olmalı):
{rules_block}

TESPİT EDİLEN BULGULAR (ham veri — yukarıdaki kuralları desteklemek için):
{json.dumps(findings, ensure_ascii=False, indent=2)}

E-POSTA KONUSU: {facts.subject or "(konu yok)"}
GÖVDE (ilk 2000 karakter): {facts.body_text[:2000]}"""


def build_few_shot_example(facts: EmailFacts, verdict: Verdict, report: Report) -> tuple[str, str]:
    """Returns (user_prompt, assistant_json_response) for a hand-approved
    few-shot pair, used to seed the conversation before the real query."""
    user = build_user_prompt(facts, verdict)
    assistant = report.model_dump_json(indent=2)
    return user, assistant


def build_messages(
    facts: EmailFacts,
    verdict: Verdict,
    few_shot: list[tuple[EmailFacts, Verdict, Report]] | None = None,
) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for fs_facts, fs_verdict, fs_report in (few_shot or []):
        user, assistant = build_few_shot_example(fs_facts, fs_verdict, fs_report)
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": assistant})
    messages.append({"role": "user", "content": build_user_prompt(facts, verdict)})
    return messages
