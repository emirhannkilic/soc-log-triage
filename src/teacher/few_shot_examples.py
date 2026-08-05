"""
Hand-written few-shot examples for the teacher prompt (v3 plan §6.1).

Rewritten 2026-08-05 after 5 prompt-only attempts failed to stop
sonuc_ve_gerekce/genel_degerlendirme from duplicating each other (see
PROGRESS.md "rapor bölümlerinin içerik derinliği"). External review
(Codex) diagnosed that at temperature=0 the model deterministically
repeats one wrong pattern regardless of instruction wording, and that
these examples — still in the OLD free-form format — were reinforcing
exactly the pattern the system prompt was trying to forbid. Rewritten to
match the new fixed-template rules literally: sonuc_ve_gerekce is one
sentence naming only categories from the allowed list, genel_degerlendirme
is exactly three sentences (scenario / expected action / harm) with zero
technical terms.

Two examples only (rich + poor), per Codex's advice: showing a "wrong"
example alongside a "right" one risks a small model imitating the wrong
one's shape, so only positive examples are shown.

Candidate 1 (data/phishing_pot/email/sample-1299.eml) and Candidate 20
(data/raw/gmail/eml/inbox-748.eml) are indices 1 and 20 in
data/holdout/review.md (1-indexed) — EXCLUDED from the Adım 6 smoke-test
set precisely because they're used here. See
scripts/generate_teacher_smoke_test.py for the smoke test set.
"""
from schemas.report import Report

# Candidate 1: data/phishing_pot/email/sample-1299.eml
# Verdict: Phishing (score 7) — dkim_missing_or_fail_domain_mismatch(2),
# return_path_mismatch(2), credential_request_with_external_link(2),
# urgency_keywords(1)
# "Rich" case: multiple independent signal categories fire together.
FEW_SHOT_PHISHING = Report(
    risk_seviyesi="Phishing",
    sonuc_ve_gerekce=(
        "Bu karar; kimlik ve marka taklidi, kimlik doğrulama uyumsuzluğu "
        "ve aciliyet ve baskı kategorilerinin birlikte değerlendirilmesine "
        "dayanır."
    ),
    genel_degerlendirme=(
        "Olası senaryo: Microsoft hesap güvenliği bildirimi kılığında "
        "kimlik bilgisi çalmaya yönelik bir kimlik avı girişimi. "
        "Alıcıdan beklenen eylem: e-postadaki doğrulama linkine tıklayıp "
        "hesap bilgilerini girmesi. "
        "Olası zarar: hesabın ele geçirilmesi ve bu hesap üzerinden "
        "başka kaynaklara yetkisiz erişim sağlanması."
    ),
    teknik_bulgular=[
        {
            "baslik": "DKIM doğrulaması yok",
            "aciklama": (
                "DKIM sonucu 'none' ve dkim_domain from_domain ile "
                "uyuşmuyor — gönderen domain'in kimliği kriptografik "
                "olarak doğrulanamıyor."
            ),
        },
        {
            "baslik": "Return-Path uyumsuzluğu",
            "aciklama": (
                "Return-Path domain'i (cumvxope.servifans.com) From "
                "domain'inden (access-accsecurity.com) farklı — bu, "
                "yanıtların/bounce'ların gönderenin iddia ettiği "
                "domain'den farklı bir yere gittiği anlamına gelir."
            ),
        },
        {
            "baslik": "Kimlik bilgisi talebi ve dış link",
            "aciklama": (
                "E-posta bir doğrulama/oturum linki içeriyor ve kullanıcıyı "
                "bu linke tıklamaya yönlendiriyor."
            ),
        },
        {
            "baslik": "Aciliyet dili",
            "aciklama": "Mesaj, 'olağandışı aktivite' vurgusuyla acil aksiyon bekletiyor.",
        },
    ],
    phishing_gostergeleri=[
        "E-posta example-bank.com benzeri bir Microsoft alan adından "
        "geliyormuş gibi görünüyor ama DKIM ile doğrulanmış bir imza yok",
        "Return-Path domain'i (cumvxope.servifans.com) From domain'inden "
        "(access-accsecurity.com) farklı",
        "Aciliyet hissi yaratan dil ve dış link kombinasyonu",
    ],
    onerilen_aksiyon="E-postayı silin, linke tıklamayın, gönderen adresi engelleyin.",
)

# Candidate 20: data/raw/gmail/eml/inbox-748.eml
# Verdict: Güvenilir (score 2) — return_path_mismatch(2)
# "Poor" case: a single, weak, ultimately benign signal — the example the
# model most needs to see, since it's the one where it's tempted to
# invent extra findings to fill space.
FEW_SHOT_GUVENILIR = Report(
    risk_seviyesi="Güvenilir",
    sonuc_ve_gerekce=(
        "Bu karar; kimlik doğrulama uyumsuzluğu kategorisinde tek ve "
        "zayıf bir sinyalin, güçlü kimlik doğrulama sonuçlarıyla birlikte "
        "değerlendirilmesine dayanır."
    ),
    genel_degerlendirme=(
        "Olası senaryo: kurumsal bir pazarlama/tanıtım bülteni, kötü "
        "niyetli bir senaryo tespit edilmedi. "
        "Alıcıdan beklenen eylem: Mevcut bulgulardan belirlenemiyor. "
        "Olası zarar: Mevcut bulgulardan belirlenemiyor."
    ),
    teknik_bulgular=[
        {
            "baslik": "DKIM doğrulaması başarılı",
            "aciklama": "DKIM sonucu 'pass' ve dkim_domain from_domain ile uyumlu.",
        },
        {
            "baslik": "Return-Path alt domain farkı",
            "aciklama": (
                "Return-Path (bounce.mailer.goldengoose.com), From domain'inin "
                "(mailer.goldengoose.com) bir alt domain'i — kurumsal toplu "
                "mail altyapılarında yaygın, zararsız bir örüntü."
            ),
        },
    ],
    phishing_gostergeleri=[],
    onerilen_aksiyon="Ek bir aksiyon gerekmiyor.",
)
