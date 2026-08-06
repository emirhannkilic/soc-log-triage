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

Third example added 2026-08-06 after manual prompt-review testing
(scripts/sample_prompt_review_set.py output) showed onerilen_aksiyon
copying FEW_SHOT_PHISHING's literal "silin/tıklamayın/engelleyin" wording
into Muhtemel Phishing reports even after two rounds of system-prompt
rules explicitly forbidding it (see PROGRESS.md, "Aşk Biter mi? Zorlu
PSM'de" test, sample data/raw/gmail/eml/inbox-7348.eml, reproduced twice
verbatim at temperature=0). Two text-only fixes both failed — the model
was imitating the only concrete onerilen_aksiyon shape it had ever seen,
regardless of what the surrounding instruction said. A genuine Muhtemel
Phishing example was needed to give it something else to imitate.

FEW_SHOT_MUHTEMEL is data/raw/gmail/eml/inbox-7240.eml (real Apple
purchase receipt, Türkçe) — chosen because it is a clean case of the
band's intended meaning: every auth/domain field is genuinely consistent
(from_domain, dkim_domain, return_path_domain, message_id_domain all
email.apple.com, SPF+DKIM+DMARC pass), but the rule engine still lands on
score 3 because credential_request fires on boilerplate "parola
tercihleri" support text and has_hidden_text fires on accessibility
markup, combined with one URL carrying a redirect-style query param
(unsupportedRedirectUrl=). Nothing here is a "kaçırılan phishing" the
model needs to escalate — it is the system correctly declining to decide
on a genuinely ambiguous, actually-benign signal combination. Not drawn
from data/holdout/ (never touched for calibration, see CLAUDE.md) or
data/training/pairs.jsonl (already used for LoRA training) or
data/prompt_review/sample.jsonl (already in the manual review pool).
"""
from schemas.report import Report

# Not a hold-out index (unlike Candidates 1/20, which come from
# data/holdout/candidates.jsonl via FEW_SHOT_INDICES in each caller) —
# this one is looked up by raw .eml path instead, since it isn't part of
# that file. Callers that build the (facts, verdict, report) triple at
# runtime (src/demo.py, src/eval/baseline.py, src/eval/finetuned.py,
# src/teacher/generate.py, src/teacher/generate_training_data.py) parse
# this path directly.
FEW_SHOT_MUHTEMEL_EML_PATH = "data/raw/gmail/eml/inbox-7240.eml"

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

# data/raw/gmail/eml/inbox-7240.eml
# Verdict: Muhtemel Phishing (score 3) —
# credential_request_with_external_link(2), url_redirect_param(2) tetiklendi
# ama SPF+DKIM+DMARC hepsi pass VE from_domain/dkim_domain/return_path_domain/
# message_id_domain hepsi email.apple.com — bonus henüz devreye girmiyor
# çünkü ayrı sinyaller kendi başlarına puanlanmaya devam ediyor
# (CLAUDE.md "all_auth_pass_and_consistent return-path/reply-to şartı
# taşımaz"). Not drawn from data/holdout/ or data/training/pairs.jsonl —
# see module docstring for the full selection rationale.
#
# "Abstention" case: the band exists so the system can decline to decide
# on a genuinely ambiguous, plausibly-benign signal combination — this is
# the example that shows onerilen_aksiyon must route to a human, not
# repeat the certain "silin" wording FEW_SHOT_PHISHING uses.
FEW_SHOT_MUHTEMEL = Report(
    risk_seviyesi="Muhtemel Phishing",
    sonuc_ve_gerekce=(
        "Bu karar; kullanıcıyı işlem yapmaya yönlendirme kategorisinde "
        "birden fazla zayıf sinyalin, tutarlı kimlik doğrulama "
        "sonuçlarıyla birlikte değerlendirilmesine dayanır."
    ),
    genel_degerlendirme=(
        "Olası senaryo: bir hizmet sağlayıcıdan gelen fatura/makbuz "
        "bildirimi, kötü niyetli bir senaryo kesin olarak doğrulanamadı "
        "ama dışlanamadı da. "
        "Alıcıdan beklenen eylem: Mevcut bulgulardan belirlenemiyor. "
        "Olası zarar: Mevcut bulgulardan belirlenemiyor."
    ),
    teknik_bulgular=[
        {
            "baslik": "Kimlik bilgisi/hesap yönetimiyle ilgili metin ve dış link",
            "aciklama": (
                "Gövde, parola/hesap ayarları yönetimine atıfta bulunan "
                "metin ve dışarıya giden linkler içeriyor — bu ikisinin "
                "birlikte görülmesi genellikle kimlik bilgisi hedefleyen "
                "mailerde de rastlanan bir örüntü, tek başına yeterli "
                "kanıt değil."
            ),
        },
        {
            "baslik": "URL'de yönlendirme parametresi",
            "aciklama": (
                "Linklerden biri, tıklandığında başka bir hedefe "
                "yönlendirebilecek bir sorgu parametresi taşıyor — bu "
                "parametre gerçek hizmetlerde de (ör. giriş sonrası "
                "orijinal sayfaya dönüş) kullanılır, açık istismar "
                "kanıtı değil."
            ),
        },
        {
            "baslik": "Kimlik doğrulama sonuçları tutarlı",
            "aciklama": (
                "SPF, DKIM ve DMARC hepsi 'pass' ve imzalayan domain "
                "From domain'i ile uyumlu — bu, yukarıdaki iki sinyalin "
                "ağırlığını azaltan, ama tek başına kararı belirlemeyen "
                "bir bulgu."
            ),
        },
    ],
    phishing_gostergeleri=[
        "Parola/hesap yönetimiyle ilgili metin ve dış link birlikte "
        "görülüyor, ama kimlik doğrulama sonuçları tutarlı",
        "Bir linkte yönlendirme parametresi var, ama açık bir istismar "
        "kanıtı yok",
    ],
    onerilen_aksiyon=(
        "Otomatik karar verilemedi; bir SOC analisti gönderen ve "
        "bağlantıları manuel olarak incelemeden e-postayla etkileşime "
        "girilmemesi önerilir."
    ),
)
