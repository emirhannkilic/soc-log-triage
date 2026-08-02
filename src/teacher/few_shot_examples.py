"""
Hand-written few-shot examples for the teacher prompt (v3 plan §6.1: "3
örnek: 1 phishing, 1 muhtemel, 1 güvenilir, elle yazılmış JSON çıktılarıyla").

These 3 hold-out candidates (indices 1, 8, 20 in data/holdout/review.md,
1-indexed) are EXCLUDED from the Adım 6 smoke-test set precisely because
they're used here — the teacher must not be shown its own test data as a
worked example. See scripts/generate_teacher_smoke_test.py for the smoke
test set (the other 27 candidates, first 20).
"""
from schemas.report import Report

# Candidate 1: data/phishing_pot/email/sample-1299.eml
# Verdict: Phishing (score 7) — dkim_missing_or_fail_domain_mismatch(2),
# return_path_mismatch(2), credential_request_with_external_link(2),
# urgency_keywords(1)
FEW_SHOT_PHISHING = Report(
    risk_seviyesi="Phishing",
    sonuc_ve_gerekce=(
        "Bu e-posta Microsoft hesap güvenliği bildirimi gibi görünse de "
        "gönderen domain'i (access-accsecurity.com) Microsoft'a ait değil "
        "ve DKIM doğrulaması yok. Return-Path adresi de From adresinden "
        "farklı bir domain'e (cumvxope.servifans.com) ait — bu, mailin "
        "gerçek Microsoft altyapısından gönderilmediğinin güçlü bir "
        "göstergesi."
    ),
    genel_degerlendirme=(
        "E-posta, alıcıyı 'olağandışı oturum açma' iddiasıyla aciliyet "
        "hissi yaratıp bir linke tıklamaya yönlendiriyor. DKIM "
        "doğrulaması bulunmuyor ve Return-Path domain'i From adresiyle "
        "uyuşmuyor, bu da gönderenin kimliğinin taklit edildiğini "
        "gösteriyor. E-postada bir kimlik doğrulama linki var ve mesaj "
        "aciliyet dili kullanıyor. Toplamda bu bulgular, hesabı ele "
        "geçirmeye yönelik bir kimlik avı girişimiyle tutarlı."
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
        "Gönderen domain'i Microsoft'a ait değil, DKIM doğrulaması yok",
        "Return-Path domain'i From adresiyle uyuşmuyor",
        "Aciliyet hissi yaratan dil ve dış link kombinasyonu",
    ],
    onerilen_aksiyon="E-postayı silin, linke tıklamayın, gönderen adresi engelleyin.",
)

# Candidate 8: data/phishing_pot/email/sample-2140.eml
# Verdict: Muhtemel Phishing (score 3) — dkim_missing_or_fail_domain_mismatch(2),
# urgency_keywords(1)
FEW_SHOT_MUHTEMEL = Report(
    risk_seviyesi="Muhtemel Phishing",
    sonuc_ve_gerekce=(
        "E-posta Trust Wallet marka adını kullanıyor ve DKIM doğrulaması "
        "bulunmuyor, ayrıca hesap askıya alınma tehdidiyle aciliyet "
        "yaratıyor. Ancak From ve Return-Path domain'leri tutarlı "
        "(support-trustwallet.com) ve URL'lerde belirgin bir marka "
        "taklidi/domain uyuşmazlığı tespit edilmedi — bu yüzden kesin "
        "bir phishing kararı için yeterli güçte sinyal yok."
    ),
    genel_degerlendirme=(
        "E-posta 'hesabınız doğrulanmadı, aksi halde askıya alınacak' "
        "tarzı bir aciliyet mesajı içeriyor ve DKIM imzası yok. Bununla "
        "birlikte gönderen domain'i ile Return-Path domain'i tutarlı, "
        "URL'lerde IP tabanlı adres veya punycode gibi güçlü teknik "
        "sahtekarlık sinyalleri bulunmuyor. Bulgular orta düzeyde "
        "şüpheli, kesin sınıflandırma için yeterli değil."
    ),
    teknik_bulgular=[
        {
            "baslik": "DKIM doğrulaması yok",
            "aciklama": "DKIM sonucu 'none' ve dkim_domain from_domain ile uyuşmuyor.",
        },
        {
            "baslik": "Aciliyet dili",
            "aciklama": "Mesaj, hesabın askıya alınacağı tehdidiyle acil aksiyon bekletiyor.",
        },
    ],
    phishing_gostergeleri=[
        "DKIM doğrulaması eksik",
        "Hesap askıya alma tehdidiyle aciliyet yaratan dil",
    ],
    onerilen_aksiyon=(
        "E-postadaki linklere tıklamadan önce SOC analistine iletin, "
        "hesabınızı resmi uygulama üzerinden doğrudan kontrol edin."
    ),
)

# Candidate 20: data/raw/gmail/eml/inbox-748.eml
# Verdict: Güvenilir (score 2) — return_path_mismatch(2)
FEW_SHOT_GUVENILIR = Report(
    risk_seviyesi="Güvenilir",
    sonuc_ve_gerekce=(
        "E-posta, Golden Goose marka adına uygun bir domain'den "
        "(mailer.goldengoose.com) gönderilmiş, DKIM doğrulaması geçiyor "
        "ve dkim_domain from_domain ile uyumlu. Return-Path domain'i "
        "(bounce.mailer.goldengoose.com) From domain'inin bir alt "
        "domain'i olarak görünüyor — bu, kurumsal toplu mail "
        "altyapılarında (bounce/tracking subdomain'leri) sık görülen, "
        "zararsız bir yapı."
    ),
    genel_degerlendirme=(
        "E-posta bir pazarlama/tanıtım bülteni niteliğinde, kimlik "
        "bilgisi talebi veya aciliyet dili içermiyor. DKIM doğrulaması "
        "başarılı ve gönderen domain'i marka adıyla tutarlı. "
        "Return-Path'in From'dan farklı bir alt domain olması, kurumsal "
        "mail gönderim servislerinde (bounce/tracking subdomain'i) "
        "standart bir uygulamadır ve tek başına şüpheli değildir."
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
