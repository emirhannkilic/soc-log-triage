"""
T6 audit: classify each of the 1500 phishing_pot records in
data/processed/phishing_facts.jsonl as "phishing" (real attempt: brand
impersonation, credential/PII request, or fake urgency + malicious action
link) vs "spam" (commercial/unwanted mail with no such intent).

Heuristic, keyword-based classifier -- no LLM budget for 1500 individual
reviews. See holdout-fix-tasks.md T6 for the motivating problem: the raw
phishing_pot dataset mixes real credential phishing with plain commercial
spam (language courses, adult dating, detox products, TV subscriptions,
horoscopes, etc.), and credential_request alone has false positives on
spam that contains boilerplate "verify"/"confirm" template garbage.

Output: data/processed/phishing_facts_spam_audit.jsonl -- same records
plus "likely_spam": bool and "spam_reason": str.
"""
import json
import re
from collections import Counter

IN_PATH = "/Users/emir/Desktop/Dosyalar/CSE/Projects/soc-log-triage/data/processed/phishing_facts.jsonl"
OUT_PATH = "/Users/emir/Desktop/Dosyalar/CSE/Projects/soc-log-triage/data/processed/phishing_facts_spam_audit.jsonl"

# ---------------------------------------------------------------------------
# Known brands commonly impersonated in credential phishing. Used to detect
# brand impersonation: brand name mentioned in subject/display_name/body but
# from_domain does NOT belong to that brand.
# ---------------------------------------------------------------------------
BRANDS = {
    "microsoft": ["microsoft.com", "outlook.com", "live.com", "office.com", "office365.com", "msn.com"],
    "office365": ["microsoft.com", "outlook.com", "office.com", "office365.com"],
    "outlook": ["microsoft.com", "outlook.com", "live.com"],
    "apple": ["apple.com", "icloud.com"],
    "icloud": ["apple.com", "icloud.com"],
    "amazon": ["amazon.com", "amazon.co.uk", "amazon.de", "amazon.fr", "amazon.ca", "amazon.it", "amazon.es"],
    "paypal": ["paypal.com"],
    "netflix": ["netflix.com"],
    "google": ["google.com", "gmail.com"],
    "gmail": ["google.com", "gmail.com"],
    "facebook": ["facebook.com", "fb.com", "meta.com"],
    "instagram": ["instagram.com", "facebook.com", "meta.com"],
    "whatsapp": ["whatsapp.com", "facebook.com", "meta.com"],
    "bank of america": ["bankofamerica.com"],
    "wells fargo": ["wellsfargo.com"],
    "chase": ["chase.com", "jpmorgan.com"],
    "banco do brasil": ["bb.com.br"],
    "bradesco": ["bradesco.com.br"],
    "itau": ["itau.com.br"],
    "santander": ["santander.com", "santander.com.br"],
    "hsbc": ["hsbc.com"],
    "dhl": ["dhl.com"],
    "fedex": ["fedex.com"],
    "ups": ["ups.com"],
    "usps": ["usps.com"],
    "correios": ["correios.com.br"],
    "docusign": ["docusign.com", "docusign.net"],
    "adobe": ["adobe.com"],
    "linkedin": ["linkedin.com"],
    "coinbase": ["coinbase.com"],
    "binance": ["binance.com"],
    "ripple": ["ripple.com"],
    "coinbase.com": ["coinbase.com"],
    "protonmail": ["protonmail.com", "proton.me"],
    "proton mail": ["protonmail.com", "proton.me"],
    "chase bank": ["chase.com"],
    "irs": ["irs.gov"],
    "ebay": ["ebay.com"],
    "steam": ["steampowered.com", "steamcommunity.com"],
    "spotify": ["spotify.com"],
    "zoom": ["zoom.us"],
    "att": ["att.com"],
    "verizon": ["verizon.com"],
    "t-mobile": ["t-mobile.com"],
    "visa": ["visa.com"],
    "mastercard": ["mastercard.com"],
    "american express": ["americanexpress.com", "aexp.com"],
    "amex": ["americanexpress.com", "aexp.com"],
    "mcafee": ["mcafee.com"],
    "metamask": ["metamask.io"],
    "ledger": ["ledger.com"],
    "trust wallet": ["trustwallet.com"],
    "kraken": ["kraken.com"],
    "crypto.com": ["crypto.com"],
    "norton": ["norton.com", "nortonlifelock.com"],
    "irs": ["irs.gov"],
    "irs.gov": ["irs.gov"],
    "usps.com": ["usps.com"],
    "royal mail": ["royalmail.com"],
    "la poste": ["laposte.fr"],
    "correios": ["correios.com.br"],
    "mercado pago": ["mercadopago.com", "mercadopago.br", "mercadolibre.com"],
    "mercado libre": ["mercadolibre.com", "mercadopago.com"],
    "nubank": ["nubank.com.br"],
    "caixa": ["caixa.gov.br"],
}

# Common single-character homoglyph swaps used in typosquat display names
# (e.g. "lRS.GOV" using lowercase L for capital I, "0utlook" using zero for
# O). Normalize before brand matching to catch these.
_HOMOGLYPH_MAP = str.maketrans({
    "l": "i", "1": "i", "0": "o", "5": "s", "3": "e", "4": "a", "@": "a",
})


def normalize_homoglyphs(text):
    return text.lower().translate(_HOMOGLYPH_MAP)

# Commercial / spam-only signal words -- typical of unwanted marketing, not
# credential phishing. These alone do NOT decide anything; they're combined
# with the absence of phishing intent signals below.
SPAM_TOPIC_PATTERNS = [
    r"\bhorosc|astrolog|zodiac|tarot|voyance|psychic\b",
    r"\bviagra|cialis|erectile|e\.?d\.? (by age|treatment|pills)|libido|weight ?loss|detox|slim(ming)?|keto\b",
    r"\bdating|singles? (near|in) you|hot (girls|singles)|adult (content|dating)|xxx|milf|cam ?girl|ragazz|penis|breast selection\b",
    r"\bcasino|slots?|jackpot|free ?spins?|no deposit\b",
    r"\blanguage course|learn (spanish|english|german|french)|speak fluent\b",
    r"\bcooking class|recipe of the (day|week)|meal kit\b",
    r"\bcash ?back|swagbucks|earn money online|make money (fast|online)|work from home\b",
    r"\bnewsletter|unsubscribe|opt.?out|marketing preferences\b",
    r"\bmortgage rates?|refinance your (home|loan)|debt relief\b",
    r"\bpharmacy|prescription drugs? online\b",
    r"\bcbd oil|supplement|anti.?aging|wrinkle|muskelaufbau|muskelwachstum|anaboloxan\b",
    r"\bfree tv|streaming subscription|iptv\b",
    r"\bsolar panel|home warranty|car insurance quote\b",
    # Prize / voucher / survey scam -- "you've won X, claim your gift/bonus,
    # fill out survey" -- classic clickbait spam, no credential/PII/account
    # takeover attempt, treated as spam per T6 task definition.
    r"\byou('| ha)ve won\b|claim your (prize|bonus|gift|reward)\b|"
    r"gutschein|geschenk gewonnen|gewonnen!|hebt gewonnen|je hebt gewonnen|"
    r"congratulations you('| ha)ve been selected|selected to receive|"
    r"complete (this|our|the) (short )?survey|take (this|our) survey|"
    r"prijs gewonnen|noodpakket gewonnen\b",
]
SPAM_TOPIC_RE = re.compile("|".join(SPAM_TOPIC_PATTERNS), re.IGNORECASE)

# Advance-fee / 419 scam signals (still "phishing-adjacent" social engineering
# fraud, but not credential/brand phishing -- treat as its own phishing
# subtype since it IS a real attempted fraud with a specific victim ask).
SCAM_419_RE = re.compile(
    r"\b(next of kin|inheritance|beneficiary|deposit box|unclaimed fund|"
    r"foreign partner|dying wish|widow of|humanitarian gesture|"
    r"transfer (the|your) (fund|money)|million (dollars|usd|euros)\b.{0,40}\b(deposit|transfer|inherit))",
    re.IGNORECASE,
)

# Real credential / PII / account-security requests, tied to a specific
# service/account context (not generic "click here to learn more").
CREDENTIAL_PATTERNS = [
    r"\bverify your (account|identity|password|payment|billing)\b",
    r"\bconfirm your (account|identity|password|payment method|billing)\b",
    r"\b(account|payment) (has been )?(suspended|locked|limited|restricted|on hold)\b",
    r"\bunusual (sign.?in|login) activity\b",
    r"\bsuspicious (activity|login) detected\b",
    r"\b(someone|a user|new device) (tried to|just)?\s*log(ged)? ?in to your account\b",
    r"\bnew device.{0,20}logged into your\b",
    r"\bupdate your (payment|billing) (information|details|method)\b",
    r"\bre.?enter your (password|credentials|card details)\b",
    r"\byour password (will|is about to) expire\b",
    r"\benter your (pin|password|otp|verification code|security code)\b",
    r"\bconfirm your (card|bank) details\b",
    r"\bsign in to (avoid|prevent|restore)\b",
    r"\baction required.{0,30}account\b",
    r"\byour (account|mailbox) (will be|has been) (deactivated|deleted|closed)\b",
    r"\bclick to (unlock|reactivate|restore) your account\b",
    # Antivirus / security-software renewal scareware -- fake urgency to push
    # a payment/card-detail renewal for a subscription that (usually) was
    # never real. Real fraud attempt (payment/card data extraction), not spam.
    r"\b(schutz|protection|subscription|abonnement|antivirus) (ist |is )?(abgelaufen|expired)\b",
    r"\bger[äa]te (jetzt |now )?(ungeschützt|unprotected)\b",
    r"\brenew (your )?(subscription|protection|antivirus)\b",
    # Government/legal-notice scare phishing (PT-BR CNH/CPF suspension scams
    # and equivalents) -- fake official notice pressuring victim to act to
    # avoid suspension/fine/debt escalation. Real social-engineering fraud
    # attempt tied to a specific (fake) government/legal action, not spam.
    r"\b(suspens[ãa]o|suspensão) d[ae] (cnh|cpf|conta)\b",
    r"\bcpf (com status |está )?suspens",
    r"\bdívida ativa detectada\b",
    r"\bbloqueio iminente\b",
    r"\bregularize (a )?(sua )?situa[çc][ãa]o\b",
    r"\bprocesso de suspens[ãa]o\b",
]
CREDENTIAL_RE = re.compile("|".join(CREDENTIAL_PATTERNS), re.IGNORECASE)

# Generic marketing CTAs that often trip the naive credential_request boolean
# via boilerplate "confirm"/"verify" fragments (e.g. "confirm your
# subscription to our newsletter", "click here to unsubscribe").
BOILERPLATE_CONFIRM_RE = re.compile(
    r"\bconfirm your (subscription|newsletter|email address to (continue receiving|subscribe))\b|"
    r"\bverify (you'?re human|you are human|your (age|interest))\b|"
    r"\bconfirm your registration\b|"
    r"\bclick (here|below) to (activate your account|confirm your (inscription|subscription))\b",
    re.IGNORECASE,
)

FAKE_URGENCY_WORDS = {
    "24 hours", "expire", "expires", "expiring", "immediately", "urgent",
    "final notice", "last chance", "act now", "suspended", "locked",
    "unusual activity", "verify now", "restricted",
}


def domain_root(d):
    if not d:
        return None
    parts = d.lower().split(".")
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return d.lower()


def detect_brand_impersonation(subject, display_name, body, from_domain):
    # Only check subject + display_name (clean, structured fields) -- body_text
    # in this dataset is frequently a concatenation of unrelated template
    # fragments (mojibake, merged multi-template scrapes), so substring
    # matches against raw body produce false brand hits (e.g. "steam" inside
    # garbled non-Latin byte soup). Word-boundary match only.
    text = f"{subject or ''} {display_name or ''}".lower()
    text_norm = normalize_homoglyphs(text)
    from_root = domain_root(from_domain)
    hits = []
    for brand, legit_domains in BRANDS.items():
        brand_norm = normalize_homoglyphs(brand)
        pattern = r"\b" + re.escape(brand) + r"\b"
        pattern_norm = r"\b" + re.escape(brand_norm) + r"\b"
        if re.search(pattern, text) or re.search(pattern_norm, text_norm):
            legit_roots = {domain_root(d) for d in legit_domains}
            if from_root not in legit_roots:
                hits.append(brand)
    return hits


def classify(rec):
    subject = rec.get("subject") or ""
    body = rec.get("body_text") or ""
    display_name = rec.get("display_name") or ""
    from_domain = rec.get("from_domain") or ""
    urls = rec.get("urls") or []
    claims_attachment = rec.get("claims_attachment", False)
    urgency_kw = rec.get("urgency_keywords") or []
    urgency_words = {u.get("keyword", "").lower() for u in urgency_kw if isinstance(u, dict)}

    # combined_head: subject + display_name, clean structured fields.
    # combined_body_start: only the first ~400 chars of body_text, where the
    # email's own actual opening line/greeting lives, before phishing_pot's
    # known cross-template boilerplate concatenation kicks in further down.
    combined_head = f"{subject} {display_name}"
    combined_body_start = body[:400]
    combined_full = f"{subject} {display_name} {body[:2000]}"

    # Check subject+display_name first; if empty/no hit, also scan the first
    # ~400 chars of body (covers cases like display_name=None but body opens
    # with "Facebook\n Hi ..., Someone tried to log in...").
    brand_hits = detect_brand_impersonation(subject, display_name, "", from_domain)
    if not brand_hits:
        brand_hits = detect_brand_impersonation(body[:400], "", "", from_domain)

    # Spam-topic check: subject/display_name first (reliable), fall back to
    # body start only if head is empty/uninformative.
    has_spam_topic = bool(SPAM_TOPIC_RE.search(combined_head)) or bool(
        SPAM_TOPIC_RE.search(combined_body_start)
    )

    # Credential ask: require the pattern to appear in subject/display_name
    # OR in the body's opening section -- NOT the full 2000-char body, which
    # is where unrelated boilerplate fragments ("password will expire",
    # "click here to verify") from merged/scraped templates live in this
    # dataset (see CLAUDE.md T1 note on parser encoding/structure issues).
    has_real_credential_ask = (
        bool(CREDENTIAL_RE.search(combined_head))
        or bool(CREDENTIAL_RE.search(combined_body_start))
        or bool(CREDENTIAL_RE.search(normalize_homoglyphs(combined_head)))
    )
    has_boilerplate_confirm = bool(BOILERPLATE_CONFIRM_RE.search(combined_full))
    has_419_scam = bool(SCAM_419_RE.search(combined_full))

    real_urgency = urgency_words & FAKE_URGENCY_WORDS
    has_action_link = len(urls) > 0
    has_fake_urgency_and_link = bool(real_urgency) and has_action_link and not has_spam_topic

    # --- Decision logic -----------------------------------------------
    # 1. Brand impersonation: strong phishing signal on its own.
    if brand_hits:
        reason = f"brand impersonation: '{brand_hits[0]}' mentioned but from_domain={from_domain!r} not owned by brand"
        return False, reason

    # 2. Real credential/account-security ask tied to specific context,
    #    not just boilerplate marketing confirm language.
    if has_real_credential_ask and not (has_spam_topic and not brand_hits):
        reason = "credential/account-security request pattern matched (verify/confirm/suspended/PIN/OTP tied to account)"
        return False, reason

    # 3. 419 / advance-fee fraud: real fraud attempt, not spam.
    if has_419_scam:
        reason = "advance-fee/419 fraud pattern (inheritance, next of kin, fund transfer request)"
        return False, reason

    # 4. Fake urgency + action link, no spam topic override.
    if has_fake_urgency_and_link and claims_attachment is not None and not has_spam_topic:
        reason = f"fake urgency keyword(s) {sorted(real_urgency)} combined with action link, no commercial topic"
        return False, reason

    # 5. Spam topic detected and none of the above phishing signals fired.
    if has_spam_topic:
        matched = SPAM_TOPIC_RE.search(combined_full)
        reason = f"commercial/unwanted topic matched ({matched.group(0)!r}), no brand impersonation or real credential ask"
        return True, reason

    # 6. credential_request flag true but only boilerplate confirm text found
    #    (the known false-positive pattern) -> spam.
    if rec.get("credential_request") and has_boilerplate_confirm and not has_real_credential_ask and not brand_hits:
        reason = "credential_request=True but only boilerplate subscribe/newsletter confirm language found, no brand or real account context"
        return True, reason

    # 7. credential_request flag true, no brand, no real pattern matched,
    #    no spam topic either -- ambiguous, lean spam since no phishing
    #    intent signal fired (conservative: don't over-count phishing).
    if rec.get("credential_request") and not brand_hits and not has_real_credential_ask:
        reason = "credential_request=True but no brand impersonation or specific account-security pattern found; likely template artifact"
        return True, reason

    # 8. No credential_request, no urgency, no brand, no spam topic keyword
    #    match -- fall back to generic marketing heuristics (unsubscribe /
    #    promotional tone words) vs default.
    generic_marketing_re = re.compile(
        r"\bunsubscribe|% off|discount code|promo code|limited time offer|"
        r"free shipping|shop now|new arrivals|flash sale\b", re.IGNORECASE)
    if generic_marketing_re.search(combined_full) and not brand_hits and not has_real_credential_ask:
        reason = "generic marketing/promotional language, no brand impersonation or credential ask"
        return True, reason

    # 9. Default: if nothing matched at all (no urls, no urgency, no
    #    credential ask, no spam keywords) -- likely a low-signal spam/junk
    #    or 419-style message we didn't pattern-match; be conservative and
    #    mark as phishing only if there's at least an action link or
    #    attachment claim (still could be malware/phish), else spam.
    if not has_action_link and not claims_attachment and not urgency_words:
        reason = "no urls, no attachment claim, no urgency keywords, no brand/credential signal -- likely low-signal spam/scam text"
        return True, reason

    reason = "no strong spam-topic match but also no confirmed brand/credential/urgency phishing signal -- default phishing (has link/attachment/urgency)"
    return False, reason


def main():
    total = 0
    spam_count = 0
    phish_count = 0
    domain_spam_counter = Counter()
    domain_total_counter = Counter()

    with open(IN_PATH) as fin, open(OUT_PATH, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            total += 1
            is_spam, reason = classify(rec)
            rec["likely_spam"] = is_spam
            rec["spam_reason"] = reason
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

            fd = rec.get("from_domain") or "(none)"
            domain_total_counter[fd] += 1
            if is_spam:
                spam_count += 1
                domain_spam_counter[fd] += 1
            else:
                phish_count += 1

    print(f"Total records: {total}")
    print(f"Classified spam: {spam_count} ({spam_count/total:.1%})")
    print(f"Classified phishing: {phish_count} ({phish_count/total:.1%})")
    print()
    print("Top 20 spam-heavy from_domains (domain -> spam_count / total_count):")
    for dom, cnt in domain_spam_counter.most_common(20):
        print(f"  {dom:40s} {cnt:4d} / {domain_total_counter[dom]:4d}")


if __name__ == "__main__":
    main()
