"""
Extracts body signal facts (v3 plan section 4.6) and meta facts (4.7) from
an email.message.Message.
"""
import re
from email.message import Message

from bs4 import BeautifulSoup

from src.parser.urls import _domain_of_url

BODY_TEXT_MAX_CHARS = 2000

# holdout-fix-tasks.md T2: keeping the two languages as separate lists so
# they can be gated by detected language — mixing them caused unrelated
# German/French spam to pick up Turkish "acil"/"hemen" hits via substring
# collisions (see _URGENCY_PATTERNS below for the actual matching fix).
_URGENCY_KEYWORDS_TR = {
    "acil", "hemen", "derhal", "son gün", "hesabınız askıya", "doğrulayın",
    "onaylayın", "şifrenizi", "kilitlenecek", "iptal edilecek", "24 saat",
    "hemen tıklayın", "şimdi harekete geçin",
}
_URGENCY_KEYWORDS_EN = {
    "urgent", "immediately", "act now", "verify your account", "suspended",
    "expire", "click here", "confirm your", "password will", "24 hours",
    "limited time", "final notice",
}

# Word-boundary patterns per keyword. Multi-word phrases ("click here") are
# naturally bounded by their own spaces plus \b at each end. Single stems
# that legitimately take suffixes (Turkish "acil" -> "acilen") get an
# explicit \w* rather than relying on \b, which Python treats Turkish
# letters (ş, ı, ğ, ü, ö, ç) as non-word characters for and so fails to
# find a boundary after suffixed forms — see src/parser/body.py's existing
# _CREDENTIAL_PATTERNS comment for the same underlying issue.
_STEM_KEYWORDS = {"acil", "hemen"}


def _keyword_pattern(keyword: str) -> re.Pattern:
    if keyword in _STEM_KEYWORDS:
        return re.compile(rf"\b{re.escape(keyword)}\w*", re.IGNORECASE)
    return re.compile(rf"\b{re.escape(keyword)}\b", re.IGNORECASE)


_URGENCY_PATTERNS_TR = {kw: _keyword_pattern(kw) for kw in _URGENCY_KEYWORDS_TR}
_URGENCY_PATTERNS_EN = {kw: _keyword_pattern(kw) for kw in _URGENCY_KEYWORDS_EN}

_CONTEXT_CHARS = 40

# holdout-fix-tasks.md T3: a single keyword hit isn't enough — the parser
# was flagging candidate 29's real Apple notification (body literally warns
# "we will never ask for your password") and candidate 20's Bosch order
# confirmation (body says "we never ask for your password by email" as a
# security notice) as credential_request=True purely because "şifre"/
# "password" appeared somewhere in the text, regardless of who's asking
# what of whom. It also MISSED candidate 15's real credential phishing
# ("kindly re-login with the attachment...") because "re-login"/"mailbox"
# weren't covered by the old target-object-only word list.
#
# Fixed by requiring three things to co-occur in the same message:
#   1. a REQUEST VERB directed at the reader (doğrula/onayla/giriş yap/
#      güncelle/gir, verify/confirm/re-login/sign in/update/enter)
#   2. a TARGET OBJECT (şifre/parola/hesap/kart/TC kimlik/kullanıcı adı/PIN,
#      password/credentials/account/card)
#   3. an ACTION CHANNEL: an external link, an attachment, or an HTML form
#      in the same message (already computed above as urls/attachments/
#      has_html_form)
# A target-object-only mention (candidate 29/20's security notices) no
# longer flags, because there's no request verb directed at the reader.
# Precision is prioritized over recall per T3's explicit instruction.
_CREDENTIAL_REQUEST_VERB_RE = re.compile(
    r"\b("
    r"doğrula\w*|onayla\w*|giri[şs]\s*yap\w*|güncelle\w*|gir(?:iniz|in)?|"
    r"re-?login|sign[\s-]?in|verify|confirm|update|enter|log\s*in|log[\s-]?in\s*again"
    r")\b",
    re.IGNORECASE,
)
_CREDENTIAL_TARGET_OBJECT_RE = re.compile(
    r"\b("
    r"şifre\w*|parola\w*|hesab\w*|kart\s*numaras[ıi]|kullan[ıi]c[ıi]\s*ad[ıi]|"
    r"pin|iban|T\.?C\.?\s*kimlik|hesap\s*numaras[ıi]|mailbox|"
    r"password|credentials?|account|card\s*number|cvv|social\s*security|"
    r"username"
    r")\b",
    re.IGNORECASE,
)

# Codex'e danışıldı 2026-08-06 (Rule Engine v2, credential bağlam
# penceresi düzeltmesi). Eski mantık, verb ve target object'i TÜM
# body_text üzerinde ayrı ayrı .search() ediyordu — ikisi birbirinden
# çok uzakta, alakasız iki cümlede geçse bile eşleşiyordu. Somut
# örnek: data/raw/gmail/eml/inbox-4742.eml (Amazon sipariş bildirimi)
# "Lütfen ödeme aracınızı güncelleyin" (index 1720) ile "Hesabım"
# (index 82, navigasyon menüsü linki) 1638 karakter uzakta, tamamen
# ilgisiz iki öğe — ama eski mantık credential_request=True üretiyordu.
# Düzeltme: verb ve target object AYNI PENCEREDE (±100 karakter) olmalı.
_CREDENTIAL_CONTEXT_WINDOW_CHARS = 100

# "asla parolanızı istemeyiz" tarzı güvenlik uyarıları verb+target
# object'i aynı cümlede taşıyabilir (T3'ün candidate 29/20 örnekleri
# bunun bir varyasyonu, o zaman "verb yok" mantığıyla düzeltilmişti —
# ama "never ask"/"always verify" gibi kalıplar READER'a değil
# GÖNDERENİN kendi politikasına atıfta bulunduğu için verb+target aynı
# pencerede de olabilir). Negasyon kelimesi verb'e yakınsa (aynı
# pencere) bu bir TALEP değil bir UYARI/POLİTİKA beyanıdır.
_CREDENTIAL_NEGATION_RE = re.compile(
    r"\b("
    r"asla|hiçbir\s*zaman|kesinlikle\s*istemeyiz|"
    r"never|will\s*not|won'?t|do\s*not\s*need\s*to"
    r")\b",
    re.IGNORECASE,
)


def _credential_request_in_context(text: str) -> bool:
    """True only if a request verb and a target object co-occur within
    _CREDENTIAL_CONTEXT_WINDOW_CHARS of each other, AND that window
    isn't a negation ("we will never ask for your password").

    The negation check widens the window by _CREDENTIAL_CONTEXT_WINDOW_CHARS
    on EACH side (not just between verb and target) — "We will NEVER ask
    you to verify your account" puts "never" before both matches, outside
    a verb-to-target-only span."""
    verb_matches = list(_CREDENTIAL_REQUEST_VERB_RE.finditer(text))
    target_matches = list(_CREDENTIAL_TARGET_OBJECT_RE.finditer(text))
    for v in verb_matches:
        for t in target_matches:
            distance = max(v.start(), t.start()) - min(v.end(), t.end())
            if distance > _CREDENTIAL_CONTEXT_WINDOW_CHARS:
                continue
            window_start = max(0, min(v.start(), t.start()) - _CREDENTIAL_CONTEXT_WINDOW_CHARS)
            window_end = min(len(text), max(v.end(), t.end()) + _CREDENTIAL_CONTEXT_WINDOW_CHARS)
            window = text[window_start:window_end]
            if _CREDENTIAL_NEGATION_RE.search(window):
                continue
            return True
    return False

# holdout-fix-tasks.md T5: candidate 15 ("Attached Re-login") has an empty
# attachments list but its body claims one exists — the "attachment" is
# actually a link, a social-engineering pattern in its own right (promising
# a file that doesn't exist). Deliberately narrow: matches the noun, not
# verbs like "attach" alone, to avoid firing on unrelated usage ("please
# see attached for context" in a legitimate reply chain still mentions the
# noun, but we're only claiming "body references an attachment" — the
# rule engine decides what to do with that combined with attachments=[]).
_ATTACHMENT_CLAIM_RE = re.compile(
    r"\b(attach(?:ed|ment)?|ekte|ek\s*olarak|ekli\s*dosya)\b",
    re.IGNORECASE,
)

# Rule Engine v2 adım 8 (CLAUDE.md), 2026-08-08. Two narrow, idiom-level
# scam-language patterns — found on data/rule_engine_v2_devset's 16
# phishing candidates the family formula was still missing entirely (0
# signals across all four families): 419/advance-fee fraud and fake
# crypto/lottery reward claims share a defining trait none of this
# project's other signals cover — they carry ZERO header misalignment
# and often zero URLs/attachments (sample-4784.eml: gmail.com, SPF/DKIM/
# DMARC all pass, 0 URLs, 0 attachments, body is pure social-engineering
# prose). No header/URL signal can ever catch this class; only body text
# can.
#
# Deliberately NOT built like _credential_request_in_context's generic
# verb+target-object+window — these are idiom-level phrases ("outstanding
# principal", "next of kin", "claim your tokens"), not generic words that
# need a context window to disambiguate. A single regex match on a
# specific multi-word phrase already IS the disambiguation; adding a
# window would only widen false-positive surface for no precision gain.
# Measured on data/rule_engine_v2_devset (100 mail): 4/50 phishing hits,
# 0/50 legitimate false positives — see config/rules.yaml's weight
# rationale for why this alone doesn't cross to Phishing.
_ADVANCE_FEE_FRAUD_RE = re.compile(
    r"\b("
    r"beneficiary|outstanding\s+(fund|principal|balance)|deposited\s+fund|"
    r"scammed?\s+victims?|next\s+of\s+kin|unclaimed\s+(fund|inheritance)|"
    r"compensation\s+(fund|payment)"
    r")\b",
    re.IGNORECASE,
)
_FAKE_REWARD_CLAIM_RE = re.compile(
    r"\b("
    r"claim\s+(your\s+)?(share|tokens?|reward|prize|allotment)|"
    r"token\s+(redistribution|allocation)|reward\s+program|"
    r"you\s*(’|')?ve\s+won|lottery\s+winner|winning\s+notification|"
    r"selected\s+as\s+a?\s*winner"
    r")\b",
    re.IGNORECASE,
)

# Rough Turkish detector, consistent with scripts/prepare_gmail_data.py.
_TURKISH_WORDS = {
    "ve", "bir", "bu", "için", "ile", "de", "da", "çok", "ama", "gibi",
    "var", "yok", "merhaba", "selam", "teşekkür", "iyi", "günler",
}


# ü/ö/ç are shared with German (für, möchten, etc.) and aren't reliable
# on their own — ı/ş/ğ have no equivalent in German/French/English and are
# a much stronger signal. Used to avoid classifying German text like
# "für mehr Informationen" as Turkish just because of "ü".
_TURKISH_ONLY_CHARS = set("ışğİŞĞ")


def detect_language(text: str) -> str:
    if any(c in _TURKISH_ONLY_CHARS for c in text):
        return "tr"
    words = set(re.findall(r"[a-zçğıöşü]+", text.lower()))
    if len(words & _TURKISH_WORDS) >= 2:
        return "tr"
    if re.search(r"[a-zA-Z]", text):
        return "en"
    return "other"


def get_body(msg: Message) -> tuple[str, bool]:
    """Returns (raw_body_content, is_html). Uses the same preference as
    scripts/process_data.py: plain text first, HTML fallback."""
    body = msg.get_body(preferencelist=("plain", "html"))
    if body is None:
        return "", False
    try:
        content = body.get_content()
    except Exception:
        return "", False
    return content, body.get_content_type() == "text/html"


# Corporate mail gateways prepend or append a fixed security banner to every
# message that arrives from outside the organisation ("EXTERNAL EMAIL —
# verify the sender…"). That banner is not part of the email under analysis:
# it is text the *defender* added, and it is identical across every external
# message the gateway handles.
#
# Leaving it in does real damage. On a real sample it was 59% of the body,
# and the LLM — asked to report technical findings — mined the banner's own
# advice checklist and presented it as evidence: "E-posta, 'Gönderici adı ve
# e-posta adresini doğrulayınız' gibi bir uyarı mesajı içeriyor." That is a
# finding about the security banner, not about the phishing email. The
# banner's vocabulary (doğrula, dikkat, kontrol, güvenlik, tıkla) also sits
# directly in the space the urgency and credential-request patterns search.
#
# Matching is anchored on the banner HEADING, not on any single keyword, so
# an ordinary email that happens to use the word "güvenlik" is untouched.
# Everything from the heading to the end of the body is removed: these
# banners are appended as a trailing block, and their internal wording varies
# far more than their heading does.
_GATEWAY_BANNER_RE = re.compile(
    r"""(?:^|\n)\s*
    (?:
        HAR[İI]C[İI]\s+E-?POSTA(?:\s+B[İI]LG[İI]LEND[İI]RMES[İI])?
      | D[İI][ŞS]\s+KAYNAKLI\s+E-?POSTA
      | EXTERNAL\s+E-?MAIL(?:\s+WARNING)?
      | CAUTION\s*:\s*(?:This\s+e-?mail|External)
      | \[?\s*EXTERNAL\s*\]?\s*:?\s*This\s+message
    )
    .*$""",
    re.IGNORECASE | re.VERBOSE | re.DOTALL,
)


def strip_gateway_banner(text: str) -> tuple[str, bool]:
    """Remove a corporate mail gateway's external-sender banner.

    Returns (cleaned_text, was_stripped). The banner is dropped rather than
    kept in a separate field: nothing downstream needs its contents, and
    every consumer of body_text (urgency patterns, credential detection,
    the LLM prompt) is better off without it.

    An empty result is returned as-is. That case is meaningful rather than
    broken: a body consisting of nothing but a gateway banner means the
    real message carried no text at all, which is exactly what
    image_only_body needs to see. Suppressing the strip to avoid an empty
    string would hide that.
    """
    match = _GATEWAY_BANNER_RE.search(text)
    if not match:
        return text, False
    return text[:match.start()].strip(), True


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator="\n")
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def extract_body_facts(
    raw_body: str,
    is_html: bool,
    has_action_channel: bool = False,
    has_attachments: bool = False,
) -> dict:
    """Returns has_html_form, form_action_domain, has_hidden_text,
    image_only_body, urgency_keywords, credential_request,
    claims_attachment, body_text, language.

    has_action_channel: True if the message has an external URL or an
    attachment — the caller (parse.py) computes this from
    extract_url_facts()/extract_attachment_facts() and passes it in, since
    those run separately from this function. Combined with has_html_form
    (computed here), this is credential_request's third required
    component — see the _CREDENTIAL_REQUEST_VERB_RE comment above.

    has_attachments: True if extract_attachment_facts() found at least one
    real attachment — needed to compute claims_attachment (body references
    an attachment that doesn't actually exist), see holdout-fix-tasks.md T5."""
    if is_html:
        soup = BeautifulSoup(raw_body, "html.parser")
        text = soup.get_text(separator="\n")
        text = re.sub(r"\n{3,}", "\n\n", text).strip()

        form_tag = soup.find("form")
        has_html_form = form_tag is not None
        # Rule Engine v2 adım 6 (CLAUDE.md), form-action domain kontrolü.
        # A form's own presence is weak on its own (legitimate surveys/
        # signup forms exist) — what actually distinguishes a credential-
        # harvesting form is where the SUBMITTED data goes. A relative
        # action ("/login") or no action attribute at all posts back to
        # the sender's own domain, which this leaves as None (nothing to
        # compare) rather than guessing a domain that isn't actually
        # written in the HTML.
        form_action_domain = None
        if form_tag is not None:
            action = form_tag.get("action")
            if action:
                form_action_domain = _domain_of_url(action)

        # T6-style ambiguity Codex flagged 2026-08-06: display:none / a
        # zero-size style span is ALSO the standard, benign technique for
        # a preheader ("gizli önizleme metni" — the one-line summary
        # email clients show next to the subject, hidden from the
        # rendered body itself). A short hidden span is normal marketing
        # boilerplate; a LARGE hidden block is what a real content-hiding
        # attack looks like (stuffing invisible text to evade spam/
        # phishing filters). Splitting on length: rule engine treats
        # image_only_body as evidence on its own, but only counts hidden
        # text as evidence when it's long enough not to be a preheader.
        _PREHEADER_MAX_CHARS = 150
        hidden_text_total_chars = 0
        for tag in soup.find_all(style=True):
            style = tag["style"].lower().replace(" ", "")
            if "display:none" in style or re.search(r"font-size:0", style):
                hidden_text_total_chars += len(tag.get_text(strip=True))
        has_hidden_text = hidden_text_total_chars > 0
        has_large_hidden_text = hidden_text_total_chars > _PREHEADER_MAX_CHARS

        has_images = bool(soup.find_all("img"))
    else:
        text = raw_body.strip()
        has_html_form = False
        form_action_domain = None
        has_hidden_text = False
        has_large_hidden_text = False
        has_images = False

    # Strip before ANY signal is computed: the banner must not reach the
    # urgency patterns, credential detection, language detection or the
    # body_text that later goes into the LLM prompt.
    text, gateway_banner_stripped = strip_gateway_banner(text)

    # Computed AFTER stripping, deliberately. An image-only email that
    # passed through a gateway carries hundreds of characters of banner
    # text, so measuring before the strip would report image_only_body as
    # False for exactly the messages the signal exists to catch.
    image_only_body = has_images and len(text.strip()) < 20

    detected_language = detect_language(text[:BODY_TEXT_MAX_CHARS])
    # Gate keyword lists by detected language (T2 point 3) — unless
    # detection is inconclusive ("other"), in which case run everything
    # rather than risk missing real signal on a language we can't identify.
    if detected_language == "tr":
        active_patterns = _URGENCY_PATTERNS_TR
    elif detected_language == "en":
        active_patterns = _URGENCY_PATTERNS_EN
    else:
        active_patterns = {**_URGENCY_PATTERNS_TR, **_URGENCY_PATTERNS_EN}

    found_urgency = []
    for keyword, pattern in active_patterns.items():
        m = pattern.search(text)
        if not m:
            continue
        start = max(0, m.start() - _CONTEXT_CHARS)
        end = min(len(text), m.end() + _CONTEXT_CHARS)
        context = text[start:end].replace("\n", " ").strip()
        found_urgency.append({"keyword": keyword, "context": context})
    found_urgency.sort(key=lambda item: item["keyword"])

    action_channel_present = has_action_channel or has_html_form
    credential_request = bool(
        action_channel_present and _credential_request_in_context(text)
    )

    claims_attachment = bool(
        not has_attachments and _ATTACHMENT_CLAIM_RE.search(text)
    )

    has_advance_fee_fraud_language = bool(_ADVANCE_FEE_FRAUD_RE.search(text))
    has_fake_reward_claim_language = bool(_FAKE_REWARD_CLAIM_RE.search(text))

    return {
        "has_html_form": has_html_form,
        "form_action_domain": form_action_domain,
        "has_hidden_text": has_hidden_text,
        "has_large_hidden_text": has_large_hidden_text,
        "image_only_body": image_only_body,
        "urgency_keywords": found_urgency,
        "credential_request": credential_request,
        "claims_attachment": claims_attachment,
        "has_advance_fee_fraud_language": has_advance_fee_fraud_language,
        "has_fake_reward_claim_language": has_fake_reward_claim_language,
        "body_text": text[:BODY_TEXT_MAX_CHARS],
        "language": detected_language,
    }
