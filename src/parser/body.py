"""
Extracts body signal facts (v3 plan section 4.6) and meta facts (4.7) from
an email.message.Message.
"""
import re
from email.message import Message

from bs4 import BeautifulSoup

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
    """Returns has_html_form, has_hidden_text, image_only_body,
    urgency_keywords, credential_request, claims_attachment, body_text,
    language.

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

        has_html_form = soup.find("form") is not None

        has_hidden_text = False
        for tag in soup.find_all(style=True):
            style = tag["style"].lower().replace(" ", "")
            if "display:none" in style or re.search(r"font-size:0", style):
                has_hidden_text = True
                break

        images = soup.find_all("img")
        image_only_body = bool(images) and len(text.strip()) < 20
    else:
        text = raw_body.strip()
        has_html_form = False
        has_hidden_text = False
        image_only_body = False

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
        action_channel_present
        and _CREDENTIAL_REQUEST_VERB_RE.search(text)
        and _CREDENTIAL_TARGET_OBJECT_RE.search(text)
    )

    claims_attachment = bool(
        not has_attachments and _ATTACHMENT_CLAIM_RE.search(text)
    )

    return {
        "has_html_form": has_html_form,
        "has_hidden_text": has_hidden_text,
        "image_only_body": image_only_body,
        "urgency_keywords": found_urgency,
        "credential_request": credential_request,
        "claims_attachment": claims_attachment,
        "body_text": text[:BODY_TEXT_MAX_CHARS],
        "language": detected_language,
    }
