"""
Extracts body signal facts (v3 plan section 4.6) and meta facts (4.7) from
an email.message.Message.
"""
import re
from email.message import Message

from bs4 import BeautifulSoup

BODY_TEXT_MAX_CHARS = 2000

_URGENCY_KEYWORDS = {
    # Turkish
    "acil", "hemen", "derhal", "son gün", "hesabınız askıya", "doğrulayın",
    "onaylayın", "şifrenizi", "kilitlenecek", "iptal edilecek", "24 saat",
    "hemen tıklayın", "şimdi harekete geçin",
    # English
    "urgent", "immediately", "act now", "verify your account", "suspended",
    "expire", "click here", "confirm your", "password will", "24 hours",
    "limited time", "final notice",
}

# No trailing \b: Python's \b treats Turkish letters (ş, ı, ğ, ü, ö, ç) as
# non-word characters, so it fails to find a boundary after suffixed forms
# like "şifrenizi" (şifre + nizi). Leading \b is kept since these patterns
# never appear as a suffix of another word.
_CREDENTIAL_PATTERNS = [
    re.compile(r"\b(şifre|parola|password)", re.IGNORECASE),
    re.compile(r"\b(kart\s*numaras[ıi]|card\s*number|cvv)", re.IGNORECASE),
    re.compile(r"\bT\.?C\.?\s*(kimlik|no)\b", re.IGNORECASE),
    re.compile(r"\bsocial security\b", re.IGNORECASE),
    re.compile(r"\b(iban|hesap numaras[ıi])", re.IGNORECASE),
]

# Rough Turkish detector, consistent with scripts/prepare_gmail_data.py.
_TURKISH_CHARS = set("çğıöşüÇĞİÖŞÜ")
_TURKISH_WORDS = {
    "ve", "bir", "bu", "için", "ile", "de", "da", "çok", "ama", "gibi",
    "var", "yok", "merhaba", "selam", "teşekkür", "iyi", "günler",
}


def detect_language(text: str) -> str:
    if any(c in _TURKISH_CHARS for c in text):
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


def extract_body_facts(raw_body: str, is_html: bool) -> dict:
    """Returns has_html_form, has_hidden_text, image_only_body,
    urgency_keywords, credential_request, body_text, language."""
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

    text_lower = text.lower()
    found_urgency = sorted({kw for kw in _URGENCY_KEYWORDS if kw in text_lower})

    credential_request = any(p.search(text) for p in _CREDENTIAL_PATTERNS)

    return {
        "has_html_form": has_html_form,
        "has_hidden_text": has_hidden_text,
        "image_only_body": image_only_body,
        "urgency_keywords": found_urgency,
        "credential_request": credential_request,
        "body_text": text[:BODY_TEXT_MAX_CHARS],
        "language": detect_language(text[:BODY_TEXT_MAX_CHARS]),
    }
