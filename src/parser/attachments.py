"""
Extracts attachment facts from an email.message.Message (v3 plan section 4.5).
"""
from email.message import Message

# Extensions that can execute code or carry a payload on their own, or that
# commonly wrap one (archives, office formats with macro support).
_RISKY_EXTENSIONS = {
    "exe", "scr", "js", "jse", "vbs", "vbe", "wsf", "wsh", "bat", "cmd",
    "com", "pif", "iso", "img", "msi", "jar", "ps1",
    "docm", "xlsm", "pptm",  # macro-enabled office formats
}


def _extension_of(filename: str) -> str | None:
    if "." not in filename:
        return None
    return filename.rsplit(".", 1)[-1].lower()


def _has_double_extension(filename: str) -> bool:
    parts = filename.split(".")
    if len(parts) < 3:
        return False
    # e.g. "fatura.pdf.exe" — last extension is risky and there's a
    # plausible-looking extension right before it.
    last_ext = parts[-1].lower()
    second_last_ext = parts[-2].lower()
    return last_ext in _RISKY_EXTENSIONS and len(second_last_ext) <= 5


def extract_attachment_facts(msg: Message) -> list[dict]:
    facts: list[dict] = []
    for part in msg.walk():
        if part.is_multipart():
            continue
        filename = part.get_filename()
        if not filename:
            continue

        payload = part.get_payload(decode=True)
        size = len(payload) if payload else None
        extension = _extension_of(filename)

        facts.append({
            "filename": filename,
            "mime_type": part.get_content_type(),
            "size": size,
            "double_extension": _has_double_extension(filename),
            "risky_type": extension in _RISKY_EXTENSIONS if extension else False,
        })

    return facts
