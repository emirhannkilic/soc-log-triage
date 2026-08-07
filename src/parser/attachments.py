"""
Extracts attachment facts from an email.message.Message (v3 plan section 4.5).
"""
from email.message import Message

from src.parser.magic import has_extension_mismatch

# Extensions that can execute code or carry a payload on their own, or that
# commonly wrap one (office formats with macro support). Archive formats
# (.zip, .rar, .7z, .iso, .img, .cab) are NOT here — an archive isn't
# inherently malicious (a legitimate order confirmation can attach a .zip),
# so it gets its own low-weight is_archive signal instead. See
# holdout-fix-tasks.md T5.
_RISKY_EXTENSIONS = {
    "exe", "scr", "js", "jse", "vbs", "vbe", "wsf", "wsh", "bat", "cmd",
    "com", "pif", "msi", "jar", "ps1",
    "docm", "xlsm", "pptm",  # macro-enabled office formats
}

_ARCHIVE_EXTENSIONS = {"zip", "rar", "7z", "iso", "img", "cab"}


def _extension_of(filename: str) -> str | None:
    if "." not in filename:
        return None
    return filename.rsplit(".", 1)[-1].lower()


def _has_double_extension(filename: str) -> bool:
    parts = filename.split(".")
    if len(parts) < 3:
        return False
    # e.g. "fatura.pdf.exe" — last extension is risky/archive and there's
    # a plausible-looking extension right before it. Archives are included
    # here (unlike risky_type) because "invoice.pdf.zip" disguising a
    # payload behind a fake document extension is the same disguise
    # pattern regardless of whether the final wrapper is an .exe or a .zip.
    last_ext = parts[-1].lower()
    second_last_ext = parts[-2].lower()
    return (
        last_ext in _RISKY_EXTENSIONS or last_ext in _ARCHIVE_EXTENSIONS
    ) and len(second_last_ext) <= 5


def extract_attachment_facts(msg: Message) -> list[dict]:
    facts: list[dict] = []
    for part in msg.walk():
        if part.is_multipart():
            continue
        filename = part.get_filename()
        if not filename:
            continue
        # inline parts with a filename are near-universally embedded
        # signature logos / HTML-body images (image001.jpg, Ilv_logo.png,
        # etc.), not attachments a recipient would open — counting them
        # inflated attachment-risk facts with pure noise (found via
        # holdout-fix-tasks.md T5 spot-check: candidate 11 showed 4 fake
        # "attachments" that were an email signature's logo images).
        # get_content_disposition() returns None for parts with no
        # Content-Disposition header at all, which real attachments
        # normally do have (disposition="attachment") — only "inline" is
        # excluded, not None, so malformed/legacy attachments without the
        # header still count.
        if part.get_content_disposition() == "inline":
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
            "is_archive": extension in _ARCHIVE_EXTENSIONS if extension else False,
            "extension_mismatch": has_extension_mismatch(payload, filename),
        })

    return facts
