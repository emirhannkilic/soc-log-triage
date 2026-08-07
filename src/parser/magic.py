"""
File-signature ("magic number") lookup for attachment payloads — Rule
Engine v2 adım 7 (CLAUDE.md), 2026-08-08.

Detects the classic disguise: a filename claims to be a harmless document
(invoice.pdf) but the actual bytes are something else (an executable, or
nothing recognizable at all) — the extension and MIME type are attacker-
controlled metadata, the leading bytes are not.

Deliberately narrow, not a general file-type sniffer: only the formats
phishing payloads actually show up as in this project's corpus (PDF,
ZIP-based Office formats — .docx/.xlsx/.pptx are ZIP containers, legacy
OLE2 Office — .doc/.xls/.ppt, common archives, Windows PE executables).
Stdlib-only, no python-magic dependency — a handful of leading-byte
signatures is enough for this narrow set.

ISO images are deliberately NOT covered: their signature ("CD001") sits at
byte offset 0x8001, not the start of the file, so a leading-bytes check
can't identify them without reading much further into a potentially large
attachment — out of scope for this pass.
"""
_SIGNATURES: dict[str, tuple[bytes, ...]] = {
    "pdf": (b"%PDF-",),
    # ZIP-based: plain .zip AND every ZIP-container Office format
    # (docx/xlsx/pptx) share this signature — content can't be told apart
    # from the leading bytes alone without unzipping, so they're grouped.
    "zip": (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08"),
    "docx": (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08"),
    "xlsx": (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08"),
    "pptx": (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08"),
    # Legacy OLE2 container: .doc/.xls/.ppt (pre-2007 Office) all share
    # this one signature too, same reasoning as the ZIP group above.
    "doc": (b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1",),
    "xls": (b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1",),
    "ppt": (b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1",),
    "rar": (b"Rar!\x1a\x07\x00", b"Rar!\x1a\x07\x01\x00"),
    "7z": (b"7z\xbc\xaf\x27\x1c",),
    "exe": (b"MZ",),
    "scr": (b"MZ",),  # .scr (screensaver) is a renamed PE executable
    "com": (b"MZ",),
    "msi": (b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1",),  # MSI is also OLE2-based
}

# Every signature this module knows how to recognize, deduplicated, used to
# tell "the payload matches a completely different known format" apart from
# "the payload doesn't match anything we can identify at all" (encrypted,
# corrupted, or a format outside this narrow list) — the former is the
# actual disguise signal, the latter is far too common in legitimate mail
# (encrypted PDFs, some scanners' export formats) to score on its own.
_ALL_KNOWN_SIGNATURES: tuple[bytes, ...] = tuple(
    sorted({sig for sigs in _SIGNATURES.values() for sig in sigs}, key=len, reverse=True)
)


def _matches_extension(payload_head: bytes, extension: str) -> bool | None:
    """True if payload_head matches extension's known signature(s), False if
    it doesn't, None if this extension has no signature on file (can't
    judge)."""
    sigs = _SIGNATURES.get(extension)
    if sigs is None:
        return None
    return any(payload_head.startswith(sig) for sig in sigs)


def has_extension_mismatch(payload: bytes | None, filename: str) -> bool:
    """True only when the payload's leading bytes confidently match a
    DIFFERENT known format than the filename's extension claims — e.g.
    "invoice.pdf" whose bytes start with "MZ" (a PE executable). Returns
    False (not a mismatch) when the extension isn't in this module's
    table, the payload is missing/empty, or the payload's format can't be
    identified at all — an unrecognized payload is not evidence of a
    SPECIFIC disguise, just of "we don't know," and scoring that would
    catch encrypted/exotic-but-legitimate attachments as false positives.
    """
    if not payload or "." not in filename:
        return False
    extension = filename.rsplit(".", 1)[-1].lower()
    matches_claimed = _matches_extension(payload, extension)
    if matches_claimed is None or matches_claimed:
        return False
    return any(payload.startswith(sig) for sig in _ALL_KNOWN_SIGNATURES)
