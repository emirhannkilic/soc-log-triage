"""
Unit tests for src/parser/magic.py — file-signature (magic number) lookup
for attachment payloads (Rule Engine v2 adım 7, CLAUDE.md, 2026-08-08).

Run with: python3 tests/test_magic.py
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.parser.magic import has_extension_mismatch


def test_matching_pdf_is_not_mismatch():
    assert has_extension_mismatch(b"%PDF-1.4\n...", "invoice.pdf") is False


def test_matching_docx_is_not_mismatch():
    assert has_extension_mismatch(b"PK\x03\x04\x14\x00...", "report.docx") is False


def test_matching_legacy_doc_is_not_mismatch():
    assert has_extension_mismatch(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1...", "old.doc") is False


def test_exe_disguised_as_pdf_is_mismatch():
    """The classic disguise: a PE executable with a .pdf filename."""
    assert has_extension_mismatch(b"MZ\x90\x00\x03\x00\x00\x00", "invoice.pdf") is True


def test_pdf_disguised_as_exe_is_mismatch():
    assert has_extension_mismatch(b"%PDF-1.4\n...", "totally-safe.exe") is True


def test_zip_disguised_as_docx_is_not_mismatch():
    """docx IS a zip container — same signature, must not false-positive."""
    assert has_extension_mismatch(b"PK\x03\x04...", "report.docx") is False


def test_unrecognized_payload_is_not_mismatch():
    """A payload matching NO known signature (encrypted, exotic-but-
    legitimate format) is not evidence of a SPECIFIC disguise — must not
    score, or every encrypted PDF becomes a false positive."""
    assert has_extension_mismatch(b"\x01\x02\x03\x04random garbage", "invoice.pdf") is False


def test_extension_with_no_signature_on_file_is_not_mismatch():
    """An extension this module doesn't track (e.g. .txt) can't be judged
    — must not claim a mismatch just because we have no baseline."""
    assert has_extension_mismatch(b"MZ\x90\x00", "notes.txt") is False


def test_no_extension_is_not_mismatch():
    assert has_extension_mismatch(b"MZ\x90\x00", "noextension") is False


def test_empty_payload_is_not_mismatch():
    assert has_extension_mismatch(b"", "invoice.pdf") is False
    assert has_extension_mismatch(None, "invoice.pdf") is False


def test_iso_is_not_covered():
    """ISO's signature sits at byte offset 0x8001, not the start of the
    file — out of scope for this pass, must never claim a mismatch for
    .iso regardless of leading bytes."""
    assert has_extension_mismatch(b"\x00" * 64, "image.iso") is False
    assert has_extension_mismatch(b"MZ\x90\x00", "image.iso") is False


if __name__ == "__main__":
    import traceback

    tests = [(name, obj) for name, obj in list(globals().items())
              if name.startswith("test_") and callable(obj)]
    passed, failed = 0, 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS: {name}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL: {name}: {e}")
            failed += 1
        except Exception:
            print(f"ERROR: {name}")
            traceback.print_exc()
            failed += 1

    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
