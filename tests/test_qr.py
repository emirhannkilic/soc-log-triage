"""
Unit tests for src/parser/qr.py — QR code URL extraction from image
attachments/inline images (Rule Engine v2 adım 7, CLAUDE.md, 2026-08-08).

Run with: python3 tests/test_qr.py
"""
import sys
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.parser.qr import extract_qr_urls_from_attachment, extract_qr_urls_from_message

_encoder = cv2.QRCodeEncoder.create()


def _qr_png_bytes(text: str) -> bytes:
    qr = _encoder.encode(text)
    ok, buf = cv2.imencode(".png", qr)
    assert ok
    return buf.tobytes()


def test_decodes_url_from_qr_image():
    payload = _qr_png_bytes("https://phish.example.tld/login?id=123")
    urls = extract_qr_urls_from_attachment(payload, "image/png")
    assert urls == ["https://phish.example.tld/login?id=123"]


def test_non_url_qr_content_returns_empty():
    """QR codes carry arbitrary text — a QR encoding a phone number or
    plain text isn't a URL signal and must not be reported as one."""
    payload = _qr_png_bytes("Just plain text, not a URL")
    assert extract_qr_urls_from_attachment(payload, "image/png") == []


def test_plain_image_with_no_qr_returns_empty():
    import numpy as np

    plain = np.full((100, 100, 3), 200, dtype="uint8")
    ok, buf = cv2.imencode(".png", plain)
    assert extract_qr_urls_from_attachment(buf.tobytes(), "image/png") == []


def test_non_image_mime_type_returns_empty():
    payload = _qr_png_bytes("https://should-not-be-checked.example.com")
    assert extract_qr_urls_from_attachment(payload, "application/pdf") == []
    assert extract_qr_urls_from_attachment(payload, None) == []


def test_corrupt_image_bytes_returns_empty_not_raise():
    assert extract_qr_urls_from_attachment(b"not an image at all", "image/png") == []


def test_scan_survives_scale_factor_flakiness():
    """Regression: OpenCV's QRCodeDetector was observed to fail to decode
    a specific QR content at a specific integer upscale factor (9x) while
    succeeding at every neighboring factor (2-8, 10-14) on the exact same
    source image — an internal perspective-correction artifact, not a
    genuinely corrupt/unreadable QR. _decode_qr_urls must try multiple
    scale factors so one unlucky factor doesn't silently produce a false
    "no QR here.\""""
    payload = _qr_png_bytes("https://phish-via-qr.evil.tld/login")
    urls = extract_qr_urls_from_attachment(payload, "image/png")
    assert urls == ["https://phish-via-qr.evil.tld/login"]


def test_extract_from_message_finds_inline_image():
    """The common quishing shape: a QR code embedded directly in the email
    body as an inline image, no separate "attachment" at all. Unlike
    src/parser/attachments.py's AttachmentFacts list (which deliberately
    excludes inline parts), QR scanning must NOT skip them."""
    msg = MIMEMultipart()
    msg["From"] = "attacker@evil.tld"
    msg["To"] = "victim@example.com"
    msg["Subject"] = "Scan to verify"
    msg.attach(MIMEText("Please scan the QR code below.", "plain"))
    img = MIMEImage(_qr_png_bytes("https://inline-quishing.evil.tld/x"), _subtype="png")
    img.add_header("Content-Disposition", "inline", filename="verify.png")
    msg.attach(img)

    import email
    raw_msg = email.message_from_bytes(msg.as_bytes())
    urls = extract_qr_urls_from_message(raw_msg)
    assert urls == ["https://inline-quishing.evil.tld/x"]


def test_extract_from_message_with_no_images_returns_empty():
    msg = MIMEMultipart()
    msg["From"] = "a@b.com"
    msg["To"] = "c@d.com"
    msg["Subject"] = "Hello"
    msg.attach(MIMEText("Just text, no images.", "plain"))

    import email
    raw_msg = email.message_from_bytes(msg.as_bytes())
    assert extract_qr_urls_from_message(raw_msg) == []


def test_extract_from_message_deduplicates():
    """The same QR (e.g. a repeated logo-adjacent code) appearing in two
    image parts must only be reported once."""
    msg = MIMEMultipart()
    msg["From"] = "a@b.com"
    msg["To"] = "c@d.com"
    msg["Subject"] = "Hello"
    msg.attach(MIMEText("body", "plain"))
    for name in ("first.png", "second.png"):
        img = MIMEImage(_qr_png_bytes("https://duplicate.evil.tld/x"), _subtype="png")
        img.add_header("Content-Disposition", "inline", filename=name)
        msg.attach(img)

    import email
    raw_msg = email.message_from_bytes(msg.as_bytes())
    urls = extract_qr_urls_from_message(raw_msg)
    assert urls == ["https://duplicate.evil.tld/x"]


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
