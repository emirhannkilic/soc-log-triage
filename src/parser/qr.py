"""
QR code URL extraction from image attachments — Rule Engine v2 adım 7
(CLAUDE.md), 2026-08-08.

"Quishing" (QR-code phishing) hides the actual destination URL inside an
image instead of an <a href> — every URL-based signal this project has
(url_shortener, url_ip_based, has_punycode, text/href mismatch) is blind
to it unless the URL is decoded out of the image first. This module does
only the decoding; the decoded URL is fed back through
src/parser/urls.py's existing _build_url_fact() so it scores through the
same signals as any other URL, not a separate one.

Uses OpenCV's built-in QRCodeDetector (already a project dependency via
the shadow-mode classifier experiment, see PROGRESS.md) rather than
adding zbar/pyzbar — no new system dependency, and this project only
needs "is there a URL in this image," not general barcode format
support.

Measured on data/phishing_pot (300-email sample): only 2/300 emails carry
a non-inline image attachment at all, so this signal will fire rarely in
this project's email-based phishing corpus — quishing's image-only-body
variant is a real technique, but this corpus mostly isn't email
carrying scannable QR images. Implemented anyway per CLAUDE.md's locked
adım 7 scope; low expected trigger rate is a corpus property, not a
reason to skip a real class of the attack.
"""
import io
import warnings
from email.message import Message

import cv2
import numpy as np
from PIL import Image

_detector = cv2.QRCodeDetector()

# QRCodeDetector needs a visible quiet zone (white border) around the
# code and a reasonably large pixel size — a QR image embedded at its
# native encoded resolution (as small as ~25x25px) decodes as empty
# without both. But no single upscale factor is reliable: empirically,
# OpenCV's detector can fail on a SPECIFIC integer scale factor for a
# specific QR content (observed: factor 9 failed to decode a QR that
# factors 2-8 and 10-14 all decoded correctly, same image, only the
# resize target differed) — an artifact of its internal perspective-
# correction, not a real unrecoverable image. Trying several factors and
# taking the first that decodes costs a few extra milliseconds and
# removes that flakiness.
_QUIET_ZONE_PX = 40
_SCALE_FACTORS = (6, 4, 8, 3, 10, 2)


def _decode_qr_urls(pil_image: Image.Image) -> list[str]:
    """Returns every http(s) URL found in QR codes within the image.
    Empty list if there's no QR code, the QR doesn't decode, or the
    decoded payload isn't a URL (QR codes carry arbitrary text — a QR
    encoding a phone number or plain text isn't a URL signal)."""
    rgb = pil_image.convert("RGB")
    arr = np.array(rgb)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]

    for factor in _SCALE_FACTORS:
        scaled = cv2.resize(bgr, (w * factor, h * factor), interpolation=cv2.INTER_NEAREST)
        padded = cv2.copyMakeBorder(
            scaled, _QUIET_ZONE_PX, _QUIET_ZONE_PX, _QUIET_ZONE_PX, _QUIET_ZONE_PX,
            cv2.BORDER_CONSTANT, value=(255, 255, 255),
        )
        try:
            ok, decoded_texts, _points, _straight = _detector.detectAndDecodeMulti(padded)
        except cv2.error:
            continue
        if not ok:
            continue
        urls = [t for t in decoded_texts if t.startswith(("http://", "https://"))]
        if urls:
            return urls
    return []


def extract_qr_urls_from_attachment(payload: bytes, mime_type: str | None) -> list[str]:
    """payload: raw attachment bytes. Returns [] for non-image attachments,
    corrupt/undecodable images, or images with no QR code — never raises,
    since this runs over untrusted attacker-supplied files."""
    if not mime_type or not mime_type.startswith("image/"):
        return []
    try:
        with warnings.catch_warnings():
            # Some real Gmail-corpus images are palette-mode with legacy
            # byte-based transparency — Pillow warns and still loads them
            # correctly, this isn't a decode failure worth surfacing.
            warnings.simplefilter("ignore", UserWarning)
            with Image.open(io.BytesIO(payload)) as img:
                img.load()
                return _decode_qr_urls(img)
    except Exception:
        return []


def extract_qr_urls_from_message(msg: Message) -> list[str]:
    """Scans EVERY image part in the message — including inline ones
    (Content-Disposition: inline), unlike src/parser/attachments.py's
    AttachmentFacts list, which deliberately excludes inline parts as
    "not something a recipient would open." A QR code embedded directly
    in the email body (the common quishing shape — no separate attachment
    at all) is exactly an inline image, so it must not be skipped here
    the way it correctly is for the attachment-risk facts. Deduplicated
    in encounter order."""
    urls: list[str] = []
    seen: set[str] = set()
    for part in msg.walk():
        if part.is_multipart():
            continue
        if not (part.get_content_type() or "").startswith("image/"):
            continue
        payload = part.get_payload(decode=True)
        if not payload:
            continue
        for url in extract_qr_urls_from_attachment(payload, part.get_content_type()):
            if url not in seen:
                seen.add(url)
                urls.append(url)
    return urls
