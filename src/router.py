"""
Minimal input router — the llm-proxy layer's first stage, scoped down to
what this repository can actually serve.

WHY THIS IS DELIBERATELY SMALL
    The wider design (llm-proxy-notes.md) routes to three personas:
    phishing, threat intel (Titus) and general cyber Q&A. Only the phishing
    pipeline exists here — parser → rule engine → Seneca → template. So this
    router answers one question honestly:

        Is this input something the phishing pipeline can process?

    Other inputs resolve to one of three explicit outcomes: the phishing
    intent is known but the email is missing, the intent is ambiguous and
    needs clarification, or the request is outside the supported scope.

TWO STAGES, IN THIS ORDER
    1. Static rules (0 ms) — file extension, or RFC 5322 headers in pasted
       text. Structural, deterministic, and enough for every input the
       phishing pipeline can actually process.
    2. Intent classifier (--classify, src/intent.py) — only for prose the
       static rules cannot resolve. Loads a ~0.9GB model.

    The order is deliberate. The router runs on every single request; a
    model sitting on that path would make the system's most frequent code
    its slowest. Stage two exists for the case static rules genuinely
    cannot answer: telling "check this IP" from "how does SPF work"
    requires understanding the request, not matching its shape.

WHAT STAGE TWO CAN AND CANNOT DO HERE
    It names the persona, which is measurable on its own. What it cannot do
    is deliver: `titus` and `cybersec_qa` have no implementation, and
    `phishing` still needs an actual email — no classification turns a
    question into something the parser can consume.

    Its real value is the message it enables: "you want a phishing
    analysis, attach the email" beats "unsupported input".

THE CASE THAT MATTERS MOST
    An analyst rarely hands over a .eml file. They paste the raw message
    into a chat box. _looks_like_raw_email covers that path, which is why
    it is not enough to check file extensions.

Usage:
    python3 src/router.py mail.eml
    python3 src/router.py --text "$(pbpaste)"
    python3 src/router.py --text "SPF nedir?" --classify
    echo "..." | python3 src/router.py --stdin
"""
import argparse
import re
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class Route(str, Enum):
    """Request-level outcomes for the phishing-only first release.

    PHISHING is kept as an alias so existing callers do not break while the
    public contract moves to the more explicit PHISHING_DIRECT name.
    """

    PHISHING_DIRECT = "phishing_direct"
    PHISHING = "phishing_direct"
    PHISHING_MISSING_EMAIL = "phishing_missing_email"
    NEEDS_CLARIFICATION = "needs_clarification"
    UNSUPPORTED = "unsupported"


class RoutingStatus(str, Enum):
    ACCEPTED = "accepted"
    MISSING_INPUT = "missing_input"
    NOT_ROUTED = "not_routed"


class ConfidenceSource(str, Enum):
    DETERMINISTIC = "deterministic"
    TRUSTED_METADATA = "trusted_metadata"
    MODEL = "model"
    NONE = "none"


@dataclass
class RoutingDecision:
    route: Route
    reason: str
    # How the decision was reached, so a caller (or a demo) can show the
    # path rather than just the outcome.
    matched_rule: str
    status: RoutingStatus
    reason_code: str
    confidence_source: ConfidenceSource
    # Populated only for PHISHING: what the pipeline should read.
    eml_path: Path | None = None
    raw_email: str | None = None
    supported_scope: str = "phishing_email_analysis"
    unsupported_scope: str | None = None

    @property
    def accepted(self) -> bool:
        return self.route is Route.PHISHING_DIRECT

    def as_dict(self) -> dict[str, str | bool | None]:
        """Stable machine-readable shape for API and logging consumers."""
        return {
            "route": self.route.value,
            "status": self.status.value,
            "reason_code": self.reason_code,
            "matched_rule": self.matched_rule,
            "confidence_source": self.confidence_source.value,
            "message": self.reason,
            "accepted": self.accepted,
            "supported_scope": self.supported_scope,
            "unsupported_scope": self.unsupported_scope,
        }


# Outlook's binary .msg format needs a separate parser, which this repository
# does not have. A text/RFC message with any extension is still accepted by
# the structural content check below.
EMAIL_EXTENSIONS = (".eml",)

# RFC 5322 header names that appear at the start of a line in a raw message.
# Matching a single one is not enough — "From:" alone shows up in ordinary
# prose ("From: the marketing team") and in quoted replies. Requiring
# several distinct headers is what separates a pasted message from text
# that merely mentions one.
_HEADER_RE = re.compile(
    r"^(From|To|Cc|Subject|Date|Received|Return-Path|Message-ID|"
    r"Reply-To|MIME-Version|Content-Type|Authentication-Results|DKIM-Signature)\s*:",
    re.MULTILINE | re.IGNORECASE,
)
_MIN_DISTINCT_HEADERS = 3

# Only the head of the input is scanned: headers live at the top of a
# message, and a long body could otherwise accumulate incidental matches.
_HEADER_SCAN_CHARS = 4000


def _distinct_headers(text: str) -> set[str]:
    return {m.group(1).lower()
            for m in _HEADER_RE.finditer(text[:_HEADER_SCAN_CHARS])}


def looks_like_raw_email(text: str) -> bool:
    """True when the text is a pasted raw email rather than a question.

    Requires _MIN_DISTINCT_HEADERS different header fields, so a message
    quoting "Subject: ..." in passing does not qualify.
    """
    if not text or not text.strip():
        return False
    return len(_distinct_headers(text)) >= _MIN_DISTINCT_HEADERS


def _decision(
    route_value: Route,
    reason: str,
    matched_rule: str,
    *,
    status: RoutingStatus,
    reason_code: str,
    confidence_source: ConfidenceSource,
    eml_path: Path | None = None,
    raw_email: str | None = None,
    unsupported_scope: str | None = None,
) -> RoutingDecision:
    return RoutingDecision(
        route=route_value,
        reason=reason,
        matched_rule=matched_rule,
        status=status,
        reason_code=reason_code,
        confidence_source=confidence_source,
        eml_path=eml_path,
        raw_email=raw_email,
        unsupported_scope=unsupported_scope,
    )


def _trusted_system_requests_phishing(message: str | None) -> bool:
    """Recognise an explicit phishing persona in trusted upstream text.

    Trust is established by the caller. End-user text must never be passed
    through this argument merely because it claims to be a system prompt.
    """
    if not message:
        return False
    return bool(re.search(r"\bphishing\b|kimlik\s+av[ıi]|oltalama",
                          message, re.IGNORECASE))


def _route_via_classifier(text: str) -> RoutingDecision:
    """Stage two: ask the small model which persona this prose belongs to.

    A `phishing` intent still cannot be served — the pipeline needs an
    actual email, and no amount of intent classification produces one from
    a question. So the classifier's value is in the message it enables:
    "you want a phishing analysis, attach the email" is more useful than
    "unsupported input".
    """
    try:
        from src.intent import classify
    except ImportError as e:
        return _decision(
            Route.NEEDS_CLARIFICATION,
            f"Girdi düz metin ve niyet sınıflandırıcı yüklenemedi ({e}). "
            f"Phishing analizi için .eml dosyasını veya ham e-postayı gönderin.",
            "classifier_unavailable",
            status=RoutingStatus.NOT_ROUTED,
            reason_code="classifier_unavailable",
            confidence_source=ConfidenceSource.NONE,
        )

    try:
        r = classify(text)
    except Exception as e:
        return _decision(
            Route.NEEDS_CLARIFICATION,
            f"Niyet sınıflandırıcı çalıştırılamadı ({type(e).__name__}: {e}). "
            f"Phishing analizi için .eml dosyasını veya ham e-postayı gönderin.",
            "classifier_unavailable",
            status=RoutingStatus.NOT_ROUTED,
            reason_code="classifier_unavailable",
            confidence_source=ConfidenceSource.NONE,
        )
    detail = f"[{r.persona} %{r.confidence*100:.0f}]"

    if r.fallback_reason:
        return _decision(
            Route.NEEDS_CLARIFICATION,
            f"{detail} İstek güvenilir biçimde yönlendirilemedi. Phishing "
            f"analizi istiyorsanız .eml dosyasını veya header'larıyla ham "
            f"e-postayı gönderin. ({r.fallback_reason})",
            "classifier_below_threshold",
            status=RoutingStatus.NOT_ROUTED,
            reason_code="ambiguous_intent",
            confidence_source=ConfidenceSource.MODEL,
        )

    if r.persona == "phishing":
        return _decision(
            Route.PHISHING_MISSING_EMAIL,
            f"{detail} Phishing analizi istendiği anlaşıldı ama girdide bir "
            f"e-posta yok. Analiz için .eml dosyasını ekleyin ya da mailin "
            f"ham halini (header'larıyla birlikte) yapıştırın.",
            "intent_phishing_no_email",
            status=RoutingStatus.MISSING_INPUT,
            reason_code="phishing_intent_no_email",
            confidence_source=ConfidenceSource.MODEL,
        )

    if r.persona == "alakasiz":
        return _decision(
            Route.UNSUPPORTED,
            f"{detail} Bu istek siber güvenlikle ilgili görünmüyor. Bu "
            f"sistemde yalnızca phishing analizi hattı çalışıyor — bir "
            f"e-posta paylaşırsanız analiz edebilirim.",
            "intent_alakasiz",
            status=RoutingStatus.NOT_ROUTED,
            reason_code="unsupported_intent",
            confidence_source=ConfidenceSource.MODEL,
            unsupported_scope="non_cyber_request",
        )

    return _decision(
        Route.UNSUPPORTED,
        f"{detail} Bu istek '{r.persona}' personasına ait görünüyor, ama o "
        f"persona bu repoda kurulmadı — yalnızca phishing hattı çalışıyor.",
        f"intent_{r.persona}",
        status=RoutingStatus.NOT_ROUTED,
        reason_code="unsupported_intent",
        confidence_source=ConfidenceSource.MODEL,
        unsupported_scope=r.persona,
    )


def route(file_path: Path | None = None, text: str | None = None,
          use_classifier: bool = False,
          trusted_route_hint: str | None = None,
          trusted_system_message: str | None = None) -> RoutingDecision:
    """Decide whether the phishing pipeline can handle this input.

    Checked in order of certainty: an email file is unambiguous, a pasted
    message is nearly so, anything else is not an email at all.

    use_classifier turns on stage two for prose that reached the end of the
    static rules. Off by default: it loads a model, and most callers only
    need the structural answer.
    """
    if file_path is not None:
        if not file_path.is_file():
            return _decision(
                Route.UNSUPPORTED,
                f"Dosya bulunamadı: {file_path}",
                "file_missing",
                status=RoutingStatus.NOT_ROUTED,
                reason_code="file_missing",
                confidence_source=ConfidenceSource.DETERMINISTIC,
            )
        if file_path.suffix.lower() in EMAIL_EXTENSIONS:
            return _decision(
                Route.PHISHING_DIRECT,
                f"E-posta dosyası ({file_path.suffix}) — phishing analizine uygun.",
                "file_extension",
                status=RoutingStatus.ACCEPTED,
                reason_code="email_file_extension",
                confidence_source=ConfidenceSource.DETERMINISTIC,
                eml_path=file_path,
            )
        # A file that is not named like an email may still be one; the
        # content check is the authority, the extension is only a shortcut.
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            return _decision(
                Route.UNSUPPORTED, f"Dosya okunamadı: {e}", "file_unreadable",
                status=RoutingStatus.NOT_ROUTED,
                reason_code="file_unreadable",
                confidence_source=ConfidenceSource.DETERMINISTIC,
            )
        if looks_like_raw_email(content):
            return _decision(
                Route.PHISHING_DIRECT,
                f"Uzantı e-posta değil ({file_path.suffix or 'yok'}) ama içerik "
                f"ham e-posta: {sorted(_distinct_headers(content))[:4]} "
                f"header'ları bulundu.",
                "file_content",
                status=RoutingStatus.ACCEPTED,
                reason_code="email_file_content",
                confidence_source=ConfidenceSource.DETERMINISTIC,
                eml_path=file_path,
            )
        if (trusted_route_hint or "").strip().lower() == "phishing":
            return _decision(
                Route.UNSUPPORTED,
                "Phishing route hint'i verildi ancak yüklenen dosya e-posta "
                "değil. .eml dosyasını veya ham e-postayı gönderin.",
                "route_payload_conflict",
                status=RoutingStatus.NOT_ROUTED,
                reason_code="route_payload_conflict",
                confidence_source=ConfidenceSource.DETERMINISTIC,
                unsupported_scope="non_email_payload",
            )
        return _decision(
            Route.UNSUPPORTED,
            f"Bu bir e-posta dosyası değil ({file_path.suffix or 'uzantısız'}). "
            f"Phishing analizi .eml ya da ham e-posta metni gerektiriyor.",
            "file_not_email",
            status=RoutingStatus.NOT_ROUTED,
            reason_code="file_not_email",
            confidence_source=ConfidenceSource.DETERMINISTIC,
            unsupported_scope="non_email_file",
        )

    if text is not None and text.strip():
        if looks_like_raw_email(text):
            found = sorted(_distinct_headers(text))
            return _decision(
                Route.PHISHING_DIRECT,
                f"Yapıştırılmış ham e-posta ({len(found)} header: "
                f"{', '.join(found[:4])}…) — phishing analizine uygun.",
                "pasted_email",
                status=RoutingStatus.ACCEPTED,
                reason_code="raw_email_headers",
                confidence_source=ConfidenceSource.DETERMINISTIC,
                raw_email=text,
            )

    # Trust is established outside this function. There is deliberately no
    # user-controlled `route_hint_trusted=true` switch: writing true cannot
    # create authority.
    normalised_hint = (trusted_route_hint or "").strip().lower()
    if normalised_hint:
        if normalised_hint == "phishing":
            return _decision(
                Route.PHISHING_MISSING_EMAIL,
                "Phishing hattı seçildi fakat analiz edilecek e-posta yok. "
                ".eml dosyasını veya header'larıyla ham e-postayı gönderin.",
                "trusted_route_hint_without_email",
                status=RoutingStatus.MISSING_INPUT,
                reason_code="trusted_route_hint",
                confidence_source=ConfidenceSource.TRUSTED_METADATA,
            )
        return _decision(
            Route.UNSUPPORTED,
            f"Güvenilir route hint '{normalised_hint}' bu ilk sürümde "
            "desteklenmiyor; yalnızca phishing e-posta analizi çalışıyor.",
            "trusted_route_hint_unsupported",
            status=RoutingStatus.NOT_ROUTED,
            reason_code="unsupported_intent",
            confidence_source=ConfidenceSource.TRUSTED_METADATA,
            unsupported_scope=normalised_hint,
        )

    if _trusted_system_requests_phishing(trusted_system_message):
        return _decision(
            Route.PHISHING_MISSING_EMAIL,
            "Güvenilir system mesajı phishing uzmanını seçiyor fakat analiz "
            "edilecek e-posta yok. .eml dosyasını veya ham e-postayı gönderin.",
            "trusted_system_intent_without_email",
            status=RoutingStatus.MISSING_INPUT,
            reason_code="trusted_system_intent",
            confidence_source=ConfidenceSource.TRUSTED_METADATA,
        )

    if text is not None and text.strip():
        # Static rules are exhausted: this is prose, and telling "check this
        # IP" from "how does SPF work" needs the request understood rather
        # than its shape matched. Stage two only runs here — the model must
        # not sit on the path every request takes.
        if use_classifier:
            return _route_via_classifier(text)

        return _decision(
            Route.UNSUPPORTED,
            "Girdi düz metin, e-posta değil. Bu sistemde yalnızca phishing "
            "analizi hattı çalışıyor ve o hat bir e-posta gerektiriyor "
            "(.eml dosyası ya da header'larıyla birlikte yapıştırılmış "
            "ham mail). Genel siber güvenlik soruları için ayrı bir persona "
            "planlanıyor ama henüz kurulmadı.",
            "text_not_email",
            status=RoutingStatus.NOT_ROUTED,
            reason_code="unsupported_intent",
            confidence_source=ConfidenceSource.DETERMINISTIC,
            unsupported_scope="plain_text_without_email",
        )

    return _decision(
        Route.UNSUPPORTED, "Girdi boş.", "empty_input",
        status=RoutingStatus.NOT_ROUTED,
        reason_code="empty_input",
        confidence_source=ConfidenceSource.NONE,
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Girdiyi phishing analiz hattına yönlendirip yönlendiremeyeceğini belirler.")
    ap.add_argument("file", nargs="?", type=Path, help="analiz edilecek dosya")
    ap.add_argument("--text", help="düz metin ya da yapıştırılmış ham e-posta")
    ap.add_argument("--stdin", action="store_true", help="girdiyi stdin'den oku")
    ap.add_argument("--classify", action="store_true",
                    help="düz metin geldiğinde niyet sınıflandırıcıyı çalıştır\n"
                         "(küçük model, ~0.9GB; statik kural karar veremezse)")
    ap.add_argument("--run", action="store_true",
                    help="PHISHING'e yönlendiyse demo'yu da çalıştır")
    args = ap.parse_args()

    text = args.text
    if args.stdin:
        text = sys.stdin.read()

    decision = route(file_path=args.file, text=text,
                     use_classifier=args.classify)

    icon = "✓" if decision.accepted else "✗"
    print(f"{icon} YÖNLENDİRME: {decision.route.value}")
    print(f"  durum : {decision.status.value}")
    print(f"  kod   : {decision.reason_code}")
    print(f"  kural  : {decision.matched_rule}")
    print(f"  gerekçe: {decision.reason}")

    if not decision.accepted:
        sys.exit(1)

    print()
    print("  → phishing alt sistemi: parser → rule engine → Seneca → template")

    if not args.run:
        return

    if decision.eml_path is None:
        print("\n--run yalnızca dosya girdisiyle çalışıyor; yapıştırılmış mail "
              "için önce .eml olarak kaydedin.", file=sys.stderr)
        sys.exit(1)

    import subprocess
    print()
    # demo.py writes its progress to stderr, which is unbuffered, while the
    # routing lines above went to buffered stdout — without this flush the
    # subprocess output appears first and the routing decision looks like it
    # happened afterwards.
    sys.stdout.flush()
    subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "src" / "demo.py"),
         str(decision.eml_path), "--no-llm"],
        check=False,
    )


if __name__ == "__main__":
    main()
