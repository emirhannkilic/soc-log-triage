"""
Minimal input router — the llm-proxy layer's first stage, scoped down to
what this repository can actually serve.

WHY THIS IS DELIBERATELY SMALL
    The wider design (llm-proxy-notes.md) routes to three personas:
    phishing, threat intel (Titus) and general cyber Q&A. Only the phishing
    pipeline exists here — parser → rule engine → Seneca → template. So this
    router answers one question honestly:

        Is this input something the phishing pipeline can process?

    Everything else resolves to UNSUPPORTED with a reason, rather than
    pretending a persona exists. A router that claims to dispatch to models
    that were never built would misrepresent the system.

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
    PHISHING = "phishing"
    UNSUPPORTED = "unsupported"


@dataclass
class RoutingDecision:
    route: Route
    reason: str
    # How the decision was reached, so a caller (or a demo) can show the
    # path rather than just the outcome.
    matched_rule: str
    # Populated only for PHISHING: what the pipeline should read.
    eml_path: Path | None = None
    raw_email: str | None = None


EMAIL_EXTENSIONS = (".eml", ".msg")

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
        return RoutingDecision(
            Route.UNSUPPORTED,
            f"Girdi düz metin ve niyet sınıflandırıcı yüklenemedi ({e}).",
            "classifier_unavailable")

    r = classify(text)
    detail = f"[{r.persona} %{r.confidence*100:.0f}]"

    if r.persona == "phishing":
        return RoutingDecision(
            Route.UNSUPPORTED,
            f"{detail} Phishing analizi istendiği anlaşıldı ama girdide bir "
            f"e-posta yok. Analiz için .eml dosyasını ekleyin ya da mailin "
            f"ham halini (header'larıyla birlikte) yapıştırın.",
            "intent_phishing_no_email")

    if r.persona == "alakasiz":
        return RoutingDecision(
            Route.UNSUPPORTED,
            f"{detail} Bu istek siber güvenlikle ilgili görünmüyor. Bu "
            f"sistemde yalnızca phishing analizi hattı çalışıyor — bir "
            f"e-posta paylaşırsanız analiz edebilirim."
            + (f" ({r.fallback_reason})" if r.fallback_reason else ""),
            "intent_alakasiz")

    return RoutingDecision(
        Route.UNSUPPORTED,
        f"{detail} Bu istek '{r.persona}' personasına ait görünüyor, ama o "
        f"persona bu repoda kurulmadı — yalnızca phishing hattı çalışıyor."
        + (f" ({r.fallback_reason})" if r.fallback_reason else ""),
        f"intent_{r.persona}")


def route(file_path: Path | None = None, text: str | None = None,
          use_classifier: bool = False) -> RoutingDecision:
    """Decide whether the phishing pipeline can handle this input.

    Checked in order of certainty: an email file is unambiguous, a pasted
    message is nearly so, anything else is not an email at all.

    use_classifier turns on stage two for prose that reached the end of the
    static rules. Off by default: it loads a model, and most callers only
    need the structural answer.
    """
    if file_path is not None:
        if not file_path.is_file():
            return RoutingDecision(
                Route.UNSUPPORTED,
                f"Dosya bulunamadı: {file_path}",
                "file_missing",
            )
        if file_path.suffix.lower() in EMAIL_EXTENSIONS:
            return RoutingDecision(
                Route.PHISHING,
                f"E-posta dosyası ({file_path.suffix}) — phishing analizine uygun.",
                "file_extension",
                eml_path=file_path,
            )
        # A file that is not named like an email may still be one; the
        # content check is the authority, the extension is only a shortcut.
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            return RoutingDecision(
                Route.UNSUPPORTED, f"Dosya okunamadı: {e}", "file_unreadable")
        if looks_like_raw_email(content):
            return RoutingDecision(
                Route.PHISHING,
                f"Uzantı e-posta değil ({file_path.suffix or 'yok'}) ama içerik "
                f"ham e-posta: {sorted(_distinct_headers(content))[:4]} "
                f"header'ları bulundu.",
                "file_content",
                eml_path=file_path,
            )
        return RoutingDecision(
            Route.UNSUPPORTED,
            f"Bu bir e-posta dosyası değil ({file_path.suffix or 'uzantısız'}). "
            f"Phishing analizi .eml/.msg ya da ham e-posta metni gerektiriyor.",
            "file_not_email",
        )

    if text is not None and text.strip():
        if looks_like_raw_email(text):
            found = sorted(_distinct_headers(text))
            return RoutingDecision(
                Route.PHISHING,
                f"Yapıştırılmış ham e-posta ({len(found)} header: "
                f"{', '.join(found[:4])}…) — phishing analizine uygun.",
                "pasted_email",
                raw_email=text,
            )

        # Static rules are exhausted: this is prose, and telling "check this
        # IP" from "how does SPF work" needs the request understood rather
        # than its shape matched. Stage two only runs here — the model must
        # not sit on the path every request takes.
        if use_classifier:
            return _route_via_classifier(text)

        return RoutingDecision(
            Route.UNSUPPORTED,
            "Girdi düz metin, e-posta değil. Bu sistemde yalnızca phishing "
            "analizi hattı çalışıyor ve o hat bir e-posta gerektiriyor "
            "(.eml dosyası ya da header'larıyla birlikte yapıştırılmış "
            "ham mail). Genel siber güvenlik soruları için ayrı bir persona "
            "planlanıyor ama henüz kurulmadı.",
            "text_not_email",
        )

    return RoutingDecision(
        Route.UNSUPPORTED, "Girdi boş.", "empty_input")


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

    icon = "✓" if decision.route is Route.PHISHING else "✗"
    print(f"{icon} YÖNLENDİRME: {decision.route.value}")
    print(f"  kural  : {decision.matched_rule}")
    print(f"  gerekçe: {decision.reason}")

    if decision.route is not Route.PHISHING:
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
