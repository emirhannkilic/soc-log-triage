"""
Web arayüzü — terminal yerine tarayıcıdan kullanmak için ince bir katman.

Hiçbir analiz mantığı burada YOK. İstek şu sırayla mevcut parçalara
devrediliyor:

    src/router.py             → girdi phishing hattına uygun mu?
    src/workflows/phishing.py → .eml → facts → karar → rapor (KARAR BURADA)
    templates/                → rapor → HTML

Yani bu dosya bir kabuk: yeni bir karar yolu açmıyor, hem "fast" hem
"hybrid" modunu src/workflows/phishing.py::analyze_phishing()'e
devrediyor — bu dosyada ikinci bir analiz implementasyonu YOK.

İKİ MOD
    fast (varsayılan): LLM hiç çağrılmaz, rapor kurallardan mekanik
        üretilir — ~1 saniye. analyze_phishing(mode="fast").
    hybrid: rule engine v1 → semantic extraction (Qwen3.5-9B) →
        decision policy → mekanik rapor + (final_verdict != "Güvenilir"
        ise) Qwen'in dar narrative katkısı — M2 Air'de ~70 saniye.
        analyze_phishing(mode="hybrid"). Model süreçte tutuluyor
        (src/llm/service.py::get_service() singleton'ı), ikinci istek
        daha hızlıdır.

    Önceki bir "llm" modu (Seneca + eski teacher prompt'u,
    src/demo.py'nin --adapter/--constrain yolu) burada ARTIK YOK — o
    yol hiçbir zaman hybrid mimariyle (Qwen3.5-9B semantic extraction +
    decision policy) bağlı değildi, ayrı bir demo/eğitim hattıydı.
    src/demo.py'de dosya/kod olarak hâlâ duruyor, sadece web
    erişimi kaldırıldı — iki farklı "LLM analizi" kavramının web
    arayüzünde karışmaması için.

Çalıştırma:
    python3 src/web.py
    python3 src/web.py --port 8080 --host 0.0.0.0
"""
import argparse
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import jinja2  # noqa: E402
from fastapi import FastAPI, File, Form, UploadFile  # noqa: E402
from fastapi.responses import HTMLResponse, JSONResponse  # noqa: E402

from schemas.facts import EmailFacts  # noqa: E402
from schemas.report import Report  # noqa: E402
from src.router import route  # noqa: E402
from src.workflows.phishing import PhishingAnalysisResult, analyze_phishing  # noqa: E402

TEMPLATE_PATH = PROJECT_ROOT / "templates" / "report.html.j2"
UI_PATH = Path(__file__).resolve().parent / "web_ui.html"

app = FastAPI(title="soc-log-triage")


def _render_report_html(report: Report, facts: EmailFacts) -> str:
    # autoescape=True rather than select_autoescape: the template is named
    # report.html.j2 and select_autoescape keys off the LAST extension
    # (".j2"), silently disabling escaping on a file that always renders
    # HTML. This was a real XSS hole found in Adım 5 — and it matters more
    # here than in the CLI, since this output goes straight into a browser.
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(TEMPLATE_PATH.parent), autoescape=True)
    template = env.get_template(TEMPLATE_PATH.name)
    return template.render(
        **report.model_dump(),
        subject=facts.subject,
        date=facts.date,
        facts=facts,
    )


def _semantic_findings_json(analysis: PhishingAnalysisResult) -> dict:
    """Serializes accepted/rejected semantic findings for the JSON
    response — kept as a small, explicit projection rather than
    model_dump()'ing the raw objects, since rejected_findings'
    `finding` field can hold an arbitrary JSON-decoded value (a dict,
    str, or even None — see src/semantic/validate.py's ValidatedFinding
    docstring), which is not always safely JSON-serializable as-is."""
    accepted = [
        {
            "type": f.type.value,
            "evidence": f.evidence,
            "explanation": f.explanation,
            "model_confidence": f.model_confidence,
        }
        for f in analysis.accepted_findings
    ]
    rejected = [
        {
            "rejection_reason": vf.rejection_reason.value if vf.rejection_reason else None,
            # finding is best-effort here — a raw, not-yet-validated
            # candidate can be almost any JSON-decoded shape.
            "finding": vf.finding if isinstance(vf.finding, (dict, str, int, float, bool, type(None)))
            else str(vf.finding),
        }
        for vf in analysis.rejected_findings
    ]
    return {"accepted": accepted, "rejected": rejected}


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return UI_PATH.read_text(encoding="utf-8")


@app.post("/analyze")
async def analyze(
    text: str = Form(default=""),
    mode: str = Form(default="fast"),
    file: UploadFile | None = File(default=None),
) -> JSONResponse:
    t0 = time.time()

    if mode not in ("fast", "hybrid"):
        return JSONResponse({"error": f"bilinmeyen mode: {mode!r}"}, status_code=400)

    # --- 1. Routing -------------------------------------------------------
    tmp_path = None
    if file is not None and file.filename:
        raw = await file.read()
        suffix = Path(file.filename).suffix or ".eml"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(raw)
            tmp_path = Path(tmp.name)
        decision = route(file_path=tmp_path)
    else:
        # use_classifier=True: static rules run first regardless (file
        # extension / pasted-header detection) and only fall through to the
        # small model when text matches neither — same behavior as
        # `router.py --classify`, just always enabled here since a web
        # submission is already a deliberate, infrequent action (unlike the
        # CLI router, which runs on every request).
        decision = route(text=text, use_classifier=True)

    routing = decision.as_dict()
    # Temporary compatibility keys for the current UI. New consumers should
    # use matched_rule/message from the stable routing contract.
    routing["rule"] = decision.matched_rule
    routing["reason"] = decision.reason

    if not decision.accepted:
        if tmp_path:
            tmp_path.unlink(missing_ok=True)
        return JSONResponse({"routing": routing, "accepted": False})

    # --- 2. Parse ---------------------------------------------------------
    # A pasted message has no file on disk, so it is written to one: the
    # parser's entry point takes a path, and reimplementing it for strings
    # would mean two code paths that could drift apart.
    if tmp_path is None:
        with tempfile.NamedTemporaryFile(suffix=".eml", mode="w",
                                         encoding="utf-8", delete=False) as tmp:
            tmp.write(decision.raw_email or "")
            tmp_path = Path(tmp.name)

    # --- 3/4. Analysis + report --------------------------------------
    # Both modes go through the SAME shared workflow
    # (src/workflows/phishing.py::analyze_phishing) — this endpoint is
    # not a second implementation of parse -> rule engine -> decision ->
    # report for either mode.
    try:
        analysis = analyze_phishing(tmp_path, mode=mode)
    except Exception as e:
        tmp_path.unlink(missing_ok=True)
        return JSONResponse(
            {"routing": routing, "accepted": True,
             "error": f"E-posta analiz edilemedi: {e}"},
            status_code=400)

    facts = analysis.facts
    rule_assessment = analysis.rule_assessment
    report = analysis.report

    signals = [{"weight": e.weight, "description": e.description}
               for e in rule_assessment.evidence]
    html = _render_report_html(report, facts)
    tmp_path.unlink(missing_ok=True)

    fd = analysis.final_decision
    response = {
        "routing": routing,
        "accepted": True,
        "mode": mode,
        # Kural motorunun HAM kararı — hybrid modda semantik katman
        # tarafından yükseltilmiş olabilecek final_verdict'ten AYRI
        # gösteriliyor, ikisi arasındaki farkın kendisi bir bilgi.
        "verdict": rule_assessment.rule_verdict,
        "score": rule_assessment.score,
        "signals": signals,
        "report_html": html,
        "elapsed": round(time.time() - t0, 1),
        # Aşağıdaki alanların TAMAMI fast modda None/boş kalır —
        # analyze_phishing(mode="fast") hiçbir zaman final_decision/
        # semantic_status üretmiyor (bkz. o fonksiyonun docstring'i).
        "semantic_status": analysis.semantic_status,
        "semantic_skip_reason": analysis.semantic_skip_reason,
        "semantic_findings": _semantic_findings_json(analysis),
        "final_verdict": fd.final_verdict if fd else None,
        "decision_path": fd.decision_path if fd else None,
        "analyst_review_required": fd.analyst_review_required if fd else None,
        "report_source": analysis.report_source,
        "narrative_status": analysis.narrative_status,
        "narrative_error_code": analysis.narrative_error_code,
    }
    return JSONResponse(response)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    import uvicorn
    print(f"\n  http://{args.host}:{args.port}\n", file=sys.stderr)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
