"""
Web arayüzü — terminal yerine tarayıcıdan kullanmak için ince bir katman.

Hiçbir analiz mantığı burada YOK. İstek şu sırayla mevcut parçalara
devrediliyor:

    src/router.py   → girdi phishing hattına uygun mu?
    src/parser      → .eml → facts
    src/rules       → facts → verdict (KARAR BURADA)
    src/teacher     → facts + verdict → Türkçe JSON rapor (opsiyonel)
    templates/      → rapor → HTML

Yani bu dosya bir kabuk: yeni bir karar yolu açmıyor, CLI'ın yaptığı işi
tarayıcıya taşıyor. Kararın nerede verildiği değişmiyor.

İKİ MOD
    hızlı (varsayılan): LLM hiç çağrılmaz, rapor kurallardan mekanik
        üretilir — ~1 saniye. Verdict, skor ve bulgular LLM'li moddaki ile
        BİREBİR aynı; farklı olan yalnızca metnin yazılmış değil mekanik
        olması.
    llm: Seneca raporu yazar — ~100 saniye. Model yüklü kalır, ikinci
        istek daha hızlıdır.

Model, ilk LLM isteğinde yükleniyor ve süreçte tutuluyor: her istekte
4GB'lık modeli yeniden yüklemek dakikalar sürerdi.

Çalıştırma:
    python3 src/web.py
    python3 src/web.py --port 8080 --host 0.0.0.0
"""
import argparse
import json
import re
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
from src.parser.parse import parse_eml  # noqa: E402
from src.report.mechanical import build_report  # noqa: E402
from src.router import route  # noqa: E402
from src.rules.adapters import from_v1  # noqa: E402
from src.rules.engine import evaluate, load_rules  # noqa: E402
from src.workflows.phishing import analyze_phishing  # noqa: E402

MODEL_PATH = (PROJECT_ROOT / "models" /
              "Seneca-Cybersecurity-LLM_x_Qwen2.5-7B-CyberSecurity-mlx-4bit")
TEMPLATE_PATH = PROJECT_ROOT / "templates" / "report.html.j2"
UI_PATH = Path(__file__).resolve().parent / "web_ui.html"

app = FastAPI(title="soc-log-triage")

# Loaded lazily and kept for the process lifetime — a 4GB model reloaded per
# request would cost minutes each time.
_model = None
_tokenizer = None
_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


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


def _extract_json(raw_text: str) -> dict | None:
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[len("json"):]
    m = _JSON_BLOCK_RE.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _load_model():
    global _model, _tokenizer
    if _model is None:
        from mlx_lm import load
        print("Seneca yükleniyor (ilk LLM isteği)…", file=sys.stderr)
        _model, _tokenizer = load(str(MODEL_PATH))
        print("Model yüklendi.", file=sys.stderr)
    return _model, _tokenizer


def _report_via_llm(facts: EmailFacts, verdict,
                    constrain: bool = False) -> tuple[Report | None, str | None]:
    """Returns (report, error). Never repairs malformed output.

    CLAUDE.md "Yapılmayacaklar" forbids patching model output: a schema
    violation is a real result, and hiding it would misrepresent what the
    model does. Adım 10 measured this failing on 2 of 27 emails.
    """
    from mlx_lm import generate

    from src.demo import _load_few_shot
    from src.teacher.prompts import build_messages

    model, tokenizer = _load_model()
    rules = load_rules()
    messages = build_messages(facts, verdict, few_shot=_load_few_shot(rules))
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)

    # Opt-in, mirroring src/demo.py: the default has to be the model's real
    # behaviour, because that is the condition every reported number was
    # measured under.
    kwargs = {"max_tokens": 1200, "verbose": False}
    if constrain:
        from mlx_vlm.structured import build_json_schema_logits_processor
        from transformers import AutoTokenizer
        # llguidance needs the HuggingFace tokenizer, not mlx_lm's
        # TokenizerWrapper — the wrapper raises "Only fast tokenizers are
        # supported". generate() still uses the wrapper.
        hf_tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
        kwargs["logits_processors"] = [
            build_json_schema_logits_processor(
                hf_tokenizer, Report.model_json_schema())
        ]

    raw = generate(model, tokenizer, prompt, **kwargs)
    parsed = _extract_json(raw)
    if parsed is None:
        return None, ("Model geçerli JSON üretmedi. Çıktı onarılmıyor "
                      "(bilinçli karar) — hızlı modu deneyin ya da tekrar "
                      "çalıştırın.")
    try:
        report = Report(**parsed)
    except Exception as e:
        return None, f"Model çıktısı rapor şemasına uymuyor: {e}"

    # The model must echo the rule engine's decision, never override it.
    # This check is what makes "the LLM does not classify" enforceable.
    if report.risk_seviyesi != verdict.verdict:
        return None, (f"Model risk_seviyesi'ni '{report.risk_seviyesi}' yazdı "
                      f"ama rule engine '{verdict.verdict}' dedi. Karar rule "
                      f"engine'in; uyuşmayan çıktı kabul edilmiyor.")
    return report, None


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return UI_PATH.read_text(encoding="utf-8")


@app.post("/analyze")
async def analyze(
    text: str = Form(default=""),
    mode: str = Form(default="fast"),
    constrain: str = Form(default=""),
    file: UploadFile | None = File(default=None),
) -> JSONResponse:
    t0 = time.time()

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

    # --- 3/4. Rule engine + report ------------------------------------
    # "fast" goes through the shared workflow (src/workflows/phishing.py)
    # so this endpoint isn't a second implementation of
    # parse -> rule engine -> report. "llm" still parses and evaluates
    # separately: _report_via_llm needs the raw v1 Verdict to build a
    # Qwen-written report, which is a different thing from
    # analyze_phishing's mode="hybrid" (semantic extraction -> decision
    # policy -> still-mechanical report, no LLM report writer yet — see
    # analyze_phishing's docstring). Neither this endpoint's "llm" mode
    # nor the semantic/decision layer are wired together yet.
    rules = load_rules()
    if mode == "llm":
        try:
            facts = parse_eml(tmp_path)
        except Exception as e:
            tmp_path.unlink(missing_ok=True)
            return JSONResponse(
                {"routing": routing, "accepted": True,
                 "error": f"E-posta ayrıştırılamadı: {e}"},
                status_code=400)
        verdict = evaluate(facts.flat_signals(), rules)
        rule_assessment = from_v1(verdict, rules)
        report, error = _report_via_llm(facts, verdict,
                                        constrain=constrain == "1")
        if report is None:
            report = build_report(rule_assessment)
    else:
        try:
            analysis = analyze_phishing(tmp_path, mode="fast")
        except Exception as e:
            tmp_path.unlink(missing_ok=True)
            return JSONResponse(
                {"routing": routing, "accepted": True,
                 "error": f"E-posta ayrıştırılamadı: {e}"},
                status_code=400)
        facts = analysis.facts
        rule_assessment = analysis.rule_assessment
        report = analysis.report
        error = None

    signals = [{"weight": e.weight, "description": e.description}
               for e in rule_assessment.evidence]
    html = _render_report_html(report, facts)
    tmp_path.unlink(missing_ok=True)

    return JSONResponse({
        "routing": routing,
        "accepted": True,
        "verdict": rule_assessment.rule_verdict,
        "score": rule_assessment.score,
        "signals": signals,
        "thresholds": rules["thresholds"],
        "report_html": html,
        "mode": mode,
        "constrained": bool(constrain == "1" and mode == "llm"),
        "llm_error": error,
        "elapsed": round(time.time() - t0, 1),
    })


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
