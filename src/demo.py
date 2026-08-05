"""
v3 plan Adım 11 — end-to-end demo. One .eml file in, one HTML report out.

Runs the full v3 pipeline in the order CLAUDE.md "Mimari" locks down:

    .eml → parser → rule engine → LLM (report writer) → Jinja2 → HTML

The division of labour is the point of the project and is preserved here:
the parser and the rule engine DECIDE (verdict + score), the LLM only
narrates that decision in Turkish, and the template — never the model —
emits HTML.

MODEL SELECTION
    Seneca is always the model. `--adapter` only controls whether the
    LoRA adapter is layered on top of it, and the default is OFF because
    Adım 10 measured the adapter as worse on both metrics:

        baseline (no adapter)   schema 100.0%   groundedness 67.4%
        0000400 (few-shot)      schema  92.6%   groundedness 54.7%

    `--adapter 0000400` is kept so the two can be demonstrated side by
    side; the flag takes a checkpoint id from models/lora_adapters/.

    Few-shot examples are always included in the prompt: that is the
    condition both the baseline and the better adapter run were measured
    under (see src/eval/finetuned.py for why both prompt modes exist).

--no-llm
    Skips the model entirely and fills the report mechanically from the
    fired rules, reusing scripts/render_holdout_reports.py's build_report.
    The verdict, score and findings are identical — only the prose is
    mechanical instead of written. Runs in about a second instead of two
    minutes, which makes it the right mode for checking the pipeline,
    the template, or a new .eml quickly.

--constrain (opt-in, default OFF)
    Constrains generation to the Report JSON schema via llguidance, making
    malformed JSON structurally impossible rather than merely asked for.

    Off by default on purpose: every number this project reports — Adım 8's
    schema compliance, Adım 10's comparison — was measured with
    unconstrained generation. A demo running under different conditions
    than the measurements would misrepresent both.

    It is not the forbidden "repair the output": nothing is fixed after the
    fact, the grammar restricts token choice during generation, and the
    risk_seviyesi check still runs afterwards. The schema guarantees shape,
    never correctness.

    When it earns its keep: on sample-8611 three consecutive unconstrained
    runs produced invalid JSON the same way — an unescaped double quote
    inside a string value — and a prompt rule against it did not help.

Usage:
    python3 src/demo.py mail.eml                      # Seneca, no adapter
    python3 src/demo.py mail.eml --no-llm             # no model at all
    python3 src/demo.py mail.eml --adapter 0000400    # with LoRA adapter
    python3 src/demo.py mail.eml --constrain          # şema kısıtlı üretim
    python3 src/demo.py mail.eml -o /tmp/rapor.html --open
"""
import argparse
import json
import re
import sys
import time
import webbrowser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import jinja2  # noqa: E402

from schemas.facts import EmailFacts  # noqa: E402
from schemas.report import Report  # noqa: E402
from src.parser.parse import parse_eml  # noqa: E402
from src.rules.engine import Verdict, evaluate, load_rules  # noqa: E402

MODEL_PATH = PROJECT_ROOT / "models" / "Seneca-Cybersecurity-LLM_x_Qwen2.5-7B-CyberSecurity-mlx-4bit"
ADAPTER_DIR = PROJECT_ROOT / "models" / "lora_adapters"
TEMPLATE_PATH = PROJECT_ROOT / "templates" / "report.html.j2"
CANDIDATES_PATH = PROJECT_ROOT / "data" / "holdout" / "candidates.jsonl"

# The same hold-out emails used as few-shot examples during teacher
# generation and in both evaluation runs. Keeping them identical here means
# the demo reproduces the conditions the reported numbers were measured
# under, rather than a third, unmeasured prompt shape.
FEW_SHOT_INDICES = (1, 20)
METADATA_KEYS = ("source_label", "_eml_path", "is_spam_not_phishing", "spam_reason")

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _render(report: Report, facts: EmailFacts, out_path: Path) -> None:
    # autoescape=True rather than select_autoescape: the template is named
    # report.html.j2 and select_autoescape keys off the LAST extension
    # (".j2"), which is not in its HTML list — it would silently disable
    # escaping on a file that always renders HTML. This was a real XSS hole
    # found in Adım 5; see tests/test_report.py.
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(TEMPLATE_PATH.parent),
        autoescape=True,
    )
    template = env.get_template(TEMPLATE_PATH.name)
    html = template.render(
        **report.model_dump(),
        subject=facts.subject,
        date=facts.date,
        # The künye block reads straight from facts, bypassing the model —
        # sender, auth results, URLs and attachments are shown as parsed,
        # so nothing in that section can be hallucinated.
        facts=facts,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")


def _dump_raw_on_failure(raw: str) -> None:
    """Diagnostic-only: on invalid/off-schema output, save what the model
    actually wrote so the failure can be inspected. Not a repair path — the
    dumped text is never fed back in or parsed differently; CLAUDE.md's
    "çıktıyı onarma" rule is about not patching output into an accepted
    report, which this does not do."""
    path = PROJECT_ROOT / "logs" / "last_llm_failure_raw.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(raw, encoding="utf-8")
    print(f"      (ham model çıktısı kaydedildi: {path})", file=sys.stderr)


def _extract_json(raw_text: str) -> dict | None:
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[len("json"):]
    match = _JSON_BLOCK_RE.search(text)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _load_few_shot(rules: dict):
    """Rebuild the (facts, verdict, report) triples used as few-shot examples."""
    from src.teacher.few_shot_examples import (
        FEW_SHOT_GUVENILIR,
        FEW_SHOT_PHISHING,
    )

    reports = {1: FEW_SHOT_PHISHING, 20: FEW_SHOT_GUVENILIR}
    with open(CANDIDATES_PATH, encoding="utf-8") as f:
        candidates = [json.loads(line) for line in f]

    few_shot = []
    for idx in FEW_SHOT_INDICES:
        cand = candidates[idx - 1]
        fs_facts = EmailFacts(**{k: v for k, v in cand.items() if k not in METADATA_KEYS})
        fs_verdict = evaluate(fs_facts.flat_signals(), rules)
        few_shot.append((fs_facts, fs_verdict, reports[idx]))
    return few_shot


def _resolve_adapter(checkpoint: str) -> Path:
    """Stage a checkpoint into a directory mlx_lm can load.

    mlx_lm wants a directory holding adapter_config.json plus
    adapters.safetensors; intermediate checkpoints are saved as
    <id>_adapters.safetensors next to one shared config, so each is copied
    into its own subdirectory under the expected filename.
    """
    weights = (ADAPTER_DIR / "adapters.safetensors" if checkpoint in ("final", "adapters")
               else ADAPTER_DIR / f"{checkpoint}_adapters.safetensors")
    if not weights.is_file():
        raise SystemExit(f"checkpoint bulunamadı: {weights}")
    config = ADAPTER_DIR / "adapter_config.json"
    if not config.is_file():
        raise SystemExit(f"adapter_config.json eksik: {ADAPTER_DIR}")

    staged = ADAPTER_DIR / f"_demo_{checkpoint}"
    staged.mkdir(exist_ok=True)
    (staged / "adapter_config.json").write_bytes(config.read_bytes())
    target = staged / "adapters.safetensors"
    if not target.is_file() or target.stat().st_size != weights.stat().st_size:
        target.write_bytes(weights.read_bytes())
    return staged


def _report_from_llm(facts: EmailFacts, verdict: Verdict, rules: dict,
                     adapter: str | None, constrain: bool = False) -> Report:
    """Generate the report with Seneca. Raises SystemExit on failure.

    No retry and no output patching, deliberately. CLAUDE.md
    "Yapılmayacaklar" forbids repairing malformed model output — a
    schema violation is a real result and hiding it would misrepresent
    what the model does. Adım 10 measured this failing on 2 of 27 emails
    without the adapter.
    """
    from mlx_lm import generate, load

    from src.teacher.prompts import build_messages

    adapter_dir = _resolve_adapter(adapter) if adapter else None
    label = f"Seneca + adapter {adapter}" if adapter else "Seneca (adapter yok)"
    print(f"[2/4] {label} yükleniyor …", file=sys.stderr)
    model, tokenizer = load(str(MODEL_PATH),
                            adapter_path=str(adapter_dir) if adapter_dir else None)

    messages = build_messages(facts, verdict, few_shot=_load_few_shot(rules))
    prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                                           add_generation_prompt=True)

    # Constrained decoding is OPT-IN (--constrain), not the default.
    #
    # The default has to be the model's real behaviour. Every number this
    # project reports — Adım 8's 100% schema compliance, Adım 10's
    # comparison — was measured with unconstrained generation, and a demo
    # that quietly runs under different conditions than the measurements
    # would misrepresent both.
    #
    # What --constrain does: llguidance restricts each token choice to what
    # the Report schema allows, making malformed JSON structurally
    # impossible instead of merely discouraged. It is not the forbidden
    # "patch the output" — nothing is repaired afterwards, the grammar
    # limits generation itself, and the risk_seviyesi check below still
    # runs. The schema guarantees shape, never correctness.
    #
    # When it earns its keep: on sample-8611 three consecutive unconstrained
    # runs produced invalid JSON the same way — an unescaped double quote
    # inside a string value ("E-posta, "güvenli" bir …"). A prompt rule
    # against it did not help, because the model does not track "I am inside
    # a JSON string right now"; the grammar tracks it externally.
    logits_processors = None
    if constrain:
        # No silent fallback: if the constraint cannot be built, say so and
        # stop. Quietly reverting to unconstrained generation would look
        # like the constraint is active while the old failure mode is still
        # there — the exact confusion this feature exists to end.
        try:
            from mlx_vlm.structured import build_json_schema_logits_processor
        except ImportError as e:
            raise SystemExit(
                f"HATA: --constrain için mlx_vlm gerekiyor ({e}).\n"
                f"Bayrağı kaldırırsanız kısıtsız (varsayılan) modda çalışır.")

        # llguidance needs the HuggingFace tokenizer itself, not mlx_lm's
        # TokenizerWrapper — passing the wrapper raises "Only fast
        # tokenizers are supported". The wrapper is still what generate()
        # uses; this is only for building the grammar.
        from transformers import AutoTokenizer
        hf_tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
        logits_processors = [
            build_json_schema_logits_processor(
                hf_tokenizer, Report.model_json_schema())
        ]

    suffix = " — şema kısıtlı" if logits_processors else ""
    print(f"[3/4] Rapor yazılıyor{suffix} (bu adım ~2 dakika sürebilir) …",
          file=sys.stderr)
    t0 = time.time()
    kwargs = {"max_tokens": 1200, "verbose": False}
    if logits_processors:
        kwargs["logits_processors"] = logits_processors
    raw = generate(model, tokenizer, prompt, **kwargs)
    print(f"      {time.time() - t0:.0f} saniye", file=sys.stderr)

    parsed = _extract_json(raw)
    if parsed is None:
        _dump_raw_on_failure(raw)
        raise SystemExit(
            "HATA: model geçerli JSON üretmedi.\n"
            "Bu bilinen bir sınır — çıktı onarılmıyor (bkz. CLAUDE.md "
            "'Yapılmayacaklar').\n"
            "--no-llm ile mekanik rapor üretebilir ya da tekrar deneyebilirsiniz."
        )
    try:
        report = Report(**parsed)
    except Exception as e:
        _dump_raw_on_failure(raw)
        raise SystemExit(f"HATA: model çıktısı rapor şemasına uymuyor: {e}")

    # The model must echo the rule engine's decision, never override it.
    # This is the check that makes "the LLM does not classify" enforceable
    # rather than merely stated.
    if report.risk_seviyesi != verdict.verdict:
        raise SystemExit(
            f"HATA: model risk_seviyesi'ni '{report.risk_seviyesi}' yazdı ama "
            f"rule engine '{verdict.verdict}' dedi. Karar rule engine'in; "
            f"uyuşmayan çıktı kabul edilmiyor."
        )
    return report


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Bir .eml dosyasını analiz edip HTML rapor üretir.")
    ap.add_argument("eml", type=Path, help="analiz edilecek .eml dosyası")
    ap.add_argument("-o", "--out", type=Path,
                    help="çıktı HTML yolu (varsayılan: <eml adı>_rapor.html)")
    ap.add_argument("--no-llm", action="store_true",
                    help="modeli hiç çağırma, raporu kurallardan mekanik üret (~1 sn)")
    ap.add_argument("--adapter", metavar="CHECKPOINT",
                    help="LoRA adapter'ı yükle (örn. 0000400). Varsayılan kapalı — "
                         "Adım 10'da adapter her iki metrikte de baseline'dan kötü çıktı.")
    ap.add_argument("--open", action="store_true",
                    help="raporu tarayıcıda aç")
    ap.add_argument("--constrain", action="store_true",
                    help="çıktıyı JSON şemasına kısıtla — geçersiz JSON'u yapısal olarak\n"
                         "imkansız kılar. Varsayılan KAPALI: ölçümler kısıtsız modda\n"
                         "yapıldı, demo da aynı koşulda çalışmalı.")
    args = ap.parse_args()

    if not args.eml.is_file():
        raise SystemExit(f"dosya bulunamadı: {args.eml}")
    if args.no_llm and args.adapter:
        raise SystemExit("--no-llm ile --adapter birlikte kullanılamaz "
                         "(--no-llm modelin tamamını atlıyor).")

    out_path = args.out or args.eml.with_name(f"{args.eml.stem}_rapor.html")

    print(f"[1/4] Parse ediliyor: {args.eml.name}", file=sys.stderr)
    facts = parse_eml(args.eml)
    rules = load_rules()
    signals = facts.flat_signals()
    verdict = evaluate(signals, rules)

    thresholds = rules["thresholds"]
    print(f"      Karar: {verdict.verdict} (skor {verdict.score}; "
          f"eşikler: >={thresholds['phishing']} Phishing, "
          f">={thresholds['suspicious']} Muhtemel)", file=sys.stderr)
    for m in verdict.matches:
        sign = "+" if m.weight >= 0 else ""
        print(f"        {sign}{m.weight:>3}  {m.description}", file=sys.stderr)
    if not verdict.matches:
        print("        (hiçbir sinyal tetiklenmedi)", file=sys.stderr)

    if args.no_llm:
        from render_holdout_reports import build_report
        print("[2/4] LLM atlandı (--no-llm)", file=sys.stderr)
        print("[3/4] Rapor kurallardan mekanik üretiliyor", file=sys.stderr)
        report = build_report(signals, verdict)
    else:
        report = _report_from_llm(facts, verdict, rules, args.adapter,
                                  constrain=args.constrain)

    _render(report, facts, out_path)
    print(f"[4/4] Rapor yazıldı: {out_path}", file=sys.stderr)

    if args.open:
        webbrowser.open(out_path.resolve().as_uri())


if __name__ == "__main__":
    main()
