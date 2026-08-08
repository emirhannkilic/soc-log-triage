"""
Live hybrid-report generation (PHISHING_ROUTING_PLAN.md section 10.5,
"Qwen çağrı 2"). Uses src/report/prompts.py (NOT the frozen
src/teacher/prompts.py) and the shared src/llm/service.py QwenService
(the same singleton src/semantic/analyze.py's "Qwen çağrı 1" already
uses within one hybrid request, per the plan's "tek model, iki çağrı").

WHAT THIS MODULE DOES NOT DO
    No retry, no output repair. CLAUDE.md's "Yapılmayacaklar" rule
    ("Model çıktısını post-processing/regex ile yamamak") and
    src/demo.py's _report_from_llm already apply this to the offline
    teacher/adapter path; this module holds the same line for the live
    hybrid path. A malformed or off-schema response is a real result,
    not something to paper over — generate_report() raises
    ReportGenerationError and stops. It is the CALLER's job
    (src/workflows/phishing.py) to fall back to the mechanical report
    and record that this happened; this module has no fallback logic
    and does not know the mechanical report exists.

generate_report() NEVER RETURNS A FAILURE VALUE
    Every failure mode is a raised ReportGenerationError, not an Optional
    return or a Report with an error field bolted on. This keeps the
    function's contract simple for its one caller: get back a valid,
    verdict-matching Report, or get an exception with a closed `code` to
    branch on. Threading a "did this actually succeed" signal through a
    successful-looking return value invites exactly the bug class this
    project's docstrings keep calling out elsewhere (a call site that
    forgets to check a status field and treats a degraded result as a
    real one).

CLOSED `code` VALUES (mirrors src/semantic/analyze.py's SemanticExtractionError)
    "model_call_failed"     the load/generate infrastructure itself
                              failed — wraps src/llm/service.py's
                              LLMServiceError, always via `raise ... from
                              exc`.
    "invalid_json"           generate() succeeded but the output could
                              not be parsed as a JSON object at all.
    "schema_invalid"         parsed JSON does not match schemas.report.Report
                              (extra field, wrong type, missing field).
    "verdict_mismatch"       the model's risk_seviyesi did not echo
                              decision.final_verdict. This is the
                              enforcement point for "the LLM does not
                              classify, it only narrates a given
                              decision" in the LIVE hybrid path — the
                              same invariant src/demo.py's
                              _report_from_llm already enforces for the
                              offline adapter path, checked here against
                              FinalDecision.final_verdict instead of a
                              raw Verdict.

TEMPERATURE=0, NO CONSTRAINED DECODING (this turn)
    Matches src/teacher/prompts.py's locked "Teacher generation ayarları"
    default and src/semantic/analyze.py's TEMPERATURE. Constrained JSON
    decoding (mlx_vlm's build_json_schema_logits_processor, already used
    by src/semantic/analyze.py and src/demo.py's --constrain) is a valid
    future option for this module too, but CLAUDE.md's alternative note
    on constrained decoding was recorded as "not yet applied, tested
    separately" for the teacher path — the same caution applies here:
    this is new code, measured unconstrained first, matching how every
    other number in this project was produced before a constraint was
    layered on top of it.
"""
import json
import re

from pydantic import ValidationError

from schemas.decision import FinalDecision
from schemas.facts import EmailFacts
from schemas.report import Report
from schemas.rule_assessment import RuleAssessment
from schemas.semantic import ValidatedSemanticFinding
from src.llm.service import LLMServiceError, QwenService, get_service
from src.report.prompts import build_messages

TEMPERATURE = 0
# src/demo.py's unconstrained runs use 1200 for the same reporting job
# (facts + decision -> Turkish Report JSON) against the larger teacher
# few-shot prompt; this prompt is shorter (no few-shot examples), but the
# target JSON shape is identical, so the same ceiling is kept rather than
# re-guessing a smaller number that risks truncating teknik_bulgular on a
# high-signal email the way CLAUDE.md's "Teacher generation ayarları"
# entry already recorded once for the offline path (max_tokens=800 cut
# off a high-signal email; raised to 1200).
MAX_TOKENS = 1200

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


class ReportGenerationError(Exception):
    """Raised by generate_report() for every failure mode it normalizes
    to — see module docstring's CLOSED `code` VALUES section. No retry,
    no repair; the caller decides what to do next (src/workflows/
    phishing.py falls back to the mechanical report)."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


def _extract_json(raw_text: str) -> dict | None:
    """Mirrors src/demo.py's _extract_json / src/semantic/analyze.py's
    extract_json_array — best-effort location of a JSON object in raw
    model output, never a repair of malformed content. Returns None
    (not an exception) on anything that doesn't parse as a JSON object;
    generate_report() is the one that turns that into
    ReportGenerationError."""
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


def generate_report(
    facts: EmailFacts,
    rule_assessment: RuleAssessment,
    decision: FinalDecision,
    accepted_findings: list[ValidatedSemanticFinding],
    service: QwenService | None = None,
) -> Report:
    """Runs Qwen3.5-9B once (via the shared QwenService — same instance
    src/semantic/analyze.py's "Qwen çağrı 1" already loaded within this
    same hybrid request, per the plan's "tek model, iki çağrı") and
    returns a Report whose risk_seviyesi is guaranteed to equal
    decision.final_verdict.

    Raises ReportGenerationError on any failure — see module docstring.
    Never retries, never repairs the model's output.

    service: injectable QwenService, defaulting to get_service() (the
    process-wide singleton) — tests pass a QwenService constructed with
    mocked load_fn/generate_fn instead of going through the singleton,
    matching src/semantic/analyze.py's analyze_semantic() convention.
    """
    if service is None:
        service = get_service()

    messages = build_messages(facts, rule_assessment, decision, accepted_findings)

    try:
        raw = service.generate(
            messages,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )
    except LLMServiceError as exc:
        raise ReportGenerationError(
            code="model_call_failed",
            message=f"rapor modeli çağrısı başarısız: {exc}",
        ) from exc

    parsed = _extract_json(raw)
    if parsed is None:
        raise ReportGenerationError(
            code="invalid_json",
            message=f"model geçerli JSON üretmedi. Ham çıktı: {raw[:500]!r}",
        )

    try:
        report = Report(**parsed)
    except ValidationError as exc:
        raise ReportGenerationError(
            code="schema_invalid",
            message=f"model çıktısı rapor şemasına uymuyor: {exc}",
        ) from exc

    # The enforcement point for "the LLM does not classify" in the live
    # hybrid path — decision.final_verdict is authoritative, never the
    # model's own risk_seviyesi. See src/demo.py's _report_from_llm for
    # the same check against a raw Verdict on the offline adapter path.
    if report.risk_seviyesi != decision.final_verdict:
        raise ReportGenerationError(
            code="verdict_mismatch",
            message=(
                f"model risk_seviyesi'ni '{report.risk_seviyesi}' yazdı ama "
                f"nihai karar '{decision.final_verdict}' idi."
            ),
        )

    return report
