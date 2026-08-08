"""
Live hybrid-narrative generation (PROGRESS.md "rapor mimarisi
değişikliği"). Uses src/report/narrative_prompts.py and the shared
src/llm/service.py QwenService (the same singleton src/semantic/
analyze.py's "Qwen çağrı 1" already uses within one hybrid request, per
PHISHING_ROUTING_PLAN.md's "tek model, iki çağrı").

NOT src/report/generate.py (removed)
    That module asked Qwen to author a full schemas.report.Report and
    re-validated risk_seviyesi (verdict_mismatch) and a category
    vocabulary (category_violation) on the output. This module asks for
    only schemas.narrative.NarrativeDraft's three sentence fragments —
    there is no risk_seviyesi field to mismatch and no category
    vocabulary to violate, so neither of those checks exists here. The
    only output-side validation is schema validation itself
    (NarrativeDraft(**parsed), extra="forbid") — see src/report/
    narrative_prompts.py's module docstring for why that is sufficient.

WHAT THIS MODULE DOES NOT DO
    No retry, no output repair — same "çıktıyı onarma yasak" rule
    src/report/generate.py held and src/semantic/analyze.py already
    applies. A malformed or off-schema response is a real result, not
    something to paper over — generate_narrative() raises
    NarrativeGenerationError and stops. It is the CALLER's job
    (src/workflows/phishing.py) to fall back to the mechanical
    genel_degerlendirme text (already produced by src/report/
    mechanical.py unconditionally) and record that this happened; this
    module has no fallback logic and does not know the mechanical
    report exists.

generate_narrative() NEVER RETURNS A FAILURE VALUE
    Every failure mode is a raised NarrativeGenerationError, not an
    Optional return — same rationale src/report/generate.py's own
    docstring already recorded for this project's report-generation
    functions in general.

CLOSED `code` VALUES
    "model_call_failed"   the load/generate infrastructure itself
                            failed — wraps src/llm/service.py's
                            LLMServiceError.
    "invalid_json"         generate() succeeded but the output could not
                            be parsed as a JSON object at all.
    "schema_invalid"       parsed JSON does not match
                            schemas.narrative.NarrativeDraft (extra
                            field, wrong type, missing field).

TEMPERATURE=0, NO CONSTRAINED DECODING (this turn)
    Matches src/semantic/analyze.py's TEMPERATURE and the removed
    src/report/generate.py's own setting. Constrained JSON decoding
    remains a future option, unapplied here for the same reason
    CLAUDE.md already recorded for the teacher path: measured
    unconstrained first.
"""
import json
import re

from pydantic import ValidationError

from schemas.decision import FinalDecision
from schemas.facts import EmailFacts
from schemas.narrative import NarrativeDraft
from schemas.rule_assessment import RuleAssessment
from schemas.semantic import ValidatedSemanticFinding
from src.llm.service import LLMServiceError, QwenService, get_service
from src.report.narrative_prompts import build_messages

TEMPERATURE = 0
# The narrative prompt is shorter than the old full-Report prompt (no
# category vocabulary, no teknik_bulgular/phishing_gostergeleri to
# produce) and the target JSON is three short sentences — a much smaller
# ceiling than src/report/generate.py's removed MAX_TOKENS=1200 is
# appropriate, but kept generous enough that no single sentence risks
# truncation on a high-signal email.
MAX_TOKENS = 400

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


class NarrativeGenerationError(Exception):
    """Raised by generate_narrative() for every failure mode it
    normalizes to — see module docstring's CLOSED `code` VALUES
    section. No retry, no repair; the caller decides what to do next
    (src/workflows/phishing.py keeps the mechanical genel_degerlendirme
    text)."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


def _extract_json(raw_text: str) -> dict | None:
    """Mirrors src/report/generate.py's own _extract_json — best-effort
    location of a JSON object in raw model output, never a repair of
    malformed content."""
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


def generate_narrative(
    facts: EmailFacts,
    rule_assessment: RuleAssessment,
    decision: FinalDecision,
    accepted_findings: list[ValidatedSemanticFinding],
    service: QwenService | None = None,
) -> NarrativeDraft:
    """Runs Qwen3.5-9B once (via the shared QwenService) and returns a
    NarrativeDraft — three sentence fragments, nothing else. Raises
    NarrativeGenerationError on any failure — see module docstring.
    Never retries, never repairs the model's output.

    Caller's responsibility (src/workflows/phishing.py): only call this
    when decision.final_verdict != "Güvenilir" — see PROGRESS.md's
    "final_verdict == Güvenilir" skip rule. This function itself has no
    opinion on final_verdict; it will build a prompt and call the model
    regardless of what final_verdict is, since that policy decision
    belongs to the workflow layer, not this generator.

    service: injectable QwenService, defaulting to get_service() (the
    process-wide singleton) — tests pass a QwenService constructed with
    mocked load_fn/generate_fn instead of going through the singleton.
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
        raise NarrativeGenerationError(
            code="model_call_failed",
            message=f"narrative modeli çağrısı başarısız: {exc}",
        ) from exc

    parsed = _extract_json(raw)
    if parsed is None:
        raise NarrativeGenerationError(
            code="invalid_json",
            message=f"model geçerli JSON üretmedi. Ham çıktı: {raw[:500]!r}",
        )

    try:
        return NarrativeDraft(**parsed)
    except ValidationError as exc:
        raise NarrativeGenerationError(
            code="schema_invalid",
            message=f"model çıktısı narrative şemasına uymuyor: {exc}",
        ) from exc
