"""
Phishing analysis workflow (PHISHING_ROUTING_PLAN.md step 4).

demo.py and web.py each re-implemented the same pipeline —
parse -> rule engine -> report -> render — with small drifts between
them (e.g. web.py built signals as plain dicts, demo.py kept RuleMatch
objects). This module is the single place that pipeline lives now; both
entry points call analyze_phishing() instead of inlining it.

First version deliberately does nothing new: it wraps the existing,
already-shipped v1 pipeline (src/rules/engine.py) behind RuleAssessment
(schemas/rule_assessment.py, PHISHING_ROUTING_PLAN.md step 3) and the
mechanical report builder (src/report/mechanical.build_report, which
itself takes RuleAssessment — not a raw v1 Verdict, see that module's
docstring for why). v2 (src/rules/engine_v2.py) stays out of this path —
CLAUDE.md's "v1→v2 geçiş kararı HÂLÂ VERİLMEDİ" is still true, v1 is the
only engine with a production decision behind it, and demo.py/web.py
were both still on v1 before this migration too.

Routing (src/router.py) happens BEFORE this function is called — this
module's job starts once an input has already been accepted as an
analyzable email. It takes a path to an .eml file, not raw text or a
RoutingDecision, so it has no dependency on how the caller obtained
that file (a real .eml on disk, or pasted text the caller already wrote
to a temp file the way web.py does).

MODE
    "fast": parse -> rule engine v1 -> RuleAssessment -> mechanical report.
        No model call, ~1 second.
    "hybrid": parse -> rule engine v1 -> RuleAssessment -> semantic
        extraction (Qwen3.5-9B, shadow-mode extractor validated against
        the email body) -> decision policy -> Qwen report generation
        (src/report/generate.py), falling back to the mechanical report
        on any failure. See HYBRID MODE and REPORT GENERATION below for
        the full contract.

HYBRID MODE
    The email is parsed exactly once — facts is reused for both the rule
    engine and the semantic extractor, never re-parsed.

    semantic_status is one of three states, distinguished deliberately
    (per the user's explicit instruction) so a caller can never confuse
    "the model wasn't asked" with "the model was asked and broke":
        "skipped"   rule_verdict was already "Phishing" — decide() cannot
                    use semantic findings to move a Phishing verdict
                    anywhere, so analyze_semantic() (tens of seconds,
                    loads a 9B model) is never called at all.
                    semantic_skip_reason="rule_verdict_already_phishing".
        "failed"    analyze_semantic() was called but raised
                    SemanticExtractionError (src/semantic/analyze.py) —
                    a malformed model response, or the underlying
                    QwenService failing (e.g. the GPU timeouts
                    PROGRESS.md documents for this same model in
                    scripts/evaluate_semantic_extractor.py, wrapped as
                    LLMServiceError and re-raised with code
                    "model_call_failed"). The deterministic rule_verdict
                    is NEVER lost in this case: decide() still runs, just
                    with an empty semantic_findings list, so final_verdict
                    falls back to rule_verdict exactly like the "skipped"
                    path. No retry — CLAUDE.md's "Yapılmayacaklar" rule
                    against patching/retrying model output applies here
                    too. Any OTHER, truly unexpected exception is not
                    caught here — it propagates, rather than being
                    silently folded into "failed" and hiding a real bug
                    in the wiring itself.
        "completed" analyze_semantic() returned normally. Its accepted
                    (validator-passed) findings feed decide(); its
                    rejected findings are carried on the result for
                    audit only — schemas/decision.py's policy never sees
                    them, matching validate.py's "a rejected candidate
                    is not evidence" rule.

    Only ValidationResult.accepted findings — already grounding-checked
    by src/semantic/validate.py — are passed to decide(). Rejected
    candidates are exposed on PhishingAnalysisResult.rejected_findings
    for observability but never reach the policy, matching
    src/decision/phishing_policy.py's "semantic_findings must already be
    validator-accepted" contract.

    context (PhishingDecisionContext) is built from facts, not from
    RuleAssessment.evidence — schemas/decision.py's module docstring
    explains why has_url must come from the parser's own EmailFacts.urls
    rather than being inferred from which url_* rule signals happened to
    fire.

    build_report() is always called with the FinalDecision — in fast
    mode it's never involved, but hybrid mode's mechanical report must
    reflect final_verdict, not the possibly-stale rule_verdict, whenever
    the policy upgraded the verdict. See src/report/mechanical.py's
    module docstring for the invariant this preserves:
    report.risk_seviyesi == final_decision.final_verdict, always, in
    both modes.

REPORT GENERATION (PHISHING_ROUTING_PLAN.md section 10.5, "Qwen çağrı 2")
    Fast mode's report is always mechanical (src/report/mechanical.py) —
    unchanged, no model call.

    Hybrid mode, after decide() produces a FinalDecision, calls
    src/report/generate.py's generate_report() to have Qwen write the
    Turkish report. generate_report() never repairs or retries its own
    output (same "çıktıyı onarma yasak" rule analyze_semantic() already
    applies) — it either returns a valid, verdict-matching Report or
    raises ReportGenerationError. This module is the one that decides
    what happens next: on ReportGenerationError, the deterministic
    mechanical report (already used by fast mode) is substituted, so a
    broken model call NEVER prevents a report from being produced — it
    only changes which report generator produced it.

    report_source and llm_report_status record which happened, so a
    caller/observer can tell "the model wrote this" from "the model was
    asked but failed and this is the deterministic fallback" without
    inspecting report content:

        mode="fast"                          -> report_source="mechanical",
                                                 llm_report_status="not_requested"
        mode="hybrid", Qwen succeeds          -> report_source="qwen",
                                                 llm_report_status="completed"
        mode="hybrid", Qwen fails (any code)  -> report_source="mechanical",
                                                 llm_report_status="failed_fallback",
                                                 llm_report_error_code=<the
                                                 ReportGenerationError's code>

    generate_report() itself has no notion of "fallback" or a status
    field — it only ever returns a successful Report or raises. Fallback
    selection and status bookkeeping are entirely this module's
    responsibility, matching generate_report()'s own module docstring
    ("generate_report() NEVER RETURNS A FAILURE VALUE").
"""
import sys
from pathlib import Path
from typing import Literal

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pydantic import BaseModel, ConfigDict, Field  # noqa: E402

from schemas.decision import FinalDecision, PhishingDecisionContext  # noqa: E402
from schemas.facts import EmailFacts  # noqa: E402
from schemas.report import Report  # noqa: E402
from schemas.rule_assessment import RuleAssessment  # noqa: E402
from schemas.semantic import ValidatedSemanticFinding  # noqa: E402
from src.decision.context import build_context  # noqa: E402
from src.decision.phishing_policy import decide  # noqa: E402
from src.parser.parse import parse_eml  # noqa: E402
from src.report.generate import ReportGenerationError, generate_report  # noqa: E402
from src.report.mechanical import build_report  # noqa: E402
from src.rules.adapters import from_v1  # noqa: E402
from src.rules.engine import evaluate, load_rules  # noqa: E402
from src.semantic.analyze import SemanticExtractionError, analyze_semantic  # noqa: E402
from src.semantic.validate import ValidatedFinding  # noqa: E402

AnalysisMode = Literal["fast", "hybrid"]
SemanticStatus = Literal["skipped", "failed", "completed"]
ReportSource = Literal["mechanical", "qwen"]
LLMReportStatus = Literal["not_requested", "completed", "failed_fallback"]

SEMANTIC_SKIP_REASON_ALREADY_PHISHING = "rule_verdict_already_phishing"


class PhishingAnalysisResult(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    mode: AnalysisMode
    facts: EmailFacts
    rule_assessment: RuleAssessment
    report: Report
    # None in fast mode — hybrid mode always sets this, even when
    # semantic_status is "skipped" or "failed" (decide() still runs with
    # an empty finding list in both of those cases, so final_verdict
    # falls back to rule_verdict rather than the field going unset).
    final_decision: FinalDecision | None = None
    semantic_status: SemanticStatus | None = None
    semantic_skip_reason: str | None = None
    accepted_findings: list[ValidatedSemanticFinding] = Field(default_factory=list)
    # Raw model output plus schema-invalid items — SemanticFindingCandidate
    # for anything that parsed but failed grounding, or the raw
    # JSON-decoded value (dict/str/None/...) for anything that didn't
    # even parse. Audit trail only, never policy input — see module
    # docstring's "completed" status.
    rejected_findings: list[ValidatedFinding] = Field(default_factory=list)
    # Defaults match fast mode's only possible values, so existing
    # callers (CLI/web, both still fast-only as of this turn) get these
    # fields for free without constructing them explicitly — see module
    # docstring's REPORT GENERATION table for the full mode/status matrix.
    report_source: ReportSource = "mechanical"
    llm_report_status: LLMReportStatus = "not_requested"
    llm_report_error_code: str | None = None


def analyze_phishing(email_input: Path, mode: AnalysisMode = "fast") -> PhishingAnalysisResult:
    """email_input: path to an .eml file that has already passed routing."""
    if mode not in ("fast", "hybrid"):
        raise ValueError(f"bilinmeyen mode: {mode!r}")

    facts = parse_eml(email_input)
    rules = load_rules()
    verdict = evaluate(facts.flat_signals(), rules)
    rule_assessment = from_v1(verdict, rules)

    if mode == "fast":
        report = build_report(rule_assessment)
        return PhishingAnalysisResult(
            mode=mode,
            facts=facts,
            rule_assessment=rule_assessment,
            report=report,
        )

    accepted_findings: list[ValidatedSemanticFinding] = []
    rejected_findings: list[ValidatedFinding] = []
    semantic_skip_reason: str | None = None

    if rule_assessment.rule_verdict == "Phishing":
        semantic_status: SemanticStatus = "skipped"
        semantic_skip_reason = SEMANTIC_SKIP_REASON_ALREADY_PHISHING
    else:
        try:
            validation_result = analyze_semantic(facts)
        except SemanticExtractionError:
            # analyze_semantic() normalizes every expected failure mode
            # (malformed model output, underlying QwenService failure —
            # e.g. the GPU/Metal timeouts PROGRESS.md documents) to this
            # one exception type. Anything else propagates instead of
            # being folded in here — an unexpected exception is a bug in
            # the wiring, not a "semantic layer degraded gracefully" case.
            semantic_status = "failed"
        else:
            semantic_status = "completed"
            accepted_findings = validation_result.accepted
            rejected_findings = validation_result.rejected

    context: PhishingDecisionContext = build_context(facts)
    final_decision = decide(rule_assessment, accepted_findings, context)

    # generate_report() never repairs/retries its own output (see its
    # module docstring) — a ReportGenerationError here means the model
    # call or its output was unusable, not that no report can be
    # produced at all. The deterministic mechanical report (the same
    # builder fast mode always uses) substitutes in that case, so
    # analyze_phishing() always returns a usable report either way; only
    # report_source/llm_report_status say which generator actually wrote
    # it.
    try:
        report = generate_report(facts, rule_assessment, final_decision, accepted_findings)
    except ReportGenerationError as exc:
        report = build_report(rule_assessment, decision=final_decision)
        report_source: ReportSource = "mechanical"
        llm_report_status: LLMReportStatus = "failed_fallback"
        llm_report_error_code = exc.code
    else:
        report_source = "qwen"
        llm_report_status = "completed"
        llm_report_error_code = None

    return PhishingAnalysisResult(
        mode=mode,
        facts=facts,
        rule_assessment=rule_assessment,
        report=report,
        final_decision=final_decision,
        semantic_status=semantic_status,
        semantic_skip_reason=semantic_skip_reason,
        accepted_findings=accepted_findings,
        rejected_findings=rejected_findings,
        report_source=report_source,
        llm_report_status=llm_report_status,
        llm_report_error_code=llm_report_error_code,
    )
