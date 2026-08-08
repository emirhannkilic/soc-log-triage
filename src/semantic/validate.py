"""
Evidence validator for SemanticFindingCandidate (PHISHING_ROUTING_PLAN.md
step 5, offset design revised in step 6 after a real model run).

Written before any model call exists (step 6-7 add the Qwen extractor).
This module's only job is: given a list of SemanticFindingCandidate
objects a model *might* produce, decide which ones are actually grounded
in the email, compute their real offsets, and turn them into
ValidatedSemanticFinding objects. It has no model dependency and no I/O —
every test in tests/test_semantic_validate.py constructs candidates by
hand.

WHY THE VALIDATOR COMPUTES start/end, NOT THE MODEL
    See schemas/semantic.py's module docstring for the full story: a
    real run showed the model cannot reliably count characters (offsets
    didn't match its own quoted evidence in 4/4 findings on one email,
    made worse by invisible Unicode in the body). So the model is never
    asked for offsets at all — SemanticFindingCandidate has no start/end
    field. This validator computes them itself with a real substring
    search (canonical_body.find(evidence)), which is the ONLY source of
    truth for where a finding is located from this point on.

WHY AMBIGUOUS EVIDENCE IS REJECTED, NOT RESOLVED
    If evidence appears more than once in canonical_body, there is no
    correct choice between the occurrences — picking the first one would
    locate the finding somewhere the model never specifically pointed
    at. RejectionReason.AMBIGUOUS_EVIDENCE rejects the whole finding
    instead. The fix belongs upstream, in the prompt asking the model to
    quote a long/specific enough span to be unique (src/semantic/
    analyze.py) — this validator does not guess on the model's behalf.

CANONICAL TEXT — read this before changing anything upstream
    canonical_body = canonicalize_body(facts.body_text)
    (src/semantic/canonical.py), always — this module does NOT call
    canonicalize_body() itself; every caller (src/semantic/analyze.py,
    scripts/build_semantic_eval_ground_truth.py) is responsible for
    normalizing before calling validate_findings()/
    validate_raw_findings(), and must pass the SAME normalized string
    the model/labeler actually saw. A real bug (PROGRESS.md, 2026-08)
    happened from exactly this drifting: facts.body_text's raw CRLF
    line endings got silently collapsed to LF by an unrelated text-mode
    file read somewhere in the labeling pipeline, so two ground-truth
    findings whose evidence was quoted against the LF version failed
    this validator's search against the (still-CRLF) raw body_text.
    canonicalize_body() existing as one shared function, rather than
    each caller doing its own ad hoc normalization, is what makes "the
    same canonical_body" an actual guarantee instead of an assumption
    that can silently stop holding.

SUBJECT/HEADER QUOTES ARE OUT OF SCOPE, NOT SUPPORTED
    schemas/semantic.py's finding schemas have no `source` field — they
    were locked as body-only (evidence defined against canonical_body
    only). The plan's step 5 checklist mentions verifying "source" when
    a finding quotes the subject or a header, but the schema this
    validator checks against doesn't carry that field, so there is
    nothing to verify a source claim against. Any finding whose evidence
    isn't found in canonical_body is rejected as NOT_FOUND_IN_BODY
    regardless of whether the quoted text actually came from the subject
    line — that's not a special case, it's the same failure as a
    hallucinated quote. If subject/header findings become a real
    requirement later, that needs a schema change in schemas/semantic.py
    first (an explicit source field), not a special case bolted on here.

WHAT THIS DOES NOT DO
    No repair, no fuzzy matching, no "close enough" offset correction,
    no picking a match on the model's behalf when evidence is ambiguous —
    CLAUDE.md's "Yapılmayacaklar" rule against patching model output
    applies here exactly like it does to Report: a finding that fails a
    check is rejected, never adjusted into passing.
"""
from dataclasses import dataclass, field
from enum import Enum

from pydantic import ValidationError

from schemas.semantic import (
    SemanticFindingCandidate,
    SemanticFindingType,
    ValidatedSemanticFinding,
)

MAX_EVIDENCE_LENGTH = 300
# ^ generous upper bound for a single quoted span — a "finding" that
# needs hundreds of characters of quote to make its point is not a
# targeted piece of evidence, it's most of the email. Chosen well above
# any realistic sentence-level quote so it only catches genuinely
# degenerate output (e.g. the model quoting the entire body).


class RejectionReason(str, Enum):
    UNKNOWN_TYPE = "unknown_type"
    SCHEMA_INVALID = "schema_invalid"
    EMPTY_EVIDENCE = "empty_evidence"
    EVIDENCE_TOO_LONG = "evidence_too_long"
    NOT_FOUND_IN_BODY = "not_found_in_body"
    AMBIGUOUS_EVIDENCE = "ambiguous_evidence"
    DUPLICATE_FINDING = "duplicate_finding"


@dataclass
class ValidatedFinding:
    # ValidatedSemanticFinding for anything accepted; the original raw
    # item (a SemanticFindingCandidate, a dict, or any other
    # JSON-decoded value — str/None/int/list) for every rejection,
    # since a rejected candidate never gets a computed offset and so
    # never becomes a ValidatedSemanticFinding.
    finding: "ValidatedSemanticFinding | SemanticFindingCandidate | object"
    accepted: bool
    rejection_reason: RejectionReason | None = None


@dataclass
class ValidationResult:
    accepted: list[ValidatedSemanticFinding] = field(default_factory=list)
    rejected: list[ValidatedFinding] = field(default_factory=list)

    @property
    def all(self) -> list[ValidatedFinding]:
        return [ValidatedFinding(f, True) for f in self.accepted] + self.rejected


def _candidate_error(
    candidate: SemanticFindingCandidate, canonical_body: str
) -> tuple[RejectionReason | None, int | None]:
    """Returns (rejection_reason, start) — start is only meaningful when
    rejection_reason is None. Locating the single match and validating it
    are the same substring search, so this does both in one place rather
    than searching twice."""
    # No unknown-type check here: SemanticFindingCandidate.type is a
    # SemanticFindingType enum field, so Pydantic already refuses to
    # construct a candidate with an unrecognized type. Raw model JSON
    # (dicts, before SemanticFindingCandidate construction) goes through
    # validate_raw_findings() below instead, which surfaces
    # RejectionReason.UNKNOWN_TYPE for exactly that case.
    if not candidate.evidence.strip():
        return RejectionReason.EMPTY_EVIDENCE, None
    if len(candidate.evidence) > MAX_EVIDENCE_LENGTH:
        return RejectionReason.EVIDENCE_TOO_LONG, None

    first_start = canonical_body.find(candidate.evidence)
    if first_start == -1:
        return RejectionReason.NOT_FOUND_IN_BODY, None

    second_start = canonical_body.find(candidate.evidence, first_start + 1)
    if second_start != -1:
        return RejectionReason.AMBIGUOUS_EVIDENCE, None

    return None, first_start


def validate_raw_findings(
    raw_findings: list, canonical_body: str
) -> ValidationResult:
    """Entry point for actual model output: a list of items parsed from
    JSON that have not been through SemanticFindingCandidate's own
    validation yet. A list element that isn't even a dict (a model can
    emit ["some string", null, 5, []] just as easily as a list of
    objects — a JSON array element has no guaranteed shape), a dict
    whose "type" isn't one of SemanticFindingType's nine values, or a
    dict that's missing/mis-typed a required field, is rejected here as
    SCHEMA_INVALID/UNKNOWN_TYPE rather than raising — the same
    "a malformed finding is dropped, not repaired" rule validate_findings
    applies to grounding failures. Everything that does parse is handed
    to validate_findings for the grounding checks.
    """
    result = ValidationResult()
    constructed: list[SemanticFindingCandidate] = []

    for raw in raw_findings:
        if not isinstance(raw, dict):
            result.rejected.append(
                ValidatedFinding(raw, False, RejectionReason.SCHEMA_INVALID)
            )
            continue
        raw_type = raw.get("type")
        if raw_type not in {t.value for t in SemanticFindingType}:
            result.rejected.append(
                ValidatedFinding(raw, False, RejectionReason.UNKNOWN_TYPE)
            )
            continue
        try:
            constructed.append(SemanticFindingCandidate(**raw))
        except (ValidationError, TypeError):
            # TypeError covers a dict with non-string keys (e.g. {1: "x"}),
            # which passes the isinstance(dict) check above but still
            # can't be splatted into **raw.
            result.rejected.append(
                ValidatedFinding(raw, False, RejectionReason.SCHEMA_INVALID)
            )

    grounded = validate_findings(constructed, canonical_body)
    result.accepted.extend(grounded.accepted)
    result.rejected.extend(grounded.rejected)
    return result


def validate_findings(
    candidates: list[SemanticFindingCandidate], canonical_body: str
) -> ValidationResult:
    """canonical_body must be facts.body_text for the SAME EmailFacts the
    model was shown — see module docstring."""
    result = ValidationResult()
    seen: set[tuple[str, str]] = set()

    for candidate in candidates:
        reason, start = _candidate_error(candidate, canonical_body)
        if reason is not None:
            result.rejected.append(ValidatedFinding(candidate, False, reason))
            continue

        key = (candidate.type.value, candidate.evidence)
        if key in seen:
            result.rejected.append(
                ValidatedFinding(candidate, False, RejectionReason.DUPLICATE_FINDING)
            )
            continue
        seen.add(key)

        result.accepted.append(ValidatedSemanticFinding(
            type=candidate.type,
            evidence=candidate.evidence,
            start=start,
            end=start + len(candidate.evidence),
            model_confidence=candidate.model_confidence,
            explanation=candidate.explanation,
        ))

    return result
