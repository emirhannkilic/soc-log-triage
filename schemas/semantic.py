"""
Semantic finding schemas (PHISHING_ROUTING_PLAN.md step 5, offset
design revised in step 6 after a real model run).

Defines what a semantic extractor (a model — Qwen, PHISHING_ROUTING_PLAN.md
step 6-7) is allowed to claim about an email BEFORE any model is wired up.
This is deliberately the reverse order of "call the model, see what comes
back, then figure out how to validate it" — the accepted shape is fixed
first so validation (src/semantic/validate.py, step 5's next file) has
something fixed to check against, and the model has no way to introduce
a finding type nobody agreed to accept.

extra="forbid" matches schemas/facts.py and schemas/report.py's
convention: a field outside this schema is a bug, not a silent
pass-through.

WHY TWO SCHEMAS, NOT ONE
    The original design had the model emit start/end character offsets
    directly. A real run (src/semantic/smoke_test.py against
    inbox-1804.eml, 2026-08) showed this doesn't work: the model's
    offsets didn't match its own evidence in any of 4 findings — LLMs
    are not reliable character counters, and this body additionally
    contained invisible Unicode (U+034F, a Gmail tracking artifact) that
    makes counting even harder. Asking the model for something it
    structurally cannot do reliably and then rejecting its output for
    getting it wrong is asking the wrong question.

    So the model never emits offsets. SemanticFindingCandidate is what
    the model is allowed to produce — evidence, type, confidence,
    explanation, no start/end. ValidatedSemanticFinding is what comes
    out of src/semantic/validate.py: the SAME fields plus start/end,
    which the validator computes itself via a real substring search
    (canonical_body.find(evidence)) rather than trusting a claimed
    number. Anything downstream of validation (the decision policy, the
    web UI's span highlighting) only ever sees ValidatedSemanticFinding
    — a finding without a verified offset does not exist past this
    boundary.

WHY AMBIGUOUS EVIDENCE IS REJECTED, NOT ACCEPTED AT ITS FIRST MATCH
    If evidence appears more than once in canonical_body, silently
    picking the first occurrence would locate the finding at a position
    the model never actually pointed at — cosmetically successful,
    substantively invented. src/semantic/validate.py rejects this case
    (RejectionReason.AMBIGUOUS_EVIDENCE) instead. The prompt
    (src/semantic/analyze.py) tells the model to quote enough text to be
    unique for exactly this reason — the fix belongs in what the model
    is asked to produce, not in the validator picking one match on its
    behalf.

WHY THE TYPE LIST IS AN ENUM, NOT A str
    CLAUDE.md's locked "LLM'e sınıflandırma yaptırmak" boundary is about
    the model not deciding Phishing/Muhtemel Phishing/Güvenilir — it
    does not forbid the model naming what kind of manipulative content
    it found. But an open string field would let the model invent new
    categories the decision policy (step 9) was never taught to weigh,
    which is a soft form of the model steering the decision by choosing
    its own vocabulary. A closed enum keeps that surface fixed and
    auditable. The list starts deliberately narrow — nine types, no
    aliases — and is only meant to grow after the shadow-mode
    measurement in step 6 shows a real gap, not speculatively.
"""
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class SemanticFindingType(str, Enum):
    CREDENTIAL_REQUEST = "credential_request"
    PAYMENT_REQUEST = "payment_request"
    AUTHORITY_IMPERSONATION = "authority_impersonation"
    BRAND_IMPERSONATION = "brand_impersonation"
    URGENCY_OR_PRESSURE = "urgency_or_pressure"
    THREAT_OR_FEAR = "threat_or_fear"
    REWARD_OR_PRIZE_LURE = "reward_or_prize_lure"
    ATTACHMENT_OR_LINK_INSTRUCTION = "attachment_or_link_instruction"
    REPLY_CHANNEL_MANIPULATION = "reply_channel_manipulation"


class SemanticFindingCandidate(BaseModel):
    """What the model is allowed to produce. No start/end — see module
    docstring for why offsets moved out of the model's responsibility."""

    # type is deliberately NOT strict: it arrives from JSON as a plain
    # string ("credential_request") and Pydantic's enum coercion from a
    # matching string value is exactly the intended path — that's not
    # the same as "silently fixing a wrong type" (see model_confidence
    # below), a raw string IS the enum's own value representation. An
    # actually unrecognized value ("credential_theft") still fails
    # validation either way.
    model_config = ConfigDict(extra="forbid")

    type: SemanticFindingType
    evidence: str
    # ^ the exact quoted substring this finding is based on, expected to
    # be long/specific enough to be a UNIQUE match in canonical_body —
    # src/semantic/analyze.py's prompt asks for this explicitly.
    # src/semantic/validate.py enforces both "appears verbatim" and
    # "appears exactly once" with a plain substring search, not a fuzzy
    # match. A finding whose evidence can't be found, or is found more
    # than once, is rejected, not repaired (same "çıktıyı onarma yasak"
    # rule src/demo.py already applies to Report).
    #
    # model_confidence is strict=True: Pydantic's default (lax) mode
    # silently coerces "0.9" -> 0.9, so a model that emits confidence as
    # a string would pass schema validation with the wrong type quietly
    # fixed for it. Model output is untrusted input, not a well-behaved
    # caller — a type mismatch here is a real defect in what the model
    # produced and belongs in RejectionReason.SCHEMA_INVALID
    # (src/semantic/validate.py), not papered over at construction time.
    model_confidence: float = Field(ge=0.0, le=1.0, strict=True)
    explanation: str


class ValidatedSemanticFinding(BaseModel):
    """Output of src/semantic/validate.py only — never constructed
    directly from model output. start/end are the validator's own
    computed offsets (canonical_body.find(evidence)), not a claim the
    model made. Anything downstream of validation (decision policy, web
    UI) takes this type, never SemanticFindingCandidate."""

    model_config = ConfigDict(extra="forbid")

    type: SemanticFindingType
    evidence: str
    start: int
    end: int
    # ^ character offsets into canonical_body such that
    # canonical_body[start:end] == evidence, computed by the validator
    # via a real substring search — see module docstring.
    model_confidence: float = Field(ge=0.0, le=1.0)
    explanation: str
