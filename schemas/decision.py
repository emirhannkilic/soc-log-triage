"""
Decision policy output and input-context schemas
(PHISHING_ROUTING_PLAN.md step 9).

PhishingDecisionContext is a deliberately NARROW view of EmailFacts —
the decision policy (src/decision/phishing_policy.py) never receives
raw EmailFacts. Only the specific deterministic facts a policy rule
actually needs are extracted here, by the workflow layer, before the
policy is called. This keeps the policy's own signature small and
auditable (every field it can see is a field some rule actually reads)
and stops a policy rule from reaching into EmailFacts for something
ad hoc that isn't tracked in `decision_reasons`/`contributing_*_ids`.

WHY has_url IS DETERMINISTIC, NOT A SEMANTIC FINDING
    The credential_request-upgrade rule needs to know "is there an
    external link in this email" — that's the parser's job
    (EmailFacts.urls), not something to re-derive from what the model
    noticed. A semantic finding of type credential_request together
    with a model-noticed link would double up on a signal source the
    policy is specifically trying to keep separate: semantic findings
    answer "what does the BODY TEXT say," context answers "what does
    the PARSER'S deterministic analysis say" — see phishing_policy.py's
    module docstring for the full rationale.

parser_credential_request IS PROVENANCE ONLY, NOT A TRIGGER
    EmailFacts.credential_request (the parser's own regex/keyword
    heuristic) is carried here for auditability — so a FinalDecision
    can be inspected against what the deterministic layer ALSO thought
    — but the policy must never branch on it directly. It already fed
    into RuleAssessment's score (v1's credential_request_with_external_
    link signal, v2's content family) — branching on it again in the
    policy would double-count the same evidence under a different name.
"""
from pydantic import BaseModel, ConfigDict

from schemas.rule_assessment import RuleVerdict


class PhishingDecisionContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    has_url: bool
    url_count: int
    url_ids: list[str]  # the URLs themselves (EmailFacts.urls[i].url) — no separate id field exists
    # Provenance only — see module docstring. The policy must not
    # branch on this field; it already fed into the rule engine's score.
    parser_credential_request: bool


class FinalDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rule_verdict: RuleVerdict
    final_verdict: RuleVerdict
    decision_path: str
    # ^ short, fixed, human-readable code for WHICH rule fired — e.g.
    # "rule_engine_only", "credential_request_plus_url_upgrade",
    # "payment_request_combination_upgrade" — see phishing_policy.py's
    # DECISION_PATH_* constants for the closed set of values.
    contributing_rule_ids: list[str]
    # ^ RuleAssessment.evidence[i].signal values that contributed to
    # the FINAL decision (not necessarily all of RuleAssessment.evidence
    # — e.g. if the rule engine's own verdict is what decided things,
    # this lists every fired signal; if a semantic upgrade decided
    # things, this can be empty, since no rule signal alone caused the
    # upgrade).
    contributing_semantic_ids: list[str]
    # ^ Identifies WHICH validated semantic findings contributed, as
    # "<type>:<start>-<end>" strings (SemanticFindingType has no
    # separate id field; start/end make an otherwise-ambiguous repeated
    # type unique within one email — see ValidatedSemanticFinding).
    analyst_review_required: bool
    # ^ True whenever final_verdict is "Muhtemel Phishing" — this band
    # means the system did not resolve the decision on its own,
    # regardless of whether that came from the rule engine alone or a
    # semantic upgrade. False for "Phishing" (acted on) and "Güvenilir"
    # (no action needed).
