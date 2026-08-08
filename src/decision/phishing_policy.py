"""
Deterministic decision policy (PHISHING_ROUTING_PLAN.md step 9).

Combines RuleAssessment (the rule engine's decision, engine-agnostic —
schemas/rule_assessment.py) with validated semantic findings (Qwen
shadow-mode extraction, already grounding-checked by
src/semantic/validate.py — schemas/semantic.py's ValidatedSemanticFinding)
into a single FinalDecision (schemas/decision.py).

WHAT THIS DOES NOT DO
    This is not a second classifier. The rule engine's verdict is never
    overridden downward, and semantic findings can only ever move a
    verdict UP one band (Güvenilir -> Muhtemel Phishing), never straight
    to Phishing and never down. CLAUDE.md's "LLM'e sınıflandırma
    yaptırmak" boundary stays intact: the semantic extractor still only
    NAMES manipulation patterns with grounded quotes, it never decides
    Phishing/Muhtemel Phishing/Güvenilir — this module is the one place
    that decision gets made, and it is a fixed, testable set of rules,
    not a model call.

model_confidence IS NEVER ADDED TO A RULE SCORE
    0.87 model confidence + 3 rule score = 3.87 is not a statistically
    meaningful number — confidence and a rule engine's additive weight
    measure different things on different scales. model_confidence is
    carried on ValidatedSemanticFinding for observability only; no rule
    below reads it.

authority_impersonation IS EXCLUDED FROM EVERY RULE, PERMANENTLY (for
now)
    Measured across three prompt-tuning rounds on the semantic
    evaluation set (data/semantic_eval, see PROGRESS.md): even after
    two rounds of prompt fixes narrowing its definition, this type
    remained the single most persistent false-positive source — the
    model kept tagging a sender's own legitimate signature, legal
    footer, or regulatory disclosure as impersonation. The user's
    explicit decision (2026-08): this type stays shadow-only — logged
    and visible, but it must never contribute to a FinalDecision. If a
    future measurement shows it's cleaned up, that requires a new,
    deliberate decision to re-include it, not a silent code change.

WHY THE PAYMENT_REQUEST RULE REQUIRES A COMBINATION, NOT
payment_request ALONE
    A bare payment_request finding is common in entirely legitimate
    mail (invoices, subscription renewals, marketing "hemen para
    yatır" CTAs — see the Papara candidate in data/semantic_eval).
    Requiring it to co-occur with reward_or_prize_lure, threat_or_fear,
    or reply_channel_manipulation targets the actual attack shape (419/
    advance-fee fraud, fake reward claims) without flagging ordinary
    commercial payment requests. urgency_or_pressure and
    brand_impersonation are deliberately NOT in this combination set —
    see test_payment_request_alone_with_urgency_stays_guvenilir for the
    negative case this was measured against.

PROVENANCE FIELDS ON PhishingDecisionContext ARE NOT TRIGGERS
    context.parser_credential_request (EmailFacts.credential_request,
    the parser's own heuristic) is carried through for auditability
    only. It already contributed to RuleAssessment's score (v1's
    credential_request_with_external_link signal, v2's content family)
    — branching on it again here would double-count the same evidence
    under a different name. Only context.has_url is read by any rule
    below.
"""
from schemas.decision import FinalDecision, PhishingDecisionContext
from schemas.rule_assessment import RuleAssessment
from schemas.semantic import SemanticFindingType, ValidatedSemanticFinding

# Closed set of decision_path values — every FinalDecision.decision_path
# must be one of these, so downstream consumers (the web UI, PROGRESS.md
# reporting) can rely on a fixed vocabulary rather than parsing free text.
DECISION_PATH_RULE_ENGINE_ONLY = "rule_engine_only"
DECISION_PATH_CREDENTIAL_REQUEST_URL_UPGRADE = "credential_request_plus_url_upgrade"
DECISION_PATH_PAYMENT_REQUEST_COMBINATION_UPGRADE = "payment_request_combination_upgrade"

# authority_impersonation is deliberately absent from every rule below —
# see module docstring. Listed here as the single source of truth for
# "the type this policy refuses to act on," so a future rule addition
# has to explicitly decide to touch this constant rather than silently
# reading SemanticFindingType.AUTHORITY_IMPERSONATION somewhere new.
EXCLUDED_FROM_POLICY = frozenset({SemanticFindingType.AUTHORITY_IMPERSONATION})

# Types that, together with a payment_request finding, indicate the
# 419/advance-fee or fake-reward shape this rule targets — see module
# docstring for why urgency_or_pressure and brand_impersonation are
# deliberately excluded from this set.
PAYMENT_REQUEST_COMBINATION_TYPES = frozenset({
    SemanticFindingType.REWARD_OR_PRIZE_LURE,
    SemanticFindingType.THREAT_OR_FEAR,
    SemanticFindingType.REPLY_CHANNEL_MANIPULATION,
})


def _semantic_id(finding: ValidatedSemanticFinding) -> str:
    return f"{finding.type.value}:{finding.start}-{finding.end}"


def _types_present(findings: list[ValidatedSemanticFinding]) -> set[SemanticFindingType]:
    return {f.type for f in findings if f.type not in EXCLUDED_FROM_POLICY}


def decide(
    rule_assessment: RuleAssessment,
    semantic_findings: list[ValidatedSemanticFinding],
    context: PhishingDecisionContext,
) -> FinalDecision:
    """semantic_findings must already be validator-accepted
    (ValidatedSemanticFinding, not SemanticFindingCandidate) — this
    function does no grounding/evidence checking itself, that already
    happened in src/semantic/validate.py."""
    rule_verdict = rule_assessment.rule_verdict

    # Phishing (including any critical-predicate match, which is
    # already folded into rule_verdict by src/rules/adapters.py) is
    # never touched — semantic findings can only move a verdict UP,
    # and Phishing is already the top band.
    if rule_verdict == "Phishing":
        return FinalDecision(
            rule_verdict=rule_verdict,
            final_verdict="Phishing",
            decision_path=DECISION_PATH_RULE_ENGINE_ONLY,
            contributing_rule_ids=[e.signal for e in rule_assessment.evidence],
            contributing_semantic_ids=[],
            analyst_review_required=False,
        )

    # Muhtemel Phishing: the rule engine already couldn't decide.
    # Semantic findings are not asked to resolve that ambiguity in this
    # first version of the policy — they can only upgrade a Güvenilir
    # verdict, not adjudicate an existing Muhtemel Phishing one. This
    # keeps the "semantic findings only ever move Güvenilir up one
    # band" rule simple and auditable rather than introducing a second,
    # different upgrade path for the middle band.
    if rule_verdict == "Muhtemel Phishing":
        return FinalDecision(
            rule_verdict=rule_verdict,
            final_verdict="Muhtemel Phishing",
            decision_path=DECISION_PATH_RULE_ENGINE_ONLY,
            contributing_rule_ids=[e.signal for e in rule_assessment.evidence],
            contributing_semantic_ids=[],
            analyst_review_required=True,
        )

    # rule_verdict == "Güvenilir" from here on — the only band semantic
    # findings are allowed to upgrade.
    present_types = _types_present(semantic_findings)

    if SemanticFindingType.CREDENTIAL_REQUEST in present_types and context.has_url:
        contributing = [
            _semantic_id(f) for f in semantic_findings
            if f.type == SemanticFindingType.CREDENTIAL_REQUEST
        ]
        return FinalDecision(
            rule_verdict=rule_verdict,
            final_verdict="Muhtemel Phishing",
            decision_path=DECISION_PATH_CREDENTIAL_REQUEST_URL_UPGRADE,
            contributing_rule_ids=[],
            contributing_semantic_ids=contributing,
            analyst_review_required=True,
        )

    if (SemanticFindingType.PAYMENT_REQUEST in present_types
            and present_types & PAYMENT_REQUEST_COMBINATION_TYPES):
        combination_types = {SemanticFindingType.PAYMENT_REQUEST} | (
            present_types & PAYMENT_REQUEST_COMBINATION_TYPES
        )
        contributing = [
            _semantic_id(f) for f in semantic_findings if f.type in combination_types
        ]
        return FinalDecision(
            rule_verdict=rule_verdict,
            final_verdict="Muhtemel Phishing",
            decision_path=DECISION_PATH_PAYMENT_REQUEST_COMBINATION_UPGRADE,
            contributing_rule_ids=[],
            contributing_semantic_ids=contributing,
            analyst_review_required=True,
        )

    # No upgrade condition met — urgency_or_pressure alone,
    # authority_impersonation (excluded entirely), or no relevant
    # findings at all. rule_verdict stands.
    return FinalDecision(
        rule_verdict=rule_verdict,
        final_verdict="Güvenilir",
        decision_path=DECISION_PATH_RULE_ENGINE_ONLY,
        contributing_rule_ids=[e.signal for e in rule_assessment.evidence],
        contributing_semantic_ids=[],
        analyst_review_required=False,
    )
