"""
Verdict -> RuleAssessment and VerdictV2 -> RuleAssessment adapters
(PHISHING_ROUTING_PLAN.md step 3).

This is the only module allowed to know both src/rules/engine.py's
Verdict shape and src/rules/engine_v2.py's VerdictV2 shape. Everything
downstream (report model, decision policy, web UI) imports only
schemas/rule_assessment.py.
"""
from schemas.rule_assessment import FamilyContribution, RuleAssessment, RuleEvidence
from src.rules.engine import Verdict
from src.rules.engine_v2 import FAMILY_NAMES, VerdictV2


def from_v1(verdict: Verdict, rules: dict) -> RuleAssessment:
    weights = rules["signals"]
    evidence = [
        RuleEvidence(
            signal=m.signal,
            description=m.description,
            weight=m.weight,
        )
        for m in verdict.matches
    ]
    thresholds = rules["thresholds"]
    decision_reasons = [
        f"score {verdict.score} vs thresholds "
        f"(phishing>={thresholds['phishing']}, suspicious>={thresholds['suspicious']})",
    ]
    return RuleAssessment(
        engine_version="v1",
        rule_verdict=verdict.verdict,
        score=verdict.score,
        total=None,
        families=[],
        critical_matches=[],
        evidence=evidence,
        decision_reasons=decision_reasons,
    )


def _v2_decision_reasons(verdict: VerdictV2) -> list[str]:
    reasons = []
    scores = {name: verdict.families[name].score for name in FAMILY_NAMES}
    if verdict.critical_matches:
        reasons.append(f"critical predicate matched: {', '.join(verdict.critical_matches)}")
    maxed = [name for name, s in scores.items() if s == 4]
    if maxed:
        reasons.append(f"family score 4 (two independent mechanisms) in: {', '.join(maxed)}")
    if verdict.total >= 5 and verdict.material_family_count >= 2:
        reasons.append(
            f"total={verdict.total} (>=5) with material_family_count="
            f"{verdict.material_family_count} (>=2)"
        )
    strong = [name for name, s in scores.items() if s >= 3]
    if strong:
        reasons.append(f"family score >=3 in: {', '.join(strong)}")
    if verdict.total >= 3 and verdict.active_family_count >= 2:
        reasons.append(
            f"total={verdict.total} (>=3) with active_family_count="
            f"{verdict.active_family_count} (>=2)"
        )
    if not reasons:
        reasons.append(
            f"no threshold met (total={verdict.total}, "
            f"active_family_count={verdict.active_family_count})"
        )
    return reasons


def _v2_evidence_entry(family_name: str, hit, weights: dict) -> RuleEvidence:
    # Almost every v2 signal name is a config/rules.yaml key and this reads
    # its real description/weight — the same source of truth from_v1
    # reads, so a signal's description/weight read identically regardless
    # of which engine fired it. hit.weight itself is NOT that config
    # weight: it's _weight_to_strength()'s normalized 1/2/3, stored
    # separately as `strength`.
    #
    # One exception: attachment_is_archive has no config/rules.yaml entry
    # at all (see src/rules/engine_v2.py's comment on payload_hits — v1's
    # is_archive_with_credential_request combined rule was dropped rather
    # than ported, so v2's archive subgroup scores the raw signal with a
    # hardcoded strength instead of a configured weight). Falls back to
    # the signal name as its own description and the normalized strength
    # as its weight, since there is no configured weight to report.
    config = weights.get(hit.signal)
    return RuleEvidence(
        signal=hit.signal,
        description=config["description"] if config else hit.signal.replace("_", " "),
        weight=config["weight"] if config else hit.weight,
        family=family_name,
        subgroup=hit.subgroup,
        strength=hit.weight,
    )


def from_v2(verdict: VerdictV2, rules: dict) -> RuleAssessment:
    weights = rules["signals"]
    families = [
        FamilyContribution(family=name, score=verdict.families[name].score)
        for name in FAMILY_NAMES
    ]
    evidence = [
        _v2_evidence_entry(family_name, hit, weights)
        for family_name in FAMILY_NAMES
        for hit in verdict.families[family_name].hits
    ]
    return RuleAssessment(
        engine_version="v2",
        rule_verdict=verdict.verdict,
        score=None,
        total=verdict.total,
        families=families,
        critical_matches=verdict.critical_matches,
        evidence=evidence,
        decision_reasons=_v2_decision_reasons(verdict),
    )
