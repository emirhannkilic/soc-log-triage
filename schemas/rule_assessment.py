"""
Engine-agnostic rule engine output (PHISHING_ROUTING_PLAN.md step 3).

src/rules/engine.py's Verdict and src/rules/engine_v2.py's VerdictV2 have
different shapes (v1: flat score + matches list; v2: per-family scores +
critical predicates). Downstream consumers — the report model, the
decision policy, and the web UI — must not depend on either shape
directly, or swapping engines means touching every consumer. They only
ever see RuleAssessment; src/rules/adapters.py is the only code that
knows about Verdict/VerdictV2 internals.

extra="forbid" matches the convention in schemas/facts.py and
schemas/report.py: a field outside this schema is a bug, not a
silent pass-through.
"""
from typing import Literal

from pydantic import BaseModel, ConfigDict

RuleVerdict = Literal["Phishing", "Muhtemel Phishing", "Güvenilir"]


class RuleEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    signal: str
    # config/rules.yaml's own description text for this signal — the
    # same human-readable explanation regardless of which engine fired
    # it. NOT a family/subgroup label; see family/subgroup below for that.
    description: str
    # config/rules.yaml's raw configured weight for this signal, e.g. -3
    # or +4. This is the SAME number for a given signal in both v1 and
    # v2 — v1 sums these directly into its score; v2 maps them through
    # _weight_to_strength() into 1/2/3 (see `strength` below) before
    # scoring, but never overwrites this field with that mapped value.
    weight: float
    # v1: always None (v1 has no family/subgroup concept).
    # v2: which of identity/url/content/payload this signal contributed
    # to.
    family: str | None = None
    # v2: which subgroup within that family (e.g.
    # "infrastructure_alignment") — see src/rules/engine_v2.py's module
    # docstring for the full family/subgroup table. None for v1.
    subgroup: str | None = None
    # v2: the normalized 1(weak)/2(moderate)/3(strong) strength
    # _weight_to_strength() mapped `weight` to before family scoring —
    # NOT the same number as `weight` above (e.g. a +4 weight maps to
    # strength 3, same as a +3 weight). None for v1, which has no such
    # mapping.
    strength: int | None = None


class FamilyContribution(BaseModel):
    model_config = ConfigDict(extra="forbid")

    family: str
    score: int


class RuleAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    engine_version: str  # "v1" | "v2"
    rule_verdict: RuleVerdict
    score: float | None  # v1: total weight. v2: None, see families/total instead.
    total: int | None  # v2: sum of family scores. v1: None.
    families: list[FamilyContribution]  # v1: []. v2: one entry per family.
    critical_matches: list[str]  # v1: always []. v2: fired critical predicate names.
    evidence: list[RuleEvidence]
    decision_reasons: list[str]
