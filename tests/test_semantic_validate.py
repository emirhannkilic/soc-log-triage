"""Unit tests for src/semantic/validate.py (PHISHING_ROUTING_PLAN.md
step 5, offset design revised in step 6). No model involved — every
candidate here is constructed by hand.

Offsets are now computed BY THE VALIDATOR (canonical_body.find(evidence)),
not supplied by the model — see schemas/semantic.py's module docstring
for why. Tests here check: grounding (evidence must be a real quote),
uniqueness (evidence must match exactly once, or the finding is
AMBIGUOUS_EVIDENCE), and that computed offsets are correct."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from schemas.semantic import SemanticFindingCandidate, SemanticFindingType  # noqa: E402
from src.semantic.validate import (  # noqa: E402
    MAX_EVIDENCE_LENGTH,
    RejectionReason,
    validate_findings,
    validate_raw_findings,
)

BODY = (
    "Sayın müşterimiz, hesabınızın güvenliği için lütfen şifrenizi doğrulayın. "
    "Aksi halde hesabınız 24 saat içinde askıya alınacaktır. "
    "https://güvenli-banka-giris.example/login adresini ziyaret edin."
)

# BODY where the word "hesabınız" (a short, generic word) appears twice —
# used for ambiguity tests.
AMBIGUOUS_BODY = (
    "hesabınız güvende değil. lütfen hesabınız ile ilgili işlem yapın."
)


def _candidate(**overrides) -> SemanticFindingCandidate:
    kwargs = dict(
        type=SemanticFindingType.CREDENTIAL_REQUEST,
        evidence="şifrenizi doğrulayın",
        model_confidence=0.8,
        explanation="asks for password confirmation",
    )
    kwargs.update(overrides)
    return SemanticFindingCandidate(**kwargs)


# --- grounding: happy path --------------------------------------------

def test_well_grounded_candidate_is_accepted_with_correct_offsets():
    c = _candidate()
    result = validate_findings([c], BODY)
    assert len(result.accepted) == 1
    accepted = result.accepted[0]
    assert accepted.evidence == c.evidence
    assert BODY[accepted.start:accepted.end] == c.evidence
    assert result.rejected == []


def test_multiple_distinct_candidates_all_accepted():
    c1 = _candidate()
    c2 = _candidate(
        type=SemanticFindingType.THREAT_OR_FEAR,
        evidence="hesabınız 24 saat içinde askıya alınacaktır",
    )
    result = validate_findings([c1, c2], BODY)
    assert len(result.accepted) == 2
    assert result.rejected == []
    for finding in result.accepted:
        assert BODY[finding.start:finding.end] == finding.evidence


def test_accepted_finding_carries_through_type_confidence_explanation():
    c = _candidate(model_confidence=0.73, explanation="özel açıklama")
    result = validate_findings([c], BODY)
    finding = result.accepted[0]
    assert finding.type == c.type
    assert finding.model_confidence == 0.73
    assert finding.explanation == "özel açıklama"


# --- evidence must be a real, unique quote ------------------------------

def test_hallucinated_evidence_not_in_body_is_rejected():
    c = _candidate(evidence="kredi kartı bilgilerinizi hemen gönderin")
    result = validate_findings([c], BODY)
    assert result.accepted == []
    assert result.rejected[0].rejection_reason == RejectionReason.NOT_FOUND_IN_BODY


def test_evidence_from_subject_or_header_is_rejected_not_specially_handled():
    """SemanticFindingCandidate has no `source` field (schemas/semantic.py) —
    a quote that isn't in body_text is rejected the same way a
    hallucinated quote is, regardless of where the model claims it came
    from."""
    c = _candidate(evidence="Şifre Sıfırlama Talebi")
    result = validate_findings([c], BODY)
    assert result.rejected[0].rejection_reason == RejectionReason.NOT_FOUND_IN_BODY


def test_ambiguous_evidence_matching_twice_is_rejected():
    """A short/generic quote that appears more than once in the body has
    no correct offset to pick — the whole finding is rejected rather
    than silently choosing the first occurrence."""
    c = _candidate(evidence="hesabınız")
    result = validate_findings([c], AMBIGUOUS_BODY)
    assert result.accepted == []
    assert result.rejected[0].rejection_reason == RejectionReason.AMBIGUOUS_EVIDENCE


def test_evidence_matching_three_times_is_still_ambiguous_not_a_crash():
    body = "tekrar tekrar tekrar"
    c = _candidate(evidence="tekrar")
    result = validate_findings([c], body)
    assert result.rejected[0].rejection_reason == RejectionReason.AMBIGUOUS_EVIDENCE


def test_unique_long_quote_in_ambiguous_body_is_still_accepted():
    """The fix for ambiguity is a longer/more specific quote — a
    candidate that IS unique must still be accepted even in a body that
    has other ambiguous substrings elsewhere."""
    c = _candidate(evidence="lütfen hesabınız ile ilgili işlem yapın")
    result = validate_findings([c], AMBIGUOUS_BODY)
    assert len(result.accepted) == 1
    assert result.rejected == []


# --- empty / too long ----------------------------------------------------

def test_empty_evidence_is_rejected():
    c = SemanticFindingCandidate(
        type=SemanticFindingType.CREDENTIAL_REQUEST,
        evidence="",
        model_confidence=0.5,
        explanation="x",
    )
    result = validate_findings([c], BODY)
    assert result.rejected[0].rejection_reason == RejectionReason.EMPTY_EVIDENCE


def test_whitespace_only_evidence_is_rejected():
    c = SemanticFindingCandidate(
        type=SemanticFindingType.CREDENTIAL_REQUEST,
        evidence="   ",
        model_confidence=0.5,
        explanation="x",
    )
    result = validate_findings([c], BODY)
    assert result.rejected[0].rejection_reason == RejectionReason.EMPTY_EVIDENCE


def test_evidence_longer_than_max_is_rejected():
    long_evidence = "a" * (MAX_EVIDENCE_LENGTH + 50)
    body = long_evidence
    c = _candidate(evidence=long_evidence)
    result = validate_findings([c], body)
    assert result.rejected[0].rejection_reason == RejectionReason.EVIDENCE_TOO_LONG


def test_evidence_at_max_length_is_accepted():
    body = "a" * MAX_EVIDENCE_LENGTH
    c = _candidate(evidence=body)
    result = validate_findings([c], body)
    assert len(result.accepted) == 1
    assert result.accepted[0].start == 0
    assert result.accepted[0].end == MAX_EVIDENCE_LENGTH


# --- duplicates ------------------------------------------------------------

def test_duplicate_candidate_same_type_and_evidence_is_rejected():
    c1 = _candidate()
    c2 = _candidate()  # identical type/evidence
    result = validate_findings([c1, c2], BODY)
    assert len(result.accepted) == 1
    assert result.rejected[0].rejection_reason == RejectionReason.DUPLICATE_FINDING


def test_same_evidence_different_type_is_not_a_duplicate():
    c1 = _candidate(type=SemanticFindingType.CREDENTIAL_REQUEST)
    c2 = _candidate(type=SemanticFindingType.URGENCY_OR_PRESSURE)
    result = validate_findings([c1, c2], BODY)
    assert len(result.accepted) == 2
    assert result.rejected == []


# --- validate_raw_findings (dict-level, pre-Pydantic) ----------------------

def test_raw_finding_with_unknown_type_is_rejected():
    raw = [{
        "type": "credential_theft",  # not one of the nine allowed values
        "evidence": "şifrenizi doğrulayın",
        "model_confidence": 0.9,
        "explanation": "x",
    }]
    result = validate_raw_findings(raw, BODY)
    assert result.accepted == []
    assert result.rejected[0].rejection_reason == RejectionReason.UNKNOWN_TYPE


def test_raw_finding_missing_required_field_is_schema_invalid():
    raw = [{
        "type": "credential_request",
        # missing evidence/model_confidence/explanation
    }]
    result = validate_raw_findings(raw, BODY)
    assert result.accepted == []
    assert result.rejected[0].rejection_reason == RejectionReason.SCHEMA_INVALID


def test_raw_finding_with_start_end_is_schema_invalid():
    """The model must never emit start/end — SemanticFindingCandidate has
    extra="forbid" and no such fields, so a raw dict that includes them
    (e.g. an older model checkpoint, or a model ignoring the prompt) is
    rejected as SCHEMA_INVALID, not silently stripped."""
    raw = [{
        "type": "credential_request",
        "evidence": "şifrenizi doğrulayın",
        "start": 10,
        "end": 30,
        "model_confidence": 0.9,
        "explanation": "x",
    }]
    result = validate_raw_findings(raw, BODY)
    assert result.accepted == []
    assert result.rejected[0].rejection_reason == RejectionReason.SCHEMA_INVALID


def test_raw_finding_valid_and_grounded_is_accepted():
    evidence = "şifrenizi doğrulayın"
    raw = [{
        "type": "credential_request",
        "evidence": evidence,
        "model_confidence": 0.9,
        "explanation": "x",
    }]
    result = validate_raw_findings(raw, BODY)
    assert len(result.accepted) == 1
    assert result.accepted[0].start == BODY.index(evidence)
    assert result.rejected == []


def test_raw_finding_valid_schema_but_ungrounded_still_rejected():
    """A dict that parses fine into SemanticFindingCandidate must still go
    through the same grounding checks as a hand-built one."""
    raw = [{
        "type": "payment_request",
        "evidence": "iban bilgilerinizi paylaşın",
        "model_confidence": 0.5,
        "explanation": "x",
    }]
    result = validate_raw_findings(raw, BODY)
    assert result.accepted == []
    assert result.rejected[0].rejection_reason == RejectionReason.NOT_FOUND_IN_BODY


def test_raw_finding_ambiguous_evidence_is_rejected():
    raw = [{
        "type": "credential_request",
        "evidence": "hesabınız",
        "model_confidence": 0.9,
        "explanation": "x",
    }]
    result = validate_raw_findings(raw, AMBIGUOUS_BODY)
    assert result.accepted == []
    assert result.rejected[0].rejection_reason == RejectionReason.AMBIGUOUS_EVIDENCE


# --- no silent type coercion on raw model output (confidence) -----------

def test_raw_finding_string_confidence_is_schema_invalid_not_coerced():
    evidence = "şifrenizi doğrulayın"
    raw = [{
        "type": "credential_request",
        "evidence": evidence,
        "model_confidence": "0.9",  # string, not float
        "explanation": "x",
    }]
    result = validate_raw_findings(raw, BODY)
    assert result.accepted == []
    assert result.rejected[0].rejection_reason == RejectionReason.SCHEMA_INVALID


def test_raw_finding_type_as_string_still_coerces_via_enum():
    evidence = "şifrenizi doğrulayın"
    raw = [{
        "type": "credential_request",  # plain string, expected input shape
        "evidence": evidence,
        "model_confidence": 0.9,
        "explanation": "x",
    }]
    result = validate_raw_findings(raw, BODY)
    assert len(result.accepted) == 1


# --- non-dict list elements must not raise ------------------------------

def test_raw_findings_list_with_string_element_is_schema_invalid():
    result = validate_raw_findings(["not a finding"], BODY)
    assert result.accepted == []
    assert result.rejected[0].rejection_reason == RejectionReason.SCHEMA_INVALID


def test_raw_findings_list_with_none_element_is_schema_invalid():
    result = validate_raw_findings([None], BODY)
    assert result.accepted == []
    assert result.rejected[0].rejection_reason == RejectionReason.SCHEMA_INVALID


def test_raw_findings_list_with_number_element_is_schema_invalid():
    result = validate_raw_findings([42], BODY)
    assert result.accepted == []
    assert result.rejected[0].rejection_reason == RejectionReason.SCHEMA_INVALID


def test_raw_findings_list_with_nested_list_element_is_schema_invalid():
    result = validate_raw_findings([["nested", "list"]], BODY)
    assert result.accepted == []
    assert result.rejected[0].rejection_reason == RejectionReason.SCHEMA_INVALID


def test_raw_findings_mixed_valid_and_malformed_elements_no_raise():
    """A model can emit a mix of well-formed and garbage list elements in
    the same array — validate_raw_findings must process the whole batch
    without raising on the bad ones."""
    evidence = "şifrenizi doğrulayın"
    good = {
        "type": "credential_request",
        "evidence": evidence,
        "model_confidence": 0.9,
        "explanation": "x",
    }
    raw = [good, "garbage", None, 5, [], {"type": "credential_request"}]
    result = validate_raw_findings(raw, BODY)
    assert len(result.accepted) == 1
    assert len(result.rejected) == 5


# --- Unicode-hostile body: invisible characters must not break search ----

def test_body_with_invisible_unicode_still_finds_evidence():
    """Real body (inbox-1804.eml) contained U+034F (combining grapheme
    joiner, a Gmail tracking artifact) before the actual content —
    exactly the kind of body where offset counting broke for the model.
    The validator's find()-based approach must still work correctly:
    the invisible characters are just more characters in the string,
    not a parsing obstacle for a plain substring search."""
    invisible = "͏" * 20
    body = f"{invisible}\nMerhaba, şifrenizi doğrulamanız gerekiyor lütfen."
    c = _candidate(evidence="şifrenizi doğrulamanız gerekiyor")
    result = validate_findings([c], body)
    assert len(result.accepted) == 1
    finding = result.accepted[0]
    assert body[finding.start:finding.end] == c.evidence
    assert finding.start == len(invisible) + len("\nMerhaba, ")


if __name__ == "__main__":
    import traceback

    tests = [(name, obj) for name, obj in list(globals().items())
              if name.startswith("test_") and callable(obj)]
    passed, failed = 0, 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS: {name}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL: {name}: {e}")
            failed += 1
        except Exception:
            print(f"ERROR: {name}")
            traceback.print_exc()
            failed += 1

    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
