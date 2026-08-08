"""Unit tests for schemas/semantic.py (PHISHING_ROUTING_PLAN.md step 5,
offset design revised in step 6).

Only tests the schemas' own constraints (closed enum, confidence bounds,
extra="forbid", no start/end on the candidate model) — does not test
evidence-in-body validation or offset computation, that's
src/semantic/validate.py's job (see tests/test_semantic_validate.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic import ValidationError  # noqa: E402

from schemas.semantic import (  # noqa: E402
    SemanticFindingCandidate,
    SemanticFindingType,
    ValidatedSemanticFinding,
)

CANDIDATE_KWARGS = dict(
    type=SemanticFindingType.CREDENTIAL_REQUEST,
    evidence="şifrenizi doğrulayın",
    model_confidence=0.9,
    explanation="Body text asks the recipient to confirm their password.",
)


def _expect_candidate_error(**overrides):
    kwargs = dict(CANDIDATE_KWARGS)
    kwargs.update(overrides)
    try:
        SemanticFindingCandidate(**kwargs)
        raise AssertionError(f"expected ValidationError for override {overrides!r}")
    except ValidationError:
        pass


# --- SemanticFindingCandidate: what the model is allowed to produce ------

def test_valid_candidate_constructs():
    finding = SemanticFindingCandidate(**CANDIDATE_KWARGS)
    assert finding.type == SemanticFindingType.CREDENTIAL_REQUEST
    assert finding.model_confidence == 0.9


def test_candidate_has_no_start_end_fields():
    """The whole point of the schema split: the model is never asked for
    offsets. Passing them must be rejected by extra="forbid", not
    silently accepted/ignored."""
    try:
        SemanticFindingCandidate(**CANDIDATE_KWARGS, start=0, end=10)
        raise AssertionError("expected ValidationError for start/end on candidate")
    except ValidationError:
        pass


def test_all_nine_types_are_accepted():
    expected = {
        "credential_request",
        "payment_request",
        "authority_impersonation",
        "brand_impersonation",
        "urgency_or_pressure",
        "threat_or_fear",
        "reward_or_prize_lure",
        "attachment_or_link_instruction",
        "reply_channel_manipulation",
    }
    actual = {t.value for t in SemanticFindingType}
    assert actual == expected, actual


def test_unknown_type_is_rejected():
    try:
        SemanticFindingCandidate(**{**CANDIDATE_KWARGS, "type": "credential_theft"})
        raise AssertionError("expected ValidationError for unknown type")
    except ValidationError:
        pass


def test_extra_field_is_rejected():
    try:
        SemanticFindingCandidate(**CANDIDATE_KWARGS, verdict="Phishing")
        raise AssertionError("expected ValidationError for extra field")
    except ValidationError:
        pass


def test_confidence_above_one_is_rejected():
    _expect_candidate_error(model_confidence=1.5)


def test_confidence_below_zero_is_rejected():
    _expect_candidate_error(model_confidence=-0.1)


def test_confidence_boundary_values_are_accepted():
    SemanticFindingCandidate(**{**CANDIDATE_KWARGS, "model_confidence": 0.0})
    SemanticFindingCandidate(**{**CANDIDATE_KWARGS, "model_confidence": 1.0})


def test_missing_required_field_is_rejected():
    kwargs = dict(CANDIDATE_KWARGS)
    del kwargs["evidence"]
    try:
        SemanticFindingCandidate(**kwargs)
        raise AssertionError("expected ValidationError for missing evidence")
    except ValidationError:
        pass


# --- strict typing: no silent coercion of model-controlled fields --------

def test_string_model_confidence_is_rejected_not_coerced():
    _expect_candidate_error(model_confidence="0.9")


def test_int_model_confidence_is_accepted():
    """1 (int) for model_confidence is a normal numeric literal, not a
    type-safety concern the way a string is."""
    finding = SemanticFindingCandidate(**{**CANDIDATE_KWARGS, "model_confidence": 1})
    assert finding.model_confidence == 1.0


def test_type_as_plain_string_still_coerces_via_enum():
    """type is intentionally NOT strict — a plain string is the normal
    way an enum value arrives from JSON, not an untrusted-type defect."""
    finding = SemanticFindingCandidate(**{**CANDIDATE_KWARGS, "type": "payment_request"})
    assert finding.type == SemanticFindingType.PAYMENT_REQUEST


# --- ValidatedSemanticFinding: only ever constructed by the validator ----

VALIDATED_KWARGS = dict(
    type=SemanticFindingType.CREDENTIAL_REQUEST,
    evidence="şifrenizi doğrulayın",
    start=10,
    end=30,
    model_confidence=0.9,
    explanation="x",
)


def test_validated_finding_requires_start_end():
    finding = ValidatedSemanticFinding(**VALIDATED_KWARGS)
    assert finding.start == 10
    assert finding.end == 30


def test_validated_finding_missing_start_is_rejected():
    kwargs = dict(VALIDATED_KWARGS)
    del kwargs["start"]
    try:
        ValidatedSemanticFinding(**kwargs)
        raise AssertionError("expected ValidationError for missing start")
    except ValidationError:
        pass


def test_validated_finding_extra_field_is_rejected():
    try:
        ValidatedSemanticFinding(**VALIDATED_KWARGS, verdict="Phishing")
        raise AssertionError("expected ValidationError for extra field")
    except ValidationError:
        pass


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
