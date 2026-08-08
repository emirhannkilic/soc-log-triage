"""Unit tests for src/semantic/analyze.py (PHISHING_ROUTING_PLAN.md
step 6). No real MODEL call anywhere in this file — prompt construction,
JSON-array extraction, and the validator hand-off are all testable
without mlx_vlm actually generating anything, and analyze_semantic()
itself is covered with the model call mocked out. One exception:
test_array_schema_compiles_with_real_llguidance_compiler runs the real
llguidance JSON-schema compiler (no model, no generation) — it exists
specifically to catch a schema-shape bug mocking would hide, see that
test's docstring."""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from schemas.facts import EmailFacts  # noqa: E402
from schemas.semantic import SemanticFindingCandidate  # noqa: E402
from src.llm.service import LLMServiceError, QwenService  # noqa: E402
from src.semantic.analyze import (  # noqa: E402
    _ALLOWED_TYPES,
    _array_schema_for,
    SemanticExtractionError,
    analyze_semantic,
    build_messages,
    build_user_prompt,
    extract_json_array,
)
from src.semantic.validate import RejectionReason, ValidationResult  # noqa: E402

BASE_FACTS_KWARGS = dict(
    spf_result="pass",
    dkim_result="pass",
    dmarc_result="pass",
    dkim_domain="example.com",
    dkim_domain_matches_from=True,
    spf_mailfrom_domain="example.com",
    spf_aligned=True,
    from_domain="example.com",
    from_source="From",
    return_path_domain="example.com",
    reply_to_domain="example.com",
    return_path_mismatch=False,
    reply_to_mismatch=False,
    display_name="Example Co",
    display_name_has_email=False,
    display_name_brand_mismatch=False,
    message_id_domain="example.com",
    message_id_domain_matches_from=True,
    received_hop_count=2,
    first_received_ip="1.2.3.4",
    urls=[],
    attachments=[],
    has_html_form=False,
    form_action_domain=None,
    has_hidden_text=False,
    has_large_hidden_text=False,
    image_only_body=False,
    urgency_keywords=[],
    credential_request=False,
    claims_attachment=False,
    has_advance_fee_fraud_language=False,
    has_fake_reward_claim_language=False,
    subject="Hesabınızı doğrulayın",
    date="Mon, 1 Jan 2024 00:00:00 +0000",
    body_text="Sayın müşterimiz, lütfen şifrenizi 24 saat içinde doğrulayın.",
    language="tr",
)


def facts(**overrides) -> EmailFacts:
    kwargs = dict(BASE_FACTS_KWARGS)
    kwargs.update(overrides)
    return EmailFacts(**kwargs)


# --- prompt construction --------------------------------------------------

def test_build_user_prompt_contains_exact_body_text():
    f = facts()
    prompt = build_user_prompt(f)
    # canonical_body must appear byte-for-byte — this is the same string
    # validate_raw_findings() will check offsets against later.
    assert f.body_text in prompt


def test_build_user_prompt_contains_display_name_and_domain():
    f = facts(display_name="Banka Güvenlik", from_domain="banka-guvenlik.tk")
    prompt = build_user_prompt(f)
    assert "Banka Güvenlik" in prompt
    assert "banka-guvenlik.tk" in prompt


def test_build_user_prompt_handles_missing_display_name():
    f = facts(display_name=None, from_domain=None)
    prompt = build_user_prompt(f)
    assert "(yok)" in prompt


def test_build_messages_has_system_and_single_user_turn():
    f = facts()
    messages = build_messages(f)
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert f.body_text in messages[1]["content"]


def test_system_prompt_lists_all_nine_allowed_types():
    for t in (
        "credential_request", "payment_request", "authority_impersonation",
        "brand_impersonation", "urgency_or_pressure", "threat_or_fear",
        "reward_or_prize_lure", "attachment_or_link_instruction",
        "reply_channel_manipulation",
    ):
        assert t in _ALLOWED_TYPES


def test_system_prompt_explicitly_disclaims_verdict_and_report():
    """Guardrail against scope creep: the prompt must explicitly tell the
    model it is not classifying or writing a report, not merely omit
    asking for one."""
    from src.semantic.analyze import SYSTEM_PROMPT
    lowered = SYSTEM_PROMPT.lower()
    assert "karar" in lowered
    assert "rapor" in lowered
    assert "yapmıyorsun" in lowered or "yapmiyorsun" in lowered


def test_system_prompt_clarifies_authority_impersonation_excludes_own_signature():
    """Regression guardrail for a real measured failure (PROGRESS.md,
    2026-08 semantic evaluation run): the model tagged a company's own
    signature/legal footer ("Saygılarımızla, Microsoft hesap ekibi",
    a GDPR/KVKK notice) as authority_impersonation — a sender using its
    own real name is the opposite of impersonation. The prompt must
    explicitly rule this out, not just define what the type IS."""
    from src.semantic.analyze import SYSTEM_PROMPT
    assert "authority_impersonation" in SYSTEM_PROMPT
    # The negative example must actually be present, not just the type
    # name — otherwise this test would pass even if the clarifying text
    # were removed.
    assert "imza" in SYSTEM_PROMPT.lower()


def test_system_prompt_clarifies_link_instruction_excludes_ordinary_links():
    """Regression guardrail for a real measured failure: the model
    flagged ordinary marketing/tracking/content links (a Netflix title
    link, a Çiçeksepeti campaign tracking URL) as
    attachment_or_link_instruction — that type requires the link to
    push a security-relevant action (verify, log in, open an
    attachment), not just be a URL."""
    from src.semantic.analyze import SYSTEM_PROMPT
    assert "attachment_or_link_instruction" in SYSTEM_PROMPT
    assert "pazarlama" in SYSTEM_PROMPT.lower() or "tracking" in SYSTEM_PROMPT.lower()


def test_system_prompt_clarifies_authority_impersonation_excludes_legal_footer():
    """Second-round regression guardrail (PROGRESS.md, 2026-08 second
    semantic evaluation run): even after the first prompt fix, the
    model kept tagging a sender's own legal/registration information
    (a company address, "6493 sayılı yasa kapsamında kurulmuş",
    "Facebook, Inc., Attention: Community Support, 1 Facebook Way") as
    authority_impersonation. This is a company disclosing ITS OWN real
    regulatory identity, not impersonating anyone — the prompt must
    explicitly rule this out AND require the impersonation claim to be
    combined with a transaction/threat/fund promise to count."""
    from src.semantic.analyze import SYSTEM_PROMPT
    lowered = SYSTEM_PROMPT.lower()
    assert "mersis" in lowered or "yasal" in lowered
    # The positive condition (impersonation claim + transaction/threat/
    # fund) must also be spelled out, not just the negative examples —
    # otherwise the model has no rule to distinguish a real positive
    # case from these negatives.
    assert "tehdit" in lowered and "fon" in lowered


def test_system_prompt_gives_positive_examples_for_threat_and_urgency():
    """Regression guardrail: threat_or_fear and urgency_or_pressure were
    measured as consistently MISSED (false negatives) across multiple
    candidates in the second evaluation run, unlike authority_
    impersonation/attachment_or_link_instruction which were measured as
    false positives. Concrete positive phrasing must be given so the
    model has something to pattern-match against, not just abstract
    definitions."""
    from src.semantic.analyze import SYSTEM_PROMPT
    assert "threat_or_fear" in SYSTEM_PROMPT
    assert "urgency_or_pressure" in SYSTEM_PROMPT
    assert "askıya alınacak" in SYSTEM_PROMPT or "yetkisiz erişim" in SYSTEM_PROMPT
    assert "24 saat" in SYSTEM_PROMPT or "iş günü" in SYSTEM_PROMPT


def test_system_prompt_requires_exact_line_break_preservation():
    """Regression guardrail: the largest single failure mode in the
    second evaluation run was NOT_FOUND_IN_BODY on quotes the model
    clearly intended correctly but reflowed/normalized (collapsing a
    line break the real body has) — a grounding failure, not a
    conceptual one. The prompt must explicitly tell the model to
    preserve line breaks inside a quote, not just say "verbatim"."""
    from src.semantic.analyze import SYSTEM_PROMPT
    lowered = SYSTEM_PROMPT.lower()
    assert "satır sonu" in lowered
    assert "birleştirme" in lowered or "temizlemeye" in lowered


# --- credential_request taxonomy fix (2026-08-08) ----------------------
#
# Regression suite for a real, repeated failure: five separate
# scripts/smoke_test_hybrid.py runs against tests/fixtures/
# hybrid_credential_upgrade.eml all produced attachment_or_link_
# instruction instead of credential_request for a sentence explicitly
# asking the reader to enter their username and password — the
# decision policy's upgrade rule (src/decision/phishing_policy.py)
# only reacts to CREDENTIAL_REQUEST, so this consistently blocked the
# Güvenilir -> Muhtemel Phishing upgrade path from ever being observed.
# Root cause (not model caprice): the OLD prompt defined
# attachment_or_link_instruction with "kimlik bilgisi girme" explicitly
# inside its own definition, and never defined credential_request at
# all despite it being in the allowed-type enum — the model had
# nowhere else to route a credential-entry request, and correctly went
# with the only definition that covered it.

def test_system_prompt_defines_credential_request_with_positive_and_negative_examples():
    """credential_request is in the ALLOWED enum (see
    test_system_prompt_lists_all_nine_allowed_types) but the prompt
    used to never define what it means — this is the taxonomy gap the
    five failed smoke-test runs traced back to. The definition must
    give both a positive example (something to pattern-match toward)
    and a negative example (a past-tense notification, which is NOT a
    request) so the model can distinguish "asking for a credential"
    from "reporting that one already changed"."""
    from src.semantic.analyze import SYSTEM_PROMPT
    assert '"credential_request":' in SYSTEM_PROMPT
    lowered = SYSTEM_PROMPT.lower()
    # Positive: a concrete verb+target example the model can match against.
    assert "parolanızı forma girin" in lowered or "doğrulama kodunu" in lowered
    # Negative: a past-tense notification must be explicitly ruled out —
    # otherwise "şifreniz güncellendi" (a notification) risks being
    # read as a request just because it mentions the same target word.
    assert "başarıyla güncellendi" in lowered or "geçmiş zaman" in lowered


def test_system_prompt_narrows_attachment_or_link_instruction_to_action_channel_only():
    """attachment_or_link_instruction's definition must no longer claim
    "kimlik bilgisi girme" as part of ITS OWN scope — that phrase used
    to sit inside this type's own definition, giving the model no
    reason to ever reach for credential_request instead. The type must
    now be scoped to the action channel (click/open/run) with an
    explicit statement that requesting the credential itself does NOT
    substitute for a separate credential_request finding.

    Uses the ORIGINAL (non-lowered) SYSTEM_PROMPT for these checks —
    Python's str.lower() turns Turkish "İ" into a combining "i̇"
    (two code points), which breaks a plain substring search against
    an ASCII "i" the way test_system_prompt_forbids_classification_
    language already had to work around elsewhere. Searching the
    original text avoids the whole class of dotless-I mismatches."""
    from src.semantic.analyze import SYSTEM_PROMPT
    assert "EYLEM KANALINI" in SYSTEM_PROMPT
    assert "YERİNE GEÇMEZ" in SYSTEM_PROMPT


def test_system_prompt_states_multi_label_rule_for_same_evidence():
    """The model must be told explicitly that a single sentence
    satisfying two types should produce TWO findings (one per type),
    and that reusing the same evidence quote across types is not a
    duplicate — src/semantic/validate.py's seen-key is (type, evidence),
    so two findings with the same evidence but different types were
    always structurally allowed; the prompt just never told the model
    this was expected, and it defaulted to picking one type."""
    from src.semantic.analyze import SYSTEM_PROMPT
    assert "ÇOKLU ETİKET KURALI" in SYSTEM_PROMPT
    assert "tekrar/duplicate" in SYSTEM_PROMPT


def test_system_prompt_multi_label_example_names_both_relevant_types():
    """The multi-label rule's own worked example must name BOTH
    attachment_or_link_instruction and credential_request together —
    a rule stated only in the abstract ("multiple types can apply") is
    exactly the kind of guidance the taxonomy gap already showed the
    model doesn't reliably generalize from; a concrete example anchors
    it to the actual failure mode being fixed."""
    from src.semantic.analyze import SYSTEM_PROMPT
    example_section = SYSTEM_PROMPT[SYSTEM_PROMPT.index("ÇOKLU ETİKET KURALI"):]
    assert "attachment_or_link_instruction" in example_section
    assert "credential_request" in example_section


# --- JSON array extraction --------------------------------------------

def test_extract_json_array_plain():
    raw = '[{"type": "credential_request", "evidence": "x"}]'
    result = extract_json_array(raw)
    assert result == [{"type": "credential_request", "evidence": "x"}]


def test_extract_json_array_in_code_fence():
    raw = '```json\n[{"type": "urgency_or_pressure", "evidence": "hemen"}]\n```'
    result = extract_json_array(raw)
    assert result == [{"type": "urgency_or_pressure", "evidence": "hemen"}]


def test_extract_json_array_empty_list():
    raw = "[]"
    assert extract_json_array(raw) == []


def test_extract_json_array_with_surrounding_prose_still_extracts():
    raw = 'Here is the result:\n[{"type": "threat_or_fear", "evidence": "y"}]\nDone.'
    result = extract_json_array(raw)
    assert result == [{"type": "threat_or_fear", "evidence": "y"}]


def test_extract_json_array_malformed_json_returns_none():
    raw = "[{not valid json"
    assert extract_json_array(raw) is None


def test_extract_json_array_top_level_object_returns_none():
    """A top-level JSON object (not array) must not be silently accepted
    as if it were a single-element list — that would be repairing
    malformed output, which this project forbids."""
    raw = '{"type": "credential_request", "evidence": "x"}'
    assert extract_json_array(raw) is None


def test_extract_json_array_no_json_at_all_returns_none():
    raw = "I cannot find any manipulative content in this email."
    assert extract_json_array(raw) is None


def test_extract_json_array_never_raises_on_garbage():
    for garbage in ("", "null", "42", '"just a string"', "[1, 2, {broken"):
        extract_json_array(garbage)  # must not raise


# --- schema shape: real llguidance compiler, no model/generation --------

def test_array_schema_hoists_defs_to_outer_root():
    schema = _array_schema_for(SemanticFindingCandidate)
    assert schema["type"] == "array"
    assert "$defs" in schema
    assert "$defs" not in schema["items"]
    assert schema["items"]["properties"]["type"]["$ref"] == "#/$defs/SemanticFindingType"


def test_array_schema_compiles_with_real_llguidance_compiler():
    """Regression test for a real failure: wrapping
    SemanticFinding.model_json_schema() as {"type": "array", "items":
    <schema>} without hoisting $defs left "$ref": "#/$defs/..." pointing
    at a document root that no longer has $defs (it was nested under
    `items` instead) — llguidance's JsonCompiler raised
    "Pointer '/$defs/SemanticFindingType' does not exist" the first time
    this was run against the real model. This test would have caught it
    without needing the model at all: JsonCompiler.compile() is a pure
    schema-to-grammar step, no generation involved."""
    import json

    import llguidance as llg

    schema = _array_schema_for(SemanticFindingCandidate)
    schema_text = json.dumps(schema)
    if hasattr(llg, "JsonCompiler"):
        llg.JsonCompiler(separators=(", ", ": "), whitespace_pattern="").compile(schema_text)
    else:
        llg.grammar_from("json_schema", schema_text)


# --- analyze_semantic() with the model call mocked via QwenService DI --

def _mock_generate_result(text: str):
    m = MagicMock()
    m.text = text
    return m


def _service_returning(text: str) -> QwenService:
    """A QwenService whose injected load_fn/generate_fn never touch
    mlx_vlm — same mocking shape the previous patch("...load_model", ...)
    approach had, just moved to constructor injection (src/llm/service.py's
    testability contract)."""
    return QwenService(
        load_fn=lambda path: (MagicMock(), MagicMock()),
        apply_chat_template_fn=lambda processor, config, messages, num_images: "PROMPT",
        generate_fn=lambda *a, **k: _mock_generate_result(text),
    )


def _service_raising(exc: Exception) -> QwenService:
    def _raise(*a, **k):
        raise exc

    return QwenService(
        load_fn=lambda path: (MagicMock(), MagicMock()),
        apply_chat_template_fn=lambda processor, config, messages, num_images: "PROMPT",
        generate_fn=_raise,
    )


def test_analyze_semantic_returns_validation_result_with_mocked_model():
    f = facts()
    evidence = "şifrenizi 24 saat içinde doğrulayın"
    assert evidence in f.body_text
    raw_json = (
        f'[{{"type": "urgency_or_pressure", "evidence": {evidence!r}, '
        f'"model_confidence": 0.8, '
        f'"explanation": "24 saatlik süre baskısı"}}]'
    ).replace("'", '"')

    with patch("src.semantic.analyze._build_logits_processor", return_value=MagicMock()):
        result = analyze_semantic(f, service=_service_returning(raw_json))

    assert isinstance(result, ValidationResult)
    assert len(result.accepted) == 1
    accepted = result.accepted[0]
    assert accepted.evidence == evidence
    # start/end are computed by the validator, not taken from the model —
    # confirm they're actually correct against the real body.
    assert f.body_text[accepted.start:accepted.end] == evidence
    assert result.rejected == []


def test_analyze_semantic_hallucinated_evidence_is_rejected_not_raised():
    """The model call succeeds and returns well-formed JSON, but the
    quote isn't actually in the body — this must come back as a
    rejected finding via validate_raw_findings, not crash and not be
    silently accepted."""
    f = facts()
    raw_json = (
        '[{"type": "payment_request", "evidence": "kredi kartı bilgilerinizi girin", '
        '"model_confidence": 0.9, "explanation": "x"}]'
    )

    with patch("src.semantic.analyze._build_logits_processor", return_value=MagicMock()):
        result = analyze_semantic(f, service=_service_returning(raw_json))

    assert result.accepted == []
    assert result.rejected[0].rejection_reason == RejectionReason.NOT_FOUND_IN_BODY


def test_analyze_semantic_model_emitting_start_end_anyway_is_rejected():
    """Some models ignore the "don't emit offsets" instruction (or an
    older prompt/checkpoint still asks for them) and include start/end
    in the JSON regardless. SemanticFindingCandidate's extra="forbid"
    means this must be rejected as SCHEMA_INVALID, not silently accepted
    with the model's (unverified) offsets, and not silently stripped
    down to just the allowed fields either — CLAUDE.md's "çıktıyı
    onarma yasak" rule."""
    f = facts()
    evidence = "şifrenizi 24 saat içinde doğrulayın"
    raw_json = (
        f'[{{"type": "urgency_or_pressure", "evidence": {evidence!r}, '
        f'"start": 999, "end": 1050, "model_confidence": 0.8, '
        f'"explanation": "x"}}]'
    ).replace("'", '"')

    with patch("src.semantic.analyze._build_logits_processor", return_value=MagicMock()):
        result = analyze_semantic(f, service=_service_returning(raw_json))

    assert result.accepted == []
    assert result.rejected[0].rejection_reason == RejectionReason.SCHEMA_INVALID


def test_analyze_semantic_no_findings_is_empty_not_an_error():
    f = facts()
    with patch("src.semantic.analyze._build_logits_processor", return_value=MagicMock()):
        result = analyze_semantic(f, service=_service_returning("[]"))

    assert result.accepted == []
    assert result.rejected == []


def test_analyze_semantic_malformed_output_raises_semantic_extraction_error():
    """No retry, no repair — same rule src/demo.py's _report_from_llm
    follows for the report path (CLAUDE.md "Yapılmayacaklar"). The
    exception type is SemanticExtractionError(code="invalid_json"), not
    SystemExit — an earlier version raised SystemExit, which is wrong:
    SystemExit is meant to terminate the whole process, not signal an
    ordinary, catchable failure to a caller (see src/semantic/analyze.py's
    module docstring)."""
    f = facts()
    with patch("src.semantic.analyze._build_logits_processor", return_value=MagicMock()):
        try:
            analyze_semantic(f, service=_service_returning("not json at all"))
            raise AssertionError("expected SemanticExtractionError")
        except SemanticExtractionError as e:
            assert e.code == "invalid_json"


def test_analyze_semantic_service_failure_wrapped_as_model_call_failed():
    """When the underlying generate call itself fails (a GPU/Metal
    timeout, PROGRESS.md), QwenService.generate() wraps it as
    LLMServiceError (with `from exc` preserving the original), and
    analyze_semantic() must in turn normalize THAT to
    SemanticExtractionError(code="model_call_failed") — again via
    `from exc` — rather than letting LLMServiceError leak past this
    module's boundary. The full chain (RuntimeError -> LLMServiceError
    -> SemanticExtractionError) must survive intact."""
    f = facts()
    original = RuntimeError("GPU Timeout")
    with patch("src.semantic.analyze._build_logits_processor", return_value=MagicMock()):
        try:
            analyze_semantic(f, service=_service_raising(original))
            raise AssertionError("expected SemanticExtractionError")
        except SemanticExtractionError as e:
            assert e.code == "model_call_failed"
            assert isinstance(e.__cause__, LLMServiceError)
            assert e.__cause__.__cause__ is original


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
