"""Unit tests for src/report/prompts.py (PHISHING_ROUTING_PLAN.md section
10.5, "Qwen çağrı 2" prompt construction). No model call anywhere in this
file — build_messages()/build_user_prompt() are pure string builders."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from schemas.decision import FinalDecision  # noqa: E402
from schemas.facts import EmailFacts, UrlFacts  # noqa: E402
from schemas.rule_assessment import RuleAssessment, RuleEvidence  # noqa: E402
from schemas.semantic import SemanticFindingType, ValidatedSemanticFinding  # noqa: E402
from src.decision.phishing_policy import (  # noqa: E402
    DECISION_PATH_CREDENTIAL_REQUEST_URL_UPGRADE,
    DECISION_PATH_RULE_ENGINE_ONLY,
)
from src.report.prompts import build_messages, build_user_prompt  # noqa: E402

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


def _evidence(signal="dkim_pass_but_domain_mismatch", weight=3.0,
              description="DKIM pass ama imzalayan domain From ile uyumsuz") -> RuleEvidence:
    return RuleEvidence(signal=signal, description=description, weight=weight)


def _assessment(evidence=None, rule_verdict="Güvenilir") -> RuleAssessment:
    return RuleAssessment(
        engine_version="v1",
        rule_verdict=rule_verdict,
        score=3.0,
        total=None,
        families=[],
        critical_matches=[],
        evidence=evidence or [_evidence()],
        decision_reasons=["test"],
    )


def _decision(
    final_verdict="Güvenilir",
    decision_path=DECISION_PATH_RULE_ENGINE_ONLY,
    contributing_rule_ids=None,
    contributing_semantic_ids=None,
    rule_verdict="Güvenilir",
) -> FinalDecision:
    return FinalDecision(
        rule_verdict=rule_verdict,
        final_verdict=final_verdict,
        decision_path=decision_path,
        contributing_rule_ids=contributing_rule_ids or [],
        contributing_semantic_ids=contributing_semantic_ids or [],
        analyst_review_required=final_verdict != "Güvenilir",
    )


def _finding(type_=SemanticFindingType.CREDENTIAL_REQUEST, evidence="şifrenizi doğrulayın",
             start=0, end=None) -> ValidatedSemanticFinding:
    return ValidatedSemanticFinding(
        type=type_,
        evidence=evidence,
        start=start,
        end=end if end is not None else len(evidence),
        model_confidence=0.9,
        explanation="test açıklaması",
    )


# --- authoritative verdict -------------------------------------------------

def test_prompt_shows_final_verdict_not_rule_verdict():
    """decision.final_verdict must be the only verdict shown to the
    model — a semantic upgrade means rule_verdict is stale, and the
    prompt must never let the model see (and possibly echo) the old
    value."""
    assessment = _assessment(rule_verdict="Güvenilir")
    decision = _decision(
        final_verdict="Muhtemel Phishing",
        decision_path=DECISION_PATH_CREDENTIAL_REQUEST_URL_UPGRADE,
        contributing_semantic_ids=["credential_request:0-24"],
    )
    finding = _finding()
    prompt = build_user_prompt(facts(), assessment, decision, [finding])

    assert "NİHAİ KARAR: Muhtemel Phishing" in prompt
    assert "NİHAİ KARAR: Güvenilir" not in prompt


# --- decision_path / contributing ids never appear as raw codes -----------

def test_raw_decision_path_code_never_appears_in_prompt():
    assessment = _assessment(rule_verdict="Güvenilir")
    decision = _decision(
        final_verdict="Muhtemel Phishing",
        decision_path=DECISION_PATH_CREDENTIAL_REQUEST_URL_UPGRADE,
        contributing_semantic_ids=["credential_request:0-24"],
    )
    finding = _finding()
    prompt = build_user_prompt(facts(), assessment, decision, [finding])

    assert DECISION_PATH_CREDENTIAL_REQUEST_URL_UPGRADE not in prompt
    # A plain-language explanation must be present instead.
    assert "kimlik bilgisi talebi" in prompt


def test_raw_contributing_id_strings_never_appear_in_prompt():
    """contributing_rule_ids/contributing_semantic_ids are internal
    bookkeeping tokens ("<type>:<start>-<end>", signal names) — they
    select which evidence/findings are shown, but must never be
    interpolated into the prompt text themselves."""
    finding = _finding(evidence="şifrenizi doğrulayın", start=5, end=29)
    semantic_id = f"{finding.type.value}:{finding.start}-{finding.end}"
    assessment = _assessment(rule_verdict="Güvenilir")
    decision = _decision(
        final_verdict="Muhtemel Phishing",
        decision_path=DECISION_PATH_CREDENTIAL_REQUEST_URL_UPGRADE,
        contributing_semantic_ids=[semantic_id],
    )
    prompt = build_user_prompt(facts(), assessment, decision, [finding])

    assert semantic_id not in prompt
    # The finding's own evidence/explanation must still surface.
    assert finding.evidence in prompt


# --- only contributing evidence/findings are included ----------------------

def test_only_contributing_rule_evidence_is_included():
    contributing = _evidence(signal="dkim_pass_but_domain_mismatch",
                              description="DKIM pass ama imzalayan domain From ile uyumsuz")
    noncontributing = _evidence(signal="url_shortener", weight=2.0,
                                 description="bağlantı kısaltıcı kullanılmış")
    assessment = _assessment(evidence=[contributing, noncontributing], rule_verdict="Phishing")
    decision = _decision(
        final_verdict="Phishing",
        rule_verdict="Phishing",
        contributing_rule_ids=["dkim_pass_but_domain_mismatch"],
    )
    prompt = build_user_prompt(facts(), assessment, decision, [])

    assert "DKIM pass ama imzalayan domain" in prompt
    assert "bağlantı kısaltıcı kullanılmış" not in prompt


def test_only_contributing_semantic_findings_are_included():
    contributing = _finding(type_=SemanticFindingType.CREDENTIAL_REQUEST,
                             evidence="şifrenizi doğrulayın")
    noncontributing = _finding(type_=SemanticFindingType.URGENCY_OR_PRESSURE,
                                evidence="hemen tıklayın", start=100, end=113)
    contributing_id = f"{contributing.type.value}:{contributing.start}-{contributing.end}"
    assessment = _assessment(rule_verdict="Güvenilir")
    decision = _decision(
        final_verdict="Muhtemel Phishing",
        decision_path=DECISION_PATH_CREDENTIAL_REQUEST_URL_UPGRADE,
        contributing_semantic_ids=[contributing_id],
    )
    prompt = build_user_prompt(facts(), assessment, decision, [contributing, noncontributing])

    assert "şifrenizi doğrulayın" in prompt
    assert "hemen tıklayın" not in prompt


def test_empty_contributing_evidence_says_so_explicitly():
    assessment = _assessment(rule_verdict="Güvenilir")
    decision = _decision(final_verdict="Güvenilir", contributing_rule_ids=[])
    prompt = build_user_prompt(facts(), assessment, decision, [])
    assert "katkı veren kural kanıtı yok" in prompt
    assert "katkı veren gövde bulgusu yok" in prompt


# --- URL/attachment blocks ---------------------------------------------

def test_url_block_lists_urls_by_name():
    url = UrlFacts(
        url="http://phishy.example/login",
        href_domain="phishy.example",
        anchor_text_domain="example.com",
        text_href_mismatch=True,
        is_ip_based=False,
        is_shortener=False,
        has_punycode=False,
        redirect_param=False,
    )
    assessment = _assessment(rule_verdict="Phishing")
    decision = _decision(final_verdict="Phishing", rule_verdict="Phishing")
    prompt = build_user_prompt(facts(urls=[url]), assessment, decision, [])
    assert "http://phishy.example/login" in prompt


# --- messages structure -----------------------------------------------

def test_build_messages_has_system_and_single_user_turn():
    assessment = _assessment()
    decision = _decision()
    messages = build_messages(facts(), assessment, decision, [])
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"


def test_system_prompt_forbids_classification_language():
    from src.report.prompts import SYSTEM_PROMPT
    lowered = SYSTEM_PROMPT.lower()
    assert "sınıflandırma" in lowered
    # Turkish dotless-I lowercasing quirk: "YAPMIYORSUN".lower() ==
    # "yapmiyorsun" (dotless i), not "yapmıyorsun" — check both forms.
    assert "yapmıyorsun" in lowered or "yapmiyorsun" in lowered


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
