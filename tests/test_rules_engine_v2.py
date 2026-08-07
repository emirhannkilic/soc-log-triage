"""Unit tests for src/rules/engine_v2.py (Rule Engine v2 — CLAUDE.md
"Rule Engine v2 — Aile Bazlı Skorlama"). Does not test src/rules/
engine.py (v1) — see tests/test_rules_engine.py for that, v1 is
unchanged and stays the production baseline."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from schemas.facts import AttachmentFacts, EmailFacts, UrgencyMatch, UrlFacts
from src.rules.engine_v2 import (
    _auth_aligned,
    _root_domain,
    _url_root_mismatch_count,
    evaluate_v2,
    load_rules,
)

RULES = load_rules()

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
    subject="Hello",
    date="Mon, 1 Jan 2024 00:00:00 +0000",
    body_text="hello",
    language="en",
)


def facts(**overrides) -> EmailFacts:
    kwargs = dict(BASE_FACTS_KWARGS)
    kwargs.update(overrides)
    return EmailFacts(**kwargs)


def test_clean_email_is_guvenilir():
    result = evaluate_v2(facts(), RULES)
    assert result.verdict == "Güvenilir", (result.verdict, result.total)


def test_urgency_alone_is_guvenilir():
    """Codex's guardrail: urgency alone (weight 1) must never on its own
    push a family score high enough to leave Güvenilir — marketing mail
    carries urgency language at similar rates to phishing."""
    f = facts(urgency_keywords=[UrgencyMatch(keyword="hemen", context="hemen tıklayın")])
    result = evaluate_v2(f, RULES)
    assert result.verdict == "Güvenilir", (result.verdict, result.families["content"].score)


def test_double_extension_and_risky_type_is_critical():
    f = facts(attachments=[
        AttachmentFacts(filename="invoice.pdf.exe", mime_type="application/x-msdownload",
                        size=1024, double_extension=True, risky_type=True, is_archive=False,
                        extension_mismatch=False)
    ])
    result = evaluate_v2(f, RULES)
    assert result.verdict == "Phishing", (result.verdict, result.critical_matches)
    assert "double_extension_and_risky_type_same_attachment" in result.critical_matches


def test_risky_type_without_double_extension_is_not_critical():
    """Risky type alone (no double extension) must not trip the critical
    predicate — it's still a real payload signal, just not automatic
    Phishing on its own (Codex: ".docm şüphelidir ama otomatik olarak
    phishing değildir")."""
    f = facts(attachments=[
        AttachmentFacts(filename="report.docm", mime_type="application/vnd.ms-word.document.macroEnabled.12",
                        size=1024, double_extension=False, risky_type=True, is_archive=False,
                        extension_mismatch=False)
    ])
    result = evaluate_v2(f, RULES)
    assert not result.critical_matches
    assert result.verdict != "Phishing", (result.verdict, result.families["payload"].score)


# --- Fix #1: hidden preheader vs real content-hiding --------------------

def test_short_hidden_text_alone_does_not_score():
    """has_hidden_text (True) but has_large_hidden_text (False) — a
    preheader. Must not contribute to content family score."""
    f = facts(has_hidden_text=True, has_large_hidden_text=False)
    result = evaluate_v2(f, RULES)
    assert result.families["content"].score == 0, result.families["content"]


def test_large_hidden_text_scores():
    f = facts(has_hidden_text=True, has_large_hidden_text=True)
    result = evaluate_v2(f, RULES)
    assert result.families["content"].score > 0, result.families["content"]


def test_image_only_body_scores_independently_of_hidden_text():
    f = facts(image_only_body=True, has_hidden_text=False, has_large_hidden_text=False)
    result = evaluate_v2(f, RULES)
    assert result.families["content"].score > 0, result.families["content"]


# --- Fix #2: conditional return_path_mismatch ----------------------------

def test_return_path_mismatch_alone_with_aligned_auth_does_not_score():
    """The documented false-positive shape: ESP-routed bounce address,
    everything else aligned. Must not fire identity on its own."""
    f = facts(return_path_mismatch=True, return_path_domain="esp-provider.com")
    result = evaluate_v2(f, RULES)
    assert result.families["identity"].score == 0, result.families["identity"]


def test_return_path_mismatch_with_misaligned_auth_scores():
    f = facts(return_path_mismatch=True, return_path_domain="esp-provider.com",
              spf_result="fail")
    result = evaluate_v2(f, RULES)
    assert result.families["identity"].score > 0, result.families["identity"]


def test_return_path_mismatch_with_free_reply_to_scores():
    f = facts(return_path_mismatch=True, return_path_domain="esp-provider.com",
              reply_to_mismatch=True, reply_to_domain="gmail.com")
    result = evaluate_v2(f, RULES)
    assert result.families["identity"].score > 0, result.families["identity"]


def test_return_path_mismatch_with_brand_mismatch_scores():
    f = facts(return_path_mismatch=True, return_path_domain="esp-provider.com",
              display_name_brand_mismatch=True)
    result = evaluate_v2(f, RULES)
    assert result.families["identity"].score > 0, result.families["identity"]


# --- Adım 6: SPF alignment -------------------------------------------------

def test_spf_pass_but_misaligned_mailfrom_scores():
    """spf_result=pass only proves the ENVELOPE domain's own SPF record
    checked out — an attacker's rented domain passes this trivially while
    claiming an unrelated From. Only fires when DKIM does NOT already
    establish alignment on its own (see the guard's docstring in
    engine_v2.py) — dkim_result must be something other than an aligned
    pass here, or the guard suppresses it."""
    f = facts(spf_result="pass", spf_aligned=False, spf_mailfrom_domain="rented-evil.tld",
              dkim_result="none", dkim_domain=None, dkim_domain_matches_from=None)
    result = evaluate_v2(f, RULES)
    assert result.families["identity"].score > 0, result.families["identity"]


def test_spf_misaligned_but_dkim_aligned_does_not_score():
    """The measured false-positive shape (data/rule_engine_v2_devset, adım
    6): a legitimate bulk sender on a shared ESP (Amazon SES, Persona
    Click) fails SPF alignment because the ESP's shared sending IP isn't
    the sender's own domain — that's normal ESP routing. When DKIM
    independently vouches for the claimed From (pass AND domain matches),
    the SPF mismatch adds no evidence and must not score. 6/6 measured
    false positives on data/rule_engine_v2_devset had exactly this shape
    before this guard existed."""
    f = facts(spf_result="pass", spf_aligned=False, spf_mailfrom_domain="amazonses.com",
              dkim_result="pass", dkim_domain="example.com", dkim_domain_matches_from=True)
    result = evaluate_v2(f, RULES)
    assert result.families["identity"].score == 0, result.families["identity"]


def test_spf_pass_and_aligned_mailfrom_does_not_score():
    f = facts(spf_result="pass", spf_aligned=True, spf_mailfrom_domain="example.com")
    result = evaluate_v2(f, RULES)
    assert result.families["identity"].score == 0, result.families["identity"]


def test_spf_pass_with_unknown_alignment_does_not_score():
    """spf_aligned=None (no smtp.mailfrom in Authentication-Results) means
    'can't tell,' not 'misaligned' — must not fire."""
    f = facts(spf_result="pass", spf_aligned=None, spf_mailfrom_domain=None)
    result = evaluate_v2(f, RULES)
    assert result.families["identity"].score == 0, result.families["identity"]


# --- Fix #3: root-domain URL comparison -----------------------------------

def test_root_domain_helper():
    assert _root_domain("email.uber.com") == "uber.com"
    # Real PSL (tldextract), not "last two labels": gov.tr is itself a
    # public suffix, so the registrable domain is turkiye.gov.tr, not gov.tr.
    assert _root_domain("www.turkiye.gov.tr") == "turkiye.gov.tr"
    assert _root_domain("uber.com") == "uber.com"
    assert _root_domain("mail.example.co.uk") == "example.co.uk"
    assert _root_domain(None) is None


def test_subdomain_of_same_org_is_not_url_mismatch():
    f = facts(urls=[
        UrlFacts(url="https://email.uber.com/x", href_domain="email.uber.com",
                 anchor_text_domain="uber.com", text_href_mismatch=True,
                 is_ip_based=False, is_shortener=False, has_punycode=False,
                 redirect_param=False)
    ])
    assert _url_root_mismatch_count(f) == 0
    result = evaluate_v2(f, RULES)
    assert result.families["url"].score == 0, result.families["url"]


def test_genuinely_different_root_domain_is_url_mismatch():
    f = facts(urls=[
        UrlFacts(url="https://evil-domain.ru/x", href_domain="evil-domain.ru",
                 anchor_text_domain="paypal.com", text_href_mismatch=True,
                 is_ip_based=False, is_shortener=False, has_punycode=False,
                 redirect_param=False)
    ])
    assert _url_root_mismatch_count(f) == 1
    result = evaluate_v2(f, RULES)
    assert result.families["url"].score > 0, result.families["url"]


# --- Adım 6: form action domain mismatch -----------------------------------

def test_form_action_mismatch_with_credential_request_scores():
    f = facts(from_domain="mybank.com", form_action_domain="evil-collector.ru",
              has_html_form=True, credential_request=True)
    result = evaluate_v2(f, RULES)
    fired_signals = {h.signal for hits in result.families.values() for h in hits.hits}
    assert "form_action_domain_mismatch" in fired_signals
    assert result.families["content"].score > 0


def test_form_action_same_root_domain_does_not_score():
    """A form posting to a subdomain of the sender's own domain
    (login.mybank.com vs mybank.com) is not a mismatch — root-domain
    compared, same as URL mismatch fix #3."""
    f = facts(from_domain="mybank.com", form_action_domain="login.mybank.com",
              has_html_form=True, credential_request=True)
    result = evaluate_v2(f, RULES)
    fired_signals = {h.signal for hits in result.families.values() for h in hits.hits}
    assert "form_action_domain_mismatch" not in fired_signals


def test_form_action_mismatch_without_credential_request_does_not_score():
    """A different form-action domain alone isn't enough — Codex's
    guardrail: has_html_form alone is weak (legitimate surveys exist).
    The mismatch only matters combined with an actual credential ask."""
    f = facts(from_domain="mybank.com", form_action_domain="evil-collector.ru",
              has_html_form=True, credential_request=False)
    result = evaluate_v2(f, RULES)
    fired_signals = {h.signal for hits in result.families.values() for h in hits.hits}
    assert "form_action_domain_mismatch" not in fired_signals


# --- Auth guard invariant --------------------------------------------------

def test_auth_aligned_and_dkim_pass_domain_mismatch_are_mutually_exclusive():
    """Codex's invariant: all_auth_pass_and_consistent (dkim_domain_
    matches_from == True) and dkim_pass_but_domain_mismatch
    (dkim_domain_matches_from == False) cannot both be true — they read
    the same field in opposite directions. If this ever fails on real
    data it's a parser bug, not a rule engine one; this test guards the
    logical invariant the engine's design depends on."""
    aligned = facts(dkim_domain_matches_from=True)
    mismatched = facts(dkim_domain_matches_from=False, dkim_result="pass")
    assert _auth_aligned(aligned) is True
    assert _auth_aligned(mismatched) is False
    # A dkim_pass_but_domain_mismatch hit requires dkim_result=="pass"
    # AND dkim_domain_matches_from is False — auth_aligned requires the
    # opposite for the same field, so they cannot co-occur.
    result_mismatched = evaluate_v2(mismatched, RULES)
    fired_signals = {h.signal for hits in result_mismatched.families.values() for h in hits.hits}
    assert "dkim_pass_but_domain_mismatch" in fired_signals
    assert _auth_aligned(mismatched) is False


# --- Adım 7: magic-byte / extension mismatch --------------------------

def test_extension_mismatch_scores_payload_family():
    f = facts(attachments=[
        AttachmentFacts(filename="invoice.pdf", mime_type="application/pdf",
                        size=2048, double_extension=False, risky_type=False,
                        is_archive=False, extension_mismatch=True)
    ])
    result = evaluate_v2(f, RULES)
    assert result.families["payload"].score > 0, result.families["payload"]


def test_extension_mismatch_and_double_extension_same_subgroup_no_double_bonus():
    """Both signals live in filename_disguise — corroboration_bonus needs
    a DIFFERENT subgroup to fire, so two filename_disguise hits alone must
    not out-score a single strong hit from two distinct subgroups."""
    f = facts(attachments=[
        AttachmentFacts(filename="invoice.pdf.scr", mime_type="application/octet-stream",
                        size=2048, double_extension=True, risky_type=True,
                        is_archive=False, extension_mismatch=True)
    ])
    result = evaluate_v2(f, RULES)
    # dangerous_type (attachment_risky_type) IS a different subgroup here,
    # so corroboration still applies — this just confirms it isn't double
    # counted from filename_disguise's two hits alone.
    assert result.families["payload"].score == 4, result.families["payload"]


def test_no_extension_mismatch_does_not_score():
    f = facts(attachments=[
        AttachmentFacts(filename="invoice.pdf", mime_type="application/pdf",
                        size=2048, double_extension=False, risky_type=False,
                        is_archive=False, extension_mismatch=False)
    ])
    result = evaluate_v2(f, RULES)
    assert result.families["payload"].score == 0, result.families["payload"]


# --- Adım 8: scam-narrative signals ----------------------------------------

def test_advance_fee_fraud_language_scores_content_family():
    f = facts(has_advance_fee_fraud_language=True)
    result = evaluate_v2(f, RULES)
    assert result.families["content"].score > 0, result.families["content"]


def test_fake_reward_claim_language_scores_content_family():
    f = facts(has_fake_reward_claim_language=True)
    result = evaluate_v2(f, RULES)
    assert result.families["content"].score > 0, result.families["content"]


def test_scam_narrative_alone_does_not_reach_phishing():
    """Deliberately low weight (+1): scam-narrative language alone must
    not be sufficient for a Phishing verdict on its own — it needs
    corroboration from another family, same guardrail as urgency_keywords
    (test_urgency_alone_is_guvenilir)."""
    f = facts(has_advance_fee_fraud_language=True, has_fake_reward_claim_language=True)
    result = evaluate_v2(f, RULES)
    assert result.verdict != "Phishing", (result.verdict, result.families["content"])


def test_advance_fee_fraud_matches_real_missed_sample_shape():
    """The exact shape that motivated adım 8: zero header misalignment,
    zero URLs, zero attachments, pure social-engineering prose
    (data/phishing_pot/email/sample-4784.eml) — the family formula
    produced 0 signals across all four families for this email before
    this signal existed."""
    f = facts(has_advance_fee_fraud_language=True, urls=[], attachments=[])
    result = evaluate_v2(f, RULES)
    assert result.families["content"].score > 0
    assert result.families["identity"].score == 0
    assert result.families["url"].score == 0
    assert result.families["payload"].score == 0


def test_thresholds_symmetry_with_v1():
    """Not a v1/v2 equivalence claim — just documents that v2's
    thresholds (5, 3) start from the same numbers as v1's config/
    rules.yaml thresholds, per CLAUDE.md's note that they're a starting
    point to be re-validated on the new dev set, not re-derived from
    scratch."""
    t = RULES["thresholds"]
    assert t["phishing"] == 5
    assert t["suspicious"] == 3


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
