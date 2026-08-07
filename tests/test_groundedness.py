"""
Unit tests for src/eval/groundedness.py, v3 plan section 7.3 metric 2
(holdout-fix-tasks.md T7).

Run with: python3 tests/test_groundedness.py
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.groundedness import check_claim, check_claims

CANDIDATES_PATH = PROJECT_ROOT / "data" / "holdout" / "candidates.jsonl"


def _candidate_signals(index: int) -> dict:
    """1-indexed, matching data/holdout/review.md's Candidate N numbering."""
    records = [json.loads(line) for line in open(CANDIDATES_PATH) if line.strip()]
    from schemas.facts import AttachmentFacts, EmailFacts, UrgencyMatch, UrlFacts
    r = dict(records[index - 1])
    r.pop("source_label", None)
    r.pop("_eml_path", None)
    r.pop("is_spam_not_phishing", None)
    r.pop("spam_reason", None)
    # data/holdout/candidates.jsonl is a frozen snapshot written before
    # later EmailFacts/AttachmentFacts fields existed (form_action_domain,
    # has_large_hidden_text, spf_mailfrom_domain, spf_aligned,
    # extension_mismatch) — CLAUDE.md locks hold-out data itself from
    # being touched, so backfill neutral defaults here rather than
    # regenerating the file.
    r.setdefault("form_action_domain", None)
    r.setdefault("has_large_hidden_text", False)
    r.setdefault("spf_mailfrom_domain", None)
    r.setdefault("spf_aligned", None)
    r.setdefault("has_advance_fee_fraud_language", False)
    r.setdefault("has_fake_reward_claim_language", False)
    for a in r.get("attachments", []):
        a.setdefault("extension_mismatch", False)
    facts = EmailFacts(
        **{k: v for k, v in r.items() if k not in ("urls", "attachments", "urgency_keywords")},
        urls=[UrlFacts(**u) for u in r["urls"]],
        attachments=[AttachmentFacts(**a) for a in r["attachments"]],
        urgency_keywords=[UrgencyMatch(**m) for m in r["urgency_keywords"]],
    )
    return facts.flat_signals()


def test_flat_signals_includes_url_mismatch_count():
    signals = _candidate_signals(7)
    assert signals["url_count"] == 8
    assert signals["url_text_href_mismatch_count"] == 0


def test_holdout_candidate_7_fabricated_mismatch_count_is_ungrounded():
    """The exact bug holdout-fix-tasks.md T7 was written for: a generated
    report claimed '28 URLs with a text/href mismatch' for this candidate
    while the facts show zero URLs with that flag set. The check must
    flag this as ungrounded — if it doesn't, the check isn't doing its job."""
    signals = _candidate_signals(7)
    claim = "Bulunan 28 URL'de text/href mismatch tespit edildi."
    result = check_claim(claim, signals)
    assert result.grounded is False


def test_true_boolean_signal_is_grounded():
    signals = {"credential_request": True, "spf_result": "fail"}
    result = check_claim("E-posta kimlik bilgisi talep ediyor.", signals)
    assert result.grounded is True


def test_false_boolean_signal_is_ungrounded():
    signals = {"credential_request": False}
    result = check_claim("E-posta kimlik bilgisi talep ediyor.", signals)
    assert result.grounded is False


def test_unrecognized_claim_is_ungrounded():
    signals = {"spf_result": "fail"}
    result = check_claim("Bu e-posta tamamen alakasız bir iddia içeriyor.", signals)
    assert result.grounded is False


def test_correct_count_claim_is_grounded():
    signals = {"url_count": 3, "url_text_href_mismatch_count": 3}
    result = check_claim("3 URL'de text/href uyuşmazlığı bulundu.", signals)
    assert result.grounded is True


def test_wrong_count_claim_is_ungrounded():
    signals = {"url_count": 8, "url_text_href_mismatch_count": 0}
    result = check_claim("5 URL'de text/href uyuşmazlığı bulundu.", signals)
    assert result.grounded is False


def test_check_claims_aggregates_ratio():
    signals = {"credential_request": True, "spf_result": "fail",
               "url_text_href_mismatch_count": 0, "url_count": 2}
    claims = [
        "E-posta kimlik bilgisi talep ediyor.",   # grounded
        "SPF doğrulaması başarısız oldu.",         # grounded
        "URL'lerde text/href uyuşmazlığı var.",    # ungrounded (count 0)
    ]
    result = check_claims(claims, signals)
    assert result["total_claims"] == 3
    assert result["grounded_claims"] == 2
    assert result["ratio"] == 2 / 3
    assert len(result["ungrounded"]) == 1


def test_empty_claims_list_has_ratio_one():
    result = check_claims([], {"spf_result": "fail"})
    assert result["total_claims"] == 0
    assert result["ratio"] == 1.0


def test_multilingual_dkim_claim_grounded_via_domain_match():
    signals = {"dkim_result": "pass", "dkim_domain_matches_from": True}
    result = check_claim("DKIM signature validated correctly.", signals)
    assert result.grounded is True


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
