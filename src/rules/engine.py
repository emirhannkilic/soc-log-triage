"""
Deterministic rule engine (v3 plan section 5, CLAUDE.md "Rule engine
eşikleri"). Takes EmailFacts.flat_signals() output, sums weights from
config/rules.yaml, returns a three-class verdict. This is the layer that
actually decides Phishing/Muhtemel Phishing/Güvenilir — the LLM downstream
never classifies, it only narrates this verdict (see CLAUDE.md "Mimari").
"""
from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_RULES_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "rules.yaml"


@dataclass
class RuleMatch:
    signal: str
    weight: float
    description: str


@dataclass
class Verdict:
    score: float
    verdict: str  # "Phishing" | "Muhtemel Phishing" | "Güvenilir"
    matches: list[RuleMatch]


def load_rules(path: Path = DEFAULT_RULES_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _auth_pass(signals: dict, field: str) -> bool:
    return signals.get(field) == "pass"


def evaluate(signals: dict, rules: dict | None = None) -> Verdict:
    """signals: EmailFacts.flat_signals() output for one email."""
    if rules is None:
        rules = load_rules()

    weights = rules["signals"]
    matches: list[RuleMatch] = []

    def fire(name: str):
        w = weights[name]["weight"]
        matches.append(RuleMatch(name, w, weights[name]["description"]))

    if signals.get("spf_result") == "fail" or signals.get("dmarc_result") == "fail":
        fire("spf_or_dmarc_fail")

    dkim_missing_or_fail = signals.get("dkim_result") in (None, "none", "fail")
    if dkim_missing_or_fail and signals.get("dkim_domain_matches_from") is False:
        fire("dkim_missing_or_fail_domain_mismatch")

    if signals.get("return_path_mismatch"):
        fire("return_path_mismatch")

    if signals.get("display_name_brand_mismatch"):
        fire("display_name_brand_mismatch")

    if signals.get("url_text_href_mismatch_count", 0) > 0:
        fire("url_text_href_mismatch")

    if signals.get("url_ip_based_count", 0) > 0:
        fire("url_ip_based")

    if signals.get("url_punycode_count", 0) > 0:
        fire("url_punycode")

    if signals.get("attachment_risky_type_count", 0) > 0:
        fire("attachment_risky_type")

    if signals.get("attachment_double_extension_count", 0) > 0:
        fire("attachment_double_extension")

    if signals.get("has_hidden_text") or signals.get("image_only_body"):
        fire("hidden_text_or_image_only")

    if signals.get("credential_request") and signals.get("url_count", 0) > 0:
        fire("credential_request_with_external_link")

    if signals.get("urgency_keyword_count", 0) > 0:
        fire("urgency_keywords")

    if signals.get("attachment_is_archive_count", 0) > 0 and signals.get("credential_request"):
        fire("is_archive_with_credential_request")

    if signals.get("claims_attachment") and signals.get("attachment_count", 0) == 0:
        fire("claims_attachment_but_empty")

    from_domain = signals.get("from_domain")
    if from_domain and "." not in from_domain:
        fire("from_domain_no_tld")

    all_pass = (
        _auth_pass(signals, "spf_result")
        and _auth_pass(signals, "dkim_result")
        and _auth_pass(signals, "dmarc_result")
    )
    consistent = (
        signals.get("dkim_domain_matches_from") is True
        and not signals.get("return_path_mismatch")
        and not signals.get("reply_to_mismatch")
    )
    if all_pass and consistent:
        fire("all_auth_pass_and_consistent")

    score = sum(m.weight for m in matches)

    thresholds = rules["thresholds"]
    if score >= thresholds["phishing"]:
        verdict = "Phishing"
    elif score >= thresholds["suspicious"]:
        verdict = "Muhtemel Phishing"
    else:
        verdict = "Güvenilir"

    return Verdict(score=score, verdict=verdict, matches=matches)
