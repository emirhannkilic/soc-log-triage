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


# Consumer webmail providers. Used only to qualify a Reply-To mismatch:
# a corporate or institutional sender whose replies are redirected to a free
# mailbox is the shape of advance-fee fraud (419) and of business email
# compromise — the sender has, or has borrowed, a trusted domain, but wants
# the conversation to continue somewhere they actually control.
_FREE_MAIL_DOMAINS = frozenset({
    "gmail.com", "googlemail.com", "yahoo.com", "yahoo.co.uk", "ymail.com",
    "hotmail.com", "hotmail.co.uk", "outlook.com", "live.com", "msn.com",
    "aol.com", "icloud.com", "me.com", "mac.com",
    "proton.me", "protonmail.com", "pm.me",
    "gmx.com", "gmx.net", "mail.com", "zoho.com",
    "yandex.com", "yandex.ru", "yandex.com.tr", "mail.ru",
})


def _reply_to_free_mail(signals: dict) -> bool:
    """Reply-To points at a free mailbox while From does not.

    The raw reply_to_mismatch signal is too noisy to score on its own — on
    the 80-email hold-out it fires for 40% of phishing but also 25% of
    legitimate mail, since plenty of real senders route replies elsewhere.
    Narrowing it to "and the reply address is consumer webmail" makes it
    clean: 3/15 phishing, 0/65 legitimate, 0/60 on the dev set.
    """
    reply_to = (signals.get("reply_to_domain") or "").lower()
    from_domain = (signals.get("from_domain") or "").lower()
    if not reply_to or not from_domain or not signals.get("reply_to_mismatch"):
        return False
    return reply_to in _FREE_MAIL_DOMAINS and from_domain not in _FREE_MAIL_DOMAINS


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

    # A VALID signature from the wrong domain. The check above only covers a
    # missing or failing DKIM, so this case slipped through entirely: the
    # signature verifies, so dkim_result is "pass", but the signing domain is
    # not the sender's. That is what third-party spoofing looks like — the
    # attacker signs with a domain they control and the From header claims
    # another. Found on a phishing sample that scored 2 (Güvenilir) while
    # DKIM was signed by ladelanoagency.com for a From of jwgmedia.com.
    if (signals.get("dkim_result") == "pass"
            and signals.get("dkim_domain_matches_from") is False):
        fire("dkim_pass_but_domain_mismatch")

    if signals.get("return_path_mismatch"):
        fire("return_path_mismatch")

    if _reply_to_free_mail(signals):
        fire("reply_to_free_mail")

    if signals.get("display_name_brand_mismatch"):
        fire("display_name_brand_mismatch")

    if signals.get("url_text_href_mismatch_count", 0) > 0:
        fire("url_text_href_mismatch")

    if signals.get("url_ip_based_count", 0) > 0:
        fire("url_ip_based")

    if signals.get("url_punycode_count", 0) > 0:
        fire("url_punycode")

    if signals.get("url_shortener_count", 0) > 0:
        fire("url_shortener")

    if signals.get("url_redirect_param_count", 0) > 0:
        fire("url_redirect_param")

    if signals.get("has_html_form"):
        fire("has_html_form")

    if signals.get("display_name_has_email"):
        fire("display_name_has_email")

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
    # Return-Path and Reply-To mismatches used to disqualify this bonus. That
    # made the bonus almost unreachable for legitimate bulk mail: every
    # sender using an ESP (Mailchimp, SES, a retailer's own mail platform)
    # routes bounces through the provider's domain, so return_path_mismatch
    # is their normal state, not a symptom. Measured on 65 hand-labelled
    # legitimate emails: 16 of the 17 false positives had SPF, DKIM and DMARC
    # all passing with a matching DKIM domain, and were only flagged because
    # this bonus never fired.
    #
    # A verifying DKIM signature from the sender's own domain is the strong
    # claim here; the envelope return path is a routing detail. The mismatch
    # signals still score on their own (+2 each) — they are just no longer
    # allowed to cancel the evidence that the sender is who they say.
    consistent = signals.get("dkim_domain_matches_from") is True
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
