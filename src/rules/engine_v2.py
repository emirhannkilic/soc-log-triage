"""
Rule Engine v2 — family-based scoring (CLAUDE.md "Rule Engine v2 — Aile
Bazlı Skorlama", locked 2026-08-06 after Codex's multi-round review).

DOES NOT REPLACE src/rules/engine.py (v1). v1 stays the production
decision maker and the frozen baseline (scripts/freeze_rule_engine_v1_baseline.py)
until v2 is measured on data/rule_engine_v2_devset/candidates.jsonl and
found to actually improve on it — CLAUDE.md's locked "GEÇİŞ ATOMİK
OLMALI" rule: the family formula, the auth guard, and the 3 known
false-positive fixes below all had to land TOGETHER, because a read-only
simulation of the family formula ALONE (auth bonus removed, FP sources
not yet fixed) produced 23/65 false positives on legitimate mail.

WHY A SEPARATE FILE, NOT A FLAG IN engine.py
    v1 must remain byte-for-byte re-runnable as the fixed comparison
    point. Branching v1/v2 logic inside one function risked exactly the
    kind of entanglement CLAUDE.md's "ara aşamalar production kararına
    bağlanmayacak" rule exists to prevent.

FAMILIES AND SUBGROUPS

    identity
      infrastructure_alignment: spf_or_dmarc_fail, dkim_missing_or_fail_domain_mismatch,
                                 dkim_pass_but_domain_mismatch,
                                 spf_pass_but_mailfrom_misaligned (adım 6),
                                 return_path_mismatch, from_domain_no_tld
      claimed_identity:         display_name_brand_mismatch, display_name_has_email
      reply_routing:            reply_to_free_mail

    url
      visible_deception:        url_text_href_mismatch (root-domain compared, see below)
      destination_obfuscation:  url_ip_based, url_punycode
      indirection:               url_shortener, url_redirect_param

    content   (what the message tries to make the RECIPIENT do)
      visual_evasion:            has_large_hidden_text, image_only_body
      credential_collection:     credential_request_with_external_link, has_html_form,
                                  form_action_domain_mismatch (adım 6, root-domain compared)
      pressure:                  urgency_keywords
      attachment_lure:           claims_attachment_but_empty
      scam_narrative:            advance_fee_fraud_language, fake_reward_claim_language
                                  (adım 8 — the only families this class of
                                  fraud ever touches; no header/URL signal
                                  can catch it)

    payload   (ONLY downloaded/carried executable content)
      dangerous_type:            attachment_risky_type
      filename_disguise:         attachment_double_extension,
                                  attachment_extension_mismatch (adım 7)
      archive:                   attachment_is_archive

THREE KNOWN FALSE-POSITIVE FIXES (all locked in CLAUDE.md, all required
before v2 can be measured — this file implements all three):

1. hidden preheader vs real content-hiding: schemas/facts.py's
   has_large_hidden_text (>150 chars of display:none/font-size:0 text)
   is scored, NOT has_hidden_text — a short preheader is standard
   marketing, not a signal. image_only_body still scores on its own.

2. return_path_mismatch is now CONDITIONAL, not scored standalone: only
   counts when combined with infrastructure misalignment (auth not
   aligned), a free-mail Reply-To, or a brand-mismatched display name —
   an ESP-routed bounce address alone (Mailchimp, SES, a retailer's own
   platform) is normal and must not fire this on its own.

3. url_text_href_mismatch now compares REGISTRABLE domains (via
   src/parser/psl.py, backed by a real Public Suffix List through
   tldextract — added adım 6, 2026-08-07), not full netlocs:
   email.uber.com vs uber.com, or www.turkiye.gov.tr vs gov.tr, are the
   SAME organization's subdomain, not a mismatch. This fixed real false
   positives found on data/rule_engine_v2_devset (inbox-5278.eml:
   email.uber.com/uber.com; inbox-1913.eml: www.turkiye.gov.tr/gov.tr;
   inbox-7421.eml: www.iyzico.com/iyzico.com). The PSL also fixes
   multi-part TLDs (co.uk, com.tr) that a "last two labels" heuristic
   got wrong.

AUTH BONUS IS NOW A GUARD, NOT A SCORE — see _auth_aligned() and its
use in _score_identity(). All_auth_pass_and_consistent is GONE as a
signal; it only suppresses return_path_mismatch's contribution, nothing
else (not reply_to_free_mail, not display_name signals, not any
content/url/payload signal).

CRITICAL PREDICATES — explicit list, not a 5th family or "weight>=4":
see CRITICAL_PREDICATES below. Currently only
attachment_double_extension AND attachment_risky_type on the SAME
attachment (e.g. invoice.pdf.exe).
"""
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from schemas.facts import EmailFacts
from src.parser.psl import registrable_domain as _root_domain

DEFAULT_RULES_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "rules.yaml"

# Same list src/rules/engine.py uses for reply_to_free_mail — duplicated
# rather than imported so v2 has no import-time dependency on v1's
# internals (v1 could change its private helpers without silently
# breaking v2, and vice versa).
_FREE_MAIL_DOMAINS = frozenset({
    "gmail.com", "googlemail.com", "yahoo.com", "yahoo.co.uk", "ymail.com",
    "hotmail.com", "hotmail.co.uk", "outlook.com", "live.com", "msn.com",
    "aol.com", "icloud.com", "me.com", "mac.com",
    "proton.me", "protonmail.com", "pm.me",
    "gmx.com", "gmx.net", "mail.com", "zoho.com",
    "yandex.com", "yandex.ru", "yandex.com.tr", "mail.ru",
})

FAMILY_NAMES = ("identity", "url", "content", "payload")


@dataclass
class SubgroupHit:
    subgroup: str
    signal: str
    weight: int  # 1=weak, 2=moderate, 3=strong (mapped from config/rules.yaml)


@dataclass
class FamilyScore:
    family: str
    score: int  # 0-4, see module docstring
    hits: list[SubgroupHit] = field(default_factory=list)


@dataclass
class VerdictV2:
    verdict: str  # "Phishing" | "Muhtemel Phishing" | "Güvenilir"
    families: dict[str, FamilyScore]
    critical_matches: list[str]
    total: int
    active_family_count: int
    material_family_count: int


def load_rules(path: Path = DEFAULT_RULES_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _auth_aligned(facts: EmailFacts) -> bool:
    return (
        facts.spf_result == "pass"
        and facts.dkim_result == "pass"
        and facts.dmarc_result == "pass"
        and facts.dkim_domain_matches_from is True
    )


def _url_root_mismatch_count(facts: EmailFacts) -> int:
    """Root-domain version of text_href_mismatch — see fix #3 in the
    module docstring. Recomputed here rather than trusted from
    UrlFacts.text_href_mismatch, which compares full netlocs."""
    count = 0
    for u in facts.urls:
        anchor_root = _root_domain(u.anchor_text_domain)
        href_root = _root_domain(u.href_domain)
        # Either side resolving to None means it's not a real registrable
        # domain to compare (a bare public suffix like "gov.tr", or no
        # domain at all) — not a basis for a mismatch claim.
        if not anchor_root or not href_root:
            continue
        if anchor_root != href_root:
            count += 1
    return count


def _weight_to_strength(weight: int) -> int:
    """1->1 (weak), 2->2 (moderate), 3 or 4->3 (strong) — see module
    docstring's family score formula."""
    if weight <= 1:
        return 1
    if weight == 2:
        return 2
    return 3


def _score_family(subgroup_hits: list[SubgroupHit]) -> FamilyScore:
    if not subgroup_hits:
        return FamilyScore(family="", score=0, hits=[])
    base = max(h.weight for h in subgroup_hits)
    distinct_subgroups = {h.subgroup for h in subgroup_hits}
    corroboration_bonus = 1 if len(distinct_subgroups) >= 2 else 0
    score = min(4, base + corroboration_bonus)
    return FamilyScore(family="", score=score, hits=subgroup_hits)


# Critical predicates: explicit, auditable conditions, NOT a signal or
# a 5th family. Each predicate is a (name, check_fn) pair; check_fn
# takes EmailFacts and returns True/False.
def _double_extension_and_risky_type(facts: EmailFacts) -> bool:
    return any(a.double_extension and a.risky_type for a in facts.attachments)


CRITICAL_PREDICATES: list[tuple[str, "callable"]] = [
    ("double_extension_and_risky_type_same_attachment", _double_extension_and_risky_type),
]


def evaluate_v2(facts: EmailFacts, rules: dict | None = None) -> VerdictV2:
    if rules is None:
        rules = load_rules()
    weights = rules["signals"]

    def w(name: str) -> int:
        return int(weights[name]["weight"])

    auth_aligned = _auth_aligned(facts)

    # --- identity ---------------------------------------------------
    identity_hits: list[SubgroupHit] = []
    if facts.spf_result == "fail" or facts.dmarc_result == "fail":
        identity_hits.append(SubgroupHit("infrastructure_alignment",
                                         "spf_or_dmarc_fail", w("spf_or_dmarc_fail")))
    dkim_missing_or_fail = facts.dkim_result in (None, "none", "fail")
    if dkim_missing_or_fail and facts.dkim_domain_matches_from is False:
        identity_hits.append(SubgroupHit("infrastructure_alignment",
                                         "dkim_missing_or_fail_domain_mismatch",
                                         w("dkim_missing_or_fail_domain_mismatch")))
    if facts.dkim_result == "pass" and facts.dkim_domain_matches_from is False:
        identity_hits.append(SubgroupHit("infrastructure_alignment",
                                         "dkim_pass_but_domain_mismatch",
                                         w("dkim_pass_but_domain_mismatch")))
    # Adım 6 (2026-08-07): SPF alignment, not just spf_result pass/fail.
    # spf_result=pass only proves the ENVELOPE sender's own domain checked
    # out; a rented/attacker-controlled domain with a valid SPF record
    # passes this trivially while claiming an unrelated From. Same
    # subgroup as dkim_pass_but_domain_mismatch — both say "the passing
    # auth mechanism doesn't actually vouch for the claimed sender."
    #
    # Guarded by "DKIM is NOT already aligned": measured on
    # data/rule_engine_v2_devset (100 mail) that firing this unconditionally
    # produces 6/6 false positives, ALL of them legitimate bulk senders on
    # a shared ESP (Amazon SES, Persona Click) — SPF is IP-based and an ESP's
    # shared sending IPs routinely fail alignment for every tenant using
    # them, that's normal ESP routing, not spoofing. DKIM is a cryptographic
    # signature over the message itself, not the sending IP — when DKIM
    # already independently vouches for the claimed From (pass AND domain
    # matches), an SPF/envelope mismatch adds no evidence. Only fire when
    # DKIM does NOT already establish that alignment, so this catches what
    # dkim_pass_but_domain_mismatch and dkim_missing_or_fail_domain_mismatch
    # can miss (e.g. DKIM absent entirely) without re-punishing the exact
    # ESP pattern return_path_mismatch's fix #2 already had to guard against.
    dkim_already_aligned = facts.dkim_result == "pass" and facts.dkim_domain_matches_from is True
    if facts.spf_result == "pass" and facts.spf_aligned is False and not dkim_already_aligned:
        identity_hits.append(SubgroupHit("infrastructure_alignment",
                                         "spf_pass_but_mailfrom_misaligned",
                                         w("spf_pass_but_mailfrom_misaligned")))
    # Fix #2: return_path_mismatch only counts when corroborated —
    # infra misalignment, a free-mail Reply-To, or a brand-mismatched
    # display name. An ESP-routed bounce address alone is normal.
    reply_to = (facts.reply_to_domain or "").lower()
    from_domain = (facts.from_domain or "").lower()
    is_free_reply_to = bool(
        reply_to and from_domain and facts.reply_to_mismatch
        and reply_to in _FREE_MAIL_DOMAINS and from_domain not in _FREE_MAIL_DOMAINS
    )
    if facts.return_path_mismatch and (not auth_aligned or is_free_reply_to
                                       or facts.display_name_brand_mismatch):
        identity_hits.append(SubgroupHit("infrastructure_alignment",
                                         "return_path_mismatch", w("return_path_mismatch")))
    if facts.from_domain and "." not in facts.from_domain:
        identity_hits.append(SubgroupHit("infrastructure_alignment",
                                         "from_domain_no_tld", w("from_domain_no_tld")))
    if facts.display_name_brand_mismatch:
        identity_hits.append(SubgroupHit("claimed_identity",
                                         "display_name_brand_mismatch",
                                         w("display_name_brand_mismatch")))
    if facts.display_name_has_email:
        identity_hits.append(SubgroupHit("claimed_identity",
                                         "display_name_has_email", w("display_name_has_email")))
    if is_free_reply_to:
        identity_hits.append(SubgroupHit("reply_routing",
                                         "reply_to_free_mail", w("reply_to_free_mail")))
    identity_hits = [SubgroupHit(h.subgroup, h.signal, _weight_to_strength(h.weight))
                     for h in identity_hits]
    identity = _score_family(identity_hits)
    identity.family = "identity"

    # --- url ----------------------------------------------------------
    url_hits: list[SubgroupHit] = []
    if _url_root_mismatch_count(facts) > 0:
        url_hits.append(SubgroupHit("visible_deception",
                                    "url_text_href_mismatch", w("url_text_href_mismatch")))
    if any(u.is_ip_based for u in facts.urls):
        url_hits.append(SubgroupHit("destination_obfuscation",
                                    "url_ip_based", w("url_ip_based")))
    if any(u.has_punycode for u in facts.urls):
        url_hits.append(SubgroupHit("destination_obfuscation",
                                    "url_punycode", w("url_punycode")))
    if any(u.is_shortener for u in facts.urls):
        url_hits.append(SubgroupHit("indirection", "url_shortener", w("url_shortener")))
    if any(u.redirect_param for u in facts.urls):
        url_hits.append(SubgroupHit("indirection",
                                    "url_redirect_param", w("url_redirect_param")))
    url_hits = [SubgroupHit(h.subgroup, h.signal, _weight_to_strength(h.weight))
               for h in url_hits]
    url_family = _score_family(url_hits)
    url_family.family = "url"

    # --- content --------------------------------------------------------
    content_hits: list[SubgroupHit] = []
    # Fix #1: has_large_hidden_text, not has_hidden_text.
    if facts.has_large_hidden_text or facts.image_only_body:
        content_hits.append(SubgroupHit("visual_evasion",
                                        "hidden_text_or_image_only",
                                        w("hidden_text_or_image_only")))
    if facts.credential_request and len(facts.urls) > 0:
        content_hits.append(SubgroupHit("credential_collection",
                                        "credential_request_with_external_link",
                                        w("credential_request_with_external_link")))
    if facts.has_html_form:
        content_hits.append(SubgroupHit("credential_collection",
                                        "has_html_form", w("has_html_form")))
    # Adım 6: form action domain mismatch — the form posts data to a
    # domain that isn't the sender's own, AND the body asks for
    # credentials. Root-domain compared (same _root_domain() as URL
    # mismatch, fix #3) so a form posting to a legitimate subdomain
    # (login.example.com vs example.com) doesn't false-positive.
    if (facts.form_action_domain and facts.from_domain
            and _root_domain(facts.form_action_domain) != _root_domain(facts.from_domain)
            and facts.credential_request):
        content_hits.append(SubgroupHit("credential_collection",
                                        "form_action_domain_mismatch",
                                        w("form_action_domain_mismatch")))
    if len(facts.urgency_keywords) > 0:
        content_hits.append(SubgroupHit("pressure",
                                        "urgency_keywords", w("urgency_keywords")))
    if facts.claims_attachment and len(facts.attachments) == 0:
        content_hits.append(SubgroupHit("attachment_lure",
                                        "claims_attachment_but_empty",
                                        w("claims_attachment_but_empty")))
    # Adım 8 (2026-08-08): 419/advance-fee fraud and fake reward claims
    # routinely carry ZERO header misalignment and often zero URLs/
    # attachments — no other family can ever catch this class, only body
    # text can. New subgroup scam_narrative (distinct from pressure/
    # credential_collection/attachment_lure — this is neither urgency
    # language nor a credential ask, it's a specific fraud storyline).
    if facts.has_advance_fee_fraud_language:
        content_hits.append(SubgroupHit("scam_narrative",
                                        "advance_fee_fraud_language",
                                        w("advance_fee_fraud_language")))
    if facts.has_fake_reward_claim_language:
        content_hits.append(SubgroupHit("scam_narrative",
                                        "fake_reward_claim_language",
                                        w("fake_reward_claim_language")))
    content_hits = [SubgroupHit(h.subgroup, h.signal, _weight_to_strength(h.weight))
                    for h in content_hits]
    content = _score_family(content_hits)
    content.family = "content"

    # --- payload --------------------------------------------------------
    payload_hits: list[SubgroupHit] = []
    if any(a.risky_type for a in facts.attachments):
        payload_hits.append(SubgroupHit("dangerous_type",
                                        "attachment_risky_type", w("attachment_risky_type")))
    if any(a.double_extension for a in facts.attachments):
        payload_hits.append(SubgroupHit("filename_disguise",
                                        "attachment_double_extension",
                                        w("attachment_double_extension")))
    # Adım 7 (2026-08-08): the filename/MIME type are attacker-controlled
    # metadata; the leading bytes are not. Same subgroup as
    # attachment_double_extension — both say "this attachment's true
    # nature is hidden behind its name," just via different mechanisms
    # (a fake second extension vs. genuinely different file content).
    if any(a.extension_mismatch for a in facts.attachments):
        payload_hits.append(SubgroupHit("filename_disguise",
                                        "attachment_extension_mismatch",
                                        w("attachment_extension_mismatch")))
    if any(a.is_archive for a in facts.attachments):
        # is_archive alone has no config/rules.yaml weight (v1 only
        # scored it combined with credential_request via
        # is_archive_with_credential_request, which v2 drops — see
        # module docstring). Weight 1 here maps to strength 1 (weak)
        # via _weight_to_strength below, same as every other hit.
        payload_hits.append(SubgroupHit("archive", "attachment_is_archive", 1))
    payload_hits = [SubgroupHit(h.subgroup, h.signal, _weight_to_strength(h.weight))
                    for h in payload_hits]
    payload = _score_family(payload_hits)
    payload.family = "payload"

    families = {"identity": identity, "url": url_family, "content": content, "payload": payload}

    critical_matches = [name for name, check in CRITICAL_PREDICATES if check(facts)]

    scores = [families[f].score for f in FAMILY_NAMES]
    total = sum(scores)
    active_family_count = sum(1 for s in scores if s >= 1)
    material_family_count = sum(1 for s in scores if s >= 2)

    if critical_matches:
        verdict = "Phishing"
    elif any(s == 4 for s in scores):
        verdict = "Phishing"
    elif total >= 5 and material_family_count >= 2:
        verdict = "Phishing"
    elif any(s >= 3 for s in scores):
        verdict = "Muhtemel Phishing"
    elif total >= 3 and active_family_count >= 2:
        verdict = "Muhtemel Phishing"
    else:
        verdict = "Güvenilir"

    return VerdictV2(
        verdict=verdict,
        families=families,
        critical_matches=critical_matches,
        total=total,
        active_family_count=active_family_count,
        material_family_count=material_family_count,
    )
