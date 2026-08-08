"""
Fixed six-category attack vocabulary shared by src/report/mechanical.py
(deterministic sonuc_ve_gerekce construction) and
src/report/narrative_prompts.py (narrative-scope prompt context).

Moved out of the old src/report/prompts.py (now removed — see
PROGRESS.md's "rapor mimarisi değişikliği") unchanged: this is rule-
engine/finding-type domain knowledge, not prompt-construction logic, so
it outlives the LLM-authored-category-sentence design it originally
shipped with. Every mapped signal here is a deliberate editorial
judgment about which category it supports, not a mechanical derivation —
a newly added config/rules.yaml signal or SemanticFindingType with no
entry here simply supports no category (safe default).
"""
from schemas.semantic import SemanticFindingType

ALL_CATEGORIES = (
    "kimlik ve marka taklidi",
    "kimlik doğrulama uyumsuzluğu",
    "içerik gizleme",
    "kullanıcıyı işlem yapmaya yönlendirme",
    "aciliyet ve baskı",
    "zararlı ek veya içerik",
)

# Keys are v1 rule engine signal names (config/rules.yaml).
CATEGORY_SIGNAL_MAP: dict[str, str] = {
    "display_name_brand_mismatch": "kimlik ve marka taklidi",
    "display_name_has_email": "kimlik ve marka taklidi",
    "spf_or_dmarc_fail": "kimlik doğrulama uyumsuzluğu",
    "dkim_missing_or_fail_domain_mismatch": "kimlik doğrulama uyumsuzluğu",
    "dkim_pass_but_domain_mismatch": "kimlik doğrulama uyumsuzluğu",
    "spf_pass_but_mailfrom_misaligned": "kimlik doğrulama uyumsuzluğu",
    "return_path_mismatch": "kimlik doğrulama uyumsuzluğu",
    "reply_to_free_mail": "kimlik doğrulama uyumsuzluğu",
    "from_domain_no_tld": "kimlik doğrulama uyumsuzluğu",
    "hidden_text_or_image_only": "içerik gizleme",
    "url_text_href_mismatch": "kullanıcıyı işlem yapmaya yönlendirme",
    "url_ip_based": "kullanıcıyı işlem yapmaya yönlendirme",
    "url_punycode": "kullanıcıyı işlem yapmaya yönlendirme",
    "url_shortener": "kullanıcıyı işlem yapmaya yönlendirme",
    "url_redirect_param": "kullanıcıyı işlem yapmaya yönlendirme",
    "has_html_form": "kullanıcıyı işlem yapmaya yönlendirme",
    "form_action_domain_mismatch": "kullanıcıyı işlem yapmaya yönlendirme",
    "credential_request_with_external_link": "kullanıcıyı işlem yapmaya yönlendirme",
    "urgency_keywords": "aciliyet ve baskı",
    "advance_fee_fraud_language": "aciliyet ve baskı",
    "fake_reward_claim_language": "aciliyet ve baskı",
    "attachment_risky_type": "zararlı ek veya içerik",
    "attachment_double_extension": "zararlı ek veya içerik",
    "attachment_extension_mismatch": "zararlı ek veya içerik",
    "is_archive_with_credential_request": "zararlı ek veya içerik",
    "claims_attachment_but_empty": "zararlı ek veya içerik",
}

CATEGORY_FINDING_MAP: dict[SemanticFindingType, str] = {
    SemanticFindingType.BRAND_IMPERSONATION: "kimlik ve marka taklidi",
    SemanticFindingType.AUTHORITY_IMPERSONATION: "kimlik ve marka taklidi",
    SemanticFindingType.CREDENTIAL_REQUEST: "kullanıcıyı işlem yapmaya yönlendirme",
    SemanticFindingType.PAYMENT_REQUEST: "kullanıcıyı işlem yapmaya yönlendirme",
    SemanticFindingType.ATTACHMENT_OR_LINK_INSTRUCTION: "kullanıcıyı işlem yapmaya yönlendirme",
    SemanticFindingType.REPLY_CHANNEL_MANIPULATION: "kullanıcıyı işlem yapmaya yönlendirme",
    SemanticFindingType.URGENCY_OR_PRESSURE: "aciliyet ve baskı",
    SemanticFindingType.THREAT_OR_FEAR: "aciliyet ve baskı",
    SemanticFindingType.REWARD_OR_PRIZE_LURE: "aciliyet ve baskı",
}


def categories_from_evidence(
    evidence_signals: list[str],
    finding_types: list[SemanticFindingType],
) -> list[str]:
    """The subset of ALL_CATEGORIES supported by the given fired rule
    signals and semantic finding types, in ALL_CATEGORIES order (not
    insertion order) so callers get a stable, deterministic ordering
    regardless of evidence order."""
    present: set[str] = set()
    for signal in evidence_signals:
        if signal in CATEGORY_SIGNAL_MAP:
            present.add(CATEGORY_SIGNAL_MAP[signal])
    for finding_type in finding_types:
        if finding_type in CATEGORY_FINDING_MAP:
            present.add(CATEGORY_FINDING_MAP[finding_type])
    return [c for c in ALL_CATEGORIES if c in present]
