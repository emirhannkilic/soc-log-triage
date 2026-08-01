"""
Pydantic schema for the deterministic parser's output (v3 plan section 4).

Every field is either a verified fact or None — the parser never guesses.
extra="forbid" enforces that no field outside this schema can ever be
produced, so the parser and the rule engine stay in lockstep with what's
documented here.
"""
from pydantic import BaseModel, ConfigDict


class UrlFacts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str
    href_domain: str | None
    anchor_text_domain: str | None
    text_href_mismatch: bool
    is_ip_based: bool
    is_shortener: bool
    has_punycode: bool
    redirect_param: bool


class AttachmentFacts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    filename: str
    mime_type: str | None
    size: int | None
    double_extension: bool
    risky_type: bool


class EmailFacts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # 4.1 Authentication
    spf_result: str | None
    dkim_result: str | None
    dmarc_result: str | None
    dkim_domain: str | None
    dkim_domain_matches_from: bool | None

    # 4.2 Address consistency
    from_domain: str | None
    return_path_domain: str | None
    reply_to_domain: str | None
    return_path_mismatch: bool
    reply_to_mismatch: bool
    display_name: str | None
    display_name_has_email: bool
    display_name_brand_mismatch: bool

    # 4.3 Message-ID and routing
    message_id_domain: str | None
    message_id_domain_matches_from: bool | None
    received_hop_count: int
    first_received_ip: str | None

    # 4.4 URLs
    urls: list[UrlFacts]

    # 4.5 Attachments
    attachments: list[AttachmentFacts]

    # 4.6 Body signals
    has_html_form: bool
    has_hidden_text: bool
    image_only_body: bool
    urgency_keywords: list[str]
    credential_request: bool

    # 4.7 Meta
    subject: str | None
    date: str | None
    body_text: str
    language: str
