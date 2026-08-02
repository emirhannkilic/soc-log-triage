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


class UrgencyMatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    keyword: str
    context: str
    # ^ the matched text plus surrounding characters, so a downstream
    # groundedness check (T7) can verify a generated report's claim ("the
    # email uses urgency language") against the actual quoted evidence
    # rather than trusting the boolean/keyword alone. See
    # holdout-fix-tasks.md T2.


class AttachmentFacts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    filename: str
    mime_type: str | None
    size: int | None
    double_extension: bool
    risky_type: bool
    is_archive: bool
    # ^ separate from risky_type: an archive isn't inherently malicious
    # (a legitimate order confirmation can attach a .zip), but it's a
    # low-weight signal on its own, and is_archive + credential_request
    # together is meaningful — see holdout-fix-tasks.md T5.


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
    from_source: str | None
    # ^ which header/method actually yielded from_domain: "From", "Sender",
    # "Return-Path", "Received-for", or None if none of them worked. Lets
    # downstream consumers (rule engine, groundedness check) distinguish
    # "no sender address exists" from "we had to fall back past From" —
    # see holdout-fix-tasks.md T1.
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
    urgency_keywords: list[UrgencyMatch]
    credential_request: bool
    claims_attachment: bool
    # ^ body text references an attachment ("attached", "ekte", "ek olarak")
    # while the attachments list is empty — promising a file that doesn't
    # exist is itself a signal (often the "attachment" is actually a link).
    # See holdout-fix-tasks.md T5.

    # 4.7 Meta
    subject: str | None
    date: str | None
    body_text: str
    language: str

    def flat_signals(self) -> dict:
        """Flattens this record into a single {signal_name: value} dict for
        src/eval/groundedness.py (holdout-fix-tasks.md T7): every scalar
        EmailFacts field by name, plus a handful of derived aggregate
        counts a generated report is likely to cite as a number
        ("28 URLs with a mismatch") rather than a per-URL flag. Nested list
        fields (urls, attachments, urgency_keywords) are summarized here,
        not exploded per-item — groundedness checking cares whether a
        claimed signal/count has ANY basis in the facts, not which specific
        URL triggered it.
        """
        signals: dict = {}
        for name, value in self.model_dump().items():
            if name in ("urls", "attachments", "urgency_keywords"):
                continue
            signals[name] = value

        signals["url_count"] = len(self.urls)
        signals["url_text_href_mismatch_count"] = sum(
            1 for u in self.urls if u.text_href_mismatch
        )
        signals["url_ip_based_count"] = sum(1 for u in self.urls if u.is_ip_based)
        signals["url_shortener_count"] = sum(1 for u in self.urls if u.is_shortener)
        signals["url_punycode_count"] = sum(1 for u in self.urls if u.has_punycode)
        signals["url_redirect_param_count"] = sum(
            1 for u in self.urls if u.redirect_param
        )

        signals["attachment_count"] = len(self.attachments)
        signals["attachment_risky_type_count"] = sum(
            1 for a in self.attachments if a.risky_type
        )
        signals["attachment_double_extension_count"] = sum(
            1 for a in self.attachments if a.double_extension
        )
        signals["attachment_is_archive_count"] = sum(
            1 for a in self.attachments if a.is_archive
        )

        signals["urgency_keyword_count"] = len(self.urgency_keywords)

        return signals
