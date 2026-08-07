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
    extension_mismatch: bool
    # ^ the payload's actual leading bytes (magic number) confidently match
    # a DIFFERENT known format than the filename's extension claims (e.g.
    # "invoice.pdf" whose bytes start with "MZ", a PE executable) — the
    # extension/MIME type are attacker-controlled metadata, this checks the
    # bytes themselves. src/parser/magic.py. False (not True) when the
    # extension isn't in that module's narrow table or the payload's
    # format can't be identified at all — an unrecognized payload isn't
    # evidence of a SPECIFIC disguise. Rule Engine v2 (CLAUDE.md, adım 7).


class EmailFacts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # 4.1 Authentication
    spf_result: str | None
    dkim_result: str | None
    dmarc_result: str | None
    dkim_domain: str | None
    dkim_domain_matches_from: bool | None
    spf_mailfrom_domain: str | None
    # ^ the envelope-sender domain SPF actually checked (Authentication-
    # Results' smtp.mailfrom=), as opposed to the visible From header a
    # recipient reads. None if Authentication-Results has no smtp.mailfrom.
    spf_aligned: bool | None
    # ^ whether spf_mailfrom_domain is the same organization (via PSL,
    # src/parser/psl.py) as from_domain. spf_result=pass only proves the
    # envelope sender's own domain checked out — it says nothing about
    # whether that domain has anything to do with the claimed From. Rule
    # Engine v2 (CLAUDE.md, adım 6) uses this for a real alignment check,
    # not just pass/fail. None if either domain is unknown.

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
    form_action_domain: str | None
    # ^ domain the HTML <form>'s action attribute posts to, if the form
    # has one and it's an absolute URL. None if there's no form, no
    # action attribute, or a relative action (posts back to the
    # sender's own domain — nothing to compare). Rule Engine v2
    # (CLAUDE.md, adım 6) compares this against from_domain: a form
    # collecting credentials but submitting them to a DIFFERENT domain
    # than the sender claims to be is the credential-harvesting shape,
    # not the form's mere presence.
    has_hidden_text: bool
    # ^ ANY display:none/font-size:0 styled text — includes short,
    # benign preheaders (the one-line summary email clients show next to
    # the subject). Not scored directly; see has_large_hidden_text.
    has_large_hidden_text: bool
    # ^ hidden text long enough (>150 chars) not to be a preheader — a
    # real content-hiding signal. Rule Engine v2 (CLAUDE.md) scores this,
    # not has_hidden_text, to stop legitimate marketing preheaders from
    # triggering the same rule as an actual hidden-content attack.
    image_only_body: bool
    urgency_keywords: list[UrgencyMatch]
    credential_request: bool
    claims_attachment: bool
    # ^ body text references an attachment ("attached", "ekte", "ek olarak")
    # while the attachments list is empty — promising a file that doesn't
    # exist is itself a signal (often the "attachment" is actually a link).
    # See holdout-fix-tasks.md T5.
    has_advance_fee_fraud_language: bool
    # ^ idiom-level 419/advance-fee fraud phrasing ("beneficiary",
    # "outstanding principal", "next of kin", "deposited fund") — this
    # scam class routinely carries ZERO header misalignment and often
    # zero URLs/attachments, so no other signal in this schema can catch
    # it. Rule Engine v2 (CLAUDE.md, adım 8).
    has_fake_reward_claim_language: bool
    # ^ idiom-level fake reward/lottery/crypto-airdrop phrasing ("claim
    # your tokens", "token redistribution", "lottery winner") — same
    # rationale as has_advance_fee_fraud_language: body-text-only signal
    # for a scam class headers/URLs don't reliably expose.

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
