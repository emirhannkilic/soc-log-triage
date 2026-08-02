"""
Groundedness check (holdout-fix-tasks.md T7, v3 plan section 7.3 metric 2):
verifies that every claim in a generated report is actually backed by the
deterministic facts the parser extracted, rather than invented.

Concrete motivating bug: a generated report for holdout candidate 7 claimed
"28 URLs with a text/href mismatch" while the facts for that email showed
zero URLs with the mismatch flag set. This is the same failure class v2's
LLM-classifies-and-writes architecture produced (a fabricated "unusual
X-Mailer header" cited on an Enron email that had no X-Mailer header at
all) — v3 is built specifically to make this class of error checkable, not
just less likely. A report that claims something facts don't support is a
groundedness failure regardless of whether the underlying verdict happens
to be correct.

Written BEFORE src/teacher/generate.py (v3 plan Adim 6-7) so every
generated training pair can be groundedness-checked as it's produced, per
CLAUDE.md's evaluation metrics: schema validity, groundedness, Turkish
quality, classification accuracy — reported separately, not blended into
one score.

Deliberately independent of schemas/report.py (v3 plan Adim 5, not yet
written): this module takes a list of claim strings, not a Report object,
so the claim-extraction step (pulling teknik_bulgular/phishing_gostergeleri
strings out of a parsed report JSON) can be added as a thin wrapper once
that schema exists, without this module's actual checking logic depending
on a schema that doesn't exist yet.
"""
import re

# Maps a claim-text pattern to the facts.flat_signals() key(s) it should be
# backed by. A claim is grounded if AT LEAST ONE mapped signal is truthy
# (True, or a nonzero count) — several source phrases can point at the same
# underlying signal (EN/TR wording, or a signal with multiple related
# flags), and only one of them needs to actually hold.
#
# Deliberately a plain keyword→signal(s) table, not NLP/embedding matching:
# holdout-fix-tasks.md T7 asks for "a mapping dict" as the practical
# starting point, and a report writer (the teacher LLM) is constrained by
# the JSON schema to a limited vocabulary of claim phrasings, not free
# prose — a keyword table covers that closed vocabulary without needing a
# more complex NLP layer. Extend this table as schemas/report.py (Adim 5)
# settles the actual field vocabulary the teacher's prompt asks it to use.
_CLAIM_SIGNAL_MAP: dict[str, tuple[str, ...]] = {
    # Authentication
    r"\bSPF\b": ("spf_result",),
    r"\bDKIM\b": ("dkim_result", "dkim_domain_matches_from"),
    r"\bDMARC\b": ("dmarc_result",),
    # Address consistency
    r"return[- ]path": ("return_path_mismatch",),
    r"reply[- ]to": ("reply_to_mismatch",),
    r"display[- ]?name.{0,20}(marka|brand|mismatch|uyum)": ("display_name_brand_mismatch",),
    r"message-?id": ("message_id_domain_matches_from",),
    # URLs
    r"text\s*/\s*href|metin.{0,10}href|anchor": ("url_text_href_mismatch_count",),
    r"\bIP\b.{0,15}(based|tabanl[ıi])": ("url_ip_based_count",),
    r"shorten|k[ıi]salt[ıi]c[ıi]": ("url_shortener_count",),
    r"punycode": ("url_punycode_count",),
    r"redirect|y[öo]nlendirme parametresi": ("url_redirect_param_count",),
    # Attachments
    r"riskli.{0,10}(dosya|uzant[ıi])|risky.{0,10}(file|extension|type)": (
        "attachment_risky_type_count",
    ),
    r"[çc]ift uzant[ıi]|double extension": ("attachment_double_extension_count",),
    r"ar[şs]iv|archive": ("attachment_is_archive_count",),
    r"eklenti.{0,15}(vaat|iddia)|claims?.{0,15}attachment": ("claims_attachment",),
    # Body signals
    r"gizli (metin|i[çc]erik)|hidden (text|content)": ("has_hidden_text",),
    r"(html )?form": ("has_html_form",),
    r"sadece g[öo]rsel|image[- ]only": ("image_only_body",),
    r"aciliyet|urgency|urgent": ("urgency_keyword_count",),
    r"kimlik bilgi(si|leri)|credential": ("credential_request",),
}

# Numeric claims ("28 URLs with a mismatch") are checked against these
# count signals specifically — a number in the claim text must match the
# actual count, not just be nonzero. See _check_numeric_claim.
_COUNT_SIGNALS = {
    "url": "url_count",
    "url_text_href_mismatch_count": "url_text_href_mismatch_count",
    "attachment": "attachment_count",
    "urgency_keyword_count": "urgency_keyword_count",
}

_NUMBER_RE = re.compile(r"\b(\d+)\b")


class ClaimCheck:
    def __init__(self, claim: str, grounded: bool, reason: str):
        self.claim = claim
        self.grounded = grounded
        self.reason = reason

    def __repr__(self) -> str:
        status = "GROUNDED" if self.grounded else "UNGROUNDED"
        return f"[{status}] {self.claim!r} — {self.reason}"


def _signal_is_true(signals: dict, key: str) -> bool:
    value = signals.get(key)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value > 0
    return bool(value)


def _matched_signal_keys(claim: str) -> list[str]:
    matched: list[str] = []
    for pattern, signal_keys in _CLAIM_SIGNAL_MAP.items():
        if re.search(pattern, claim, re.IGNORECASE):
            matched.extend(signal_keys)
    return matched


def _check_numeric_claim(claim: str, signals: dict, matched_keys: list[str]) -> str | None:
    """If the claim cites a specific number ('28 URLs'), verify it against
    the matching count signal. Returns a failure reason string if the
    number doesn't match, or None if there's nothing to check (no number
    in the claim, or no count signal among the matched keys)."""
    numbers = _NUMBER_RE.findall(claim)
    if not numbers:
        return None
    count_keys = [k for k in matched_keys if k.endswith("_count")]
    if not count_keys:
        return None
    claimed = int(numbers[0])
    for key in count_keys:
        actual = signals.get(key)
        if isinstance(actual, int) and actual == claimed:
            return None
    actuals = {k: signals.get(k) for k in count_keys}
    return f"claims {claimed}, but facts show {actuals}"


def check_claim(claim: str, signals: dict) -> ClaimCheck:
    matched_keys = _matched_signal_keys(claim)
    if not matched_keys:
        return ClaimCheck(
            claim, grounded=False,
            reason="claim text didn't match any known signal — either an "
                   "unrecognized phrasing (extend _CLAIM_SIGNAL_MAP) or a "
                   "genuinely unsupported claim",
        )

    numeric_failure = _check_numeric_claim(claim, signals, matched_keys)
    if numeric_failure:
        return ClaimCheck(claim, grounded=False, reason=numeric_failure)

    if any(_signal_is_true(signals, key) for key in matched_keys):
        return ClaimCheck(
            claim, grounded=True,
            reason=f"backed by {[k for k in matched_keys if _signal_is_true(signals, k)]}",
        )

    return ClaimCheck(
        claim, grounded=False,
        reason=f"matched signal(s) {matched_keys} but all are False/zero in facts",
    )


def check_claims(claims: list[str], signals: dict) -> dict:
    """Returns {"grounded_claims": int, "total_claims": int, "ratio": float,
    "ungrounded": list[ClaimCheck]} for a full report's worth of claims."""
    results = [check_claim(c, signals) for c in claims]
    grounded = [r for r in results if r.grounded]
    ungrounded = [r for r in results if not r.grounded]
    total = len(results)
    return {
        "grounded_claims": len(grounded),
        "total_claims": total,
        "ratio": (len(grounded) / total) if total else 1.0,
        "ungrounded": ungrounded,
    }
