"""
Renders data/holdout/candidates.jsonl into two human-readable markdown
files for manual review (v3 plan section 3.3, redesigned per
holdout-fix-tasks.md T8):

data/holdout/review.md — the labeling worksheet. GROUND TRUTH is left
blank for the reviewer to fill in; nothing here anchors the reviewer's
judgment toward a particular answer. The .eml path is the first line of
each block, ahead of any parsed fact, because T1-T6 repeatedly found the
parser's own facts wrong or misleading in ways a human labeling from facts
alone would have missed (mojibake, dropped attachments, mislabeled spam)
— labeling must be done by reading the raw email, with facts as a
secondary aid, not the other way around. Ground truth is binary
(phishing/legitimate), matching source_label, per T8 Sorun 2: a
"Muhtemel Phishing" option here would let the reviewer anchor to "the
rule engine probably can't decide this one" instead of independently
judging what the email actually is — that's circular (calibrating the
system's grey zone against a label chosen because it's grey).

data/holdout/review_suggestions.md — a separate file, deliberately NOT
consulted while filling in review.md. Lists the risk-relevant facts
already computed by the parser (credential_request, urgency_keywords,
url anomalies, etc.) without a verdict or recommendation attached, so it
can't anchor the labeling pass the way pre-filled verdict suggestions did
before this redesign. Read AFTER labeling, to see where the parser's raw
signals agreed or disagreed with the independent human judgment; that
disagreement rate is itself useful information for calibrating the rule
engine (Adim 4), not a defect to fix away.

Both are derived from data/holdout/candidates.jsonl.
"""
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CANDIDATES_PATH = PROJECT_ROOT / "data" / "holdout" / "candidates.jsonl"
REVIEW_PATH = PROJECT_ROOT / "data" / "holdout" / "review.md"
SUGGESTIONS_PATH = PROJECT_ROOT / "data" / "holdout" / "review_suggestions.md"


def render_review_record(i: int, r: dict) -> str:
    """Labeling worksheet block: .eml path first, ground truth last and
    blank. No verdict, no rule-engine-style reasoning — see module
    docstring for why those live in review_suggestions.md instead."""
    lines = [
        f"## Candidate {i}",
        "",
        f"- eml path: `{r['_eml_path']}`",
        f"- subject: {r.get('subject')!r}",
        f"- from_domain: {r.get('from_domain')} | display_name: {r.get('display_name')!r}",
    ]

    body_preview = (r.get("body_text") or "")[:300].replace("\n", " ")
    lines.append(f"- body preview: {body_preview!r}")
    lines.append("")
    lines.append("GROUND TRUTH (phishing / legitimate): _[fill in]_")
    lines.append("")
    lines.append("---")
    lines.append("")
    return "\n".join(lines)


def _signal_lines(r: dict) -> list[str]:
    """Raw fact digest, no verdict language — just which risk-relevant
    signals are present. Deliberately excludes source_label and
    is_spam_not_phishing (the very things review.md's blank ground-truth
    field exists to independently re-derive)."""
    lines = [
        f"- spf={r.get('spf_result')} dkim={r.get('dkim_result')} dmarc={r.get('dmarc_result')}",
        f"- dkim_domain_matches_from: {r.get('dkim_domain_matches_from')}",
        f"- return_path_mismatch: {r.get('return_path_mismatch')} | "
        f"reply_to_mismatch: {r.get('reply_to_mismatch')}",
        f"- display_name_brand_mismatch: {r.get('display_name_brand_mismatch')} | "
        f"display_name_has_email: {r.get('display_name_has_email')}",
        f"- message_id_domain_matches_from: {r.get('message_id_domain_matches_from')}",
        f"- received_hop_count: {r.get('received_hop_count')} | "
        f"first_received_ip: {r.get('first_received_ip')}",
        f"- urgency_keywords: {[m['keyword'] for m in r.get('urgency_keywords', [])]}",
        f"- credential_request: {r.get('credential_request')}",
        f"- claims_attachment: {r.get('claims_attachment')}",
        f"- has_html_form: {r.get('has_html_form')} | "
        f"has_hidden_text: {r.get('has_hidden_text')} | "
        f"image_only_body: {r.get('image_only_body')}",
    ]

    urls = r.get("urls", [])
    if urls:
        lines.append(f"- urls ({len(urls)}):")
        for u in urls[:5]:
            flags = []
            if u.get("text_href_mismatch"):
                flags.append("TEXT/HREF MISMATCH")
            if u.get("is_ip_based"):
                flags.append("IP-BASED")
            if u.get("is_shortener"):
                flags.append("SHORTENER")
            if u.get("has_punycode"):
                flags.append("PUNYCODE")
            if u.get("redirect_param"):
                flags.append("REDIRECT-PARAM")
            flag_str = f" [{', '.join(flags)}]" if flags else ""
            lines.append(f"  - {u['href_domain']}{flag_str}")
        if len(urls) > 5:
            lines.append(f"  - ... and {len(urls) - 5} more")
    else:
        lines.append("- urls: none")

    attachments = r.get("attachments", [])
    if attachments:
        lines.append(f"- attachments ({len(attachments)}):")
        for a in attachments:
            flags = []
            if a.get("double_extension"):
                flags.append("DOUBLE EXT")
            if a.get("risky_type"):
                flags.append("RISKY TYPE")
            if a.get("is_archive"):
                flags.append("ARCHIVE")
            flag_str = f" [{', '.join(flags)}]" if flags else ""
            lines.append(f"  - {a['filename']}{flag_str}")
    else:
        lines.append("- attachments: none")

    return lines


def render_suggestion_record(i: int, r: dict) -> str:
    lines = [f"## Candidate {i}", ""]
    lines.extend(_signal_lines(r))
    lines.append("")
    lines.append("---")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    records = [json.loads(line) for line in open(CANDIDATES_PATH) if line.strip()]

    review_out = [
        "# Hold-out Review — v3 plan section 3.3\n",
        f"{len(records)} candidates. For each: open the .eml path and read it "
        "directly — do not open review_suggestions.md until every GROUND TRUTH "
        "below is filled in. Ground truth is binary: phishing or legitimate.\n",
        "---\n",
    ]
    suggestions_out = [
        "# Hold-out Signal Summary — v3 plan section 3.3\n",
        f"{len(records)} candidates. Raw parser facts, no verdict attached. "
        "Read this only AFTER completing review.md, to compare the parser's "
        "signals against your independent judgment.\n",
        "---\n",
    ]
    for i, r in enumerate(records, start=1):
        review_out.append(render_review_record(i, r))
        suggestions_out.append(render_suggestion_record(i, r))

    REVIEW_PATH.write_text("\n".join(review_out), encoding="utf-8")
    SUGGESTIONS_PATH.write_text("\n".join(suggestions_out), encoding="utf-8")
    print(f"Wrote {REVIEW_PATH}")
    print(f"Wrote {SUGGESTIONS_PATH}")


if __name__ == "__main__":
    main()
