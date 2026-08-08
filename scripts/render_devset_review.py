"""
Renders data/rule_engine_v2_devset/candidates.jsonl's PHISHING half into
two human-readable markdown files for manual review — same design as
scripts/render_holdout_review.py (v3 plan section 3.3 / holdout-fix-tasks.md
T8), applied to the Rule Engine v2 dev set.

WHY ONLY THE PHISHING HALF
    The dev set's legitimate 50 come from the user's own Gmail Takeout
    export — the same source and the same trust basis the hold-out's
    legitimate half already relies on without a separate hand-verification
    pass (CLAUDE.md never re-labels Gmail-sourced legitimate mail by hand).
    The phishing 50 come from phishing_pot, where CLAUDE.md's own audit
    (scripts/audit_spam_vs_phishing.py) estimates ~43% of the corpus is
    plain commercial spam, not phishing — source_label alone is not
    ground truth for that half, exactly the problem T6/T8 solved for the
    hold-out. This applies the same fix to the dev set before the v1→v2
    recall/false-positive numbers already measured on it (CLAUDE.md,
    "Rule Engine v2 — Aile Bazlı Skorlama") get treated as more than
    "agreement with source_label."

data/rule_engine_v2_devset/review.md — the labeling worksheet. GROUND
TRUTH is left blank; nothing here anchors the reviewer (or an external
model asked to draft labels) toward a particular answer. Ternary, not
binary, unlike the hold-out: "phishing", "spam" (commercial/junk, no
phishing mechanics), or "unclear" — matching audit_spam_vs_phishing.py's
own finding that a meaningful share of this corpus is neither. Collapsing
"spam" and "unclear" into "legitimate" would misrepresent what a spam
email actually is; keeping them distinct from "phishing" is what actually
matters for measuring rule-engine recall (both should be excluded from
"how many REAL phishing emails did we catch").

data/rule_engine_v2_devset/review_suggestions.md — the parser's raw
signals, no verdict, read only AFTER labeling — identical purpose to the
hold-out's file of the same role.

Usage:
    python3 scripts/render_devset_review.py
    python3 scripts/render_devset_review.py --force   # overwrite existing labels
"""
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CANDIDATES_PATH = PROJECT_ROOT / "data" / "rule_engine_v2_devset" / "candidates.jsonl"
REVIEW_PATH = PROJECT_ROOT / "data" / "rule_engine_v2_devset" / "review.md"
SUGGESTIONS_PATH = PROJECT_ROOT / "data" / "rule_engine_v2_devset" / "review_suggestions.md"


def render_review_record(i: int, r: dict) -> str:
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
    lines.append("GROUND TRUTH (phishing / spam / unclear): _[fill in]_")
    lines.append("")
    lines.append("---")
    lines.append("")
    return "\n".join(lines)


def _signal_lines(r: dict) -> list[str]:
    lines = [
        f"- spf={r.get('spf_result')} dkim={r.get('dkim_result')} dmarc={r.get('dmarc_result')}",
        f"- dkim_domain_matches_from: {r.get('dkim_domain_matches_from')}",
        f"- return_path_mismatch: {r.get('return_path_mismatch')} | "
        f"reply_to_mismatch: {r.get('reply_to_mismatch')}",
        f"- display_name_brand_mismatch: {r.get('display_name_brand_mismatch')} | "
        f"display_name_has_email: {r.get('display_name_has_email')}",
        f"- message_id_domain_matches_from: {r.get('message_id_domain_matches_from')}",
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

    # audit_spam_vs_phishing.py's own guess is shown here, not in
    # review.md — same anchoring concern review_suggestions.md already
    # exists to isolate, applied to this dev set's extra field.
    lines.append(f"- is_spam_not_phishing (audit heuristic guess): "
                 f"{r.get('is_spam_not_phishing')} | spam_reason: {r.get('spam_reason')!r}")

    return lines


def render_suggestion_record(i: int, r: dict) -> str:
    lines = [f"## Candidate {i}", ""]
    lines.extend(_signal_lines(r))
    lines.append("")
    lines.append("---")
    lines.append("")
    return "\n".join(lines)


def _count_filled_labels(path: Path) -> int:
    if not path.is_file():
        return 0
    filled = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if "GROUND TRUTH" not in line:
            continue
        _, _, after = line.partition("GROUND TRUTH")
        if re.search(r"\b(phishing|spam|unclear)\b", after, re.IGNORECASE):
            filled += 1
    return filled


def main() -> None:
    records = [json.loads(line) for line in open(CANDIDATES_PATH) if line.strip()]
    phishing_records = [r for r in records if r["source_label"] == "phishing"]

    already = _count_filled_labels(REVIEW_PATH)
    if already and "--force" not in sys.argv:
        raise SystemExit(
            f"{REVIEW_PATH} zaten {already} etiketlenmiş kayıt içeriyor ve bu\n"
            f"dosya elle düzenleniyor — üzerine yazmak o etiketleri siler.\n\n"
            f"--force ile sıfırdan üretilebilir (ETİKETLER KAYBOLUR).\n"
        )

    review_out = [
        "# Dev Set Review (phishing half only) — Rule Engine v2\n",
        f"{len(phishing_records)} candidates, all source_label=phishing. Legitimate half "
        "(50 mails, from the user's own Gmail Takeout) is NOT included here — same trust "
        "basis the hold-out already relies on for its legitimate half.\n",
        "For each: open the .eml path and read it directly — do not open "
        "review_suggestions.md until every GROUND TRUTH below is filled in. "
        "Ground truth is one of: phishing, spam (commercial/junk with no phishing "
        "mechanics — brand impersonation, credential request, or fake urgency+link), "
        "unclear.\n",
        "---\n",
    ]
    suggestions_out = [
        "# Dev Set Signal Summary (phishing half only) — Rule Engine v2\n",
        f"{len(phishing_records)} candidates. Raw parser facts, no verdict attached. "
        "Read this only AFTER completing review.md.\n",
        "---\n",
    ]
    for i, r in enumerate(phishing_records, start=1):
        review_out.append(render_review_record(i, r))
        suggestions_out.append(render_suggestion_record(i, r))

    REVIEW_PATH.write_text("\n".join(review_out), encoding="utf-8")
    SUGGESTIONS_PATH.write_text("\n".join(suggestions_out), encoding="utf-8")
    print(f"Wrote {REVIEW_PATH} ({len(phishing_records)} candidates)")
    print(f"Wrote {SUGGESTIONS_PATH}")


if __name__ == "__main__":
    main()
