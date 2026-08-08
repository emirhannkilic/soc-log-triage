"""
Renders an 18-email semantic evaluation labeling worksheet
(PHISHING_ROUTING_PLAN.md step 6/8) — same review.md design as
scripts/render_holdout_review.py and scripts/render_devset_review.py,
adapted for FINDING-LEVEL ground truth instead of a single email-level
verdict.

WHY A NEW, SEPARATE SET (not hold-out, not the rule-engine-v2 dev set)
    Hold-out and the dev set both carry phishing/legitimate (or
    phishing/spam/unclear) VERDICT labels, not finding-level labels —
    they can't answer "did the extractor find the RIGHT semantic
    findings," only "did the rule engine get the final verdict right."
    CLAUDE.md's "kalibrasyon hold-out'ta yapılmaz" rule also means
    hold-out can't be reused here even if it could answer the question.

SELECTION (fixed 18 emails, chosen by hand — see PROGRESS.md for the
full rationale)
    3 per axis x 6 axes: net phishing, legitimate marketing/urgency
    language, credential-free account notifications, 419/advance-fee +
    fake-reward scams, real authority/brand notifications, no-signal
    mail — except marketing (4) and no-signal (2), reflecting that
    marketing/urgency language false-positives are this extractor's
    biggest known risk. None overlap with data/holdout or
    data/rule_engine_v2_devset's already-used .eml paths. The list is
    hardcoded below, not re-derived by a script, because the selection
    criteria (no repeated campaign/sender, mechanism diversity, TR/EN
    mix) needed human judgment a script can't reproduce.

GROUND TRUTH FORMAT — why YAML-ish blocks, not one line per email
    Unlike hold-out/dev-set's single verdict per email, a semantic
    finding has a type AND a verbatim quote, and a quote can contain
    colons, quotes, and newlines — a single "GROUND TRUTH: X" line
    (hold-out/dev-set's format) can't safely hold that. Each email's
    ground truth is a YAML block instead:

        GROUND_TRUTH:
          status: labeled
          findings:
            - type: credential_request
              evidence: |-
                <exact quote from the email body>
              reason: <one-sentence justification>
          reason: <only used when findings is empty or status is unclear>

    status is "labeled" (normal case) or "unclear" (the reviewer can't
    decide — excluded from the main precision/recall count, tallied
    separately, per the plan). findings: [] is valid and expected for
    legitimate mail — reviewers must not force a finding onto a clean
    email. See data/semantic_eval/README.md (written alongside this
    script's output) for the full labeling instructions kept in one
    place rather than repeated in every record.

start/end ARE NEVER HAND-WRITTEN. scripts/build_semantic_eval_ground_truth.py
computes them from evidence via the same find()-based approach
src/semantic/validate.py uses, so hand-typed offsets can never drift
from the actual quote.

Usage:
    python3 scripts/render_semantic_eval_review.py
    python3 scripts/render_semantic_eval_review.py --force
"""
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.parser.parse import parse_eml  # noqa: E402
from src.semantic.canonical import canonicalize_body  # noqa: E402

OUT_DIR = PROJECT_ROOT / "data" / "semantic_eval"
REVIEW_PATH = OUT_DIR / "review.md"
README_PATH = OUT_DIR / "README.md"

# (axis, eml relative path) — 18 total, see module docstring.
SELECTION: list[tuple[str, str]] = [
    ("net_phishing", "data/phishing_pot/email/sample-6426.eml"),
    ("net_phishing", "data/phishing_pot/email/sample-333.eml"),
    ("net_phishing", "data/phishing_pot/email/sample-1170.eml"),
    ("fraud_or_reward", "data/phishing_pot/email/sample-1183.eml"),
    ("fraud_or_reward", "data/phishing_pot/email/sample-119.eml"),
    ("fraud_or_reward", "data/phishing_pot/email/sample-795.eml"),
    ("legit_marketing_urgency", "data/raw/gmail/eml/inbox-3286.eml"),
    ("legit_marketing_urgency", "data/raw/gmail/eml/inbox-4999.eml"),
    ("legit_marketing_urgency", "data/raw/gmail/eml/inbox-5167.eml"),
    ("legit_marketing_urgency", "data/raw/gmail/eml/inbox-8117.eml"),
    ("neutral_notification", "data/raw/gmail/eml/inbox-9705.eml"),
    ("neutral_notification", "data/raw/gmail/eml/inbox-3871.eml"),
    ("neutral_notification", "data/raw/gmail/eml/inbox-9419.eml"),
    ("authority_brand", "data/raw/gmail/eml/inbox-807.eml"),
    ("authority_brand", "data/raw/gmail/eml/inbox-10055.eml"),
    ("authority_brand", "data/raw/gmail/eml/inbox-9963.eml"),
    ("no_signal", "data/raw/gmail/eml/inbox-893.eml"),
    ("no_signal", "data/raw/gmail/eml/inbox-5754.eml"),
]

ALLOWED_TYPES = (
    "credential_request", "payment_request", "authority_impersonation",
    "brand_impersonation", "urgency_or_pressure", "threat_or_fear",
    "reward_or_prize_lure", "attachment_or_link_instruction",
    "reply_channel_manipulation",
)

INSTRUCTIONS = f"""# Semantic Evaluation Review

18 candidates, 6 axes (3 each, except legit_marketing_urgency=4 and
no_signal=2 — marketing/urgency language false-positives are this
extractor's biggest known risk, see PROGRESS.md). Selection is fixed in
scripts/render_semantic_eval_review.py's SELECTION list; none overlap
with data/holdout or data/rule_engine_v2_devset.

For each candidate: open the .eml path and read the body directly. Do
NOT look at any model output before finalizing GROUND_TRUTH — label
first, compare against Qwen's output only after every record below is
filled in.

Allowed finding types (exactly these nine, nothing else):
{chr(10).join(f"  - {t}" for t in ALLOWED_TYPES)}

GROUND_TRUTH format — fill in the block under each candidate:

    GROUND_TRUTH:
      status: labeled
      findings:
        - type: credential_request
          evidence: |-
            <paste the EXACT quote from the email body, word for word>
          reason: <one sentence: why this counts as this type>
        - type: urgency_or_pressure
          evidence: |-
            <another exact quote, if there's a second finding>
          reason: <one sentence>
      reason: <only needed if findings is empty, or status is unclear>

Rules:
  - type must be one of the nine allowed values above, nothing else.
  - evidence must be copied verbatim from the email body (the exact
    text a conversion script will later search for with a plain
    substring match) — do not paraphrase, translate, or summarize.
  - Do NOT write start/end yourself. A conversion script computes them
    deterministically from evidence.
  - A single email can have multiple findings, including more than one
    evidence quote for the SAME type if genuinely different mechanisms
    are both present (e.g. two separate urgency phrases).
  - A phishing-labeled email having `findings: []` is a VALID outcome —
    do not force a finding onto an email just because it's phishing.
    Some phishing (e.g. pure header/infrastructure spoofing with no
    manipulative body text) legitimately has zero semantic findings.
  - If you genuinely cannot decide, set `status: unclear` and leave
    `findings: []` — unclear candidates are excluded from the main
    precision/recall count and tallied separately, not treated as
    "no findings."
  - Every finding needs its own one-sentence `reason`.

No GROUND TRUTH clue, source label, or axis name is shown per-candidate
below — axis names only appear in this instructions block, not next to
any individual email, to avoid anchoring the label toward what the axis
"expects" to contain.

---
"""


def render_record(i: int, eml_path: str) -> str:
    facts = parse_eml(PROJECT_ROOT / eml_path)
    # canonicalize_body(), not raw facts.body_text — this is the exact
    # string a labeler's quoted evidence must be found in later by
    # scripts/build_semantic_eval_ground_truth.py, and the exact string
    # the model sees via src/semantic/analyze.py's build_user_prompt().
    # See src/semantic/canonical.py's module docstring for the CRLF/LF
    # bug this exists to prevent.
    body = canonicalize_body(facts.body_text)
    lines = [
        f"## Candidate {i:03d}",
        "",
        f"EML_PATH: `{eml_path}`",
        f"SUBJECT: {facts.subject!r}",
        f"FROM_DOMAIN: {facts.from_domain} | DISPLAY_NAME: {facts.display_name!r}",
        "",
        "BODY:",
        "```",
        body,
        "```",
        "",
        "GROUND_TRUTH:",
        "  status: _[labeled|unclear]_",
        "  findings: []",
        "  reason: _[fill in]_",
        "",
        "---",
        "",
    ]
    return "\n".join(lines)


_CANDIDATE_HEADER_RE = re.compile(r"^## Candidate \d+")
_PLACEHOLDER_STATUS_RE = re.compile(r"^\s*status:\s*_\[labeled\|unclear\]_\s*$")
_FILLED_STATUS_RE = re.compile(r"^\s*status:\s*(labeled|unclear)\s*$")


def _count_labeled(path: Path) -> int:
    """Counts only ACTUAL filled-in `status:` lines inside a candidate's
    own GROUND_TRUTH block (i.e. after a "## Candidate N" header) — not
    the unfilled placeholder (`status: _[labeled|unclear]_`)
    render_record() writes, and not the word "labeled"/"unclear"
    appearing anywhere in the instructions prose ABOVE the first
    candidate (e.g. the format example or the "set `status: unclear`"
    rule)."""
    if not path.is_file():
        return 0
    filled = 0
    past_instructions = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if _CANDIDATE_HEADER_RE.match(line):
            past_instructions = True
        if not past_instructions:
            continue
        if _PLACEHOLDER_STATUS_RE.match(line):
            continue
        if _FILLED_STATUS_RE.match(line):
            filled += 1
    return filled


def main() -> None:
    already = _count_labeled(REVIEW_PATH)
    if already and "--force" not in sys.argv:
        raise SystemExit(
            f"{REVIEW_PATH} already has {already} labeled/unclear records and is\n"
            f"hand-edited — overwriting would delete those labels.\n\n"
            f"Re-run with --force to regenerate from scratch (LABELS ARE LOST).\n"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    out = [INSTRUCTIONS]
    for i, (_axis, eml_path) in enumerate(SELECTION, start=1):
        out.append(render_record(i, eml_path))

    REVIEW_PATH.write_text("\n".join(out), encoding="utf-8")
    print(f"Wrote {REVIEW_PATH} ({len(SELECTION)} candidates)")


if __name__ == "__main__":
    main()
