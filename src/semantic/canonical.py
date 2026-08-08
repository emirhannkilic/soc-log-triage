"""
Single source of truth for canonical_body normalization
(PHISHING_ROUTING_PLAN.md step 8).

WHY THIS EXISTS
    facts.body_text can contain CRLF ("\\r\\n") line endings — real
    .eml bodies do. A real bug traced to this: scripts/
    render_semantic_eval_review.py wrote review.md with body_text's
    original CRLF intact (Path.write_text does not touch embedded
    newlines), but the labeling round-trip later read that file back
    with a text-mode open()/read_text() call — Python's universal
    newlines translation silently turns "\\r\\n" into "\\n" on
    ANY text-mode read, with no way to opt out short of opening in
    binary mode. The human labeler (and the model reading the same
    rendered text) then saw and quoted the LF-only version, which no
    longer matched facts.body_text's real CRLF bytes character-for-
    character — src/semantic/validate.py's substring search rejected
    two otherwise-correct ground-truth findings as NOT_FOUND_IN_BODY.

    The fix is not patching those two findings — it's removing the
    distinction that broke them. CRLF vs LF is not a meaningful
    difference to a human reading an email or to the model reading the
    same prompt text; forcing every consumer to agree on ONE
    representation up front removes the possibility of this class of
    bug recurring anywhere else the body gets copied through a
    text-mode read (a markdown render, a copy-paste, a future export
    format).

WHAT canonicalize_body() DOES AND DOES NOT DO
    Does: normalizes line endings only — "\\r\\n" -> "\\n", and a lone
    "\\r" (old Mac-style, or a stray carriage return not part of a
    CRLF pair) -> "\\n".
    Does NOT: collapse whitespace, strip anything, change case, or
    apply any Unicode normalization (NFC/NFKC/etc). Adding either of
    those would risk shifting character offsets between what a human
    saw and what the parser produced, or masking a real signal (e.g.
    src/parser's own hidden-text/preheader detection relies on
    whitespace being unmodified). This function's ONLY job is making
    "\\r\\n" and "\\n" stop being two different things.

EVERY CONSUMER OF facts.body_text AS A SEMANTIC canonical_body MUST
CALL THIS — src/semantic/analyze.py (the prompt shown to the model),
src/semantic/validate.py (what evidence is searched against),
scripts/render_semantic_eval_review.py (what a human labeler reads),
and scripts/build_semantic_eval_ground_truth.py (what ground-truth
evidence is searched against) all call canonicalize_body() on
facts.body_text before doing anything else with it. Offsets
(ValidatedSemanticFinding.start/end) are always relative to the
NORMALIZED string, never the raw facts.body_text — this is worth
repeating because it is exactly the kind of assumption that silently
drifts if one call site is ever changed without the others.

IDEMPOTENCY
    canonicalize_body(canonicalize_body(x)) == canonicalize_body(x) —
    calling it twice (e.g. once when rendering a worksheet, again when
    later loading that same already-normalized text) must be safe and
    a no-op the second time. Enforced by test_idempotent_on_already_lf.
"""
import re

_CRLF_OR_CR_RE = re.compile(r"\r\n|\r")


def canonicalize_body(body_text: str) -> str:
    return _CRLF_OR_CR_RE.sub("\n", body_text)
