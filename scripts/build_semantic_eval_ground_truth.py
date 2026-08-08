"""
Converts data/semantic_eval/review.md's filled-in GROUND_TRUTH blocks
into data/semantic_eval/ground_truth.json (PHISHING_ROUTING_PLAN.md
step 8, the step after scripts/render_semantic_eval_review.py's
worksheet was hand-labeled — see PROGRESS.md).

WHY start/end ARE COMPUTED HERE, NOT READ FROM review.md
    review.md never asked the labeler for offsets (see its own
    instructions block) — only `type` + a verbatim `evidence` quote.
    This script computes start/end the exact same way
    src/semantic/validate.py does for real model output:
    SemanticFindingCandidate objects go through validate_findings(),
    which locates each evidence string via a real substring search and
    rejects anything not found or ambiguous (matches more than once).
    Ground truth is held to the SAME grounding standard the model's
    output is — a label a human wrote that doesn't actually appear
    verbatim in the body, or that matches more than one place, is a
    labeling bug, not usable ground truth, and this script fails loudly
    on it rather than silently accepting or guessing an offset.

CANONICAL BODY SOURCE — re-parsed from the .eml via
canonicalize_body(parse_eml(...).body_text), not read back out of
review.md's rendered BODY block. Re-parsing keeps ground_truth.json's
canonical text tied to the actual parser output a real evaluation run
will see, instead of a second, textually-extracted copy that could
silently diverge (e.g. if review.md's fenced code block ever got
hand-edited).

    canonicalize_body() (src/semantic/canonical.py) is NOT optional
    here — a real bug (PROGRESS.md, 2026-08) happened from assuming
    "re-parsing the same .eml gives the same string by construction"
    without it: facts.body_text can contain CRLF, but review.md got
    read back through a text-mode open() at some point in the labeling
    round-trip, which silently turns "\r\n" into "\n" — the labeler's
    quoted evidence matched the LF version they actually saw, not
    body_text's raw CRLF. Two otherwise-correct ground-truth findings
    failed validate_findings() against uncanonicalized body_text until
    this call was added. canonicalize_body() is the single normalization
    step that makes "the .eml and review.md describe the same text"
    actually true instead of assumed.

OUTPUT SHAPE (data/semantic_eval/ground_truth.json)
    A JSON array, one record per candidate:
        {
          "candidate": 1,
          "eml_path": "data/phishing_pot/email/sample-6426.eml",
          "status": "labeled" | "unclear",
          "findings": [
            {"type": "...", "evidence": "...", "start": 0, "end": 0,
             "reason": "..."}
          ]
        }
    No model_confidence field — that's a model-output-only concept
    (schemas/semantic.py's SemanticFindingCandidate), meaningless for a
    hand-written label. `reason` is kept (not part of
    ValidatedSemanticFinding) because it's the human justification for
    the label, useful for anyone auditing a disagreement later.

Usage:
    python3 scripts/build_semantic_eval_ground_truth.py
    python3 scripts/build_semantic_eval_ground_truth.py --force
"""
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml  # noqa: E402

from schemas.semantic import SemanticFindingCandidate  # noqa: E402
from src.parser.parse import parse_eml  # noqa: E402
from src.semantic.canonical import canonicalize_body  # noqa: E402
from src.semantic.validate import validate_findings  # noqa: E402

REVIEW_PATH = PROJECT_ROOT / "data" / "semantic_eval" / "review.md"
OUT_PATH = PROJECT_ROOT / "data" / "semantic_eval" / "ground_truth.json"

_CANDIDATE_SPLIT_RE = re.compile(r"\n## Candidate (\d+)\n")
_EML_PATH_RE = re.compile(r"^EML_PATH: `(.+)`$", re.MULTILINE)
_GROUND_TRUTH_RE = re.compile(r"(GROUND_TRUTH:\n.*?)(?=\n---|\Z)", re.DOTALL)


def _parse_candidate_blocks(review_text: str) -> dict[int, str]:
    parts = _CANDIDATE_SPLIT_RE.split(review_text)[1:]
    blocks = {}
    for i in range(0, len(parts), 2):
        blocks[int(parts[i])] = parts[i + 1]
    return blocks


def _extract_eml_path(block: str, candidate: int) -> str:
    m = _EML_PATH_RE.search(block)
    if not m:
        raise SystemExit(f"Candidate {candidate}: EML_PATH satırı bulunamadı.")
    return m.group(1)


def _extract_ground_truth_yaml(block: str, candidate: int) -> dict:
    m = _GROUND_TRUTH_RE.search(block)
    if not m:
        raise SystemExit(f"Candidate {candidate}: GROUND_TRUTH bloğu bulunamadı.")
    try:
        parsed = yaml.safe_load(m.group(1))
    except yaml.YAMLError as e:
        raise SystemExit(f"Candidate {candidate}: GROUND_TRUTH YAML parse hatası: {e}") from e
    # yaml.safe_load parses the whole "GROUND_TRUTH:\n  status: ...\n  ..."
    # block as {"GROUND_TRUTH": {"status": ..., ...}} — GROUND_TRUTH is the
    # outer key, its value is the actual record.
    if not isinstance(parsed, dict) or "GROUND_TRUTH" not in parsed:
        raise SystemExit(f"Candidate {candidate}: GROUND_TRUTH şekli beklenmedik: {parsed!r}")
    inner = parsed["GROUND_TRUTH"]
    if not isinstance(inner, dict) or "status" not in inner:
        raise SystemExit(f"Candidate {candidate}: GROUND_TRUTH içeriği beklenmedik: {inner!r}")
    return inner


def _build_record(candidate: int, eml_path: str, gt: dict) -> dict:
    status = gt.get("status")
    if status not in ("labeled", "unclear"):
        raise SystemExit(
            f"Candidate {candidate}: status {status!r} — henüz etiketlenmemiş "
            f"(placeholder '_[labeled|unclear]_' kaldı) ya da geçersiz. "
            f"Tüm candidate'ler etiketlenmeden bu script çalıştırılamaz."
        )

    raw_findings = gt.get("findings") or []
    facts = parse_eml(PROJECT_ROOT / eml_path)
    # canonicalize_body(), not raw facts.body_text — must be the exact
    # same normalized string scripts/render_semantic_eval_review.py
    # showed the labeler. See src/semantic/canonical.py.
    canonical_body = canonicalize_body(facts.body_text)

    candidates = []
    reasons_by_key: dict[tuple[str, str], str] = {}
    for f in raw_findings:
        if "type" not in f or "evidence" not in f:
            raise SystemExit(
                f"Candidate {candidate}: eksik alan içeren finding: {f!r}"
            )
        candidates.append(SemanticFindingCandidate(
            type=f["type"],
            evidence=f["evidence"],
            model_confidence=1.0,  # placeholder — not meaningful for a hand label
            explanation=f.get("reason", ""),
        ))
        reasons_by_key[(f["type"], f["evidence"])] = f.get("reason", "")

    result = validate_findings(candidates, canonical_body)
    if result.rejected:
        lines = [
            f"Candidate {candidate} ({eml_path}): {len(result.rejected)} ground-truth "
            f"finding(s) failed grounding — this is a LABELING BUG, not a model failure:"
        ]
        for vf in result.rejected:
            evidence = getattr(vf.finding, "evidence", repr(vf.finding))
            lines.append(f"  [{vf.rejection_reason.value}] {evidence!r}")
        raise SystemExit("\n".join(lines))

    findings_out = []
    for finding in result.accepted:
        reason = reasons_by_key.get((finding.type.value, finding.evidence), "")
        findings_out.append({
            "type": finding.type.value,
            "evidence": finding.evidence,
            "start": finding.start,
            "end": finding.end,
            "reason": reason,
        })

    return {
        "candidate": candidate,
        "eml_path": eml_path,
        "status": status,
        "findings": findings_out,
    }


def main() -> None:
    if OUT_PATH.is_file() and "--force" not in sys.argv:
        raise SystemExit(
            f"{OUT_PATH} zaten mevcut — üzerine yazmak için --force kullanın."
        )

    review_text = REVIEW_PATH.read_text(encoding="utf-8")
    blocks = _parse_candidate_blocks(review_text)

    records = []
    for candidate in sorted(blocks):
        block = blocks[candidate]
        eml_path = _extract_eml_path(block, candidate)
        gt = _extract_ground_truth_yaml(block, candidate)
        records.append(_build_record(candidate, eml_path, gt))

    OUT_PATH.write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    labeled = sum(1 for r in records if r["status"] == "labeled")
    unclear = sum(1 for r in records if r["status"] == "unclear")
    total_findings = sum(len(r["findings"]) for r in records)
    empty = sum(1 for r in records if r["status"] == "labeled" and not r["findings"])
    print(f"Wrote {OUT_PATH} ({len(records)} candidates)")
    print(f"  labeled={labeled} unclear={unclear} "
          f"total_findings={total_findings} labeled_with_zero_findings={empty}")


if __name__ == "__main__":
    main()
