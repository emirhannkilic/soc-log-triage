"""
Ad-hoc single-email smoke test for src/workflows/phishing.py's hybrid
mode against the REAL Qwen3.5-9B model — no mocking anywhere in this
script. Mirrors src/semantic/smoke_test.py's role (real model output
before trusting the mocked unit tests), extended to the full "tek
model, iki çağrı" chain: semantic extraction -> decision policy ->
Qwen report generation, falling back to the mechanical report on
failure (src/workflows/phishing.py's own contract — unchanged here).

ONE EMAIL PER PROCESS, DELIBERATELY
    This script takes exactly one required positional .eml path and
    processes exactly one email per run — running several emails back
    to back in the same process is NOT supported here. CLAUDE.md's
    "Uzun süren GPU işleri" section records two real macOS kernel
    panics from MLX/Metal state during sustained GPU work on this
    machine; chaining multiple real model runs in one process is exactly
    the kind of sustained-state pattern that risk applies to. Comparing
    several emails means running this script several times, once per
    terminal invocation, each a fresh process.

NOT RUN BY THIS SESSION
    Per CLAUDE.md's "Ağır/Uzun Süren Script Çalıştırmaları" rule, a real
    Qwen3.5-9B load + two inference calls is the user's terminal to run,
    not something run in the background here. This script is written to
    be handed to the user, not executed as part of writing it.

WHAT THIS DOES NOT DO
    No retry, no output repair, no new model call beyond what
    analyze_phishing(mode="hybrid") already makes. This script only
    calls that one function and prints its result — CLAUDE.md's
    "Yapılmayacaklar" rule against patching model output applies here
    exactly as it does to the pipeline itself.

THREE-WAY EXIT CODE (revised after a real run conflated two different
questions: "did the deterministic decision/workflow behave correctly"
and "did the Qwen-authored narrative succeed"; updated again for
PROGRESS.md's "rapor mimarisi değişikliği" — report_source's possible
values changed from "mechanical"/"qwen" to "mechanical"/"mechanical_
with_qwen_narrative", and the model-side failure this catches is now
NarrativeGenerationError, not ReportGenerationError)
    A real run against tests/fixtures/hybrid_credential_upgrade.eml
    (2026-08-08, against the OLD full-Report-authoring path since
    removed) hit a category_violation guard — the model abandoned the
    fixed category vocabulary in its report text. The workflow's OWN
    fallback contract (src/workflows/phishing.py) worked exactly as
    designed: it caught the generation error, kept build_report()'s own
    mechanical genel_degerlendirme text, and the report.risk_seviyesi ==
    final_decision.final_verdict invariant held throughout. That is a
    successful, controlled degradation — not a smoke-test failure — but
    a two-way (0 or 1) exit code has no way to say so; it would report
    this run as a hard failure, indistinguishable from
    analyze_phishing() itself raising. "The upgrade path was verified"
    and "the Qwen narrative succeeded" are NOT the same claim, and
    collapsing them either hides real report defects (if fallback
    silently counted as pass) or mislabels a working policy/workflow as
    broken (if fallback silently counted as fail). Three exit codes keep
    them separate:

    0   FULL SUCCESS — analyze_phishing() completed, the verdict
        invariant held, EVERY --expect-* check that was passed matched
        (if none were passed, this just means the run didn't crash and
        the invariant held), AND either the Qwen narrative was accepted
        (report_source="mechanical_with_qwen_narrative",
        narrative_status="completed") OR final_verdict=="Güvenilir"
        (narrative_status="not_requested" by design — see
        src/workflows/phishing.py's own module docstring for why a
        Güvenilir verdict never calls the narrative generator at all;
        this is not a degraded outcome for this script's purposes).
    1   HARD FAILURE — analyze_phishing() raised, the verdict invariant
        did not hold (a bug this script's whole purpose is to catch,
        since build_report() should never allow this), hybrid mode
        produced no final_decision at all, or an --expect-* check that
        WAS passed did not match reality (e.g. --expect-final-verdict
        "Muhtemel Phishing" was given but the run produced
        "Güvenilir" — the deterministic decision path itself did not
        behave as expected, a real regression to investigate).
    2   PASS WITH FALLBACK — everything a hard failure would check
        passed (workflow ran clean, invariant held, all given
        --expect-* checks matched), final_verdict != "Güvenilir" (a
        narrative call WAS attempted), but narrative_status=
        "failed_fallback" — the Qwen-authored narrative was rejected by
        generate_narrative()'s own validation (any
        NarrativeGenerationError code) and build_report()'s own
        mechanical genel_degerlendirme text was kept, exactly per
        src/workflows/phishing.py's documented contract. This is a
        REAL, separate signal: the decision/workflow layer is healthy,
        but the narrative-writing layer needs attention. Never silently
        folded into 0 or 1.

    The mechanical fallback result is still printed in full regardless
    of which code is chosen — an exit code reports the outcome
    category, it never hides the run's actual output.

--expect-* FLAGS (all optional; omitting all of them makes this a pure
wiring/invariant smoke test with no claim about what the DECISION
should have been)
    --expect-rule-verdict TEXT      e.g. "Güvenilir"
    --expect-final-verdict TEXT     e.g. "Muhtemel Phishing"
    --expect-decision-path TEXT     e.g. "credential_request_plus_url_upgrade"
    Each given flag is checked against the corresponding FinalDecision
    field and printed as its own PASS/FAIL line; any mismatch forces
    exit code 1 regardless of report_source, because it means the
    deterministic policy did not do what this specific fixture was
    built to demonstrate — a real regression, not a report-quality
    question report_source alone could ever capture.

Usage:
    python3 scripts/smoke_test_hybrid.py path/to/mail.eml
    python3 scripts/smoke_test_hybrid.py tests/fixtures/hybrid_credential_upgrade.eml \\
        --expect-rule-verdict "Güvenilir" \\
        --expect-final-verdict "Muhtemel Phishing" \\
        --expect-decision-path "credential_request_plus_url_upgrade"
"""
import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.workflows.phishing import analyze_phishing  # noqa: E402


def _print_header(title: str) -> None:
    print(f"\n=== {title} ===")


def _print_expectation(label: str, expected, actual) -> bool:
    """Prints one PASS/FAIL line for a single --expect-* check and
    returns whether it matched. Only called for flags the user actually
    passed — an omitted expectation prints nothing and cannot fail."""
    ok = expected == actual
    status = "PASS" if ok else "FAIL"
    print(f"{label}: {status} (expected={expected!r}, actual={actual!r})")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("eml", type=Path, help="tek bir .eml dosyasının yolu")
    ap.add_argument("--expect-rule-verdict", default=None,
                     help="beklenen RuleAssessment.rule_verdict (ör. 'Güvenilir')")
    ap.add_argument("--expect-final-verdict", default=None,
                     help="beklenen FinalDecision.final_verdict (ör. 'Muhtemel Phishing')")
    ap.add_argument("--expect-decision-path", default=None,
                     help="beklenen FinalDecision.decision_path (ör. "
                          "'credential_request_plus_url_upgrade')")
    args = ap.parse_args()

    if not args.eml.is_file():
        raise SystemExit(f"dosya bulunamadı: {args.eml}")

    print(f"EML: {args.eml}", file=sys.stderr)
    print("analyze_phishing(mode='hybrid') çalıştırılıyor — gerçek Qwen3.5-9B, "
          "ilk yükleme birkaç dakika sürebilir ...", file=sys.stderr)
    t0 = time.time()
    try:
        result = analyze_phishing(args.eml, mode="hybrid")
    except Exception as exc:
        print(f"\nHATA: analyze_phishing() exception fırlattı: {exc!r}", file=sys.stderr)
        print("WORKFLOW_CHECK: FAIL")
        print("SMOKE_RESULT: FAIL")
        print("EXIT_CODE: 1")
        sys.exit(1)
    elapsed = time.time() - t0
    print(f"toplam süre: {elapsed:.1f} saniye", file=sys.stderr)

    _print_header("RULE ENGINE")
    print(f"engine_version: {result.rule_assessment.engine_version}")
    print(f"rule_verdict:   {result.rule_assessment.rule_verdict}")
    print(f"score:          {result.rule_assessment.score}")
    print(f"total:          {result.rule_assessment.total}")

    _print_header("SEMANTIC EXTRACTION")
    print(f"semantic_status:      {result.semantic_status}")
    print(f"semantic_skip_reason: {result.semantic_skip_reason}")

    print(f"\n--- kabul edilen bulgular ({len(result.accepted_findings)}) ---")
    for f in result.accepted_findings:
        print(f"  [{f.type.value}] {f.evidence!r} (conf={f.model_confidence:.2f})")
        print(f"    -> {f.explanation}")

    print(f"\n--- reddedilen bulgular ({len(result.rejected_findings)}) ---")
    for vf in result.rejected_findings:
        finding = vf.finding
        label = getattr(finding, "evidence", repr(finding))
        print(f"  [{vf.rejection_reason.value}] {label!r}")

    _print_header("DECISION POLICY")
    fd = result.final_decision
    if fd is None:
        print("final_decision: None (fast mode özelliği — hybrid modda beklenmez)")
    else:
        print(f"rule_verdict:              {fd.rule_verdict}")
        print(f"final_verdict:             {fd.final_verdict}")
        print(f"decision_path:             {fd.decision_path}")
        print(f"analyst_review_required:   {fd.analyst_review_required}")
        print(f"contributing_rule_ids:     {fd.contributing_rule_ids}")
        print(f"contributing_semantic_ids: {fd.contributing_semantic_ids}")

    _print_header("NARRATIVE GENERATION")
    print(f"report_source:        {result.report_source}")
    print(f"narrative_status:     {result.narrative_status}")
    print(f"narrative_error_code: {result.narrative_error_code}")

    _print_header("REPORT (tam JSON)")
    print(result.report.model_dump_json(indent=2, exclude_none=False))

    _print_header("SONUÇ DEĞERLENDİRMESİ")

    # WORKFLOW_CHECK — the invariant that must hold regardless of what
    # the model did: hybrid mode always produces a final_decision, and
    # the report's risk_seviyesi always echoes it, mechanical fallback
    # or not (src/report/mechanical.py's own contract).
    workflow_ok = fd is not None and result.report.risk_seviyesi == fd.final_verdict
    print(f"WORKFLOW_CHECK: {'PASS' if workflow_ok else 'FAIL'} "
          f"(final_decision present: {fd is not None}, "
          f"risk_seviyesi=={fd.final_verdict if fd else None!r}: "
          f"{result.report.risk_seviyesi!r})")

    # UPGRADE_CHECK — only meaningful if the caller passed at least one
    # --expect-* flag; an omitted flag can't fail, so a run with none
    # given always reports N/A here (this script becomes a pure
    # wiring/invariant test, making no claim about what the DECISION
    # should have been).
    expectations_given = any([
        args.expect_rule_verdict, args.expect_final_verdict, args.expect_decision_path,
    ])
    expectations_ok = True
    if not expectations_given:
        print("UPGRADE_CHECK: N/A (hiçbir --expect-* verilmedi)")
    elif fd is None:
        print("UPGRADE_CHECK: FAIL (final_decision yok, beklenti kontrol edilemedi)")
        expectations_ok = False
    else:
        if args.expect_rule_verdict is not None:
            expectations_ok &= _print_expectation(
                "  rule_verdict", args.expect_rule_verdict, fd.rule_verdict)
        if args.expect_final_verdict is not None:
            expectations_ok &= _print_expectation(
                "  final_verdict", args.expect_final_verdict, fd.final_verdict)
        if args.expect_decision_path is not None:
            expectations_ok &= _print_expectation(
                "  decision_path", args.expect_decision_path, fd.decision_path)
        print(f"UPGRADE_CHECK: {'PASS' if expectations_ok else 'FAIL'}")

    # REPORT_CHECK — independent of UPGRADE_CHECK. A mechanical
    # fallback is DEGRADED, not FAIL; final_verdict=="Güvenilir" (no
    # narrative ever requested — src/workflows/phishing.py's own
    # by-design skip) is NEITHER — see module docstring's exit-code
    # rationale for why these three outcomes must never collapse into a
    # pass/fail boolean.
    narrative_never_requested_by_design = (
        fd is not None and fd.final_verdict == "Güvenilir"
        and result.narrative_status == "not_requested"
    )
    qwen_succeeded = (
        result.report_source == "mechanical_with_qwen_narrative"
        and result.narrative_status == "completed"
    )
    if narrative_never_requested_by_design:
        print("REPORT_CHECK: PASS (final_verdict=Güvenilir — narrative never requested by design)")
    elif qwen_succeeded:
        print("REPORT_CHECK: PASS")
    else:
        print(f"REPORT_CHECK: DEGRADED — {result.narrative_error_code}")

    hard_failure = not workflow_ok or not expectations_ok
    if hard_failure:
        smoke_result, exit_code = "FAIL", 1
    elif not qwen_succeeded and not narrative_never_requested_by_design:
        smoke_result, exit_code = "PASS_WITH_FALLBACK", 2
    else:
        smoke_result, exit_code = "PASS", 0

    print(f"SMOKE_RESULT: {smoke_result}")
    print(f"EXIT_CODE: {exit_code}")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
