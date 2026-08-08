"""
Development-set hybrid pipeline RELIABILITY measurement — NOT a
finding-quality evaluation. Runs src/workflows/phishing.py's
analyze_phishing(mode="hybrid") end-to-end (real Qwen3.5-9B, no
mocking) on the 18 emails in data/semantic_eval/ground_truth.json and
counts how often each stage of the "tek model, iki çağrı" pipeline
succeeds, degrades, or hard-fails — NOT whether the semantic findings
or report content were CORRECT (that is scripts/evaluate_semantic_
extractor.py's job, a separate, already-existing script measuring
precision/recall against this same ground truth's finding labels).

WHY "DEVELOPMENT FALLBACK RATE," NOT "FINAL RELIABILITY RATE"
    data/semantic_eval's 18 candidates were the SAME set src/semantic/
    analyze.py's SYSTEM_PROMPT was iterated against across three
    prompt-tuning rounds (see that module's TİP TANIMLARI section and
    PROGRESS.md's semantic evaluation history) — this set has already
    shaped the prompt it's now being used to measure. A number produced
    here describes reliability ON THE SET THE PROMPT WAS DEVELOPED
    AGAINST, not an independent held-out measurement. This is the same
    distinction CLAUDE.md's rule-engine calibration history draws
    between a dev set (repeatedly touched, safe to iterate against) and
    a hold-out (touched once, for measurement only) — every number this
    script produces must be reported as "development fallback rate,"
    never "final" or "test" reliability, until a SEPARATE, untouched
    sample is measured after prompts/config are frozen.

WHY smoke_test_hybrid.py's THREE PAST RUNS ARE NOT THIS MEASUREMENT
    scripts/smoke_test_hybrid.py exists to prove specific code PATHS
    work (the skip-when-Phishing path, the credential_request upgrade
    path) on hand-built or hand-picked single fixtures — 3/3 passing
    there is evidence those specific branches are wired correctly, not
    a reliability RATE. A rate needs a fixed, independent-of-this-code
    sample size; three fixtures chosen specifically because they
    exercise particular branches would silently overrepresent whichever
    paths were easiest to construct a fixture for.

TWO SEPARATE model_call_failed COUNTERS — NEVER MERGED
    "the underlying QwenService itself failed" can happen at TWO
    different points in one candidate's run, and collapsing them into
    one counter would hide which stage actually broke:
        upstream_model_failure    analyze_semantic() (the FIRST Qwen
                                    call) raised SemanticExtractionError
                                    (code="model_call_failed"). Per
                                    src/workflows/phishing.py's own
                                    contract, generate_narrative() (the
                                    SECOND call) is then never even
                                    attempted — see that module's
                                    "REPORT GENERATION" docstring
                                    section. narrative_error_code is
                                    recorded as None here, not
                                    "model_call_failed" a second time,
                                    because no narrative call was made
                                    to fail.
        narrative_generation_failure generate_narrative() (the SECOND
                                    call) itself raised
                                    NarrativeGenerationError(code="model_
                                    call_failed") — the first call
                                    succeeded, the infrastructure was
                                    healthy enough to try again, and
                                    THIS specific call is what broke.
    Both are real "the GPU/Metal call failed" events, but at different
    points with different consequences (one skips the second call
    entirely, the other means the second call was attempted and lost) —
    a single merged counter would make it impossible to tell whether
    the semantic layer or the narrative layer needs attention.

PROCESS-PER-CANDIDATE ISOLATION (same rationale as scripts/
evaluate_semantic_extractor.py's own docstring, which found and fixed
a real GPU/Metal state accumulation bug this way)
    The PARENT process (this script's default `analyze` mode) never
    imports src/workflows/phishing.py, never loads a model, and never
    calls analyze_phishing() itself. For each of the 18 candidates it
    spawns a fresh `python3 evaluate_hybrid_reliability.py
    --worker-candidate N --worker-eml-path ... --worker-out <tmp>`
    subprocess. That subprocess loads Qwen3.5-9B via src/llm/service.py's
    get_service() ONCE and calls analyze_phishing(mode="hybrid") on
    exactly one email — the semantic call and the report call within
    that ONE candidate's run correctly SHARE the same QwenService
    instance (this is the real "tek model, iki çağrı" contract being
    measured), but no state or GPU/Metal context survives across
    candidates, and one candidate's process crashing can never take
    down the rest of the batch.

NO RETRY, ANYWHERE
    A worker subprocess calls analyze_phishing() exactly once. If it
    raises, that is recorded as a workflow_hard_failure for this
    candidate and the batch continues to the next one — CLAUDE.md's
    "Yapılmayacaklar" rule against patching/retrying model output
    applies to a whole pipeline run just as much as to a single
    malformed JSON response.

CACHE INVALIDATION ON PROMPT/MODEL/CONFIG CHANGE
    A cached result is only reused if compute_cache_key() (hashing
    src/semantic/analyze.py's SYSTEM_PROMPT/MAX_TOKENS/TEMPERATURE,
    src/report/prompts.py's SYSTEM_PROMPT_TEMPLATE, src/report/
    generate.py's MAX_TOKENS/TEMPERATURE, config/rules.yaml's raw
    bytes, and src/llm/service.py's MODEL_PATH) matches the key stored
    alongside that candidate's cached entry. Changing ANY of these
    (e.g. this session's credential_request taxonomy fix) silently
    invalidates every existing cache entry rather than mixing results
    measured under different prompts into one report — a stale cache
    hit here would misrepresent which prompt version a given fallback
    rate actually describes.

WHAT IS NEVER WRITTEN TO DISK
    Per explicit instruction: no raw email body, no subject, no
    constructed model prompt text, and no full report JSON is ever
    written to the cache or the output file. Each candidate's stored
    record is limited to: candidate id, eml_path (already a path
    reference in the checked-in ground_truth.json, not new PII
    exposure), engine_version/rule_verdict/final_verdict/decision_path
    (fixed enum-shaped values, not free text), semantic_status,
    semantic error code, report_source, narrative_status,
    narrative_error_code, and elapsed_seconds per stage. This is
    sufficient to compute every metric this module reports without
    ever persisting the content that produced them.

Usage (per this project's own rule for long-running GPU work — the
    user runs this, not this session):
    caffeinate -dims .venv/bin/python scripts/evaluate_hybrid_reliability.py \\
        --dataset data/semantic_eval/ground_truth.json
    caffeinate -dims .venv/bin/python scripts/evaluate_hybrid_reliability.py \\
        --dataset data/semantic_eval/ground_truth.json --no-resume
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "semantic_eval" / "ground_truth.json"
CACHE_PATH = PROJECT_ROOT / "data" / "semantic_eval" / "_hybrid_reliability_cache.json"


def compute_cache_key() -> str:
    """Hashes every prompt/model/config surface that affects the
    pipeline's behavior — see module docstring's CACHE INVALIDATION
    section. Deliberately imports the actual constants (not re-reading
    source files as text) so this can never drift from what the worker
    subprocess will really execute."""
    from src.llm.service import MODEL_PATH
    from src.report.narrative import MAX_TOKENS as NARRATIVE_MAX_TOKENS
    from src.report.narrative import TEMPERATURE as NARRATIVE_TEMPERATURE
    from src.report.narrative_prompts import SYSTEM_PROMPT_TEMPLATE
    from src.semantic.analyze import MAX_TOKENS as SEMANTIC_MAX_TOKENS
    from src.semantic.analyze import SYSTEM_PROMPT as SEMANTIC_SYSTEM_PROMPT
    from src.semantic.analyze import TEMPERATURE as SEMANTIC_TEMPERATURE

    rules_yaml_path = PROJECT_ROOT / "config" / "rules.yaml"
    rules_yaml_bytes = rules_yaml_path.read_bytes()

    hasher = hashlib.sha256()
    for piece in (
        SEMANTIC_SYSTEM_PROMPT.encode("utf-8"),
        str(SEMANTIC_MAX_TOKENS).encode("utf-8"),
        str(SEMANTIC_TEMPERATURE).encode("utf-8"),
        SYSTEM_PROMPT_TEMPLATE.encode("utf-8"),
        str(NARRATIVE_MAX_TOKENS).encode("utf-8"),
        str(NARRATIVE_TEMPERATURE).encode("utf-8"),
        rules_yaml_bytes,
        str(MODEL_PATH).encode("utf-8"),
    ):
        hasher.update(piece)
        hasher.update(b"\x00")
    return hasher.hexdigest()[:16]


def _load_cache() -> dict:
    if not CACHE_PATH.is_file():
        return {"cache_key": None, "results": {}}
    raw = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    raw["results"] = {int(k): v for k, v in raw.get("results", {}).items()}
    return raw


def _save_cache(cache_key: str, results: dict[int, dict]) -> None:
    """Atomic write — same rationale as scripts/evaluate_semantic_
    extractor.py's _save_cache: a crash mid-write must never corrupt or
    truncate previously-saved candidates. Written after EVERY
    candidate, not just at the end."""
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = CACHE_PATH.with_suffix(".json.tmp")
    payload = {"cache_key": cache_key, "results": {str(k): v for k, v in results.items()}}
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp_path, CACHE_PATH)


def _run_single_candidate_worker(candidate: int, eml_path_str: str, out_path: Path) -> None:
    """Runs in the WORKER subprocess only. Loads the model (via
    analyze_phishing's own lazy get_service() singleton) and calls
    analyze_phishing(mode="hybrid") on exactly one email, exactly once
    — no retry. Writes a metadata-only record (see module docstring's
    "WHAT IS NEVER WRITTEN TO DISK") as JSON to out_path."""
    from src.workflows.phishing import analyze_phishing

    eml_path = PROJECT_ROOT / eml_path_str
    t0 = time.time()
    try:
        result = analyze_phishing(eml_path, mode="hybrid")
    except Exception as exc:
        elapsed = time.time() - t0
        output = {
            "workflow_hard_failure": True,
            "error_type": type(exc).__name__,
            "error_message": str(exc)[:500],
            "elapsed_seconds": elapsed,
        }
    else:
        elapsed = time.time() - t0
        fd = result.final_decision
        output = {
            "workflow_hard_failure": False,
            "engine_version": result.rule_assessment.engine_version,
            "rule_verdict": result.rule_assessment.rule_verdict,
            "final_verdict": fd.final_verdict if fd else None,
            "decision_path": fd.decision_path if fd else None,
            "semantic_status": result.semantic_status,
            "semantic_skip_reason": result.semantic_skip_reason,
            # accepted/rejected finding COUNTS only — never the
            # evidence text itself (that would be quoted email body
            # content, exactly what this script must never persist).
            "accepted_finding_count": len(result.accepted_findings),
            "rejected_finding_count": len(result.rejected_findings),
            "report_source": result.report_source,
            "narrative_status": result.narrative_status,
            "narrative_error_code": result.narrative_error_code,
            "elapsed_seconds": elapsed,
        }
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")


def _run_all(dataset: list[dict], *, resume: bool, cache_key: str) -> dict[int, dict]:
    """Orchestrator — runs in the PARENT process, never loads the model
    or imports src/workflows/phishing.py itself. See module docstring's
    PROCESS-PER-CANDIDATE ISOLATION section."""
    cache = _load_cache() if resume else {"cache_key": None, "results": {}}
    if cache["cache_key"] != cache_key:
        if cache["results"]:
            print(
                f"Cache key mismatch (prompt/model/config changed since last run) — "
                f"ignoring {len(cache['results'])} cached result(s), running all "
                f"candidates fresh.",
                file=sys.stderr,
            )
        results: dict[int, dict] = {}
    else:
        results = cache["results"]
        if results:
            print(f"Resuming from cache: {len(results)} candidate(s) already have results.",
                  file=sys.stderr)

    import tempfile

    for record in dataset:
        candidate = record["candidate"]
        if record.get("status") == "unclear":
            print(f"[{candidate}/{len(dataset)}] SKIP (ground truth status=unclear)",
                  file=sys.stderr)
            continue
        if candidate in results:
            print(f"[{candidate}/{len(dataset)}] SKIP (cached result present)", file=sys.stderr)
            continue

        eml_path = record["eml_path"]
        print(f"[{candidate}/{len(dataset)}] {eml_path} (isolated subprocess) ...",
              file=sys.stderr)

        with tempfile.TemporaryDirectory() as tmpdir:
            worker_out = Path(tmpdir) / "result.json"
            t0 = time.time()
            proc = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()),
                 "--worker-candidate", str(candidate),
                 "--worker-eml-path", eml_path,
                 "--worker-out", str(worker_out)],
                capture_output=True,
                text=True,
            )
            elapsed = time.time() - t0

            if proc.returncode == 0 and worker_out.is_file():
                output = json.loads(worker_out.read_text(encoding="utf-8"))
                if output.get("workflow_hard_failure"):
                    print(f"        HARD FAILURE after {elapsed:.0f}s (caught in worker): "
                          f"{output['error_type']}: {output['error_message']}", file=sys.stderr)
                else:
                    print(
                        f"        {elapsed:.0f}s — rule_verdict={output['rule_verdict']} "
                        f"final_verdict={output['final_verdict']} "
                        f"report_source={output['report_source']}",
                        file=sys.stderr,
                    )
            else:
                stderr_tail = "\n".join(proc.stderr.strip().splitlines()[-20:])
                print(f"        SUBPROCESS CRASHED after {elapsed:.0f}s "
                      f"(exit code {proc.returncode}): {stderr_tail[-300:]}", file=sys.stderr)
                output = {
                    "workflow_hard_failure": True,
                    "error_type": "SubprocessCrash",
                    "error_message": f"exit code {proc.returncode}: {stderr_tail[-300:]}",
                    "elapsed_seconds": elapsed,
                }

        results[candidate] = output
        _save_cache(cache_key, results)

    return results


def _compute_metrics(dataset: list[dict], results: dict[int, dict]) -> dict:
    total = len(results)
    hard_failures = {c: r for c, r in results.items() if r["workflow_hard_failure"]}
    ok = {c: r for c, r in results.items() if not r["workflow_hard_failure"]}

    semantic_status_counts = {"completed": 0, "skipped": 0, "failed": 0, "none": 0}
    # upstream_model_failure vs narrative_generation_failure — see module
    # docstring's "TWO SEPARATE model_call_failed COUNTERS" section.
    upstream_model_failure_count = 0
    narrative_error_code_counts = {
        "model_call_failed": 0, "invalid_json": 0, "schema_invalid": 0,
    }
    narrative_completed_count = 0
    # "Güvenilir" verdicts never even attempt a narrative call (src/
    # workflows/phishing.py's by-design skip) — this is NOT a failure and
    # must not be counted in mechanical_fallback_count, which is reserved
    # for cases where a narrative call was attempted and failed.
    narrative_not_requested_guvenilir_count = 0
    mechanical_fallback_count = 0
    rule_verdict_breakdown: dict[str, dict] = {}

    for r in ok.values():
        status = r["semantic_status"] or "none"
        semantic_status_counts[status] = semantic_status_counts.get(status, 0) + 1

        rv = r["rule_verdict"]
        bucket = rule_verdict_breakdown.setdefault(rv, {
            "total": 0, "narrative_completed": 0, "mechanical_fallback": 0,
            "narrative_not_requested_guvenilir": 0,
        })
        bucket["total"] += 1

        if r["report_source"] == "mechanical_with_qwen_narrative" and r["narrative_status"] == "completed":
            narrative_completed_count += 1
            bucket["narrative_completed"] += 1
            continue

        if r["final_verdict"] == "Güvenilir" and r["narrative_status"] == "not_requested":
            narrative_not_requested_guvenilir_count += 1
            bucket["narrative_not_requested_guvenilir"] += 1
            continue

        mechanical_fallback_count += 1
        bucket["mechanical_fallback"] += 1

        # upstream_model_failure vs narrative_generation_failure — see
        # module docstring's "TWO SEPARATE model_call_failed COUNTERS"
        # section. src/workflows/phishing.py's analyze_phishing() sets
        # narrative_error_code = semantic_error_code (verbatim, code
        # "model_call_failed") ONLY on the short-circuit path where the
        # FIRST call (semantic extraction) already failed and the
        # SECOND call (generate_narrative()) was never attempted at all —
        # this is the ONE situation where a code="model_call_failed"
        # on this field does NOT mean the narrative call itself broke.
        # Any OTHER narrative_error_code (including a genuine
        # narrative-stage "model_call_failed", which only happens when
        # semantic_status is "completed" or "skipped" — i.e. the first
        # call succeeded or was never needed) means generate_narrative()
        # was actually invoked and its own failure is what's being
        # counted here.
        code = r["narrative_error_code"]
        if r["semantic_status"] == "failed" and code == "model_call_failed":
            upstream_model_failure_count += 1
        elif code in narrative_error_code_counts:
            narrative_error_code_counts[code] += 1

    narrative_acceptance_rate = narrative_completed_count / total if total else None
    fallback_rate = mechanical_fallback_count / total if total else None
    end_to_end_usable_rate = len(ok) / total if total else None
    # "usable result" = the workflow produced SOME report (mechanical or
    # mechanical_with_qwen_narrative) with the invariant intact — every
    # non-hard-failure candidate qualifies, since src/workflows/
    # phishing.py guarantees a report either way. This is deliberately
    # NOT the same as narrative_acceptance_rate.

    return {
        "total_candidates": total,
        "workflow_hard_failures": len(hard_failures),
        "hard_failure_candidate_ids": sorted(hard_failures),
        "semantic_status_counts": semantic_status_counts,
        "upstream_model_failure_count": upstream_model_failure_count,
        "narrative_completed_count": narrative_completed_count,
        "narrative_not_requested_guvenilir_count": narrative_not_requested_guvenilir_count,
        "mechanical_fallback_count": mechanical_fallback_count,
        "narrative_error_code_counts": narrative_error_code_counts,
        "narrative_acceptance_rate": narrative_acceptance_rate,
        "fallback_rate": fallback_rate,
        "end_to_end_usable_result_rate": end_to_end_usable_rate,
        "rule_verdict_breakdown": rule_verdict_breakdown,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH,
                     help="ground_truth.json yolu (varsayılan: data/semantic_eval/ground_truth.json)")
    ap.add_argument("--out", type=Path,
                     default=PROJECT_ROOT / "data" / "semantic_eval" / "hybrid_reliability_results.json",
                     help="çıktı JSON yolu")
    ap.add_argument("--no-resume", action="store_true",
                     help="önbelleği yok say, tüm adayları sıfırdan çalıştır")
    # --worker-* flags are internal — see module docstring's
    # PROCESS-PER-CANDIDATE ISOLATION section. Not meant for direct use.
    ap.add_argument("--worker-candidate", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--worker-eml-path", type=str, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--worker-out", type=Path, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.worker_candidate is not None:
        if args.worker_eml_path is None or args.worker_out is None:
            raise SystemExit("--worker-candidate requires --worker-eml-path and --worker-out")
        _run_single_candidate_worker(args.worker_candidate, args.worker_eml_path, args.worker_out)
        return

    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    unclear_count = sum(1 for r in dataset if r.get("status") == "unclear")
    print(f"{len(dataset)} candidates in dataset, {unclear_count} unclear (excluded)",
          file=sys.stderr)

    cache_key = compute_cache_key()
    print(f"cache_key={cache_key}", file=sys.stderr)

    results = _run_all(dataset, resume=not args.no_resume, cache_key=cache_key)
    metrics = _compute_metrics(dataset, results)

    output = {
        "note": (
            "DEVELOPMENT fallback/reliability rate — measured on the SAME "
            "18-candidate set src/semantic/analyze.py's prompt was iterated "
            "against. NOT a final/independent reliability measurement. See "
            "this script's module docstring."
        ),
        "cache_key": cache_key,
        "dataset_path": str(args.dataset),
        "candidates_unclear_excluded": unclear_count,
        "metrics": metrics,
        "per_candidate": results,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nWrote {args.out}", file=sys.stderr)
    print("\n=== DEVELOPMENT FALLBACK RATE (data/semantic_eval, 18 candidates) ===",
          file=sys.stderr)
    print(json.dumps(metrics, ensure_ascii=False, indent=2), file=sys.stderr)


if __name__ == "__main__":
    main()
