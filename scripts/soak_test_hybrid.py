"""
Single-process Qwen SOAK test — deliberately the opposite isolation
choice from scripts/evaluate_hybrid_reliability.py.

WHY THIS SCRIPT EXISTS, AND WHY IT IS NOT evaluate_hybrid_reliability.py
    evaluate_hybrid_reliability.py's PROCESS-PER-CANDIDATE ISOLATION
    section explains why THAT script spawns one fresh subprocess per
    candidate: no MLX/Metal state survives across candidates, so a
    18-candidate run cannot tell you whether repeated inference calls
    inside one long-lived process accumulate bad state. That was a
    deliberate exclusion, not an oversight — PROGRESS.md's "Sonraki
    görev: tek-process Qwen soak testi" note names this as the
    remaining, still-unanswered question after that script shipped:
    does src/llm/service.py's QwenService, used repeatedly within one
    process the way a real long-running server would use it, degrade,
    leak memory, or start failing after several consecutive real
    Qwen3.5-9B calls?

    This script answers exactly that question and nothing else. It is
    not a quality/precision measurement (that is scripts/evaluate_
    semantic_extractor.py's and evaluate_hybrid_reliability.py's job)
    and it is not a wiring smoke test (that is scripts/smoke_test_
    hybrid.py's job, one email per process by design). It measures
    state accumulation: whether call N behaves differently from call 1
    purely because of how many real model calls preceded it in the
    same process.

ONE PROCESS, ONE QwenService SINGLETON, TWO CYCLES OVER 6 CANDIDATES
    All 12 requests run inside this single Python process and share
    the exact same src.llm.service.get_service() singleton instance —
    no subprocess, no process-per-candidate, no fresh interpreter
    between requests. The 6 candidates are pulled from data/semantic_
    eval/ground_truth.json, filtered to the ones hybrid_reliability_
    results.json already recorded as rule_verdict="Güvenilir" — the
    ONLY bucket where analyze_semantic() is not skipped (rule_verdict
    == "Phishing" short-circuits the first Qwen call entirely, see
    src/workflows/phishing.py) and where generate_report()'s second
    call is actually attempted, so a "Güvenilir" candidate is the one
    guaranteed to drive two real Qwen calls per request instead of
    zero or one. Repeating the same 6 candidates for cycle 2 is
    intentional, not a shortcut — the point is accumulated PROCESS
    STATE, not content diversity; a second exposure to the exact same
    input isolates state drift from anything content-dependent.

NO RETRY, ANYWHERE
    Each of the 12 requests calls analyze_phishing(mode="hybrid")
    exactly once. Same rule as every other script in this family
    (CLAUDE.md's "Yapılmayacaklar" — no patching/retrying model
    output) applied to a whole request, not just a malformed response.

FLUSH-PER-REQUEST JSONL, NO RESUME/CACHE
    Every request's record is appended to the output JSONL and
    fsync'd immediately after that request completes — a crash on
    request 9 must not lose requests 1-8. Unlike evaluate_hybrid_
    reliability.py, THIS script has no cache and no --resume: resuming
    would mean starting a fresh process partway through, which is
    exactly the single-process state accumulation this test exists to
    observe. A killed run is not resumable; it is simply a shorter,
    still-valid soak (see FINAL SUMMARY's honest handling of a partial
    run) that must be started over in a new process, not patched
    together from two.

WHAT IS NEVER WRITTEN TO DISK
    Per explicit instruction, no raw email body, subject, constructed
    model prompt, or full report text is ever written to the JSONL or
    the terminal. Each record is limited to: sequence number, cycle,
    candidate id, eml_path (already a checked-in path reference in
    ground_truth.json), rule_verdict, final_verdict, semantic_status/
    error code, report_source/status/error code, elapsed seconds,
    process peak RSS, and system swap usage — the same category of
    metadata-only record evaluate_hybrid_reliability.py already
    established as sufficient and safe.

category_violation DOES NOT STOP THE SOAK
    Per instruction: category_violation is an output-QUALITY failure
    (the model abandoned the fixed category vocabulary in its report
    text — see generate_report()'s own docstring), not an
    infrastructure failure. evaluate_hybrid_reliability.py's own
    finding was that ALL 9 of its fallbacks were exactly this code,
    concentrated in Güvenilir verdicts — precisely the bucket this
    soak test draws its 6 candidates from, so category_violation
    should be EXPECTED to recur here and must never be treated as a
    reason to abort.

model_call_failed RECOVERY-PROBE PROTOCOL
    The first model_call_failed (in EITHER the semantic call or the
    report call, at any point in the run) does not stop the soak. It
    marks the run as "degraded" and the very next request is run as a
    labeled recovery probe: does the same process, same QwenService
    instance, produce a HEALTHY result on the next call, or does the
    infrastructure stay broken? If that probe ALSO comes back
    model_call_failed, that is two consecutive infrastructure failures
    and the run stops immediately (see next section) — the process's
    Qwen state is presumed unrecoverable and continuing would just
    keep re-observing the same broken state. If the probe recovers,
    the soak continues normally and a second first-failure can still
    trigger a new probe later in the run.

TWO CONSECUTIVE model_call_failed -> EARLY STOP
    This is the one hard stop condition tied to infrastructure health.
    It fires the moment two model_call_failed results land back to
    back (whether that is the initial failure + a failed recovery
    probe, or any other adjacent pair later in the run) — CLAUDE.md's
    two-real-kernel-panic history during sustained GPU work is exactly
    the failure class this guards against; a soak test whose entire
    purpose is to catch state degradation must not itself keep
    hammering a process that has already shown it twice in a row.

UNEXPECTED EXCEPTION -> RECORD AND STOP
    Anything analyze_phishing() raises that is NOT one of its own
    documented, caught failure modes (i.e. it propagated all the way
    out here) is a hard failure outside this script's contract to
    interpret. It is recorded in full (type + message, still no email
    content) and the run stops immediately — CLAUDE.md's rule against
    silently working around unexplained failures applies to a soak
    harness exactly as much as to application code.

Usage (per this project's own rule for long-running GPU work — the
    user runs this, not this session):
    caffeinate -dims .venv/bin/python scripts/soak_test_hybrid.py
"""
import argparse
import json
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

GROUND_TRUTH_PATH = PROJECT_ROOT / "data" / "semantic_eval" / "ground_truth.json"
RELIABILITY_RESULTS_PATH = PROJECT_ROOT / "data" / "semantic_eval" / "hybrid_reliability_results.json"
DEFAULT_OUT_PATH = PROJECT_ROOT / "data" / "semantic_eval" / "soak_test_results.jsonl"
CANDIDATE_COUNT = 6
CYCLE_COUNT = 2


def _select_candidates() -> list[dict]:
    """Picks the first CANDIDATE_COUNT ground_truth.json entries whose
    rule_verdict was recorded as 'Güvenilir' in the existing hybrid
    reliability run — see module docstring for why this bucket is the
    only one that reliably exercises both Qwen calls per request."""
    ground_truth = json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))
    reliability = json.loads(RELIABILITY_RESULTS_PATH.read_text(encoding="utf-8"))
    per_candidate = reliability["per_candidate"]

    by_id = {r["candidate"]: r for r in ground_truth}
    guvenilir_ids = sorted(
        int(cid) for cid, rec in per_candidate.items()
        if rec.get("rule_verdict") == "Güvenilir"
    )
    selected_ids = guvenilir_ids[:CANDIDATE_COUNT]
    if len(selected_ids) < CANDIDATE_COUNT:
        raise SystemExit(
            f"hybrid_reliability_results.json içinde yalnızca "
            f"{len(selected_ids)} Güvenilir aday var, {CANDIDATE_COUNT} gerekli."
        )
    return [by_id[cid] for cid in selected_ids]


def _ru_maxrss_mb() -> float:
    """CUMULATIVE peak RSS since process start, via stdlib resource —
    no psutil dependency. This is NOT a point-in-time reading: ru_maxrss
    only ever goes up for the life of the process, so once the model
    load pushes it past ~2GB on request 1, every later request reports
    the same number regardless of what happens after — a real 2026-08-08
    soak run confirmed exactly this (identical value across all 12
    requests). That makes this field USELESS for detecting whether
    memory grows across repeated calls within a process; it can only
    ever confirm a floor, never track accumulation. Kept (not removed)
    because it is still a legitimate "did we ever exceed X" signal and
    existing recorded runs already used this name — but do not read
    per-request repetition of this value as "no growth happened," only
    as "this field cannot see growth." A live/point-in-time RSS reading
    (e.g. `ps -o rss= -p <pid>`, sampled fresh per request) would be
    needed to actually answer the accumulation question; deferred to a
    future measurement rather than done here.

    ru_maxrss is bytes on macOS, KB on Linux; this project only runs on
    the user's Mac Air M2 (CLAUDE.md), so macOS's byte scale is used
    unconditionally rather than guessing at portability this script
    will never need."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return peak / (1024 * 1024)
    return peak / 1024


def _swap_used_mb() -> float | None:
    """Current system-wide swap usage via `sysctl vm.swapusage`
    (macOS-only, matches this project's single target machine — see
    CLAUDE.md's repeated M2 Air/16GB references). Returns None rather
    than raising if the platform or command is unavailable, since swap
    visibility is a nice-to-have for this soak test, not a correctness
    requirement."""
    if platform.system() != "Darwin":
        return None
    try:
        proc = subprocess.run(
            ["sysctl", "vm.swapusage"], capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    # Example: "vm.swapusage: total = 4096.00M  used = 2455.00M  free = 1641.00M  (encrypted)"
    parts = proc.stdout.split("used = ")
    if len(parts) < 2:
        return None
    used_str = parts[1].split()[0]  # e.g. "2455.00M"
    try:
        return float(used_str.rstrip("M"))
    except ValueError:
        return None


def _append_jsonl(path: Path, record: dict) -> None:
    """Appends one line and flushes + fsyncs immediately — a crash on
    request N must not lose records for requests 1..N-1. No cache, no
    resume; see module docstring's FLUSH-PER-REQUEST JSONL section."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        fh.flush()
        import os
        os.fsync(fh.fileno())


def _run_one_request(seq: int, cycle: int, candidate: dict) -> dict:
    """Runs analyze_phishing(mode='hybrid') exactly once, in THIS
    process, against the shared QwenService singleton. No retry."""
    from src.workflows.phishing import analyze_phishing

    eml_path = PROJECT_ROOT / candidate["eml_path"]
    t0 = time.time()
    try:
        result = analyze_phishing(eml_path, mode="hybrid")
    except Exception as exc:
        elapsed = time.time() - t0
        return {
            "seq": seq,
            "cycle": cycle,
            "candidate": candidate["candidate"],
            "eml_path": candidate["eml_path"],
            "hard_failure": True,
            "error_type": type(exc).__name__,
            "error_message": str(exc)[:500],
            "elapsed_seconds": elapsed,
            "ru_maxrss_mb": _ru_maxrss_mb(),
            "swap_used_mb": _swap_used_mb(),
        }

    elapsed = time.time() - t0
    fd = result.final_decision
    return {
        "seq": seq,
        "cycle": cycle,
        "candidate": candidate["candidate"],
        "eml_path": candidate["eml_path"],
        "hard_failure": False,
        "rule_verdict": result.rule_assessment.rule_verdict,
        "final_verdict": fd.final_verdict if fd else None,
        "semantic_status": result.semantic_status,
        "report_source": result.report_source,
        "narrative_status": result.narrative_status,
        "narrative_error_code": result.narrative_error_code,
        "elapsed_seconds": elapsed,
        "ru_maxrss_mb": _ru_maxrss_mb(),
        "swap_used_mb": _swap_used_mb(),
    }


def _is_model_call_failed(record: dict) -> bool:
    """True if THIS request hit model_call_failed at either Qwen call
    site — the semantic call (semantic_status='failed' with that code,
    reconstructed here from llm_report_error_code's short-circuit
    behavior — see src/workflows/phishing.py's own contract: when the
    FIRST call fails with model_call_failed, llm_report_error_code is
    set to that same code and generate_report() is never attempted) or
    the report call itself (narrative_error_code='model_call_failed'
    with semantic_status other than 'failed')."""
    if record["hard_failure"]:
        return False
    return record.get("narrative_error_code") == "model_call_failed"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH,
                     help="sonuçların yazılacağı JSONL yolu (varsayılan: "
                          "data/semantic_eval/soak_test_results.jsonl)")
    args = ap.parse_args()

    if args.out.exists():
        raise SystemExit(
            f"{args.out} zaten var — bu script resume/append yapmaz (bkz. modül "
            f"docstring'i), taşıyın veya silin ve yeniden çalıştırın."
        )

    candidates = _select_candidates()
    print(f"{len(candidates)} aday seçildi (rule_verdict=Güvenilir, "
          f"hybrid_reliability_results.json'dan): "
          f"{[c['candidate'] for c in candidates]}", file=sys.stderr)
    print(f"{CYCLE_COUNT} cycle x {len(candidates)} aday = "
          f"{CYCLE_COUNT * len(candidates)} toplam istek, tek process içinde.",
          file=sys.stderr)

    start_rss = _ru_maxrss_mb()
    start_swap = _swap_used_mb()
    print(f"başlangıç peak RSS: {start_rss:.1f} MB, swap: {start_swap} MB",
          file=sys.stderr)

    records: list[dict] = []
    recovery_probe_pending = False
    stop_reason: str | None = None
    first_infra_failure_seq: int | None = None
    had_consecutive_infra_failure = False
    recovery_observed: bool | None = None  # None = never tested

    seq = 0
    for cycle in range(1, CYCLE_COUNT + 1):
        for candidate in candidates:
            seq += 1
            label = " (RECOVERY PROBE)" if recovery_probe_pending else ""
            print(f"[{seq}/{CYCLE_COUNT * len(candidates)}] cycle={cycle} "
                  f"candidate={candidate['candidate']}{label} ...", file=sys.stderr)

            record = _run_one_request(seq, cycle, candidate)
            records.append(record)
            _append_jsonl(args.out, record)

            if record["hard_failure"]:
                print(f"        UNEXPECTED EXCEPTION: {record['error_type']}: "
                      f"{record['error_message']}", file=sys.stderr)
                stop_reason = "unexpected_exception"
                break

            print(f"        {record['elapsed_seconds']:.0f}s — "
                  f"rule_verdict={record['rule_verdict']} "
                  f"final_verdict={record['final_verdict']} "
                  f"semantic_status={record['semantic_status']} "
                  f"report_source={record['report_source']} "
                  f"narrative_status={record['narrative_status']} "
                  f"narrative_error_code={record['narrative_error_code']} "
                  f"peak_rss={record['ru_maxrss_mb']:.1f}MB",
                  file=sys.stderr)

            this_failed = _is_model_call_failed(record)

            if recovery_probe_pending:
                recovery_probe_pending = False
                if this_failed:
                    print("        RECOVERY PROBE FAILED — two consecutive "
                          "model_call_failed, stopping early.", file=sys.stderr)
                    recovery_observed = False
                    had_consecutive_infra_failure = True
                    stop_reason = "two_consecutive_model_call_failed"
                    break
                else:
                    print("        recovery probe OK — Qwen state recovered, "
                          "continuing normally.", file=sys.stderr)
                    recovery_observed = True
            elif this_failed:
                if first_infra_failure_seq is None:
                    first_infra_failure_seq = seq
                print("        model_call_failed — next request runs as a "
                      "labeled recovery probe.", file=sys.stderr)
                recovery_probe_pending = True

        if stop_reason:
            break

    end_rss = _ru_maxrss_mb()
    end_swap = _swap_used_mb()

    ok_records = [r for r in records if not r["hard_failure"]]
    durations = sorted(r["elapsed_seconds"] for r in ok_records)

    def _pct(sorted_vals: list[float], p: float) -> float | None:
        if not sorted_vals:
            return None
        idx = min(len(sorted_vals) - 1, int(len(sorted_vals) * p))
        return sorted_vals[idx]

    summary = {
        "note": (
            "SINGLE-PROCESS soak test — one QwenService singleton shared across "
            "every request in this run. See scripts/soak_test_hybrid.py's module "
            "docstring; NOT comparable to evaluate_hybrid_reliability.py's "
            "per-candidate-subprocess measurement."
        ),
        "requests_completed": len(records),
        "requests_planned": CYCLE_COUNT * len(candidates),
        "stopped_early": stop_reason is not None,
        "stop_reason": stop_reason,
        "first_infra_failure_seq": first_infra_failure_seq,
        "had_consecutive_infra_failure": had_consecutive_infra_failure,
        "recovery_observed": recovery_observed,
        "start_ru_maxrss_mb": start_rss,
        "end_ru_maxrss_mb": end_rss,
        "max_ru_maxrss_mb": max((r["ru_maxrss_mb"] for r in records), default=None),
        "start_swap_used_mb": start_swap,
        "end_swap_used_mb": end_swap,
        "duration_seconds_min": durations[0] if durations else None,
        "duration_seconds_median": _pct(durations, 0.5),
        "duration_seconds_max": durations[-1] if durations else None,
    }

    print("\n=== SOAK TEST ÖZET ===", file=sys.stderr)
    print(json.dumps(summary, ensure_ascii=False, indent=2), file=sys.stderr)
    print(f"\nSonuçlar (JSONL, her istek sonrası flush edildi): {args.out}",
          file=sys.stderr)

    summary_path = args.out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Özet: {summary_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
