"""
Runs src/semantic/analyze.py's Qwen extractor on all 18
data/semantic_eval candidates and measures it against
data/semantic_eval/ground_truth.json (PHISHING_ROUTING_PLAN.md step 8).

Still shadow mode: this script measures the extractor's own accuracy,
it does not touch a rule engine verdict and nothing here feeds a
decision. See src/semantic/analyze.py's module docstring.

TWO SEPARATE EVALUATION LAYERS — DO NOT MIX THEM
    1. TYPE-LEVEL (primary). Unit of measurement is (candidate_id,
       finding_type) — NOT individual finding instances. For each of
       the 9 allowed types x each labeled candidate: did ground truth
       say this type is present (bool), did the model's ACCEPTED
       output say the same (bool)? However many separate quotes either
       side has for that type collapses to one boolean each. This
       answers "did the extractor notice the right KINDS of
       manipulation," independent of how well it phrased/located them.
       Reports micro and macro precision/recall/F1, plus a per-type
       breakdown (TP/FP/FN/TN) — no single "accuracy" number, per
       CLAUDE.md's precedent for the rule engine's own P/R/abstention
       reporting.

    2. SPAN QUALITY (secondary, does NOT feed type-level TP/FP/FN).
       Within a (candidate, type) where BOTH ground truth and the model
       have at least one span, spans are matched to MAXIMIZE total IoU
       via one-to-one bipartite matching (scipy-free — 9 types x small
       per-type counts, brute-force permutation search is fine at this
       scale) — no span is matched twice. Reports mean/median IoU,
       exact-match rate, containment rate (one span fully inside the
       other), zero-overlap rate, and how many predicted/ground-truth
       spans went unmatched (more of one side than the other). Also
       reports coverage-adjusted IoU, which counts every UNMATCHED
       ground-truth span as IoU=0 rather than excluding it — a
       type-level TP where the model's spans don't cover half the
       ground-truth findings should not look as good as one where they
       do.

    A type can be a type-level TP with terrible span quality (right
    category, wrong/vague quote) or the reverse is impossible (no span
    quality without agreeing on the type first) — keeping these
    separate is why the type-level pass never looks at offsets at all.

ONLY VALIDATOR-ACCEPTED FINDINGS COUNT AS MODEL PREDICTIONS
    result.rejected findings from analyze_semantic() (ungrounded,
    ambiguous, malformed) are logged for visibility but never counted
    toward type-level TP or span quality — an evidence quote the
    validator already rejected is not a claim the model gets credit or
    blame for at the type level; it is a separate, structural failure
    (grounding rate) already exercised by tests/test_semantic_validate.py
    and tests/test_semantic_analyze.py.

LEGITIMATE FALSE-FINDING RATE
    Restricted to the 13/18 candidates NOT in the net_phishing or
    fraud_or_reward axes (legit_marketing_urgency, neutral_notification,
    authority_brand, no_signal — see data/semantic_eval/README.md).
    Counts how many of the model's ACCEPTED findings on those 13 landed
    on a (candidate, type) ground truth says has NO finding — the
    single number this whole set was built to watch most closely (see
    PROGRESS.md: marketing/urgency language is this extractor's known
    biggest risk).

unclear CANDIDATES ARE FULLY EXCLUDED from every metric above, not
folded into either side — same rule data/holdout and the rule-engine-v2
dev set apply to their own "can't decide" labels.

FAULT TOLERANCE AND RESUMABILITY
    A real run (2026-08) hit analyze_semantic()'s SystemExit on
    Candidate 1 (truncated JSON — see src/semantic/analyze.py's
    MAX_TOKENS comment) and, before this was fixed, took the entire
    18-candidate batch down with it. Each candidate's model call is
    wrapped in its own try/except, a failure is recorded as
    {"error": "..."} for that one candidate and EXCLUDED from every
    metric (same treatment as "unclear"), and the loop continues.
    Results are also cached to data/semantic_eval/
    _model_results_cache.json after every candidate — re-running the
    script (the default) resumes from that cache instead of re-querying
    the model for candidates that already succeeded. Pass --no-resume
    to force a clean run of all 18 (e.g. after changing MAX_TOKENS).

PROCESS-PER-CANDIDATE ISOLATION (2026-08, second real bug)
    A separate real run hit a "RuntimeError: [METAL] Command buffer
    execution failed: GPU Timeout Error" from mlx_vlm/Metal starting at
    Candidate 10 — and then EVERY subsequent candidate in that same
    process failed identically, most in 0 seconds (never even reaching
    generate()). Re-running Candidate 10 alone in a fresh process
    succeeded in 43s. This confirms the failure is NOT about any
    particular email — it's the long-lived model process itself
    accumulating some unrecoverable GPU/Metal state after enough
    generation calls, after which every further call in that same
    process fails immediately.

    The fix is not a retry inside one process (that would just hit the
    same accumulated bad state again) — it's giving every candidate its
    OWN process, each starting with a clean GPU/Metal state:
    - The parent process (main(), running the default `analyze` mode)
      never imports or calls analyze_semantic() itself and never loads
      the model. For each candidate it spawns
      `python3 evaluate_semantic_extractor.py --worker-candidate N
      --worker-out <tmp path>` as a fresh subprocess.
    - That subprocess (main(), running in `--worker-candidate` mode)
      loads the model, runs analyze_semantic() on exactly ONE
      candidate, writes its result as JSON to --worker-out, and exits —
      taking the model and any accumulated Metal state down with it
      when the process ends, regardless of success or failure.
    - The parent reads the worker's output file, and ALSO records the
      subprocess's exit code, stderr tail, and wall-clock latency —
      not just whether it produced a result — so a worker crash still
      leaves a diagnosable trail instead of a silent gap.
    - Successful worker output is moved into CACHE_PATH's on-disk JSON
      atomically (write to a temp file in the same directory, then
      os.replace()) so a crash mid-write of the cache file itself can
      never corrupt or truncate previously-saved candidates.
    - --resume (the default) still skips any candidate already present
      in the cache with no "error" key — a worker's fresh cost (model
      load + generation) is only paid for candidates that don't already
      have a usable result. --no-resume forces every candidate to be
      re-run in a fresh subprocess regardless of what's cached.

Usage:
    python3 scripts/evaluate_semantic_extractor.py
    python3 scripts/evaluate_semantic_extractor.py --out results.json
    python3 scripts/evaluate_semantic_extractor.py --no-resume
"""
import argparse
import itertools
import json
import os
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from schemas.semantic import SemanticFindingType  # noqa: E402

GROUND_TRUTH_PATH = PROJECT_ROOT / "data" / "semantic_eval" / "ground_truth.json"

ALL_TYPES = [t.value for t in SemanticFindingType]

# Same axis assignment as scripts/render_semantic_eval_review.py's
# SELECTION, duplicated here (not imported) because that module's
# SELECTION is keyed by eml_path/axis pairs for rendering, and this
# script only needs the axis-per-candidate mapping — importing the
# whole render module would also pull in its file-writing side effects.
CANDIDATE_AXIS = {
    1: "net_phishing", 2: "net_phishing", 3: "net_phishing",
    4: "fraud_or_reward", 5: "fraud_or_reward", 6: "fraud_or_reward",
    7: "legit_marketing_urgency", 8: "legit_marketing_urgency",
    9: "legit_marketing_urgency", 10: "legit_marketing_urgency",
    11: "neutral_notification", 12: "neutral_notification", 13: "neutral_notification",
    14: "authority_brand", 15: "authority_brand", 16: "authority_brand",
    17: "no_signal", 18: "no_signal",
}
LEGITIMATE_AXES = {
    "legit_marketing_urgency", "neutral_notification", "authority_brand", "no_signal",
}


def _iou(a: tuple[int, int], b: tuple[int, int]) -> float:
    start = max(a[0], b[0])
    end = min(a[1], b[1])
    intersection = max(0, end - start)
    union = (a[1] - a[0]) + (b[1] - b[0]) - intersection
    if union <= 0:
        return 0.0
    return intersection / union


_MAX_BRUTE_FORCE_SPANS = 6
# ^ ground truth never has more than 1 finding of the same type in the
# same candidate on this set (verified against data/semantic_eval/
# ground_truth.json) — this only guards against the model producing an
# unexpectedly large number of same-type findings, where permutations
# over both sides would otherwise blow up combinatorially.


def _best_matching(gt_spans: list[tuple[int, int]],
                    pred_spans: list[tuple[int, int]]) -> list[tuple[int, int, float]]:
    """One-to-one matching maximizing total IoU, brute force over
    permutations — fine at this scale (per-type counts within a single
    candidate are always small, never more than a handful). Returns a
    list of (gt_index, pred_index, iou) for matched pairs. Unmatched
    indices (whichever side is longer) are simply absent."""
    if not gt_spans or not pred_spans:
        return []
    if len(gt_spans) > _MAX_BRUTE_FORCE_SPANS or len(pred_spans) > _MAX_BRUTE_FORCE_SPANS:
        raise ValueError(
            f"_best_matching: too many spans for brute-force matching "
            f"(gt={len(gt_spans)}, pred={len(pred_spans)}, max={_MAX_BRUTE_FORCE_SPANS}) "
            f"— this is a real, unexpectedly large finding count, not a bug to route "
            f"around silently; inspect the model output before extending this limit."
        )
    n = min(len(gt_spans), len(pred_spans))
    best_total = -1.0
    best_pairs: list[tuple[int, int, float]] = []
    pred_indices = range(len(pred_spans))
    for gt_subset in itertools.permutations(range(len(gt_spans)), n):
        for pred_subset in itertools.permutations(pred_indices, n):
            total = 0.0
            pairs = []
            for gi, pi in zip(gt_subset, pred_subset):
                iou = _iou(gt_spans[gi], pred_spans[pi])
                total += iou
                pairs.append((gi, pi, iou))
            if total > best_total:
                best_total = total
                best_pairs = pairs
    return best_pairs


CACHE_PATH = PROJECT_ROOT / "data" / "semantic_eval" / "_model_results_cache.json"
# ^ per-candidate raw model output, written after EVERY candidate (not
# just at the end) — a crash or a SystemExit from one candidate must not
# throw away the others that already succeeded and cost real GPU time
# to produce. See PROCESS-PER-CANDIDATE ISOLATION in the module
# docstring for why each candidate's result also comes from its own
# subprocess rather than a long-lived in-process model.


def _load_cache() -> dict[int, dict]:
    if not CACHE_PATH.is_file():
        return {}
    raw = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return {int(k): v for k, v in raw.items()}


def _save_cache(results: dict[int, dict]) -> None:
    """Atomic write: a temp file in the SAME directory (so os.replace()
    is a same-filesystem rename, not a cross-filesystem copy) is written
    first and then swapped into place. Without this, a crash mid-write
    of CACHE_PATH itself — unlikely but not impossible given how often
    this script has hit unexpected process-level failures — could leave
    a truncated/corrupt cache file and silently lose every previously
    cached candidate, not just the one being written."""
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = CACHE_PATH.with_suffix(".json.tmp")
    tmp_path.write_text(
        json.dumps({str(k): v for k, v in results.items()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(tmp_path, CACHE_PATH)


def _run_single_candidate_worker(candidate: int, eml_path_str: str, out_path: Path) -> None:
    """Runs in the WORKER subprocess only (invoked via --worker-candidate).
    Loads the model, runs analyze_semantic() on exactly this one
    candidate, and writes the result (or an error) as JSON to out_path.
    Never touches CACHE_PATH directly — the parent process owns the
    cache and merges this worker's output into it after the subprocess
    exits, so a worker can never race another worker or corrupt the
    shared cache file."""
    from src.parser.parse import parse_eml
    from src.semantic.analyze import analyze_semantic

    eml_path = PROJECT_ROOT / eml_path_str
    facts = parse_eml(eml_path)
    t0 = time.time()
    try:
        result = analyze_semantic(facts)
    except (SystemExit, Exception) as e:
        elapsed = time.time() - t0
        output = {
            "error": f"{type(e).__name__}: {e}",
            "elapsed_seconds": elapsed,
            "traceback": traceback.format_exc(),
        }
    else:
        elapsed = time.time() - t0
        output = {
            "accepted": [
                {"type": f.type.value, "evidence": f.evidence,
                 "start": f.start, "end": f.end,
                 "model_confidence": f.model_confidence}
                for f in result.accepted
            ],
            "rejected": [
                {"rejection_reason": vf.rejection_reason.value,
                 "evidence": getattr(vf.finding, "evidence", repr(vf.finding))}
                for vf in result.rejected
            ],
            "elapsed_seconds": elapsed,
        }
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")


def _run_model_on_all(ground_truth: list[dict], *, resume: bool = True) -> dict[int, dict]:
    """Orchestrator — runs in the PARENT process, never loads the model
    itself. For each candidate not already cached, spawns a fresh
    `python3 evaluate_semantic_extractor.py --worker-candidate N
    --worker-out <tmp>` subprocess (see module docstring's
    PROCESS-PER-CANDIDATE ISOLATION section for why: a real run showed
    the in-process model accumulates unrecoverable GPU/Metal state after
    enough calls, after which every further candidate in that SAME
    process fails immediately — re-running the same candidate alone in
    a fresh process succeeded). Each subprocess's exit code, stderr
    tail, and wall-clock time are recorded regardless of whether it
    produced usable output, so a hard crash (not just a caught
    exception) still leaves a diagnosable trail. Results are cached
    after every candidate (see _save_cache's atomic-write docstring).
    """
    results = _load_cache() if resume else {}
    if results:
        print(f"Resuming from cache: {len(results)} candidate(s) already have results.",
              file=sys.stderr)

    import tempfile

    for record in ground_truth:
        candidate = record["candidate"]
        if record["status"] == "unclear":
            print(f"[{candidate}/18] SKIP (ground truth status=unclear)", file=sys.stderr)
            continue
        if candidate in results and "error" not in results[candidate]:
            print(f"[{candidate}/18] SKIP (cached result present)", file=sys.stderr)
            continue

        eml_path = record["eml_path"]
        print(f"[{candidate}/18] {eml_path} (isolated subprocess) ...", file=sys.stderr)

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
                if "error" in output:
                    print(f"        FAILED after {elapsed:.0f}s (worker caught): "
                          f"{output['error']}", file=sys.stderr)
                else:
                    print(f"        {elapsed:.0f}s — {len(output['accepted'])} accepted, "
                          f"{len(output['rejected'])} rejected", file=sys.stderr)
            else:
                # Subprocess crashed hard enough that it never wrote its
                # output file, or exited non-zero — capture as much
                # diagnostic signal as exists rather than silently
                # treating this the same as a clean in-worker failure.
                stderr_tail = "\n".join(proc.stderr.strip().splitlines()[-20:])
                print(f"        SUBPROCESS FAILED after {elapsed:.0f}s "
                      f"(exit code {proc.returncode}): {stderr_tail[-300:]}", file=sys.stderr)
                output = {
                    "error": f"subprocess exit code {proc.returncode}",
                    "elapsed_seconds": elapsed,
                    "subprocess_stderr_tail": stderr_tail,
                }

        results[candidate] = output
        _save_cache(results)

    return results


def _type_level_metrics(ground_truth: list[dict], model_results: dict[int, dict]) -> dict:
    """(candidate, type) confusion matrix — see module docstring."""
    per_type = {t: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for t in ALL_TYPES}
    micro = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}

    for record in ground_truth:
        candidate = record["candidate"]
        if candidate not in model_results:
            continue  # unclear, skipped
        gt_types = {f["type"] for f in record["findings"]}
        pred_types = {f["type"] for f in model_results[candidate]["accepted"]}
        for t in ALL_TYPES:
            gt_has = t in gt_types
            pred_has = t in pred_types
            if gt_has and pred_has:
                key = "tp"
            elif pred_has and not gt_has:
                key = "fp"
            elif gt_has and not pred_has:
                key = "fn"
            else:
                key = "tn"
            per_type[t][key] += 1
            micro[key] += 1

    def _prf1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) else 0.0)
        return precision, recall, f1

    per_type_out = {}
    macro_p, macro_r, macro_f1 = [], [], []
    for t, c in per_type.items():
        p, r, f1 = _prf1(c["tp"], c["fp"], c["fn"])
        per_type_out[t] = {**c, "precision": p, "recall": r, "f1": f1}
        # Only average over types that actually appear in ground truth or
        # predictions at least once — a type with tp=fp=fn=0 (never
        # relevant on this small a set) would silently drag macro scores
        # toward the "predict nothing" baseline otherwise.
        if c["tp"] + c["fp"] + c["fn"] > 0:
            macro_p.append(p)
            macro_r.append(r)
            macro_f1.append(f1)

    micro_p, micro_r, micro_f1 = _prf1(micro["tp"], micro["fp"], micro["fn"])

    return {
        "micro": {**micro, "precision": micro_p, "recall": micro_r, "f1": micro_f1},
        "macro": {
            "precision": statistics.mean(macro_p) if macro_p else 0.0,
            "recall": statistics.mean(macro_r) if macro_r else 0.0,
            "f1": statistics.mean(macro_f1) if macro_f1 else 0.0,
            "types_included": len(macro_p),
        },
        "per_type": per_type_out,
    }


def _span_quality_metrics(ground_truth: list[dict], model_results: dict[int, dict]) -> dict:
    """Secondary pass — see module docstring. Only runs within a
    (candidate, type) where both sides have at least one span; never
    affects type-level TP/FP/FN."""
    all_ious: list[float] = []
    coverage_adjusted: list[float] = []
    exact_matches = 0
    containments = 0
    zero_overlaps = 0
    unmatched_gt = 0
    unmatched_pred = 0
    matched_pairs = 0

    for record in ground_truth:
        candidate = record["candidate"]
        if candidate not in model_results:
            continue
        gt_by_type: dict[str, list[tuple[int, int]]] = {}
        for f in record["findings"]:
            gt_by_type.setdefault(f["type"], []).append((f["start"], f["end"]))
        pred_by_type: dict[str, list[tuple[int, int]]] = {}
        for f in model_results[candidate]["accepted"]:
            pred_by_type.setdefault(f["type"], []).append((f["start"], f["end"]))

        for t in set(gt_by_type) | set(pred_by_type):
            gt_spans = gt_by_type.get(t, [])
            pred_spans = pred_by_type.get(t, [])
            if not gt_spans or not pred_spans:
                # No overlap possible to measure — every gt span here is
                # unmatched (type-level FN) and counts as 0 for coverage.
                coverage_adjusted.extend([0.0] * len(gt_spans))
                unmatched_gt += len(gt_spans)
                unmatched_pred += len(pred_spans)
                continue

            pairs = _best_matching(gt_spans, pred_spans)
            matched_gt = {gi for gi, _, _ in pairs}
            matched_pred = {pi for _, pi, _ in pairs}
            for gi, pi, iou in pairs:
                all_ious.append(iou)
                coverage_adjusted.append(iou)
                matched_pairs += 1
                if gt_spans[gi] == pred_spans[pi]:
                    exact_matches += 1
                elif iou == 0.0:
                    zero_overlaps += 1
                else:
                    a, b = gt_spans[gi], pred_spans[pi]
                    if (a[0] <= b[0] and a[1] >= b[1]) or (b[0] <= a[0] and b[1] >= a[1]):
                        containments += 1
            for gi in range(len(gt_spans)):
                if gi not in matched_gt:
                    unmatched_gt += 1
                    coverage_adjusted.append(0.0)
            for pi in range(len(pred_spans)):
                if pi not in matched_pred:
                    unmatched_pred += 1

    return {
        "matched_pairs": matched_pairs,
        "mean_iou": statistics.mean(all_ious) if all_ious else None,
        "median_iou": statistics.median(all_ious) if all_ious else None,
        "coverage_adjusted_mean_iou": (
            statistics.mean(coverage_adjusted) if coverage_adjusted else None
        ),
        "exact_match_rate": exact_matches / matched_pairs if matched_pairs else None,
        "containment_rate": containments / matched_pairs if matched_pairs else None,
        "zero_overlap_rate": zero_overlaps / matched_pairs if matched_pairs else None,
        "unmatched_ground_truth_spans": unmatched_gt,
        "unmatched_predicted_spans": unmatched_pred,
    }


def _legitimate_false_finding_rate(ground_truth: list[dict],
                                    model_results: dict[int, dict]) -> dict:
    legit_candidates = [
        r for r in ground_truth
        if CANDIDATE_AXIS[r["candidate"]] in LEGITIMATE_AXES and r["candidate"] in model_results
    ]
    total_false_findings = 0
    total_accepted_findings = 0
    candidates_with_any_false_finding = 0
    details = []

    for record in legit_candidates:
        candidate = record["candidate"]
        gt_types = {f["type"] for f in record["findings"]}
        accepted = model_results[candidate]["accepted"]
        total_accepted_findings += len(accepted)
        false_here = [f for f in accepted if f["type"] not in gt_types]
        total_false_findings += len(false_here)
        if false_here:
            candidates_with_any_false_finding += 1
            details.append({
                "candidate": candidate,
                "eml_path": record["eml_path"],
                "axis": CANDIDATE_AXIS[candidate],
                "false_findings": [
                    {"type": f["type"], "evidence": f["evidence"]} for f in false_here
                ],
            })

    return {
        "legitimate_candidates_evaluated": len(legit_candidates),
        "total_accepted_findings_on_legitimate_mail": total_accepted_findings,
        "total_false_findings": total_false_findings,
        "candidates_with_any_false_finding": candidates_with_any_false_finding,
        "false_finding_rate_per_candidate": (
            candidates_with_any_false_finding / len(legit_candidates)
            if legit_candidates else None
        ),
        "details": details,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path,
                     default=PROJECT_ROOT / "data" / "semantic_eval" / "eval_results.json",
                     help="çıktı JSON yolu")
    ap.add_argument("--no-resume", action="store_true",
                     help="önbellekteki (varsa) sonuçları yok say, tüm 18 candidate'i "
                          "sıfırdan çalıştır (örn. MAX_TOKENS değiştikten sonra)")
    # --worker-* flags are internal: _run_model_on_all() (the parent/
    # orchestrator process) invokes THIS SAME script as a subprocess
    # with these flags set, one fresh process per candidate. Not meant
    # to be passed by a human directly — see module docstring's
    # PROCESS-PER-CANDIDATE ISOLATION section for why this exists.
    ap.add_argument("--worker-candidate", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--worker-eml-path", type=str, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--worker-out", type=Path, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.worker_candidate is not None:
        if args.worker_eml_path is None or args.worker_out is None:
            raise SystemExit("--worker-candidate requires --worker-eml-path and --worker-out")
        _run_single_candidate_worker(args.worker_candidate, args.worker_eml_path, args.worker_out)
        return

    ground_truth = json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))
    unclear_count = sum(1 for r in ground_truth if r["status"] == "unclear")

    print(f"{len(ground_truth)} candidates, {unclear_count} unclear (excluded)",
          file=sys.stderr)
    all_results = _run_model_on_all(ground_truth, resume=not args.no_resume)

    # Failed candidates (a real, per-candidate SystemExit from
    # analyze_semantic() — see _run_model_on_all's docstring) are
    # excluded from every metric exactly like "unclear" ground truth is:
    # there is no model output to measure, so silently including them
    # in candidates_evaluated would misrepresent how many candidates
    # actually produced a comparable result.
    failed = {c: r for c, r in all_results.items() if "error" in r}
    model_results = {c: r for c, r in all_results.items() if "error" not in r}
    if failed:
        print(f"\n{len(failed)} candidate(s) FAILED (excluded from metrics): "
              f"{sorted(failed)}", file=sys.stderr)

    type_level = _type_level_metrics(ground_truth, model_results)
    span_quality = _span_quality_metrics(ground_truth, model_results)
    legit_false_findings = _legitimate_false_finding_rate(ground_truth, model_results)

    total_rejected = sum(len(r["rejected"]) for r in model_results.values())
    total_accepted = sum(len(r["accepted"]) for r in model_results.values())

    output = {
        "candidates_evaluated": len(model_results),
        "candidates_unclear_excluded": unclear_count,
        "candidates_failed_excluded": len(failed),
        "failed_candidate_ids": sorted(failed),
        "total_accepted_findings": total_accepted,
        "total_rejected_findings": total_rejected,
        "type_level": type_level,
        "span_quality": span_quality,
        "legitimate_false_finding_rate": legit_false_findings,
        "raw_model_results": all_results,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nWrote {args.out}", file=sys.stderr)
    print(f"\n=== TYPE-LEVEL (micro) ===", file=sys.stderr)
    m = type_level["micro"]
    print(f"precision={m['precision']:.3f} recall={m['recall']:.3f} f1={m['f1']:.3f} "
          f"(tp={m['tp']} fp={m['fp']} fn={m['fn']} tn={m['tn']})", file=sys.stderr)
    print(f"\n=== TYPE-LEVEL (macro, {type_level['macro']['types_included']} types) ===",
          file=sys.stderr)
    ma = type_level["macro"]
    print(f"precision={ma['precision']:.3f} recall={ma['recall']:.3f} f1={ma['f1']:.3f}",
          file=sys.stderr)
    print(f"\n=== SPAN QUALITY ===", file=sys.stderr)
    print(json.dumps(span_quality, indent=2), file=sys.stderr)
    print(f"\n=== LEGITIMATE FALSE-FINDING RATE ===", file=sys.stderr)
    print(f"{legit_false_findings['candidates_with_any_false_finding']}/"
          f"{legit_false_findings['legitimate_candidates_evaluated']} legitimate "
          f"candidates had >=1 false finding "
          f"({legit_false_findings['total_false_findings']} false findings total)",
          file=sys.stderr)


if __name__ == "__main__":
    main()
