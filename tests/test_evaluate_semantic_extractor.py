"""Unit tests for scripts/evaluate_semantic_extractor.py's metric
computation (PHISHING_ROUTING_PLAN.md step 8). No real model call
anywhere — model_results dicts are constructed by hand to exercise the
type-level, span-quality, and legitimate-false-finding-rate logic in
isolation, and _run_model_on_all's fault-tolerance/caching is tested
with analyze_semantic() mocked out."""
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_SPEC = importlib.util.spec_from_file_location(
    "evaluate_semantic_extractor",
    Path(__file__).resolve().parent.parent / "scripts" / "evaluate_semantic_extractor.py",
)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

_iou = _MODULE._iou
_best_matching = _MODULE._best_matching
_type_level_metrics = _MODULE._type_level_metrics
_span_quality_metrics = _MODULE._span_quality_metrics
_legitimate_false_finding_rate = _MODULE._legitimate_false_finding_rate
_run_model_on_all = _MODULE._run_model_on_all
CANDIDATE_AXIS = _MODULE.CANDIDATE_AXIS


def _gt_finding(type_, start, end):
    return {"type": type_, "evidence": "x", "start": start, "end": end, "reason": "x"}


def _pred_finding(type_, start, end):
    return {"type": type_, "evidence": "x", "start": start, "end": end,
            "model_confidence": 0.9}


# --- _iou ------------------------------------------------------------

def test_iou_identical_spans_is_one():
    assert _iou((10, 20), (10, 20)) == 1.0


def test_iou_disjoint_spans_is_zero():
    assert _iou((0, 10), (20, 30)) == 0.0


def test_iou_partial_overlap():
    # intersection [10,15) len=5, union [0,20) len=20
    assert _iou((0, 15), (10, 20)) == 5 / 20


def test_iou_containment():
    # intersection = inner span, union = outer span
    assert _iou((0, 100), (10, 20)) == 10 / 100


def test_iou_touching_but_not_overlapping_is_zero():
    assert _iou((0, 10), (10, 20)) == 0.0


# --- _best_matching ----------------------------------------------------

def test_best_matching_single_pair():
    pairs = _best_matching([(0, 10)], [(0, 10)])
    assert pairs == [(0, 0, 1.0)]


def test_best_matching_picks_higher_total_iou_assignment():
    # gt0 overlaps pred1 well, gt1 overlaps pred0 well — matching must
    # NOT just pair by index order.
    gt = [(0, 10), (100, 110)]
    pred = [(100, 110), (0, 10)]
    pairs = _best_matching(gt, pred)
    total_iou = sum(iou for _, _, iou in pairs)
    assert total_iou == 2.0  # both perfect matches, cross-paired
    matched = {(gi, pi) for gi, pi, _ in pairs}
    assert (0, 1) in matched
    assert (1, 0) in matched


def test_best_matching_unequal_counts_leaves_some_unmatched():
    pairs = _best_matching([(0, 10)], [(0, 10), (50, 60)])
    assert len(pairs) == 1


def test_best_matching_empty_side_returns_empty():
    assert _best_matching([], [(0, 10)]) == []
    assert _best_matching([(0, 10)], []) == []


def test_best_matching_raises_on_too_many_spans():
    big = [(i, i + 1) for i in range(10)]
    try:
        _best_matching(big, big)
        raise AssertionError("expected ValueError for too many spans")
    except ValueError:
        pass


# --- _type_level_metrics ------------------------------------------------

def test_type_level_perfect_match_is_all_tp():
    ground_truth = [
        {"candidate": 1, "status": "labeled",
         "findings": [_gt_finding("credential_request", 0, 10)]},
    ]
    model_results = {
        1: {"accepted": [_pred_finding("credential_request", 0, 10)], "rejected": []},
    }
    metrics = _type_level_metrics(ground_truth, model_results)
    assert metrics["micro"]["tp"] == 1
    assert metrics["micro"]["fp"] == 0
    assert metrics["micro"]["fn"] == 0
    assert metrics["micro"]["precision"] == 1.0
    assert metrics["micro"]["recall"] == 1.0


def test_type_level_missed_finding_is_fn():
    ground_truth = [
        {"candidate": 1, "status": "labeled",
         "findings": [_gt_finding("threat_or_fear", 0, 10)]},
    ]
    model_results = {1: {"accepted": [], "rejected": []}}
    metrics = _type_level_metrics(ground_truth, model_results)
    assert metrics["micro"]["fn"] == 1
    assert metrics["micro"]["tp"] == 0
    assert metrics["micro"]["recall"] == 0.0


def test_type_level_extra_finding_is_fp():
    ground_truth = [{"candidate": 1, "status": "labeled", "findings": []}]
    model_results = {
        1: {"accepted": [_pred_finding("urgency_or_pressure", 0, 10)], "rejected": []},
    }
    metrics = _type_level_metrics(ground_truth, model_results)
    assert metrics["micro"]["fp"] == 1
    assert metrics["micro"]["precision"] == 0.0


def test_type_level_multiple_same_type_findings_collapse_to_one_tp():
    """Two urgency_or_pressure quotes in ground truth + two in the
    model's output must still count as ONE (candidate, type) TP, not
    two — the unit of measurement is the pair, not the instance."""
    ground_truth = [
        {"candidate": 1, "status": "labeled", "findings": [
            _gt_finding("urgency_or_pressure", 0, 10),
            _gt_finding("urgency_or_pressure", 50, 60),
        ]},
    ]
    model_results = {
        1: {"accepted": [
            _pred_finding("urgency_or_pressure", 0, 10),
            _pred_finding("urgency_or_pressure", 999, 1010),
        ], "rejected": []},
    }
    metrics = _type_level_metrics(ground_truth, model_results)
    assert metrics["micro"]["tp"] == 1
    assert metrics["micro"]["fp"] == 0
    assert metrics["micro"]["fn"] == 0


def test_type_level_unclear_candidate_excluded_entirely():
    ground_truth = [
        {"candidate": 1, "status": "unclear", "findings": []},
    ]
    model_results = {}  # never run, per _run_model_on_all's skip logic
    metrics = _type_level_metrics(ground_truth, model_results)
    # every type should be tn=0 tp=0 fp=0 fn=0 — candidate never counted
    for t, c in metrics["per_type"].items():
        assert c["tp"] == c["fp"] == c["fn"] == c["tn"] == 0


def test_type_level_true_negative_counted_for_absent_type():
    ground_truth = [{"candidate": 1, "status": "labeled", "findings": []}]
    model_results = {1: {"accepted": [], "rejected": []}}
    metrics = _type_level_metrics(ground_truth, model_results)
    assert metrics["micro"]["tn"] == 9  # all 9 types correctly absent


def test_type_level_macro_excludes_types_never_seen():
    """A type with zero tp/fp/fn across the whole set (never in ground
    truth or predictions) must not drag macro P/R toward 0 or 1 by
    being force-included."""
    ground_truth = [
        {"candidate": 1, "status": "labeled",
         "findings": [_gt_finding("payment_request", 0, 10)]},
    ]
    model_results = {1: {"accepted": [_pred_finding("payment_request", 0, 10)], "rejected": []}}
    metrics = _type_level_metrics(ground_truth, model_results)
    assert metrics["macro"]["types_included"] == 1
    assert metrics["macro"]["precision"] == 1.0
    assert metrics["macro"]["recall"] == 1.0


# --- _span_quality_metrics ----------------------------------------------

def test_span_quality_exact_match():
    ground_truth = [
        {"candidate": 1, "status": "labeled",
         "findings": [_gt_finding("credential_request", 10, 20)]},
    ]
    model_results = {1: {"accepted": [_pred_finding("credential_request", 10, 20)], "rejected": []}}
    result = _span_quality_metrics(ground_truth, model_results)
    assert result["matched_pairs"] == 1
    assert result["mean_iou"] == 1.0
    assert result["exact_match_rate"] == 1.0
    assert result["zero_overlap_rate"] == 0.0


def test_span_quality_no_prediction_counts_unmatched_gt():
    ground_truth = [
        {"candidate": 1, "status": "labeled",
         "findings": [_gt_finding("threat_or_fear", 10, 20)]},
    ]
    model_results = {1: {"accepted": [], "rejected": []}}
    result = _span_quality_metrics(ground_truth, model_results)
    assert result["matched_pairs"] == 0
    assert result["unmatched_ground_truth_spans"] == 1
    assert result["coverage_adjusted_mean_iou"] == 0.0


def test_span_quality_partial_overlap_not_exact_not_zero():
    ground_truth = [
        {"candidate": 1, "status": "labeled",
         "findings": [_gt_finding("threat_or_fear", 0, 20)]},
    ]
    model_results = {1: {"accepted": [_pred_finding("threat_or_fear", 10, 30)], "rejected": []}}
    result = _span_quality_metrics(ground_truth, model_results)
    assert 0 < result["mean_iou"] < 1
    assert result["exact_match_rate"] == 0.0
    assert result["zero_overlap_rate"] == 0.0


def test_span_quality_containment_detected():
    ground_truth = [
        {"candidate": 1, "status": "labeled",
         "findings": [_gt_finding("threat_or_fear", 0, 100)]},
    ]
    model_results = {1: {"accepted": [_pred_finding("threat_or_fear", 10, 20)], "rejected": []}}
    result = _span_quality_metrics(ground_truth, model_results)
    assert result["containment_rate"] == 1.0


def test_span_quality_ignores_unclear_candidates():
    ground_truth = [
        {"candidate": 1, "status": "unclear",
         "findings": [_gt_finding("threat_or_fear", 0, 10)]},
    ]
    model_results = {}
    result = _span_quality_metrics(ground_truth, model_results)
    assert result["matched_pairs"] == 0
    assert result["unmatched_ground_truth_spans"] == 0


# --- _legitimate_false_finding_rate --------------------------------------

def test_legitimate_false_finding_rate_clean_candidate_counts_zero():
    ground_truth = [
        {"candidate": 17, "status": "labeled", "eml_path": "x.eml", "findings": []},
    ]
    model_results = {17: {"accepted": [], "rejected": []}}
    result = _legitimate_false_finding_rate(ground_truth, model_results)
    assert result["legitimate_candidates_evaluated"] == 1
    assert result["total_false_findings"] == 0
    assert result["false_finding_rate_per_candidate"] == 0.0


def test_legitimate_false_finding_rate_flags_unexpected_finding():
    ground_truth = [
        {"candidate": 9, "status": "labeled", "eml_path": "x.eml", "findings": []},
    ]
    model_results = {
        9: {"accepted": [_pred_finding("urgency_or_pressure", 0, 10)], "rejected": []},
    }
    result = _legitimate_false_finding_rate(ground_truth, model_results)
    assert result["total_false_findings"] == 1
    assert result["candidates_with_any_false_finding"] == 1
    assert result["false_finding_rate_per_candidate"] == 1.0
    assert result["details"][0]["candidate"] == 9


def test_legitimate_false_finding_rate_excludes_phishing_axes():
    """A finding on a net_phishing/fraud_or_reward candidate must never
    be counted here, even if it's a real false positive relative to
    ground truth — this metric is specifically about legitimate mail."""
    ground_truth = [
        {"candidate": 1, "status": "labeled", "eml_path": "x.eml", "findings": []},  # net_phishing
    ]
    model_results = {
        1: {"accepted": [_pred_finding("urgency_or_pressure", 0, 10)], "rejected": []},
    }
    result = _legitimate_false_finding_rate(ground_truth, model_results)
    assert result["legitimate_candidates_evaluated"] == 0
    assert result["total_false_findings"] == 0


def test_legitimate_false_finding_rate_matching_type_is_not_false():
    ground_truth = [
        {"candidate": 14, "status": "labeled", "eml_path": "x.eml",
         "findings": [_gt_finding("authority_impersonation", 0, 10)]},  # authority_brand
    ]
    model_results = {
        14: {"accepted": [_pred_finding("authority_impersonation", 5, 15)], "rejected": []},
    }
    result = _legitimate_false_finding_rate(ground_truth, model_results)
    assert result["total_false_findings"] == 0


def test_candidate_axis_covers_all_18():
    assert set(CANDIDATE_AXIS.keys()) == set(range(1, 19))


# --- _run_model_on_all: process-per-candidate isolation, fault
# tolerance and caching. subprocess.run() is mocked — no real worker
# subprocess is ever spawned in these tests, and _run_single_candidate_
# worker() (the function that DOES run in a worker subprocess) is
# tested separately below, also without touching a real model.

def _fake_completed_process(returncode=0, stderr=""):
    cp = MagicMock()
    cp.returncode = returncode
    cp.stderr = stderr
    return cp


def _write_worker_success(out_path: Path, accepted=None, rejected=None):
    out_path.write_text(json.dumps({
        "accepted": accepted or [],
        "rejected": rejected or [],
        "elapsed_seconds": 1.0,
    }), encoding="utf-8")


def _write_worker_error(out_path: Path, message="HATA: geçersiz JSON"):
    out_path.write_text(json.dumps({
        "error": message, "elapsed_seconds": 1.0,
    }), encoding="utf-8")


def test_run_model_on_all_spawns_one_subprocess_per_candidate():
    """The core of the process-per-candidate fix: each candidate must
    get its own subprocess.run() call with --worker-candidate set to
    that candidate's id — not a single long-lived in-process model call."""
    ground_truth = [
        {"candidate": 1, "status": "labeled", "eml_path": "data/x1.eml", "findings": []},
        {"candidate": 2, "status": "labeled", "eml_path": "data/x2.eml", "findings": []},
    ]
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        out_path = Path(cmd[cmd.index("--worker-out") + 1])
        _write_worker_success(out_path)
        return _fake_completed_process()

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = Path(tmpdir) / "cache.json"
        with patch.object(_MODULE, "CACHE_PATH", cache_path), \
             patch.object(_MODULE.subprocess, "run", side_effect=fake_run):
            _run_model_on_all(ground_truth, resume=False)

    assert len(calls) == 2
    worker_candidates = [
        cmd[cmd.index("--worker-candidate") + 1] for cmd in calls
    ]
    assert sorted(worker_candidates) == ["1", "2"]


def test_run_model_on_all_one_failure_does_not_stop_the_batch():
    """The real bug: one candidate's worker failing must not prevent
    the remaining candidates from being spawned."""
    ground_truth = [
        {"candidate": 1, "status": "labeled", "eml_path": "data/x1.eml", "findings": []},
        {"candidate": 2, "status": "labeled", "eml_path": "data/x2.eml", "findings": []},
        {"candidate": 3, "status": "labeled", "eml_path": "data/x3.eml", "findings": []},
    ]

    def fake_run(cmd, **kwargs):
        candidate = cmd[cmd.index("--worker-candidate") + 1]
        out_path = Path(cmd[cmd.index("--worker-out") + 1])
        if candidate == "2":
            _write_worker_error(out_path)
        else:
            _write_worker_success(out_path)
        return _fake_completed_process()

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = Path(tmpdir) / "cache.json"
        with patch.object(_MODULE, "CACHE_PATH", cache_path), \
             patch.object(_MODULE.subprocess, "run", side_effect=fake_run):
            results = _run_model_on_all(ground_truth, resume=False)

    assert 1 in results and "error" not in results[1]
    assert 2 in results and "error" in results[2]
    assert 3 in results and "error" not in results[3]


def test_run_model_on_all_hard_subprocess_crash_is_recorded_not_raised():
    """Regression test for the real Candidate 10-18 mass failure: a
    worker subprocess that crashes hard enough to exit non-zero WITHOUT
    ever writing its output file (e.g. an unhandled RuntimeError from
    mlx_vlm/Metal before the try/except in the worker function even
    runs, or the process being killed) must still be recorded as a
    per-candidate error with diagnostic stderr, not raise in the
    parent or silently produce no result at all."""
    ground_truth = [
        {"candidate": 1, "status": "labeled", "eml_path": "data/x1.eml", "findings": []},
    ]

    def fake_run(cmd, **kwargs):
        # Deliberately does NOT write the --worker-out file, simulating
        # a hard crash before the worker could write anything. The
        # error line is placed near the END (only the last 20 stderr
        # lines are kept, mirroring a real traceback where the useful
        # exception message is at the bottom) so this also exercises
        # that the tail-truncation keeps the actually useful part.
        stderr = "\n".join(f"line {i}" for i in range(30)) + \
            "\nRuntimeError: [METAL] Command buffer execution failed: GPU Timeout Error"
        return _fake_completed_process(returncode=1, stderr=stderr)

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = Path(tmpdir) / "cache.json"
        with patch.object(_MODULE, "CACHE_PATH", cache_path), \
             patch.object(_MODULE.subprocess, "run", side_effect=fake_run):
            results = _run_model_on_all(ground_truth, resume=False)

    assert "error" in results[1]
    assert "exit code 1" in results[1]["error"]
    assert "subprocess_stderr_tail" in results[1]
    assert "GPU Timeout" in results[1]["subprocess_stderr_tail"]


def test_run_model_on_all_resumes_from_cache_without_respawning():
    ground_truth = [
        {"candidate": 1, "status": "labeled", "eml_path": "data/x1.eml", "findings": []},
    ]
    call_count = {"n": 0}

    def fake_run(cmd, **kwargs):
        call_count["n"] += 1
        out_path = Path(cmd[cmd.index("--worker-out") + 1])
        _write_worker_success(out_path, accepted=[
            {"type": "urgency_or_pressure", "evidence": "x", "start": 0, "end": 1,
             "model_confidence": 0.9},
        ])
        return _fake_completed_process()

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = Path(tmpdir) / "cache.json"
        with patch.object(_MODULE, "CACHE_PATH", cache_path), \
             patch.object(_MODULE.subprocess, "run", side_effect=fake_run):
            first = _run_model_on_all(ground_truth, resume=True)
            assert call_count["n"] == 1
            assert cache_path.is_file()

            second = _run_model_on_all(ground_truth, resume=True)
            # no new subprocess spawned — result came from cache
            assert call_count["n"] == 1
            assert second[1]["accepted"][0]["type"] == "urgency_or_pressure"

    assert first[1]["accepted"][0]["type"] == "urgency_or_pressure"


def test_run_model_on_all_resume_retries_previously_failed_candidate():
    """A candidate that failed in a prior run must be retried on
    resume — unlike a successful candidate, a failure has no usable
    result cached, so silently treating it as "done" would permanently
    lose that candidate. This is exactly what happened in the real
    Candidate 10-18 mass-timeout run and needed --no-resume to fully
    clear; resume should handle it automatically instead."""
    ground_truth = [
        {"candidate": 1, "status": "labeled", "eml_path": "data/x1.eml", "findings": []},
    ]
    call_count = {"n": 0}

    def fake_run(cmd, **kwargs):
        call_count["n"] += 1
        out_path = Path(cmd[cmd.index("--worker-out") + 1])
        if call_count["n"] == 1:
            _write_worker_error(out_path)
        else:
            _write_worker_success(out_path)
        return _fake_completed_process()

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = Path(tmpdir) / "cache.json"
        with patch.object(_MODULE, "CACHE_PATH", cache_path), \
             patch.object(_MODULE.subprocess, "run", side_effect=fake_run):
            first = _run_model_on_all(ground_truth, resume=True)
            assert "error" in first[1]
            assert call_count["n"] == 1

            second = _run_model_on_all(ground_truth, resume=True)
            assert call_count["n"] == 2  # retried, not skipped
            assert "error" not in second[1]


def test_run_model_on_all_no_resume_reruns_everything():
    ground_truth = [
        {"candidate": 1, "status": "labeled", "eml_path": "data/x1.eml", "findings": []},
    ]
    call_count = {"n": 0}

    def fake_run(cmd, **kwargs):
        call_count["n"] += 1
        out_path = Path(cmd[cmd.index("--worker-out") + 1])
        _write_worker_success(out_path)
        return _fake_completed_process()

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = Path(tmpdir) / "cache.json"
        with patch.object(_MODULE, "CACHE_PATH", cache_path), \
             patch.object(_MODULE.subprocess, "run", side_effect=fake_run):
            _run_model_on_all(ground_truth, resume=True)
            assert call_count["n"] == 1
            _run_model_on_all(ground_truth, resume=False)
            assert call_count["n"] == 2


def test_run_model_on_all_skips_unclear_without_spawning():
    ground_truth = [
        {"candidate": 1, "status": "unclear", "eml_path": "data/x1.eml", "findings": []},
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = Path(tmpdir) / "cache.json"
        with patch.object(_MODULE, "CACHE_PATH", cache_path), \
             patch.object(_MODULE.subprocess, "run") as mock_run:
            results = _run_model_on_all(ground_truth, resume=False)
    mock_run.assert_not_called()
    assert 1 not in results


# --- _run_single_candidate_worker: the function that DOES run inside a
# worker subprocess. Tested with analyze_semantic mocked, no real model.

def _fake_validated_finding(type_="urgency_or_pressure", evidence="x", start=0, end=1):
    f = MagicMock()
    f.type.value = type_
    f.evidence = evidence
    f.start = start
    f.end = end
    f.model_confidence = 0.9
    return f


def _fake_validation_result(accepted=None, rejected=None):
    r = MagicMock()
    r.accepted = accepted or []
    r.rejected = rejected or []
    return r


def test_worker_writes_success_output():
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "out.json"
        fake_result = _fake_validation_result(accepted=[_fake_validated_finding()])
        with patch("src.semantic.analyze.analyze_semantic", return_value=fake_result), \
             patch("src.parser.parse.parse_eml", return_value=MagicMock()):
            _MODULE._run_single_candidate_worker(1, "data/x1.eml", out_path)
        output = json.loads(out_path.read_text())
    assert output["accepted"][0]["type"] == "urgency_or_pressure"
    assert "error" not in output


def test_worker_writes_error_output_on_exception_not_raise():
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "out.json"
        with patch("src.semantic.analyze.analyze_semantic",
                   side_effect=RuntimeError("GPU Timeout Error")), \
             patch("src.parser.parse.parse_eml", return_value=MagicMock()):
            _MODULE._run_single_candidate_worker(1, "data/x1.eml", out_path)
        output = json.loads(out_path.read_text())
    assert "error" in output
    assert "RuntimeError" in output["error"]
    assert "traceback" in output


def test_failed_candidates_excluded_from_type_level_metrics():
    """A failed candidate must be treated exactly like "unclear" by
    every downstream metric — main()'s job (not _type_level_metrics'
    directly) is filtering "error" entries out before calling these,
    so this test exercises that model_results simply omitting a failed
    candidate produces the same result as it never having run."""
    ground_truth = [
        {"candidate": 1, "status": "labeled",
         "findings": [_gt_finding("credential_request", 0, 10)]},
    ]
    # Simulates main()'s filtering: candidate 1 failed, so it's absent
    # from model_results entirely (not present with an "error" key).
    model_results = {}
    metrics = _type_level_metrics(ground_truth, model_results)
    assert metrics["micro"]["tp"] == 0
    assert metrics["micro"]["fp"] == 0
    assert metrics["micro"]["fn"] == 0  # candidate 1 not counted at all


if __name__ == "__main__":
    import traceback

    tests = [(name, obj) for name, obj in list(globals().items())
              if name.startswith("test_") and callable(obj)]
    passed, failed = 0, 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS: {name}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL: {name}: {e}")
            failed += 1
        except Exception:
            print(f"ERROR: {name}")
            traceback.print_exc()
            failed += 1

    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
