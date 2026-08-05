"""
v3 plan Adım 10 — post-fine-tuning evaluation. Runs the LoRA-adapted
Seneca over the same hold-out emails, with the same prompt structure and
the same metrics as src/eval/baseline.py (Adım 8), so the two numbers are
directly comparable.

Everything that could bias the comparison is held identical to the
baseline run: same 27 evaluation emails (the 3 few-shot indices are
excluded from scoring in both), same temperature=0, same max_tokens,
same groundedness checker, and NO retry on failure — one shot per email,
because the point is to measure how often the model succeeds on its own.

TWO THINGS DIFFER FROM THE BASELINE, both deliberate:

1. The adapter is loaded (`load(model, adapter_path=...)`).

2. `--prompt-mode` selects whether few-shot examples are included, and
   this is the subtle one. The baseline was measured WITH few-shot
   (3 examples in the prompt), but LoRA training used
   `build_messages(..., few_shot=None)` — the adapter learned the
   facts→JSON mapping from a bare prompt. So:

     --prompt-mode few-shot  → matches the BASELINE's conditions.
                               Use this for the honest A/B comparison.
     --prompt-mode bare      → matches the ADAPTER's TRAINING conditions.
                               Use this to see what the adapter actually
                               learned to do.
     --prompt-mode both      → runs both (default).

   Reporting only one of these would be misleading in opposite
   directions: few-shot alone understates the adapter (it is being fed a
   prompt shape it never trained on), bare alone overstates the
   improvement (the baseline never got a bare-prompt score to lose to).
   `both` is the default so the comparison table is complete.

Metrics (CLAUDE.md "Değerlendirme metrikleri", plan §7.3) — reported
separately, never blended into a single number:
1. schema validity — % of outputs that parse as JSON and satisfy Report
2. groundedness — via src/eval/groundedness.py, computable only for
   schema-valid outputs
3. Turkish quality — NOT scored here; raw outputs are written to disk for
   a human 1-5 rating pass
4. classification accuracy — explicitly NOT this script's metric (that
   belongs to the rule engine, Adım 4). The verdict handed to the model
   IS the rule engine's decision; all that is checked here is whether the
   model echoes it back in risk_seviyesi, which is instruction-following,
   not detection accuracy.

Usage:
    # both checkpoints, both prompt modes (~3h on an M2 Air)
    caffeinate -dims python3 src/eval/finetuned.py 2>&1 | tee logs/eval_step10.log

    # a single checkpoint / mode
    python3 src/eval/finetuned.py --checkpoints 0000250 --prompt-mode few-shot
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from mlx_lm import generate, load  # noqa: E402

from schemas.facts import EmailFacts  # noqa: E402
from schemas.report import Report  # noqa: E402
from src.eval.groundedness import check_claims  # noqa: E402
from src.rules.engine import evaluate, load_rules  # noqa: E402
from src.teacher.few_shot_examples import (  # noqa: E402
    FEW_SHOT_GUVENILIR,
    FEW_SHOT_PHISHING,
)
from src.teacher.prompts import build_messages  # noqa: E402

MODEL_PATH = PROJECT_ROOT / "models" / "Seneca-Cybersecurity-LLM_x_Qwen2.5-7B-CyberSecurity-mlx-4bit"
ADAPTER_DIR = PROJECT_ROOT / "models" / "lora_adapters"
CANDIDATES_PATH = PROJECT_ROOT / "data" / "holdout" / "candidates.jsonl"
OUT_DIR = PROJECT_ROOT / "data" / "training"

# Adım 8 baseline (un-fine-tuned Seneca, few-shot prompt, 27 emails).
# Hardcoded so the comparison table prints without re-running the baseline;
# see PROGRESS.md "v3 Adım 8".
BASELINE = {
    "n": 27,
    "schema_valid_ratio": 1.0,
    "groundedness": 0.674,
}

METADATA_KEYS = ("source_label", "_eml_path", "is_spam_not_phishing", "spam_reason")
FEW_SHOT_INDICES = {1, 20}
FEW_SHOT_REPORTS = {1: FEW_SHOT_PHISHING, 20: FEW_SHOT_GUVENILIR}

# Checkpoints worth comparing: the val-loss minimum and the final weights.
#
# The val-loss minimum was measured AT iter 250 (1.393), but checkpoints are
# written every 20 iterations (save_every: 20) — there is no 0000250 file.
# 0000240 is the nearest saved checkpoint before that measurement, so it is
# the closest thing to "the weights that scored 1.393".
#
# NOTE: 0000240 is NOT assumed to be "the best". Val loss was computed over
# only 5 examples (val_batches: 5), so the 1.393 vs 1.424 gap is well within
# noise. That is precisely why both are measured here with the real metrics.
DEFAULT_CHECKPOINTS = ("0000240", "0000400")

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _load_candidates() -> list[dict]:
    with open(CANDIDATES_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _facts_and_verdict(cand: dict, rules: dict):
    facts_dict = {k: v for k, v in cand.items() if k not in METADATA_KEYS}
    facts = EmailFacts(**facts_dict)
    verdict = evaluate(facts.flat_signals(), rules)
    return facts, verdict


def _extract_json(raw_text: str) -> dict | None:
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[len("json"):]
    match = _JSON_BLOCK_RE.search(text)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _resolve_adapter(checkpoint: str) -> Path:
    """Map a checkpoint id to a directory mlx_lm can load.

    mlx_lm's `adapter_path` expects a DIRECTORY containing
    adapter_config.json plus adapters.safetensors — it will not take a
    bare .safetensors file. Intermediate checkpoints are saved as
    `0000250_adapters.safetensors` alongside a single shared
    adapter_config.json, so each one is staged into its own subdirectory
    with the weights renamed to the expected filename.
    """
    if checkpoint in ("final", "adapters"):
        weights = ADAPTER_DIR / "adapters.safetensors"
    else:
        weights = ADAPTER_DIR / f"{checkpoint}_adapters.safetensors"
    if not weights.is_file():
        raise SystemExit(f"checkpoint not found: {weights}")

    config = ADAPTER_DIR / "adapter_config.json"
    if not config.is_file():
        raise SystemExit(f"adapter_config.json missing in {ADAPTER_DIR}")

    staged = ADAPTER_DIR / f"_eval_{checkpoint}"
    staged.mkdir(exist_ok=True)
    (staged / "adapter_config.json").write_bytes(config.read_bytes())
    target = staged / "adapters.safetensors"
    # Re-copy only when stale; these files are ~88MB each.
    if not target.is_file() or target.stat().st_size != weights.stat().st_size:
        target.write_bytes(weights.read_bytes())
    return staged


def run_one(model, tokenizer, candidates, rules, few_shot, eval_indices,
            out_path: Path, label: str) -> dict:
    schema_valid = 0
    verdict_matches = 0
    groundedness_ratios = []

    with open(out_path, "w", encoding="utf-8") as out_f:
        for n, idx in enumerate(eval_indices, start=1):
            cand = candidates[idx - 1]
            facts, verdict = _facts_and_verdict(cand, rules)
            messages = build_messages(facts, verdict, few_shot=few_shot)
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            t0 = time.time()
            raw = generate(model, tokenizer, prompt, max_tokens=1200, verbose=False)
            elapsed = time.time() - t0

            parsed = _extract_json(raw)
            record = {
                "run": label,
                "candidate_index": idx,
                "eml_path": cand["_eml_path"],
                "verdict": verdict.verdict,
                "score": verdict.score,
                "elapsed_sec": round(elapsed, 1),
                "raw": raw,
            }

            if parsed is None:
                record["schema_valid"] = False
                record["reason"] = "invalid_json"
            else:
                try:
                    report = Report(**parsed)
                    record["schema_valid"] = True
                    schema_valid += 1
                    record["report"] = report.model_dump()
                    record["risk_seviyesi_matches_verdict"] = (
                        report.risk_seviyesi == verdict.verdict
                    )
                    if record["risk_seviyesi_matches_verdict"]:
                        verdict_matches += 1

                    claims = [f.aciklama for f in report.teknik_bulgular] + report.phishing_gostergeleri
                    grounded = check_claims(claims, facts.flat_signals())
                    record["groundedness_ratio"] = grounded["ratio"]
                    record["groundedness_detail"] = {
                        "grounded_claims": grounded["grounded_claims"],
                        "total_claims": grounded["total_claims"],
                    }
                    groundedness_ratios.append(grounded["ratio"])
                except Exception as e:
                    record["schema_valid"] = False
                    record["reason"] = f"schema_error: {e}"

            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()

            status = "SCHEMA_VALID" if record["schema_valid"] else "INVALID"
            print(f"  [{label}] [{n}/{len(eval_indices)}] cand {idx}: {status} ({elapsed:.1f}s)",
                  file=sys.stderr)

    total = len(eval_indices)
    return {
        "label": label,
        "n": total,
        "schema_valid": schema_valid,
        "schema_valid_ratio": schema_valid / total if total else 0.0,
        "verdict_matches": verdict_matches,
        "groundedness": (
            sum(groundedness_ratios) / len(groundedness_ratios)
            if groundedness_ratios else None
        ),
        "groundedness_n": len(groundedness_ratios),
        "out_path": str(out_path),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", default=",".join(DEFAULT_CHECKPOINTS),
                    help="comma-separated checkpoint ids, e.g. 0000250,0000400 or 'final'")
    ap.add_argument("--prompt-mode", default="both",
                    choices=("few-shot", "bare", "both"),
                    help="few-shot = baseline's conditions; bare = adapter's training conditions")
    args = ap.parse_args()

    checkpoints = [c.strip() for c in args.checkpoints.split(",") if c.strip()]
    modes = ["few-shot", "bare"] if args.prompt_mode == "both" else [args.prompt_mode]

    rules = load_rules()
    candidates = _load_candidates()

    few_shot_examples = []
    for idx in sorted(FEW_SHOT_INDICES):
        facts, verdict = _facts_and_verdict(candidates[idx - 1], rules)
        few_shot_examples.append((facts, verdict, FEW_SHOT_REPORTS[idx]))

    eval_indices = [i for i in range(1, len(candidates) + 1) if i not in FEW_SHOT_INDICES]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    total_runs = len(checkpoints) * len(modes)
    print(f"Plan: {len(checkpoints)} checkpoint × {len(modes)} prompt-mode = {total_runs} run, "
          f"{len(eval_indices)} e-posta/run", file=sys.stderr)

    summaries = []
    for checkpoint in checkpoints:
        adapter_dir = _resolve_adapter(checkpoint)
        print(f"\nLoading Seneca + adapter {checkpoint} ...", file=sys.stderr)
        model, tokenizer = load(str(MODEL_PATH), adapter_path=str(adapter_dir))
        print("Model loaded.", file=sys.stderr)

        for mode in modes:
            label = f"{checkpoint}/{mode}"
            few_shot = few_shot_examples if mode == "few-shot" else None
            out_path = OUT_DIR / f"finetuned_{checkpoint}_{mode.replace('-', '')}.jsonl"
            print(f"\n--- {label} ---", file=sys.stderr)
            summaries.append(
                run_one(model, tokenizer, candidates, rules, few_shot,
                        eval_indices, out_path, label)
            )

        # Drop references before loading the next checkpoint's weights;
        # two 7B models resident at once will not fit in 16GB.
        del model, tokenizer

    print("\n\n=== Adım 10 — fine-tune sonrası değerlendirme ===")
    print(f"Hold-out: {len(eval_indices)} e-posta (few-shot olarak kullanılan "
          f"{sorted(FEW_SHOT_INDICES)} hariç)\n")

    print(f"{'run':<22} {'şema uyumu':>14} {'groundedness':>14}")
    print("-" * 52)
    print(f"{'BASELINE (fine-tune yok)':<22} "
          f"{BASELINE['schema_valid_ratio']:>13.1%} "
          f"{BASELINE['groundedness']:>13.1%}")
    for s in summaries:
        ground = f"{s['groundedness']:.1%}" if s["groundedness"] is not None else "—"
        print(f"{s['label']:<22} "
              f"{s['schema_valid_ratio']:>13.1%} "
              f"{ground:>14}")

    print("\nNotlar:")
    print("  - few-shot modu baseline ile AYNI koşul → adil A/B kıyası bu satır.")
    print("  - bare modu adapter'ın EĞİTİM koşulu (LoRA few-shot görmedi).")
    print("  - groundedness ham değer; baseline'ın %85.2'lik 'düzeltilmiş'")
    print("    rakamı, check_claims'in tanımadığı ifadeler elendikten sonraydı")
    print("    (bkz. PROGRESS.md 'v3 Adım 8'). Kıyas ham-ham yapılmalı.")
    print("  - Türkçe kalite elle değerlendirilecek; çıktılar:")
    for s in summaries:
        print(f"      {s['out_path']}")
    print("  - Sınıflandırma doğruluğu bu script'in metriği DEĞİL (bkz. Adım 4).")


if __name__ == "__main__":
    main()
