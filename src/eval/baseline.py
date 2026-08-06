"""
v3 plan Adım 8 — baseline measurement. Runs the NOT-yet-fine-tuned Seneca
(SenecaLLM_x_Qwen2.5-7B-CyberSecurity, loaded via mlx_lm — plain text
model, unlike the teacher's mlx_vlm) over the full 30-email hold-out with
the same prompt structure the teacher uses (src/teacher/prompts.py), and
records the plan's §7.3 metrics BEFORE LoRA training (Adım 9) so there's
something to compare the fine-tuned result against in Adım 10.

Deliberately does NOT retry on JSON/schema failure the way
src/teacher/generate.py does — the whole point of a baseline is to see
how often the un-tuned model fails on its own, not to paper over it with
a second attempt. One shot per email, temperature=0.

Metrics (CLAUDE.md "Değerlendirme metrikleri", plan §7.3), reported
separately, never blended:
1. schema validity — % of outputs that parse as JSON and satisfy Report
2. groundedness — via src/eval/groundedness.py, only computable for
   schema-valid outputs (an invalid JSON has no claims to check)
3. Turkish quality — NOT scored here; this script writes the raw model
   outputs to a file for a human to rate 1-5 against schemas/report.py's
   30 hand-written few-shot-quality references (there are only 3 of
   those — src/teacher/few_shot_examples.py — so this is a coarse,
   manual pass, not an automated score)
4. classification accuracy — explicitly NOT this script's job (that's
   the rule engine's metric, already measured in Adım 4); the verdict
   given to the model IS the rule engine's decision, this script only
   checks whether the model's risk_seviyesi field echoes it back
   correctly (part of schema/instruction-following, not a separate
   accuracy claim)

Usage:
    python3 src/eval/baseline.py
"""
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
from src.parser.parse import parse_eml  # noqa: E402
from src.rules.engine import evaluate, load_rules  # noqa: E402
from src.teacher.few_shot_examples import (  # noqa: E402
    FEW_SHOT_GUVENILIR,
    FEW_SHOT_MUHTEMEL,
    FEW_SHOT_MUHTEMEL_EML_PATH,
    FEW_SHOT_PHISHING,
)
from src.teacher.prompts import build_messages  # noqa: E402

MODEL_PATH = PROJECT_ROOT / "models" / "Seneca-Cybersecurity-LLM_x_Qwen2.5-7B-CyberSecurity-mlx-4bit"
CANDIDATES_PATH = PROJECT_ROOT / "data" / "holdout" / "candidates.jsonl"
OUT_PATH = PROJECT_ROOT / "data" / "training" / "baseline_seneca.jsonl"

METADATA_KEYS = ("source_label", "_eml_path", "is_spam_not_phishing", "spam_reason")
FEW_SHOT_INDICES = {1, 20}
FEW_SHOT_REPORTS = {1: FEW_SHOT_PHISHING, 20: FEW_SHOT_GUVENILIR}

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


def main():
    rules = load_rules()
    candidates = _load_candidates()

    few_shot = []
    for idx in sorted(FEW_SHOT_INDICES):
        facts, verdict = _facts_and_verdict(candidates[idx - 1], rules)
        few_shot.append((facts, verdict, FEW_SHOT_REPORTS[idx]))
    muhtemel_facts = parse_eml(PROJECT_ROOT / FEW_SHOT_MUHTEMEL_EML_PATH)
    few_shot.append((muhtemel_facts, evaluate(muhtemel_facts.flat_signals(), rules), FEW_SHOT_MUHTEMEL))

    eval_indices = [i for i in range(1, len(candidates) + 1) if i not in FEW_SHOT_INDICES]

    print(f"Loading Seneca (NOT fine-tuned) from {MODEL_PATH} ...", file=sys.stderr)
    model, tokenizer = load(str(MODEL_PATH))
    print("Model loaded.", file=sys.stderr)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    schema_valid = 0
    verdict_matches = 0
    groundedness_ratios = []
    results = []

    with open(OUT_PATH, "w", encoding="utf-8") as out_f:
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
                    signals = facts.flat_signals()
                    grounded = check_claims(claims, signals)
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
            results.append(record)

            status = "SCHEMA_VALID" if record["schema_valid"] else "INVALID"
            print(f"[{n}/{len(eval_indices)}] candidate {idx}: {status} ({elapsed:.1f}s)",
                  file=sys.stderr)

    total = len(eval_indices)
    print(f"\n=== Adım 8 baseline sonucu (Seneca, fine-tune EDİLMEMİŞ) ===")
    print(f"n = {total}")
    print(f"1. Şema uyumu: {schema_valid}/{total} = {schema_valid/total:.1%}")
    if schema_valid:
        print(f"   (bunların {verdict_matches}/{schema_valid} = {verdict_matches/schema_valid:.1%}'i "
              f"risk_seviyesi'ni doğru yansıttı)")
    if groundedness_ratios:
        avg_ground = sum(groundedness_ratios) / len(groundedness_ratios)
        print(f"2. Ortalama groundedness (şema-geçerli {len(groundedness_ratios)} örnek üzerinde): "
              f"{avg_ground:.1%}")
    else:
        print("2. Groundedness: hesaplanamadı (hiç şema-geçerli örnek yok)")
    print(f"3. Türkçe kalite: elle değerlendirilecek — çıktı {OUT_PATH}")
    print(f"4. Sınıflandırma doğruluğu: bu script'in metriği DEĞİL — bkz. Adım 4 (rule engine)")
    print(f"\nÇıktı: {OUT_PATH}")


if __name__ == "__main__":
    main()
