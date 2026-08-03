"""
v3 plan Adım 9 — converts data/training/pairs.jsonl (Adım 7's teacher
output) into the {"messages": [...]} chat format mlx_lm.lora's
ChatDataset expects (see mlx_lm/LORA.md#Data), split into
train/valid/test under data/training/lora_data/.

Only status="ok" records are used — the 1 dropped example from Adım 7
is excluded. messages is built via src/teacher/prompts.py's
build_messages(), WITHOUT few-shot examples (unlike teacher generation,
which needed them to steer an untrained model's format/tone) — LoRA
learns the facts->completion mapping directly into the adapter's
weights, so repeating 3 full-report few-shot examples in every training
example only bloated it to ~8000 tokens (vs max_seq_length=4096).

IMPORTANT: this must stay raw {"messages": [...]}, NOT a pre-rendered
"prompt" string. An earlier version rendered messages through
tokenizer.apply_chat_template(tokenize=False) into a "prompt" field
and paired it with a "completion" field, matching mlx_lm's
CompletionsDataset format — but CompletionsDataset re-wraps prompt as
a *user message* and applies the chat template a SECOND time, nesting
one rendered chat inside another. That produced garbled token
sequences and silently zeroed out training (Train loss 0.000,
Trained Tokens 0 for the entire run). ChatDataset's {"messages": [...]}
format applies the template exactly once and (with mask_prompt: true
in config/lora.yaml) masks everything but the assistant turn out of
the loss, which is what we want: the model should only be scored on
producing the JSON, not on reproducing the prompt.

completion (the assistant turn's content) is the teacher's raw JSON
string — the "report" field's raw serialization would round-trip
differently than the exact bytes the teacher emitted, so this uses
model_dump_json() on the validated Report for a clean, canonical
target rather than the teacher's raw text, which could contain code
fences/stray whitespace an already-validated schema doesn't need to
reproduce.

Split: 90% train, 10% valid, per plan §7.1. No held-out "test.jsonl"
content beyond what mlx_lm.lora's --test flag needs structurally (an
empty list still satisfies the loader) — the actual test set for this
project is data/holdout/candidates.jsonl, evaluated separately in
Adım 10, not through mlx_lm's own --test mechanism.

Usage:
    python3 src/teacher/prepare_lora_data.py
"""
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from schemas.facts import EmailFacts  # noqa: E402
from schemas.report import Report  # noqa: E402
from src.rules.engine import evaluate, load_rules  # noqa: E402
from src.teacher.generate_training_data import _load_facts_pool  # noqa: E402
from src.teacher.prompts import build_messages  # noqa: E402

PAIRS_PATH = PROJECT_ROOT / "data" / "training" / "pairs.jsonl"
OUT_DIR = PROJECT_ROOT / "data" / "training" / "lora_data"

VAL_FRACTION = 0.10
SEED = 7


def main():
    rules = load_rules()

    pool = _load_facts_pool()
    pool_by_path = {row["_eml_path"]: row for row in pool}

    with open(PAIRS_PATH, encoding="utf-8") as f:
        pairs = [json.loads(line) for line in f]
    pairs = [p for p in pairs if p["status"] == "ok"]
    print(f"{len(pairs)} usable training pairs (dropped excluded).")

    examples = []
    missing = 0
    for pair in pairs:
        row = pool_by_path.get(pair["eml_path"])
        if row is None:
            missing += 1
            continue
        facts_dict = {k: v for k, v in row.items() if k != "_eml_path"}
        facts = EmailFacts(**facts_dict)
        verdict = evaluate(facts.flat_signals(), rules)

        # No few-shot: see module docstring.
        messages = build_messages(facts, verdict, few_shot=None)
        report = Report(**pair["report"])
        completion = report.model_dump_json(indent=2)
        messages.append({"role": "assistant", "content": completion})

        examples.append({"messages": messages})

    if missing:
        print(f"WARNING: {missing} pairs had no matching facts row in the pool, skipped.")

    rng = random.Random(SEED)
    rng.shuffle(examples)
    n_val = max(1, round(len(examples) * VAL_FRACTION))
    valid_set = examples[:n_val]
    train_set = examples[n_val:]

    # mlx_lm.lora's loader unconditionally indexes test.jsonl[0] even when
    # --test is never passed (create_dataset does `sample = data[0]`), so
    # an empty file crashes the run. This project's real test set is
    # data/holdout/candidates.jsonl (Adım 10), not mlx_lm's own --test
    # mechanism — reusing the valid split here only satisfies that loader
    # requirement, it is never read for an actual held-out metric.
    test_set = valid_set

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, split in (("train", train_set), ("valid", valid_set), ("test", test_set)):
        out_path = OUT_DIR / f"{name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for ex in split:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"{name}: {len(split)} examples -> {out_path}")


if __name__ == "__main__":
    main()
