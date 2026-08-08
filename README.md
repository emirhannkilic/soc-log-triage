# soc-log-triage

[English](README.md) | [Türkçe](README.tr.md)

A proof-of-concept phishing triage pipeline: a deterministic parser and rule
engine decide whether an email is phishing, an optional semantic layer reads
the body text for signals the rule engine structurally cannot see, and a
local Qwen3.5-9B model writes at most a short narrative for the analyst-facing
report. **The model never classifies, in either mode.**

---

## Pipeline (current)

```
.eml file, or a raw message pasted as text
    │
    ▼
Router               structural: file extension, or ≥3 RFC 5322 headers
    │
    ▼
Parser                email stdlib + BeautifulSoup → EmailFacts
    │
    ▼
Rule Engine            weighted score, config/rules.yaml → RuleAssessment
    │
    ▼
Semantic Extractor      (hybrid mode only) Qwen3.5-9B reads the body text
(hybrid)                for signals the rule engine cannot see — findings
    │                   are typed, not a verdict
    ▼
Evidence Validator      every finding must be an exact substring of the
                        body; anything the model paraphrased or invented
                        is rejected before it can reach a decision
    │
    ▼
Deterministic           combines RuleAssessment + validated findings into
Decision Policy         final_verdict — a fixed, auditable rule set, not
                        a model call
    │
    ▼
Deterministic Report    risk_seviyesi, category rationale, technical
                        findings, and the recommended SOC action are ALL
                        built mechanically — the model has no write
                        access to any of them, in either mode
    │
    ▼ (only if final_verdict != "Güvenilir")
Optional Qwen           three short sentences — likely scenario, what the
Narrative               email asks the recipient to do, likely harm — slotted
                        into one field of an otherwise-complete report
    │
    ▼
Jinja2 template          → HTML report
```

Two modes reach the same report shape through different amounts of this
pipeline:

| | fast | hybrid |
|---|---|---|
| Stages run | Router → Parser → Rule Engine → Deterministic Report | all of the above |
| Model calls | 0 | up to 2 (semantic extraction + narrative) |
| Time (M2 Air) | ~1 second | ~60–270 seconds |
| When the model is skipped | always | `rule_verdict` already "Phishing" (semantic call); `final_verdict` is "Güvenilir" (narrative call) |
| Default | **yes** | opt-in only |

`fast` is the default in both the CLI (`src/demo.py`) and the web UI
(`src/web.py`) — a user has to explicitly ask for `hybrid` before a model is
loaded at all.

**What "the model never classifies" means concretely:** `risk_seviyesi`
(Phishing / Muhtemel Phishing / Güvenilir), the category rationale, every
technical finding, and the recommended SOC action are produced by
`src/report/mechanical.py` — deterministic, template-driven, identical in
both modes. Qwen's only possible contribution, in hybrid mode, when the
verdict is not "Güvenilir", is three sentence fragments
(`schemas/narrative.py::NarrativeDraft`) that
`src/report/assemble.py::apply_narrative()` slots into exactly one field
(`genel_degerlendirme`), replacing generic fallback text with a more specific
one. **That narrative text is not treated as reliable output on its own** —
it carries no weight in the decision, is schema-constrained to three fixed
fields (no room for a smuggled category or verdict claim), and if the model's
call fails or produces anything off-schema, the report keeps its mechanical
fallback text with no retry and no repair (`narrative_status="failed_fallback"`,
visible on the result, never silently hidden).

---

## The core design decision

Most "LLM for phishing detection" demos ask a language model to read a raw
email and output a verdict. This project deliberately does not, because that
approach was tried here first and it failed measurably — twice, in two
different ways, at two different stages.

**First failure (v2, classification):** 70 real emails fed to a 4-bit
quantized 7B model, asked for a classification directly. **35% accuracy on a
3-class problem — worse than random.** Worse than the number itself was the
failure mode: when the model could not determine a verdict, it invented
supporting evidence. In one case it labelled an email as phishing and
justified it by citing an "unusual X-Mailer header" — the email had no
X-Mailer header at all.

**Second failure (hybrid v1, category rationale):** even after classification
moved entirely to the deterministic rule engine + decision policy, the report
writer was still asked to pick a rationale category (a closed six-item
vocabulary) for **why** a verdict held. Development-set measurement found the
model abandoning that fixed vocabulary in 9 of 18 candidates — 69% of them
"Güvenilir" verdicts, where justifying a *clean* result through an
attack-shaped category list consistently pushed the model to invent phrasing
outside it. The fix was the same lesson applied one layer deeper: the
category rationale is now built mechanically too
(`src/report/categories.py`), and the model's only remaining surface is three
narrative sentences with no vocabulary to violate (see the Pipeline section
above).

The root cause, both times, is not model size. Signals like SPF/DKIM/DMARC
results and domain string comparisons are **outputs of a deterministic
protocol**: `pass`/`fail`/`none`, two strings being equal or not. Which
category a fixed rule fired under is equally deterministic — it was already
computed by the rule engine before the model ever saw it. Asking a language
model to reproduce either kind of answer is the wrong tool for the category,
no matter how large the model.

Two rules follow from this and are enforced throughout, in both modes:

1. **The LLM does not classify.** `risk_seviyesi`, the category rationale,
   every technical finding, and the recommended action are built by
   `src/report/mechanical.py` before any model is ever called. In hybrid
   mode, an accepted semantic finding can only move the verdict through a
   fixed, auditable decision-policy rule (`src/decision/phishing_policy.py`)
   — never through the report-writing model itself.
2. **The LLM emits JSON, not free text or HTML.** Every model call — semantic
   extraction, narrative generation — is schema-validated
   (`schemas/semantic.py`, `schemas/narrative.py`) before its output can
   reach a report. Anything off-schema is dropped, not repaired; the earlier
   Seneca+LoRA report writer (see "Earlier iteration" below) had 9 of 70
   outputs with no parseable classification at all and 5 different output
   formats — schema validation eliminates that class of problem by
   construction rather than patching around it.

This is a **demo / proof-of-concept**, not a production system. The models were
chosen to run on a 16 GB fanless laptop. The goal is an honest, measurable
demonstration of the approach — not the best achievable accuracy.

### Earlier iteration: Seneca + LoRA report writer (v3, superseded)

Before the hybrid pipeline above, this project fine-tuned a local 7B model
(Seneca-Cybersecurity-LLM) with a LoRA adapter to write the Turkish report
from `facts + verdict`, with no semantic extraction step and no decision
policy — the rule engine's verdict was final, and the model's only job was
prose. That path (`src/demo.py`'s `--adapter`/`--constrain`/`--no-llm` flags,
`src/teacher/`) still exists in the codebase and is documented below for
reference, but it is **not connected to the hybrid pipeline** — `--hybrid`
and `--adapter` are mutually exclusive on the CLI, and the web UI only
exposes the hybrid path. The LoRA results below (overfitting, the baseline
comparison) are historical record of why that specific fine-tuning attempt
did not carry its weight, not a claim about the current pipeline.

---

## Results

### Hybrid pipeline: does the semantic layer help?

Measured by re-analyzing 18 already-cached hybrid runs (real Qwen3.5-9B, from
an earlier reliability measurement) against their source label (phishing
corpus vs. legitimate mailbox export) — **not a new model run**, and this is
the same 18-email development set `src/semantic/analyze.py`'s prompt was
iterated against, so these numbers describe observed behaviour on that set,
not an independent benchmark:

| Metric | Value |
|---|---|
| Decisions changed by the semantic layer | 1/18 |
| Wrong-direction upgrades (legitimate mail pushed past "Güvenilir") | 0/18 in this sample |
| Correct-direction upgrades (phishing mail moved off "Güvenilir") | 1/18 |
| Semantic/model error rate | 0/18 of the cached records processed cleanly |
| Hybrid latency, recorded in the earlier runs | 60–268 s (median 139 s) |
| Fast-mode latency | ~1 s |

**Semantic extraction upgraded the one phishing sample the rule engine had
missed, and produced no wrong-direction upgrade in this sample.** The
upgraded email had SPF/DKIM/DMARC all passing (so the rule engine alone
called it "Güvenilir") while its body contained a directly quoted, validated
credential request — exactly the blind spot the rule engine has by
construction (see "Known limitations" below). Zero wrong-direction upgrades
in 18 emails is a description of this sample, not a bound on the true
rate — a much larger, independently-drawn set would be needed to claim a
population-level false-upgrade rate. **This development set is used for
observation only, never for recalibrating the rule engine or the semantic
prompt** — doing so would fit the pipeline to the set it is being measured
against rather than to the general problem.

### Rule engine (the component that actually classifies)

Measured on a hand-labelled hold-out set of **80 emails** (15 phishing, 65
legitimate), with 22 weighted signals and thresholds at ≥5 / 3–4 / <3:

| Metric | Value | Meaning |
|---|---|---|
| Recall | 86.7% (13/15) | Phishing caught above the upper threshold |
| False-positive rate | 12.3% (8/65) | Legitimate mail wrongly flagged |
| Abstention rate | 7.5% (6/80) | Landed mid-band, deferred to an analyst |

A single "accuracy" figure is deliberately not reported: the engine emits
three classes while ground truth is binary, and the mid-band is abstention —
in a SOC context correct behaviour, not error.

#### How the false-positive rate went from "0.0%" to 12.3%

An earlier version of this README reported **0.0% false positives**. That
number was measured on 15 legitimate emails, and it was wrong — not
miscalculated, but meaningless at that sample size. It is left described here
rather than quietly replaced, because how it broke is the more useful result.

The caveat at the time was that a Wilson 95% interval put the upper bound near
20%. Expanding the legitimate side to 65 hand-labelled emails put the real
figure at **26.2%** — above even that bound.

The emails it got wrong were not obscure: `google.com`, `email.openai.com`,
`discord.com`, `client.louisvuitton.com`, `tr-info.adidas.com`. Sixteen of the
seventeen had SPF, DKIM and DMARC all passing with a matching DKIM domain.
They were flagged because the one signal that rewards a verified sender
(`all_auth_pass_and_consistent`, −3) also required Return-Path to match From —
and every sender using an email service provider routes bounces through the
provider's domain. Legitimate bulk mail could not earn the bonus by
construction.

Two fixes brought it to 12.3%: dropping the Return-Path condition from that
bonus, and adding a signal for a *valid* DKIM signature from the wrong domain
(third-party spoofing, which the "missing or failing DKIM" rule never covered).

**Both were calibrated on a separate 60-email dev set, never on the hold-out.**
Tuning against the hold-out would have turned it into a training set. The dev
set landed at 10.0% and the hold-out at 12.3% — close enough to suggest a real
improvement rather than a fit to one sample.

#### What the numbers still are, and are not

The original weights and thresholds were tuned on the first 30 emails, so
those remain a **calibration** result. The 50 legitimate emails added later
were never used for tuning, which makes the false-positive figure the closest
thing here to an independent measurement. Recall is still measured on the
original 15 phishing emails and carries a correspondingly wide interval.

### LoRA fine-tuning (the report writer)

229 training pairs generated by the teacher model, 206 train / 23 validation,
400 iterations (≈1.94 epochs).

**The adapter overfit.** Validation loss never dropped below its starting
point:

| Iter | 1 | 50 | 100 | 150 | 200 | 250 | 300 | 350 | 400 |
|---|---|---|---|---|---|---|---|---|---|
| Val loss | 1.301 | 1.343 | 1.491 | 1.462 | 1.406 | **1.393** | 1.535 | 1.439 | 1.424 |

Training loss sat at 0.000 from iteration 40 onward. With 206 samples, ~576
token targets, `batch_size=1`, and 23M trainable parameters, the model
memorised the training set rather than generalising.

This is recorded as a finding, not hidden. Validation loss is not this
project's success metric — the evaluation criteria are schema compliance,
groundedness, and Turkish quality (see below), and a memorising model can
still apply a format correctly. The margin for improvement was narrow to begin
with: the **un-fine-tuned** baseline already scored 100% schema compliance.

### Baseline (un-fine-tuned Seneca, for comparison)

| Metric | Result |
|---|---|
| Schema compliance | 27/27 = 100% |
| Groundedness (raw) | 67.4% |
| Groundedness (adjusted) | 85.2% |

The gap between raw and adjusted groundedness is a limitation of the checker's
phrase dictionary, not the model: 28 of 41 "ungrounded" claims used wording the
regex simply does not recognise. The real hallucination rate — claims the
checker understood and the facts contradict — is ~15% (13/88).

---

## Evaluation criteria

Because the LLM does not classify, classification accuracy is the *parser's*
metric and is reported separately. The model is judged on:

1. **Schema compliance** — how much of the output is valid JSON matching the
   report schema.
2. **Groundedness** — programmatic check that every technical claim in the
   report has a counterpart in `facts` (`src/eval/groundedness.py`).
3. **Turkish quality** — manual 1–5 rubric. BLEU/ROUGE are not used; they
   measure overlap with a reference, not whether a security report reads
   correctly to an analyst.
4. **Classification accuracy** — the parser's metric, reported separately.

---

## Repository layout

```
config/
  rules.yaml             scoring weights and thresholds (not hardcoded)
  lora.yaml              LoRA hyperparameters (earlier iteration, see below)
schemas/
  facts.py               EmailFacts — the parser's output contract
  report.py              Report — the final, always-complete report contract
  semantic.py            SemanticFindingCandidate / ValidatedSemanticFinding
  decision.py            PhishingDecisionContext / FinalDecision
  narrative.py           NarrativeDraft — the model's ENTIRE hybrid-mode output surface
src/
  demo.py                .eml in, HTML report out — one command; --hybrid opt-in
  web.py                 same pipeline behind a browser UI
  web_ui.html            the UI itself (single page, no build step)
  router.py              is this input something the pipeline can process?
  intent.py              persona classifier for prose the router can't resolve
  workflows/
    phishing.py            analyze_phishing() — the ONE place fast/hybrid live;
                            CLI and web both call this, no second implementation
  parser/                 deterministic feature extraction
    headers.py              SPF/DKIM/DMARC, address consistency, brand names
    urls.py                 text/href mismatch, IP-based, punycode, shorteners
    attachments.py          risky and double extensions, archives
    body.py                 hidden text, image-only bodies, gateway banners
  rules/engine.py         weighted scoring → RuleAssessment
  semantic/
    analyze.py              Qwen3.5-9B semantic extraction (hybrid mode)
    validate.py              exact-substring grounding — a finding that
                             cannot be found verbatim in the body is rejected
  decision/
    phishing_policy.py       decide() — the ONLY place a semantic finding
                             can move a verdict, a fixed rule set, not a model call
  report/
    mechanical.py            builds ALL of risk_seviyesi/category rationale/
                             findings/action, deterministically, always
    narrative.py             Qwen's narrative call — schema-in, schema-out
    narrative_prompts.py     PII-minimized prompt construction (no raw body/subject)
    assemble.py              apply_narrative() — slots the narrative into one field
    categories.py            the fixed category vocabulary, mechanically applied
  llm/service.py          shared QwenService — one model load per process,
                          reused across the semantic + narrative calls
  teacher/                training-data generation (earlier iteration, see below)
    generate_training_data.py
    prepare_lora_data.py
  eval/
    baseline.py             un-fine-tuned measurement (earlier iteration)
    finetuned.py             post-fine-tuning comparison (earlier iteration)
    groundedness.py          claim-vs-facts verification
scripts/
  anonymize.py            redacts the mailbox owner's identity
  check_anonymization.py  verification pass
  select_holdout.py       hold-out sampling
  expand_holdout_legitimate.py    append-only hold-out growth
  evaluate_hybrid_reliability.py  process-isolated hybrid pipeline reliability measurement
  smoke_test_hybrid.py            single-email, real-model smoke test
templates/
  report.html.j2          Jinja2 → HTML report
tests/                    500+ unit tests, no real model calls
```

---

## Setup

Requires Python 3.14 and Apple Silicon (the MLX runtime is Apple-specific).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Anonymization config

The pipeline redacts exactly one identity: the mailbox owner's own name and
email address. Everything else — sender domains, IPs, third-party addresses —
is deliberately kept real, because anonymizing domains would teach the model
nothing about real domain structure while providing no privacy benefit
(sender and brand domains are already public).

That identity is read from the environment and is never committed:

```bash
cp .env.anonymize.example .env.anonymize
# then edit .env.anonymize with the real values
```

`.env.anonymize` is gitignored. If it is absent the pipeline degrades safely:
no personal name is redacted, rather than corrupting the corpus.

### Analysing an email

```bash
# fast mode (default): parse → rule engine → deterministic report → HTML  (~1 s)
python3 src/demo.py mail.eml --open

# hybrid mode: + semantic extraction + decision policy + narrative  (~60-270 s on M2 Air)
python3 src/demo.py mail.eml --hybrid --open
```

`fast` is the right mode for checking a new email, the template, or the
pipeline day to day — no model is loaded, the verdict and every technical
finding are identical to what `hybrid` would produce, only the narrative
sentence is absent. Use `--hybrid` when the email's body text might contain a
signal the rule engine structurally cannot see (see "Known limitations"
below) and it is worth the wait.

`--hybrid` prints the pipeline's stages separately to the terminal — rule
verdict, semantic findings (accepted/rejected), final verdict, decision path,
narrative status — so it is visible at every step whether the model
contributed anything or the deterministic layers decided everything alone.

Two more flags exist from the **earlier**, superseded Seneca+LoRA report
writer (see "Earlier iteration" above) and are mutually exclusive with
`--hybrid`:

- `--adapter 0000400` layers the LoRA adapter on top of Seneca. Off by
  default — [it measured worse](#lora-fine-tuning-the-report-writer) on both
  metrics.
- `--constrain` restricts generation to the report schema via `llguidance`,
  making malformed JSON structurally impossible. Also off by default: every
  number reported here was measured without it, and a demo running under
  different conditions than the measurements would misrepresent both. It
  earns its keep on emails where the model repeatedly emits an unescaped
  quote inside a string and the JSON will not parse.

### Browser UI

```bash
python3 src/web.py          # http://127.0.0.1:8000
```

Paste a raw email or drag a `.eml` file in. Shows the routing decision, the
rule engine's verdict with every signal that fired and its weight, and the
rendered report. A "Hybrid analiz" toggle (off by default, same fast/hybrid
distinction as the CLI) reveals the semantic findings, final verdict, decision
path, and narrative status as separate cards when checked.

No analysis logic lives in the web layer — both modes call the same
`analyze_phishing()` (`src/workflows/phishing.py`) the CLI calls; this file is
a thin FastAPI shell around it, not a second implementation.

### Routing

The router returns four machine-readable outcomes: `phishing_direct`,
`phishing_missing_email`, `needs_clarification`, and `unsupported`. A valid
`.eml`/raw message bypasses the intent model. An authenticated upstream may
provide `trusted_route_hint="phishing"`; this metadata must not be inferred
from end-user prose.

```bash
python3 src/router.py mail.eml              # can the pipeline take this?
python3 src/router.py --text "$(pbpaste)"   # pasted email
python3 src/router.py --text "SPF nedir?" --classify   # + intent classifier
```

### Maintenance

```bash
# unit tests
for t in tests/test_*.py; do python3 "$t"; done

# parse and anonymize a corpus
python3 scripts/parse_and_anonymize.py

# verify no owner identity survived
python3 scripts/check_anonymization.py

# LoRA training (wrap long GPU jobs to prevent sleep — see below)
caffeinate -dims mlx_lm.lora --config config/lora.yaml 2>&1 | tee logs/train.log
```

---

## Notes for anyone reproducing this on a laptop

Two hard-won operational lessons, both cost hours here:

**Prevent sleep with `caffeinate -dims`, not `-i`.** Two macOS kernel panics
(`completeMemory() prepare count underflow` @ `IOGPUMemory.cpp:550`) took down
training runs. The panic report's timestamps showed the decisive clue: sleep at
19:05, wake at 19:18, panic at 19:39. Metal buffers left in a prepared state
across a sleep transition corrupt the driver's accounting. `caffeinate -i`
alone is insufficient — it holds off idle sleep only. Keep the power adapter
connected; on battery, macOS enforces some sleep policies regardless.

**`Train loss 0.000` does not necessarily mean the model is learning nothing.**
mlx_lm formats this figure as `{:.3f}`, so a real loss of 0.0004 prints as
`0.000`. The `Trained Tokens` and `Tokens/sec` counters can also emit garbage
(a run here reported 1.06 billion tokens over a 229-sample dataset). The
reliable check is to load a checkpoint and inspect the `lora_b` tensors: they
are initialised to zero and only become non-zero once gradients have actually
been applied.

Also worth knowing: `Peak mem` in mlx_lm is not physical RAM usage. It is the
cumulative peak *allocation* seen through unified memory, swap included, and it
never decreases. Seeing `42.881 GB` on a 16 GB machine is not a misreading —
it is a sign of heavy swapping.

---

## Known limitations

- **Recall rests on 15 phishing emails.** The legitimate side was expanded to
  65, but the phishing side was not — growing it needs hand-labelling
  adversarial samples, since the corpus is an estimated ~43% plain commercial
  spam and the source folder cannot be trusted as a label. 86.7% therefore
  carries a wide interval.
- Weights and thresholds were originally calibrated on the first 30 emails, so
  those remain calibration results. Later fixes were tuned on a separate dev
  set (see above).
- **The rule engine alone reads the envelope, not the letter.** Nineteen of
  its 22 signals look at headers, URLs or attachments. An email with clean
  authentication, no links and no attachments is invisible to it in `fast`
  mode however obviously fraudulent the text is. Two known misses are exactly
  this: Portuguese legal-threat social engineering forwarded through genuine
  infrastructure with SPF, DKIM and DMARC all passing, and a 419 advance-fee
  scam sent from a real `.edu.tr` account. **`hybrid` mode exists specifically
  to close part of this gap** — the semantic layer reads body text the rule
  engine cannot — but it is opt-in, costs ~60–270 seconds on an M2 Air, and
  the 18-email observation above (1/18 upgrades, 0/18 wrong-direction) is too
  small a sample to bound how much of the gap it actually closes in general.
- **The narrative sentence is not a claim to trust on its own.** It is not
  fact-checked against `EmailFacts` the way the offline teacher/adapter path's
  groundedness metric checks technical claims (see "Evaluation criteria"
  above) — there is no equivalent groundedness check for the three narrative
  sentences yet. It carries zero weight in the verdict, cannot introduce a
  new technical finding or category (the schema has no field for either), and
  a failed or off-schema narrative call keeps the mechanical fallback text
  rather than blocking the report (visibly, via `narrative_status`, never
  hidden) — but the sentences themselves should be read as a scenario
  summary, not as independently verified findings.
- 9 of 229 earlier-iteration LoRA training samples (3.9%) exceed the
  4096-token sequence limit and have their target JSON truncated — see
  "Earlier iteration" above; this does not affect the current hybrid
  pipeline, which does not fine-tune a model at all.
- The earlier iteration's fine-tuned adapter overfit; see the results
  section. The current pipeline uses Qwen3.5-9B unmodified, no fine-tuning.
- **The router is stage one only.** It answers "is this an email?" from
  structure. The intent classifier behind it (`--classify`) can name
  `titus` or `cybersec_qa`, but neither persona is implemented here — it says
  so instead of pretending to dispatch.

---

## License and data

No email data is committed to this repository. `data/` and `models/` are
gitignored. The phishing corpus comes from the public
[`rf-peixoto/phishing_pot`](https://github.com/rf-peixoto/phishing_pot)
dataset; the legitimate corpus is a personal mailbox export and is not
redistributable.
