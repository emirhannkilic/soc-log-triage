# soc-log-triage

[English](README.md) | [Türkçe](README.tr.md)

A proof-of-concept phishing triage pipeline: a deterministic parser and rule
engine decide whether an email is phishing, and a locally fine-tuned 7B model
writes the analyst-facing report in Turkish. The model never classifies.

---

## The core design decision

Most "LLM for phishing detection" demos ask a language model to read a raw
email and output a verdict. This project deliberately does not, because that
approach was tried here first and it failed measurably.

An earlier iteration (v2) fed 70 real emails to a 4-bit quantized 7B model and
asked for a classification. It scored **35% accuracy on a 3-class problem —
worse than random.** Worse than the number itself was the failure mode: when
the model could not determine a verdict, it invented supporting evidence. In
one case it labelled an email as phishing and justified it by citing an
"unusual X-Mailer header" — the email had no X-Mailer header at all.

The root cause is not model size. Signals like SPF/DKIM/DMARC results and
domain string comparisons are **outputs of a deterministic protocol**:
`pass`/`fail`/`none`, two strings being equal or not. Asking a language model
to compute a weighted sum over them is the wrong tool for the category, no
matter how large the model. So responsibility is split:

```
.eml file, or a raw message pasted as text
    │
    ▼
┌──────────────────────┐
│  router              │  structural: file extension, or ≥3 RFC 5322 headers
│  (deterministic)     │  → accept, or explain what is missing
└──────────────────────┘
    │
    ▼
┌──────────────────────┐
│  feature parser      │  Python stdlib email + BeautifulSoup
│  (deterministic)     │  → facts: dict
└──────────────────────┘
    │
    ▼
┌──────────────────────┐
│  rule engine         │  weighted score, thresholds in config/rules.yaml
│  (deterministic)     │  → Phishing | Probable Phishing | Legitimate
└──────────────────────┘
    │
    ▼  facts + verdict
┌──────────────────────┐
│  LLM                 │  training: Qwen3.5-9B (teacher)
│  (report writer)     │  inference: Seneca 7B + LoRA (student)
└──────────────────────┘
    │
    ▼  JSON (never HTML)
┌──────────────────────┐
│  Jinja2 template     │  → HTML report
└──────────────────────┘
```

Two rules follow from this and are enforced throughout:

1. **The LLM does not classify.** It receives a verdict that has already been
   decided and explains it. The `risk_seviyesi` field in its output must match
   the rule engine's verdict exactly, or the sample is dropped.
2. **The LLM emits JSON, not HTML.** The template renders the HTML. In v2, 9
   out of 70 outputs had no parseable classification at all and 5 different
   output formats appeared; this eliminates that class of problem by
   construction.

This is a **demo / proof-of-concept**, not a production system. The models were
chosen to run on a 16 GB fanless laptop. The goal is an honest, measurable
demonstration of the approach — not the best achievable accuracy.

---

## Results

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
  rules.yaml            scoring weights and thresholds (not hardcoded)
  lora.yaml             LoRA hyperparameters
schemas/
  facts.py              EmailFacts — the parser's output contract
  report.py             Report — the LLM's output contract
src/
  demo.py               .eml in, HTML report out — one command
  web.py                same pipeline behind a browser UI
  web_ui.html           the UI itself (single page, no build step)
  router.py             is this input something the pipeline can process?
  intent.py             persona classifier for prose the router can't resolve
  parser/               deterministic feature extraction
    headers.py            SPF/DKIM/DMARC, address consistency, brand names
    urls.py               text/href mismatch, IP-based, punycode, shorteners
    attachments.py        risky and double extensions, archives
    body.py               hidden text, image-only bodies, gateway banners
  rules/engine.py       weighted scoring → verdict
  teacher/              training-data generation with the teacher model
    generate_training_data.py
    prepare_lora_data.py
  eval/
    baseline.py           un-fine-tuned measurement
    finetuned.py          post-fine-tuning comparison
    groundedness.py       claim-vs-facts verification
scripts/
  anonymize.py          redacts the mailbox owner's identity
  check_anonymization.py verification pass
  select_holdout.py     hold-out sampling
  expand_holdout_legitimate.py  append-only hold-out growth
templates/
  report.html.j2        Jinja2 → HTML report
tests/                  103 unit tests
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
# full pipeline: parse → rules → Seneca writes the report → HTML  (~100 s)
python3 src/demo.py mail.eml --open

# same verdict, same score, same findings — prose written mechanically  (~1 s)
python3 src/demo.py mail.eml --no-llm --open
```

`--no-llm` is the right mode for checking a new email, the template, or the
pipeline: only the wording changes, never the decision.

Two more flags:

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
rendered report. The LLM and schema-constraint toggles mirror the CLI flags.

No analysis logic lives in the web layer — it calls the same router, parser,
rule engine and template as the CLI.

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
- **The engine reads the envelope, not the letter.** Nineteen of its 22
  signals look at headers, URLs or attachments. An email with clean
  authentication, no links and no attachments is invisible to it however
  obviously fraudulent the text is. Two known misses are exactly this:
  Portuguese legal-threat social engineering forwarded through genuine
  infrastructure with SPF, DKIM and DMARC all passing, and a 419 advance-fee
  scam sent from a real `.edu.tr` account. The second was partly recovered by
  a `reply_to_free_mail` signal (corporate sender, replies redirected to
  consumer webmail) — but only the subset that redirects replies.
- 9 of 229 training samples (3.9%) exceed the 4096-token sequence limit and
  have their target JSON truncated.
- The fine-tuned adapter overfit; see the results section.
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
