"""
Shadow-mode phishing classifier (PROGRESS.md "Plan sonrası — çok dilli
phishing classifier mekanizması", Codex'e danışıldı 2026-08-05).

SHADOW MODE — THIS NEVER FEEDS THE RULE ENGINE OR THE DECISION.
CLAUDE.md's locked rule: the rule engine decides (deterministic,
score-based); this project does not let an LLM/ML model classify. This
module does not change that. It runs a pre-trained English-trained BERT
phishing classifier ALONGSIDE the rule engine, purely to observe: does it
catch things the rule engine misses, or just echo the same signals less
precisely? That question is answered by comparing logged output against
a labeled set — never by wiring the score in. See PROGRESS.md's "shadow
mode" section for the evaluation plan and Codex's own warning: this
model is English-trained and must not be treated as authoritative on
Turkish mail without a separately-measured, language-split accuracy
check first.

Three candidate backends, all Codex's suggestions, all binary-or-reducible
phishing probability:

- "ealvaradob" (ealvaradob/bert-finetuned-phishing) — the original ANA
  ADAY. A 14-mail balanced shadow-mode sample (2026-08-06, PROGRESS.md)
  showed it wrong on 5/14, including 2 that never even went through
  translation (pure English marketing/newsletter mail scored as
  phishing) — the failure isn't language, it's the model treating
  legitimate marketing/CTA language as suspicious regardless of language.
- "cybersectony" (cybersectony/phishing-email-detection-distilbert_v2.4.1)
  — second candidate, tried BECAUSE of that result. Smaller/faster
  (DistilBERT vs BERT). Natively 4-class (legitimate_email, phishing_url,
  legitimate_url, phishing_url_alt per the model card — config.json ships
  with generic LABEL_0..3, real order taken from the model card's own
  usage example, verified 2026-08-06); reduced here to a single
  P(phishing) by summing the phishing-labeled classes (1 and 3). On the
  same 10-mail comparison it was wrong on 3/10 vs ealvaradob's 5/10 —
  better, especially on Turkish-translated mail, but shares the SAME
  blind spot on pure-English marketing/newsletter mail (2/10, no
  translation involved either).
- "aamoshdahal" (aamoshdahal/email-phishing-distilbert-finetuned) —
  third candidate, Codex's last-listed option, being tried to see
  whether the marketing/newsletter false-positive pattern is a property
  of THIS dataset family (both above were trained on similar-shaped
  phishing corpora) or something a differently-trained model avoids.

All three reduced to a single P(phishing) in [0, 1] so they return the
same ShadowResult shape and are directly comparable in one table.

Pipeline (mirrors the routing table in the doc this was adapted from):
    text
      -> detect_language (src/classifier/language.py)
      -> "en"                       -> classify directly
      -> supported, confident       -> translate_to_english, then classify
      -> unsupported/low-confidence -> SKIPPED, ml_usable=False
Never raises on a low-confidence/unsupported language — that is an
expected, common outcome (image-only mail, code-mixed text, a language
outside SUPPORTED_LANGUAGES), not a failure.
"""
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

BACKENDS = {
    "ealvaradob": {
        "model_name": "ealvaradob/bert-finetuned-phishing",
        "model_dir": PROJECT_ROOT / "models" / "bert-finetuned-phishing",
    },
    "cybersectony": {
        "model_name": "cybersectony/phishing-email-detection-distilbert_v2.4.1",
        "model_dir": PROJECT_ROOT / "models" / "phishing-distilbert-cybersectony",
    },
    "aamoshdahal": {
        "model_name": "aamoshdahal/email-phishing-distilbert-finetuned",
        "model_dir": PROJECT_ROOT / "models" / "phishing-distilbert-aamoshdahal",
    },
}
DEFAULT_BACKEND = "ealvaradob"

# Below this many characters, language detection itself is unreliable
# (fasttext's own guidance) — skip rather than trust a coin-flip label.
MIN_TEXT_CHARS = 20

# Keyed by backend name so switching backends mid-process (e.g. a
# comparison script running both) doesn't silently reuse the wrong model.
_loaded: dict[str, tuple] = {}


@dataclass
class ShadowResult:
    usable: bool
    phishing_probability: float | None
    language: str | None
    language_confidence: float | None
    translated: bool
    backend: str
    skip_reason: str | None = None


def _load(backend: str):
    if backend not in _loaded:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        spec = BACKENDS[backend]
        model_dir = spec["model_dir"]
        if not model_dir.is_dir():
            raise SystemExit(
                f"HATA: phishing classifier bulunamadı: {model_dir}\n"
                f"İndirme: hf download {spec['model_name']} --local-dir {model_dir}"
            )
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).to(device)
        model.eval()
        _loaded[backend] = (model, tokenizer)
    return _loaded[backend]


def _score_english(text: str, backend: str) -> float:
    """Returns P(phishing) in [0, 1]. Assumes text is already English."""
    import torch

    model, tokenizer = _load(backend)
    device = next(model.parameters()).device
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=512).to(device)
    with torch.inference_mode():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]

    if backend == "ealvaradob":
        # Model card: id2label {0: "benign", 1: "phishing"} — index 1 is
        # P(phishing) directly (verified against model.config.id2label,
        # see PROGRESS.md 2026-08-06 sanity check).
        return float(probs[1])

    if backend == "cybersectony":
        # config.json ships with generic LABEL_0..LABEL_3 (verified
        # 2026-08-06 — model.config.id2label has no real names), so the
        # order comes from the model card's own usage example instead:
        # index 0=legitimate_email, 1=phishing_url, 2=legitimate_url,
        # 3=phishing_url_alt. There is no single "phishing" class the way
        # ealvaradob has one — both phishing-labeled classes (1 and 3)
        # are summed into one P(phishing) so this backend returns the
        # same shape as the other and the two are comparable in one
        # table.
        return float(probs[1] + probs[3])

    if backend == "aamoshdahal":
        # config.json ships with generic LABEL_0/LABEL_1 (verified
        # 2026-08-06 — no real names in model.config.id2label), so the
        # order comes from the model card's own usage example instead:
        # labels = ["legitimate", "phishing"] — index 1 is P(phishing).
        return float(probs[1])

    raise ValueError(f"bilinmeyen backend: {backend}")


def classify(subject: str, body: str, backend: str = DEFAULT_BACKEND) -> ShadowResult:
    from src.classifier.language import detect_language

    original_text = f"{subject}\n\n{body}"
    if len(original_text.strip()) < MIN_TEXT_CHARS:
        return ShadowResult(False, None, None, None, False, backend,
                            skip_reason="body_too_short_for_language_detection")

    language, confidence = detect_language(original_text)

    if language == "en":
        score = _score_english(original_text, backend)
        return ShadowResult(True, score, language, confidence, False, backend)

    from src.classifier.translate import (
        MIN_LANGUAGE_CONFIDENCE,
        SUPPORTED_LANGUAGES,
        translate_to_english,
    )

    if confidence < MIN_LANGUAGE_CONFIDENCE:
        return ShadowResult(False, None, language, confidence, False, backend,
                            skip_reason="low_language_confidence")

    if language not in SUPPORTED_LANGUAGES:
        return ShadowResult(False, None, language, confidence, False, backend,
                            skip_reason="unsupported_language")

    translated_text = translate_to_english(original_text, language)
    score = _score_english(translated_text, backend)
    return ShadowResult(True, score, language, confidence, True, backend)


if __name__ == "__main__":
    import argparse

    from src.parser.parse import parse_eml

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("eml", type=Path)
    ap.add_argument("--backend", choices=list(BACKENDS), default=DEFAULT_BACKEND)
    args = ap.parse_args()

    facts = parse_eml(args.eml)
    result = classify(facts.subject or "", facts.body_text, backend=args.backend)
    print(f"backend: {result.backend}", file=sys.stderr)
    print(f"dil: {result.language} (güven {result.language_confidence})", file=sys.stderr)
    print(f"çevrildi: {result.translated}", file=sys.stderr)
    if result.usable:
        print(f"ml_phishing_probability: {result.phishing_probability:.4f}", file=sys.stderr)
    else:
        print(f"KULLANILAMADI: {result.skip_reason}", file=sys.stderr)
