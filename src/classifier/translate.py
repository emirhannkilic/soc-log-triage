"""
Translation for the shadow-mode phishing classifier (PROGRESS.md
"Plan sonrası — çok dilli phishing classifier mekanizması"). Only feeds
src/classifier/phishing.py — never the rule engine, never the report LLM.

Uses facebook/m2m100_418M (MIT licensed; the larger nllb-200 alternative
is CC-BY-NC and its model card says it isn't intended for production use,
so m2m100 is the cleaner fit here) to translate non-English body text to
English before it reaches the English-trained phishing classifier.

SECURITY: this module's output must NEVER reach the rule engine. The rule
engine already sees the original, unmasked, untranslated facts — that
decision is untouched by anything here. Masking below (URLs, email
addresses, IPs) exists only so the translation model doesn't rewrite or
"normalize" a real indicator (a phishing domain, an attacker's address)
while translating surrounding prose; the classifier only needs to judge
the LANGUAGE of the social-engineering text, not the literal indicators,
which the rule engine already covers via src/parser.
"""
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_NAME = "facebook/m2m100_418M"
MODEL_DIR = PROJECT_ROOT / "models" / "m2m100_418M"

# Order matters: URL before bare email/IP, since a URL can itself contain
# an email-like or IP-like substring in its path/query that would
# otherwise get double-masked into nonsense.
_URL_RE = re.compile(r"https?://\S+")
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")

# facebook/m2m100_418M's language codes are ISO 639-1, matching what
# src/classifier/language.py's fasttext model returns (lid.176 also uses
# ISO 639-1), so no code-mapping table is needed between the two.
SUPPORTED_LANGUAGES = {
    "tr", "de", "fr", "es", "it", "ar", "ru",
    "uk", "fa", "zh", "ja", "ko", "pt", "nl",
}

# Below this fasttext confidence, trusting the detected language is a
# coin flip — shared by every consumer of this module (src/classifier/
# phishing.py, src/classifier/nli_signals.py) so the threshold can't
# drift between them.
MIN_LANGUAGE_CONFIDENCE = 0.80

_model = None
_tokenizer = None


def _mask(text: str) -> str:
    text = _URL_RE.sub("[URL]", text)
    text = _EMAIL_RE.sub("[EMAIL]", text)
    text = _IP_RE.sub("[IP]", text)
    return text


def _load():
    global _model, _tokenizer
    if _model is None:
        import torch
        from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

        if not MODEL_DIR.is_dir():
            raise SystemExit(
                f"HATA: çeviri modeli bulunamadı: {MODEL_DIR}\n"
                f"İndirme: huggingface-cli download {MODEL_NAME} "
                f"--local-dir {MODEL_DIR}"
            )
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        _tokenizer = M2M100Tokenizer.from_pretrained(str(MODEL_DIR))
        _model = M2M100ForConditionalGeneration.from_pretrained(str(MODEL_DIR)).to(device)
        _model.eval()
        # The checkpoint's generation_config ships max_length=200, which
        # conflicts with the max_new_tokens=512 passed to generate()
        # below and prints a "Both ... have been set" warning on every
        # single call — harmless (max_new_tokens always wins) but with
        # src/classifier/nli_signals.py now translating per-window
        # instead of once per email, this fires dozens of times per
        # mail. Clearing it here removes the conflict at the source
        # instead of suppressing the (correct) warning.
        _model.generation_config.max_length = None
    return _model, _tokenizer


def translate_to_english(text: str, source_language: str) -> str:
    """Masks URLs/emails/IPs, translates the rest to English.

    Truncates to 512 tokens (M2M100's practical limit for this size model)
    — callers needing more should chunk before calling, not raise here;
    a shadow-mode signal on a truncated translation is still informative,
    silently failing on a long email would just drop the signal.
    """
    import torch

    model, tokenizer = _load()
    masked = _mask(text)

    tokenizer.src_lang = source_language
    device = next(model.parameters()).device
    inputs = tokenizer(masked, return_tensors="pt", truncation=True,
                       max_length=512).to(device)

    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.get_lang_id("en"),
            num_beams=1,
            do_sample=False,
            # Greedy decoding on longer/noisier marketing-style bodies
            # (found on data/raw/gmail/eml/inbox-6485.eml) locks into a
            # repeating loop — "a tragic story of a young player ..."
            # repeated ~10x with no new content — which then reads as
            # anomalous/suspicious to the phishing classifier for reasons
            # that have nothing to do with the mail's real content.
            # no_repeat_ngram_size blocks any 3-gram from repeating,
            # breaking the loop without switching to beam search (slower)
            # or sampling (non-deterministic, unwanted for a shadow-mode
            # signal that should be reproducible).
            no_repeat_ngram_size=3,
            max_new_tokens=512,
        )

    return tokenizer.batch_decode(generated, skip_special_tokens=True)[0]


if __name__ == "__main__":
    text = "Merhaba, hesabınızı https://ornek.com/dogrula adresinden doğrulayın."
    print(f"girdi:  {text}", file=sys.stderr)
    print(f"maskeli: {_mask(text)}", file=sys.stderr)
    print(f"çeviri: {translate_to_english(text, 'tr')}", file=sys.stderr)
