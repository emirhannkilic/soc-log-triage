"""
Language detection for the shadow-mode phishing classifier (PROGRESS.md
"Plan sonrası — çok dilli phishing classifier mekanizması"). Not part of
the rule engine or the report pipeline — this only feeds
src/classifier/phishing.py's decision on whether to translate before
scoring.

Uses FastText's lid.176 (https://fasttext.cc/docs/en/language-identification.html)
because it is small (.ftz: ~917KB, .bin: ~126MB), fast (single forward
pass, no GPU), and covers 176 languages — Codex's original suggestion.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "lid.176.ftz"

_model = None


def _load_model():
    global _model
    if _model is None:
        import fasttext
        if not MODEL_PATH.is_file():
            raise SystemExit(
                f"HATA: dil tespiti modeli bulunamadı: {MODEL_PATH}\n"
                f"İndirme: curl -Lo {MODEL_PATH} "
                f"https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
            )
        # fasttext prints a stray "Warning : `load_model` does not return
        # WordVectorModel..." to stderr on some versions; harmless, not
        # suppressed here since CLAUDE.md prefers visible over silent.
        _model = fasttext.load_model(str(MODEL_PATH))
    return _model


def detect_language(text: str) -> tuple[str, float]:
    """Returns (ISO 639-1 language code, confidence 0-1).

    Empty/whitespace-only text returns ("und", 0.0) rather than calling
    into fasttext, which raises on empty input.
    """
    normalized = text.replace("\n", " ").strip()
    if not normalized:
        return "und", 0.0

    model = _load_model()
    # Not model.predict(): fasttext-wheel 0.9.2's predict() ends in
    # `np.array(probs, copy=False)` where probs is a plain tuple — NumPy
    # 2.x raises ValueError on that (can't avoid a copy from a tuple),
    # a known upstream incompatibility with no fasttext-wheel fix yet.
    # Calling the underlying binding directly and doing our own (copying)
    # array conversion sidesteps the broken line without patching NumPy
    # or pinning it below what torch/transformers here expect.
    entry = normalized + "\n"
    predictions = model.f.predict(entry, 1, 0.0, "strict")
    if not predictions:
        return "und", 0.0
    probs, labels = zip(*predictions)
    language = labels[0].replace("__label__", "")
    confidence = float(probs[0])
    return language, confidence


if __name__ == "__main__":
    # Quick self-check, mirrors src/intent.py's --selftest pattern.
    samples = [
        ("Bu bir örnek Türkçe cümledir.", "tr"),
        ("This is an example English sentence.", "en"),
        ("Dies ist ein deutscher Beispielsatz.", "de"),
    ]
    ok = 0
    for text, expected in samples:
        lang, conf = detect_language(text)
        status = "OK" if lang == expected else "FAIL"
        ok += lang == expected
        print(f"[{status}] {text!r} -> {lang} ({conf:.2f}), beklenen {expected}",
              file=sys.stderr)
    print(f"{ok}/{len(samples)} doğru", file=sys.stderr)
