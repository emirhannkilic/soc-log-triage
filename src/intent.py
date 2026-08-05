"""
Intent classifier — the router's second stage, for input the static rules
cannot resolve.

WHERE THIS SITS
    src/router.py answers "is this an email?" from structure alone: a file
    extension, or RFC 5322 headers in pasted text. That covers every input
    the phishing pipeline can actually process, in about zero milliseconds.

    What it cannot do is read plain prose. "Bu IP'yi kontrol eder misin"
    and "SPF nasıl çalışır" are both text with no headers, and telling them
    apart needs to understand the request, not its shape. That is this
    module's only job.

    The order matters and is deliberate (see LLM_SECURITY_PIPELINE.html):
    static rules first, model only when they come up empty. The router runs
    on every request; making it load a model each time would put the
    system's most frequent path on its slowest code.

WHY A SEPARATE, SMALL MODEL
    Seneca is 4GB and takes ~100 seconds to write a report. Routing is a
    three-way category choice, not reasoning — it should cost milliseconds.
    Qwen2.5-1.5B-Instruct at 4-bit is ~0.88GB and answers in ~100-200ms
    once resident.

WHY LOGIT COMPARISON INSTEAD OF GENERATION
    The model never generates text here. A single forward pass produces the
    logits for the next token, and the three candidate labels' first tokens
    are compared directly. This removes the entire class of failures the
    project has already measured: on the hold-out, the fine-tuned Seneca
    produced invalid JSON on 20 of 27 emails when asked to follow a format.
    A classifier that cannot emit anything except one of three known tokens
    cannot go off-format at all.

WHAT ROUTING TO A PERSONA MEANS RIGHT NOW
    Only the phishing pipeline exists. `titus` and `cybersec_qa` are
    recognised intents but have nothing behind them, and this module says so
    rather than pretending otherwise — see UNBUILT_PERSONAS.

Usage:
    python3 src/intent.py --text "SPF nasıl çalışır?"
    python3 src/intent.py --selftest
"""
import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MODEL_ID = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
MODEL_DIR = PROJECT_ROOT / "models" / "Qwen2.5-1.5B-Instruct-4bit"

# Personas the classifier can name. Only the first has an implementation.
# "alakasiz" is not a persona — it is the classifier's escape hatch for
# input that belongs to none of the other three (see WHY A FOURTH LABEL).
PERSONAS = ("phishing", "titus", "cybersec_qa", "alakasiz")
UNBUILT_PERSONAS = frozenset({"titus", "cybersec_qa"})

# Confidence bands. The costs are asymmetric: sending a general question to
# the phishing pipeline crashes a parser that expects an email, while
# sending an email question to Q&A merely wastes a turn. So an uncertain
# classification falls back to the harmless side rather than the specific
# one.
CONFIDENT = 0.70
UNCERTAIN = 0.40

# WHY A FOURTH LABEL: softmax over only 3 candidate tokens (see classify())
# always sums to 100% across THOSE three, even when the input matches none
# of them — small, meaningless differences in the raw logits get amplified
# into a confident-looking split. Measured concretely: "selam" (a bare
# greeting) scored titus 83% with the 3-label version, comfortably above
# CONFIDENT, and got routed as a threat-intel request. A 4th "alakasiz"
# (irrelevant) label gives the softmax somewhere to put that mass instead
# of forcing it onto the three real personas — it does not fix low-signal
# input, it gives the classifier an honest way to say "none of the above"
# rather than picking the least-wrong option with false confidence.
_SYSTEM = """Sen bir SOC asistanının yönlendiricisisin. Kullanıcının \
isteğini oku ve HANGİ UZMANA gitmesi gerektiğine karar ver.

phishing    — bir e-postanın/mesajın sahte olup olmadığı, kimlik avı analizi
titus       — bir IP, domain, dosya hash'i ya da IoC hakkında tehdit istihbaratı
cybersec_qa — genel siber güvenlik sorusu, kavram açıklaması, danışmanlık
alakasiz    — selamlaşma, teşekkür, sohbet ya da siber güvenlikle hiç \
ilgisi olmayan herhangi bir istek

SADECE bu dört kelimeden birini yaz, başka hiçbir şey yazma."""

_FEW_SHOT = [
    ("Şu maili inceler misin, sahte mi?", "phishing"),
    ("185.220.101.5 zararlı bir IP mi?", "titus"),
    ("SPF ve DKIM arasındaki fark nedir?", "cybersec_qa"),
    ("Bana gelen bir mesaj bankamdan geliyormuş gibi görünüyor", "phishing"),
    ("Bu hash VirusTotal'da ne çıkıyor: 44d88612fea8a8f36de82e1278abb02f", "titus"),
    ("Kurumumuzda MFA'yı nasıl yaygınlaştırmalıyız?", "cybersec_qa"),
    ("selam", "alakasiz"),
    ("teşekkürler, iyi günler", "alakasiz"),
    ("bugün hava nasıl?", "alakasiz"),
]

_model = None
_tokenizer = None


@dataclass
class IntentResult:
    persona: str
    confidence: float
    scores: dict[str, float]
    # True when the chosen persona has no implementation behind it.
    unbuilt: bool
    # Set when confidence was too low to act on the top choice.
    fallback_reason: str | None = None


def _resolve_model_path() -> str:
    """Prefer a local copy, fall back to the Hub id (which downloads once)."""
    if MODEL_DIR.is_dir() and any(MODEL_DIR.glob("*.safetensors")):
        return str(MODEL_DIR)
    return MODEL_ID


def _load():
    global _model, _tokenizer
    if _model is None:
        from mlx_lm import load
        path = _resolve_model_path()
        print(f"Yönlendirici model yükleniyor: {path}", file=sys.stderr)
        _model, _tokenizer = load(path)
    return _model, _tokenizer


def _label_first_tokens(tokenizer) -> dict[str, int]:
    """First token id of each persona label.

    Comparing only the first token is enough because the three labels
    diverge immediately ("ph", "tit", "cy"). It also keeps this to one
    forward pass — no generation loop, no sampling.
    """
    ids = {}
    for p in PERSONAS:
        toks = tokenizer.encode(p, add_special_tokens=False)
        if not toks:
            raise RuntimeError(f"tokenizer '{p}' etiketini kodlayamadı")
        ids[p] = toks[0]
    if len({*ids.values()}) != len(PERSONAS):
        raise RuntimeError(
            "persona etiketleri aynı ilk token'ı paylaşıyor — ilk-token "
            "karşılaştırması bunları ayırt edemez")
    return ids


def classify(text: str) -> IntentResult:
    """Pick a persona for free-form text. One forward pass, no generation."""
    import mlx.core as mx

    model, tokenizer = _load()

    messages = [{"role": "system", "content": _SYSTEM}]
    for q, a in _FEW_SHOT:
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": text})

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    tokens = mx.array(tokenizer.encode(prompt))

    logits = model(tokens[None])[0, -1, :]
    label_ids = _label_first_tokens(tokenizer)

    # Softmax over ONLY the four label tokens: the question is which of
    # these is most likely, not how they rank against the whole vocabulary.
    # Normalising over the full vocab would report confidences near zero
    # for every label and make the thresholds meaningless.
    label_logits = mx.array([logits[i].item() for i in label_ids.values()])
    probs = mx.softmax(label_logits)
    scores = {p: float(probs[i]) for i, p in enumerate(label_ids)}

    top = max(scores, key=scores.get)
    conf = scores[top]

    if conf >= CONFIDENT:
        return IntentResult(top, conf, scores, top in UNBUILT_PERSONAS)

    if conf >= UNCERTAIN:
        return IntentResult(
            "cybersec_qa", conf, scores, "cybersec_qa" in UNBUILT_PERSONAS,
            fallback_reason=(
                f"'{top}' en yüksek ({conf:.0%}) ama {CONFIDENT:.0%} eşiğinin "
                f"altında — en zararsız personaya düşüldü"))

    return IntentResult(
        top, conf, scores, top in UNBUILT_PERSONAS,
        fallback_reason=(
            f"hiçbir persona {UNCERTAIN:.0%} eşiğini geçemedi "
            f"(en yüksek '{top}' {conf:.0%}) — kullanıcıya sorulmalı"))


# Sanity cases for --selftest. Not a unit test: running it downloads and
# loads a model, which tests/ deliberately never does.
_SELFTEST = [
    ("Bu mail sahte mi acaba?", "phishing"),
    ("Ekteki mesaj bankamdan geliyormuş gibi ama şüpheliyim", "phishing"),
    ("evil-domain.ru hakkında ne biliyorsun?", "titus"),
    ("Şu IP'yi kontrol eder misin: 45.13.1.2", "titus"),
    ("DMARC politikası nasıl kurulur?", "cybersec_qa"),
    ("Zero trust mimarisi nedir?", "cybersec_qa"),
    # Regression cases for the 3-label blind spot found 2026-08-05: with
    # only phishing/titus/cybersec_qa as candidates, softmax was forced to
    # split 100% across them even for input matching none — "selam" alone
    # scored titus 83%. These would have caught it before it shipped.
    ("selam", "alakasiz"),
    ("teşekkürler, iyi günler", "alakasiz"),
    ("bugün hava nasıl?", "alakasiz"),
]


def _selftest() -> int:
    ok = 0
    for text, expected in _SELFTEST:
        r = classify(text)
        hit = r.persona == expected
        ok += hit
        mark = "✓" if hit else "✗"
        print(f"  {mark} {text[:48]:<50} → {r.persona:<12} "
              f"({r.confidence:.0%}, beklenen {expected})")
    print(f"\n{ok}/{len(_SELFTEST)} doğru")
    return 0 if ok == len(_SELFTEST) else 1


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Düz metnin hangi personaya ait olduğunu belirler.")
    ap.add_argument("--text", help="sınıflandırılacak metin")
    ap.add_argument("--selftest", action="store_true",
                    help="örnek cümlelerle doğruluk kontrolü")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(_selftest())
    if not args.text:
        ap.error("--text ya da --selftest gerekli")

    import time
    t0 = time.time()
    r = classify(args.text)
    elapsed = time.time() - t0

    print(f"persona : {r.persona}")
    print(f"güven   : {r.confidence:.0%}")
    print("dağılım :", ", ".join(f"{k} {v:.0%}" for k, v in
                                 sorted(r.scores.items(), key=lambda x: -x[1])))
    print(f"süre    : {elapsed*1000:.0f} ms")
    if r.fallback_reason:
        print(f"not     : {r.fallback_reason}")
    if r.unbuilt:
        print(f"UYARI   : '{r.persona}' personası bu repoda kurulmadı — "
              f"yalnızca phishing hattı çalışıyor.")


if __name__ == "__main__":
    main()
