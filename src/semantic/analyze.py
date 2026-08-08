"""
Qwen semantic extractor — SHADOW MODE (PHISHING_ROUTING_PLAN.md step 6).

The model's ONLY job here is producing SemanticFindingCandidate objects
(schemas/semantic.py): what manipulative content the email's body
contains, quoted. It does not write a report, does not decide a verdict,
and — since a real run showed the model can't reliably do it (see
schemas/semantic.py's module docstring) — does NOT compute character
offsets either; src/semantic/validate.py does that from the model's
quote via a real substring search. CLAUDE.md's "LLM'e sınıflandırma
yaptırmak" boundary still applies to the verdict question; naming a
manipulation pattern is not the same as deciding Phishing/Muhtemel
Phishing/Güvenilir.

SHADOW MODE, NOT DECISION INPUT
    Nothing in this module reads or writes a rule engine verdict.
    analyze_semantic() takes facts, produces findings, and stops. The
    caller (not yet built — PHISHING_ROUTING_PLAN.md step 9's decision
    policy) is responsible for keeping the rule engine's verdict as the
    real decision and only ever CONSULTING these findings, never being
    overridden by them at this stage. Right now there is no caller that
    wires this into a verdict at all — this file's purpose is only to
    observe what the model finds, e.g. how it reacts to urgency language
    in legitimate marketing email, before any decision logic depends on
    it.

MODEL
    models/Qwen3.5-9B-MLX-4bit, loaded via mlx_vlm (NOT mlx_lm — same
    constraint as src/teacher/generate.py, this is a multimodal loader
    used text-only). No re-download logic here: CLAUDE.md's "Ağır/Uzun
    Süren Script Çalıştırmaları" rule means model loading and generation
    are the user's terminal to run, not something this module retries or
    manages downloads for.

CANONICAL TEXT
    canonical_body = canonicalize_body(facts.body_text)
    (src/semantic/canonical.py) — NOT raw facts.body_text. A real bug
    (PROGRESS.md, 2026-08) traced a labeling mismatch to exactly this:
    facts.body_text can contain CRLF, and a text-mode file read
    anywhere in the pipeline (e.g. re-reading a rendered worksheet)
    silently turns "\r\n" into "\n", producing two different strings
    for what was supposed to be the same body. canonicalize_body() is
    the single normalization step every consumer — this module's
    prompt, src/semantic/validate.py's grounding search, and the
    offline labeling scripts — must call, so "the same canonical_body"
    is actually guaranteed rather than assumed.

CONSTRAINED JSON
    build_json_schema_logits_processor (mlx_vlm.structured, already used
    by src/demo.py's --constrain and src/teacher/constrained_test.py)
    restricts generation to a JSON array of SemanticFindingCandidate-
    shaped objects, making a malformed top-level shape structurally
    impossible. It does NOT guarantee evidence is a real quote, or that
    it's unique — that is exactly what src/semantic/validate.py checks
    afterwards. The schema guarantees shape, never grounding.

LAZY MODEL LOADING
    Model loading itself now lives in src/llm/service.py (QwenService,
    get_service()) — this module no longer owns a `_model`/`_processor`
    singleton or calls mlx_vlm.load()/generate() directly. get_service()
    is only called from analyze_semantic(), and analyze.py itself does
    nothing at import time — so importing this module (e.g. from a test
    that only exercises build_user_prompt or parse_raw_findings) never
    touches the model or the filesystem paths it needs.

ERRORS — SemanticExtractionError, NOT SystemExit
    An earlier version of this module raised SystemExit on a malformed
    model response. That was wrong: SystemExit is a BaseException meant
    to terminate the whole CLI process, not a normal error a caller is
    expected to catch — it is not part of CLAUDE.md's "don't repair
    model output" rule, it was simply the wrong exception type. This
    version raises SemanticExtractionError(code, message) instead, with
    three closed `code` values a caller can branch on:
        "model_call_failed"      the load/generate infrastructure itself
                                  failed — wraps src/llm/service.py's
                                  LLMServiceError (e.g. a GPU/Metal
                                  timeout), always via `raise ... from exc`
                                  so the original traceback survives.
        "invalid_json"           generate() succeeded but the output
                                  couldn't be parsed as a JSON array at
                                  all (extract_json_array returned None).
        "invalid_output_schema"  reserved for a parsed-but-malformed
                                  array shape distinct from "not JSON at
                                  all" — not raised by this module today
                                  (validate_raw_findings already handles
                                  per-item schema problems as rejected
                                  findings, not a hard failure), but kept
                                  as a defined code so a caller's
                                  exhaustive `code` handling doesn't need
                                  to change if that changes later.
    The model's output is still never repaired or retried — a malformed
    response is rejected via this exception, exactly as before, just
    with the correct exception type.
"""
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from schemas.facts import EmailFacts  # noqa: E402
from schemas.semantic import SemanticFindingCandidate, SemanticFindingType  # noqa: E402
from src.llm.service import LLMServiceError, get_service  # noqa: E402
from src.semantic.canonical import canonicalize_body  # noqa: E402
from src.semantic.validate import ValidationResult, validate_raw_findings  # noqa: E402


class SemanticExtractionError(Exception):
    """Raised by analyze_semantic() for every failure mode it normalizes
    to — see module docstring's ERRORS section for the three `code`
    values and what each means."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


TEMPERATURE = 0
# Started at 350 per the plan's initial estimate — measured too low on
# a real run: a single email with 2 findings (each with a full-sentence
# Turkish `explanation`) got cut off mid-JSON before the array even
# closed (smoke test on inbox-1804.eml, 2026-08). Raised to 700 — still
# too low: the first full evaluation run (scripts/
# evaluate_semantic_extractor.py, 2026-08) cut off mid-finding on
# Candidate 1, which needed only ~500 characters (~140 tokens by a
# naive chars/3.5 estimate) to reach that cutoff. CLAUDE.md's LoRA
# training experience already recorded that char-based token estimates
# run low for Turkish text (measured max ~4877 tokens against a
# ~3030-token naive estimate) — same pattern here, so this was raised
# generously (1500) instead of re-guessing a slightly bigger number
# that could just as easily be wrong again on a candidate with more
# findings (ground truth has up to 5 findings in one email). Still
# bounded, not open-ended — this is a shadow-mode extractor producing
# short quoted findings, not free prose.
MAX_TOKENS = 1500

_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)

_ALLOWED_TYPES = ", ".join(t.value for t in SemanticFindingType)

SYSTEM_PROMPT = f"""Sen bir e-postanın GÖVDE metnini analiz edip, o metinde hangi \
manipülasyon/sosyal mühendislik kalıplarının GEÇTİĞİNİ tespit eden bir asistansın.

GÖREVİN SADECE BULGU LİSTESİ ÜRETMEK — bir karar/sınıflandırma (phishing mi \
değil mi, güvenli mi tehlikeli mi) YAPMIYORSUN, rapor YAZMIYORSUN. Sadece \
"bu metinde şu tür bir ifade var, kanıtı bu" diyorsun.

KATI KURALLAR:
1. "evidence" alanı, sana verilen GÖVDE metninden BİREBİR, KARAKTER \
KARAKTERİNE (satır sonları, boşluklar DAHİL) kopyalanmış bir alıntı \
OLMAK ZORUNDA — parafraz, özet, çeviri, cümleyi tek satıra birleştirme, \
fazladan/eksik boşluk YASAK. Uzun bir alıntı içinde GÖVDE'de bir satır \
sonu VARSA, senin alıntında da AYNI YERDE bir satır sonu OLMALI — \
metni "düzeltmeye" ya da "temizlemeye" ÇALIŞMA, GÖVDE'de nasıl \
duruyorsa TIPKI ÖYLE kopyala. Alıntı GÖVDE'de aynen (bu kurallara \
uyarak) geçmiyorsa o bulgu tamamen reddedilir — EMİN DEĞİLSEN kısa \
ve kesin bir alt-cümle seç, riskli uzun bir alıntı YERİNE.
2. "evidence" GÖVDE'de TEK BİR YERDE geçecek kadar UZUN ve SPESİFİK olmalı — \
kısa/genel bir kelime veya kalıp GÖVDE'de birden fazla yerde geçiyorsa o bulgu \
da reddedilir. Bu yüzden mümkün olduğunca TAM CÜMLE ya da en az birkaç kelimelik \
özgün bir ifade alıntıla, tek bir kelime YETERLİ DEĞİLDİR.
3. "type" SADECE şu listeden biri olabilir, başka bir kategori UYDURMA: \
{_ALLOWED_TYPES}

TİP TANIMLARI VE SIK YAPILAN HATALAR — dikkatle oku, ölçümde en çok bu \
tiplerde hata yapıldı:

- "authority_impersonation": mesaj, GERÇEKTE OLMADIĞI bir otoriteymiş \
GİBİ davranarak alıcıyı bir İŞLEME (para/bilgi vermeye, bir eyleme) \
YÖNLENDİRİYOR ya da bir TEHDİT/FON VAADİYLE birleştiriyor — sahte bir \
hukuki/resmi/kurumsal kimlik TAKINIP bunu kullanıyor (örn. "FBI Cyber \
Crime Division" adına para transferi istemek, "Federal Reserve" adına \
bir fonun sana ait olduğunu iddia etmek). BU DEĞİLDİR (üç ayrı durum, \
ÜÇÜ DE ARANMAZ):
  (a) bir şirketin/servisin KENDİ imzası ("Saygılarımızla, Microsoft \
      hesap ekibi", "Teşekkürler, Netflix"),
  (b) gizlilik/KVKK metni, adres defterine ekleme talebi, standart bir \
      kurumsal alt bilgi,
  (c) kurumun KENDİ yasal/resmi kimlik bilgilerini paylaşması — şirket \
      adresi, MERSİS numarası, "6493 sayılı yasa kapsamında kurulmuş", \
      "Facebook, Inc., Attention: Community Support, 1 Facebook Way" \
      gibi bir footer/adres bloğu. Bu bir OTORİTE İDDİASI DEĞİL, \
      standart bir kurumsal bilgilendirmedir — TEK BAŞINA asla bulgu \
      SAYILMAZ.
  Gerçek gönderen kendi adını/markasını/yasal bilgisini kullanıyorsa \
  (taklit ETMİYORSA) bu bulgu DEĞİLDİR — sahte bir otorite iddiası + bir \
  işlem/tehdit/fon vaadi BİRLİKTE ARANIR, salt bir imza/künye/yasal \
  metin ARANMAZ.
- "attachment_or_link_instruction": alıcıyı ZARARLI/ŞÜPHELİ bir eyleme \
(kimlik bilgisi girme, ek açma, "hesabını doğrula" gibi bir sayfaya \
tıklama) yönlendiren AÇIK bir talimat. BU DEĞİLDİR: gövdedeki her URL \
ya da her "tıkla" ifadesi — bir haber/ürün linki, bir "detaylar için \
tıklayın" pazarlama linki, ya da bir takip/tracking linki TEK BAŞINA \
bulgu SAYILMAZ. Sadece linkin/ekin AÇIKÇA bir güvenlik eylemi (doğrulama, \
giriş, ek açma) istediği durumlarda bu tipi kullan.
- "threat_or_fear": alıcıda KAYIP/ZARAR/SORUN korkusu yaratan bir ifade \
— "hesabınıza yetkisiz erişim tespit edildi", "hesabınız askıya \
alınacak", "işleminiz iptal edilecek", "bu fırsatı kaybedeceksiniz" \
gibi. Bir OLAY/DURUM anlatan ve bunun SONUCUNDA olumsuz bir şey \
olabileceğini ima eden cümleleri ARA — sadece "güvenlik" kelimesinin \
geçmesi YETERLİ DEĞİL, gerçek bir tehdit/kayıp anlatısı GEREKİR.
- "urgency_or_pressure": alıcıyı HIZLI hareket etmeye zorlayan bir süre \
sınırı, kıtlık ya da aciliyet ifadesi — "24 saat içinde", "5 iş günü \
içinde", "son gün", "stoklar tükenmeden", "hemen". Bu tip GÖVDE'de \
GEÇEN sayı/süre/aciliyet kelimesini alıntıla, kendi yorumunu ekleme.

4. SADECE GÖVDE metninde GERÇEKTEN geçen ifadeler için bulgu üret. Konu \
satırından, header'lardan ya da kendi tahmininden bulgu UYDURMA.
5. "model_confidence" 0.0-1.0 arası bir sayı, ne kadar emin olduğunu belirtir.
6. "explanation" alanı, bu alıntının neden bu tipte bir bulgu sayıldığını \
kısaca (1 cümle) Türkçe açıklar.
7. GÖVDE'de bu tür bir ifade hiç yoksa BOŞ LİSTE döndür — zorla bulgu uydurma. \
Meşru/ticari bir e-postada SIK SIK hiçbir bulgu OLMAMALIDIR — bu normaldir, \
her mailde mutlaka bir şey bulmak ZORUNDA DEĞİLSİN.
8. SADECE bir JSON dizisi döndür, başka hiçbir metin/açıklama ekleme. Karakter \
konumu/offset İSTEMİYORUZ — sadece alıntının kendisini ver.

JSON şeması (dizi elemanı):
{{"type": "...", "evidence": "...", "model_confidence": 0.0, "explanation": "..."}}"""


def _facts_context_block(facts: EmailFacts) -> str:
    """Minimal identity/brand context — NOT the full flat_signals(). This
    extractor only needs enough to recognize e.g. authority/brand
    impersonation language; header/URL/attachment technical signals are
    the rule engine's job (src/rules/engine.py), not this model's."""
    lines = [
        f"Gönderen görünen ad: {facts.display_name or '(yok)'}",
        f"Gönderen domain: {facts.from_domain or '(yok)'}",
        f"Konu: {facts.subject or '(yok)'}",
    ]
    return "\n".join(lines)


def build_user_prompt(facts: EmailFacts) -> str:
    """Uses canonicalize_body(facts.body_text), not the raw field —
    analyze_semantic() passes that SAME normalized string to
    validate_raw_findings(). See module docstring's CANONICAL TEXT
    section."""
    return f"""{_facts_context_block(facts)}

GÖVDE:
{canonicalize_body(facts.body_text)}"""


def build_messages(facts: EmailFacts) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(facts)},
    ]


def extract_json_array(raw_text: str) -> list | None:
    """Best-effort extraction of a JSON array from raw model output —
    mirrors src/demo.py's _extract_json, adapted for a top-level array
    instead of a top-level object. Returns None (not an exception, not a
    repaired guess) on anything that doesn't parse as a JSON array —
    caller treats that the same as "zero findings produced", not a
    crash. This is extraction, not repair: no content is invented or
    corrected, only located and parsed as-is."""
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[len("json"):]
    match = _JSON_ARRAY_RE.search(text)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, list) else None


def _array_schema_for(model_cls) -> dict:
    """Wrap a Pydantic model's JSON schema as the `items` of a top-level
    array schema, without breaking its internal $ref pointers.

    model_json_schema() emits $defs (e.g. the SemanticFindingType enum)
    at ITS OWN root, with $ref: "#/$defs/...". Naively nesting that whole
    object under {"type": "array", "items": <schema>} leaves $defs
    sitting under `items` while the $ref is still root-relative
    ("#/$defs/...") — the reference no longer resolves from the new
    document root, which is exactly the
    "Pointer '/$defs/SemanticFindingType' does not exist" failure
    llguidance's JsonCompiler raised when this wrapped the raw schema
    directly. $defs must be hoisted to the OUTER schema's root instead,
    where "#/$defs/..." actually points once the object is embedded
    under `items`.
    """
    item_schema = model_cls.model_json_schema()
    defs = item_schema.pop("$defs", None)
    schema: dict = {"type": "array", "items": item_schema}
    if defs:
        schema["$defs"] = defs
    return schema


def _build_logits_processor(processor):
    from mlx_vlm.structured import build_json_schema_logits_processor

    schema = _array_schema_for(SemanticFindingCandidate)
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    return build_json_schema_logits_processor(tokenizer, schema)


def analyze_semantic(facts: EmailFacts, service=None) -> ValidationResult:
    """Runs the model once (via src/llm/service.py's shared QwenService),
    extracts a JSON array of SemanticFindingCandidate-shaped dicts, and
    returns the ValidationResult (accepted ValidatedSemanticFinding
    objects with validator-computed offsets, plus rejected candidates)
    from src/semantic/validate.py. Never touches a verdict — see module
    docstring's SHADOW MODE section.

    service: injectable QwenService, defaulting to get_service() (the
    process-wide singleton) — tests pass a QwenService constructed with
    mocked load_fn/generate_fn instead of going through the singleton.

    Raises SemanticExtractionError on any failure — see module
    docstring's ERRORS section. No retry, no output patching (CLAUDE.md
    "Yapılmayacaklar").
    """
    if service is None:
        service = get_service()

    try:
        model, processor = service.load()
        logits_processor = _build_logits_processor(processor)
        messages = build_messages(facts)
        raw = service.generate(
            messages,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            logits_processor=logits_processor,
        )
    except LLMServiceError as exc:
        raise SemanticExtractionError(
            code="model_call_failed",
            message=f"semantic extractor model çağrısı başarısız: {exc}",
        ) from exc

    raw_findings = extract_json_array(raw)
    if raw_findings is None:
        raise SemanticExtractionError(
            code="invalid_json",
            message=(
                "semantic extractor geçerli bir JSON dizisi üretmedi. "
                f"Ham çıktı: {raw[:500]!r}"
            ),
        )

    return validate_raw_findings(raw_findings, canonicalize_body(facts.body_text))
