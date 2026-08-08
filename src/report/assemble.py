"""
Deterministic assembler that slots a Qwen-authored NarrativeDraft into
an already-complete, mechanically-authored Report (PROGRESS.md "rapor
mimarisi değişikliği" — narrative-only architecture).

apply_narrative() IS A PURE SUBSTITUTION, NOT A MERGE
    report (from src/report/mechanical.py's build_report(), always
    called first and unconditionally in every mode) already has all six
    fields fully populated — risk_seviyesi, sonuc_ve_gerekce,
    teknik_bulgular, phishing_gostergeleri, and onerilen_aksiyon are
    final as soon as build_report() returns, and genel_degerlendirme
    already holds mechanical.py's own generic three-sentence fallback
    text. apply_narrative() replaces ONLY genel_degerlendirme's three
    sentence slots with the model's three narrower sentences — every
    other field on the returned Report is the exact same value
    (verified byte-for-byte by tests/test_report_assemble.py), because
    Report is a Pydantic model and model_copy(update=...) only ever
    touches the keys named in `update`.

WHY THE MODEL CANNOT REACH THE OTHER FIVE FIELDS EVEN IN PRINCIPLE
    NarrativeDraft (schemas/narrative.py) has no field named
    risk_seviyesi, sonuc_ve_gerekce, teknik_bulgular,
    phishing_gostergeleri, or onerilen_aksiyon — extra="forbid" means a
    model output that tried to include one of those would fail schema
    validation in src/report/narrative.py's generate_narrative() before
    ever reaching this function. This function's own signature
    reinforces the same boundary: it takes a NarrativeDraft, not a dict,
    so there is no code path here that could accidentally read an
    unexpected key off model output and assign it to the wrong Report
    field.

genel_degerlendirme's UPGRADE-EXPLANATION SUFFIX IS PRESERVED
    src/report/mechanical.py's _build_genel_degerlendirme() appends a
    fourth sentence ("Rule engine kararı ... güncellendi.") when a
    semantic upgrade occurred — this is deterministic bookkeeping, not
    narrative, and it is NOT part of what apply_narrative() replaces.
    Only the fixed "Olası senaryo: ... Alıcıdan beklenen eylem: ...
    Olası zarar: ..." three-sentence PREFIX is swapped; anything mechanical.py
    appended after it survives unchanged. Implemented by locating and
    replacing exactly that prefix rather than reconstructing the whole
    field from scratch — reconstructing here would require duplicating
    mechanical.py's upgrade-suffix logic and risk the two silently
    drifting apart.
"""
from schemas.narrative import NarrativeDraft
from schemas.report import Report


def _ensure_trailing_period(sentence: str) -> str:
    """The model is instructed to write exactly one full sentence
    (SYSTEM_PROMPT_TEMPLATE's own rule) but that is not structurally
    enforced the way NarrativeDraft's field types are — a real output
    could omit or double up the trailing period. Stripped and
    re-appended exactly once here so the fixed template's own periods
    (after each "{...}" slot) never combine with the model's into "..".
    """
    stripped = sentence.strip()
    return stripped if stripped.endswith(".") else stripped + "."


def apply_narrative(report: Report, draft: NarrativeDraft) -> Report:
    """Returns a NEW Report with genel_degerlendirme's three-sentence
    prefix replaced by draft's content; every other field (including any
    upgrade-explanation suffix already appended to genel_degerlendirme)
    is unchanged from `report`. Does not mutate `report` in place."""
    prefix_start = report.genel_degerlendirme.find("Olası senaryo: ")
    if prefix_start == -1:
        # mechanical.py always writes this exact prefix — a missing
        # prefix means report was not built by build_report() as
        # expected, a caller contract violation this function should
        # never silently paper over.
        raise ValueError(
            "report.genel_degerlendirme mekanik 'Olası senaryo: ...' önekiyle "
            "başlamıyor — apply_narrative() sadece src/report/mechanical.py'nin "
            "build_report() çıktısı üzerinde çalışabilir."
        )

    suffix_marker = " Rule engine kararı '"
    suffix_start = report.genel_degerlendirme.find(suffix_marker)
    suffix = report.genel_degerlendirme[suffix_start:] if suffix_start != -1 else ""

    senaryo, eylem, zarar = (
        _ensure_trailing_period(draft.olasi_senaryo),
        _ensure_trailing_period(draft.mailin_talep_ettigi_eylem),
        _ensure_trailing_period(draft.olasi_zarar),
    )
    new_prefix = (
        f"Olası senaryo: {senaryo} "
        f"Alıcıdan beklenen eylem: {eylem} "
        f"Olası zarar: {zarar}"
    )

    return report.model_copy(update={"genel_degerlendirme": new_prefix + suffix})
