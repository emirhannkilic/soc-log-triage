"""Unit tests for src/report/assemble.py's apply_narrative() (PROGRESS.md
"rapor mimarisi değişikliği" — deterministic assembler, adım 4)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from schemas.narrative import NarrativeDraft  # noqa: E402
from schemas.report import Report  # noqa: E402
from src.report.assemble import apply_narrative  # noqa: E402

_BASE_REPORT_KWARGS = dict(
    risk_seviyesi="Phishing",
    sonuc_ve_gerekce="Bu karar; kimlik doğrulama uyumsuzluğu kategorisinin değerlendirilmesine dayanır.",
    genel_degerlendirme=(
        "Olası senaryo: e-posta, kimlik avı amacıyla hazırlanmış göstergeler taşıyor. "
        "Alıcıdan beklenen eylem: mevcut bulgulardan otomatik olarak çıkarılamadı. "
        "Olası zarar: kimlik bilgisi kaybı riski."
    ),
    teknik_bulgular=[{"baslik": "spf or dmarc fail", "aciklama": "SPF/DMARC fail (+3 puan)"}],
    phishing_gostergeleri=["SPF/DMARC fail"],
    onerilen_aksiyon="E-postayı silin, linklere tıklamayın, ekleri açmayın ve gönderen adresi engelleyin.",
)

_DRAFT = NarrativeDraft(
    olasi_senaryo="Alıcı, bankasından geldiğini iddia eden bir e-posta alıyor.",
    mailin_talep_ettigi_eylem="Alıcının bir bağlantıya tıklayıp giriş bilgilerini girmesi isteniyor.",
    olasi_zarar="Girilen kimlik bilgileri saldırgan tarafından ele geçirilebilir.",
)


def test_apply_narrative_replaces_only_genel_degerlendirme():
    report = Report(**_BASE_REPORT_KWARGS)
    result = apply_narrative(report, _DRAFT)

    assert result.genel_degerlendirme == (
        "Olası senaryo: Alıcı, bankasından geldiğini iddia eden bir e-posta alıyor. "
        "Alıcıdan beklenen eylem: Alıcının bir bağlantıya tıklayıp giriş bilgilerini "
        "girmesi isteniyor. "
        "Olası zarar: Girilen kimlik bilgileri saldırgan tarafından ele geçirilebilir."
    )


def test_apply_narrative_leaves_other_five_fields_byte_for_byte_identical():
    report = Report(**_BASE_REPORT_KWARGS)
    result = apply_narrative(report, _DRAFT)

    assert result.risk_seviyesi == report.risk_seviyesi
    assert result.sonuc_ve_gerekce == report.sonuc_ve_gerekce
    assert result.teknik_bulgular == report.teknik_bulgular
    assert result.phishing_gostergeleri == report.phishing_gostergeleri
    assert result.onerilen_aksiyon == report.onerilen_aksiyon
    # Original object itself must be untouched — apply_narrative() must
    # not mutate its input.
    assert report.genel_degerlendirme == _BASE_REPORT_KWARGS["genel_degerlendirme"]


def test_apply_narrative_preserves_upgrade_explanation_suffix():
    """src/report/mechanical.py appends a fourth sentence when a
    semantic upgrade occurred — apply_narrative() must keep that suffix
    intact, only swapping the three-sentence prefix before it."""
    kwargs = dict(_BASE_REPORT_KWARGS)
    kwargs["genel_degerlendirme"] = (
        "Olası senaryo: e-posta bazı şüpheli göstergeler taşıyor. "
        "Alıcıdan beklenen eylem: mevcut bulgulardan otomatik olarak çıkarılamadı. "
        "Olası zarar: kesin bir zarar tahmini yapılamıyor. "
        "Rule engine kararı 'Güvenilir' idi, ancak e-posta gövdesinde doğrulanmış bir "
        "kimlik bilgisi talebi ve dış link birlikte tespit edildiği için nihai karar "
        "'Muhtemel Phishing' olarak güncellendi."
    )
    report = Report(**kwargs)
    result = apply_narrative(report, _DRAFT)

    assert result.genel_degerlendirme.startswith(
        "Olası senaryo: Alıcı, bankasından geldiğini iddia eden bir e-posta alıyor."
    )
    assert result.genel_degerlendirme.endswith(
        "nihai karar 'Muhtemel Phishing' olarak güncellendi."
    )
    assert "Rule engine kararı 'Güvenilir' idi" in result.genel_degerlendirme


def test_apply_narrative_returns_a_new_object_not_a_mutation():
    report = Report(**_BASE_REPORT_KWARGS)
    result = apply_narrative(report, _DRAFT)
    assert result is not report


def test_apply_narrative_raises_on_non_mechanical_prefix():
    """apply_narrative() only operates on build_report()'s own fixed
    prefix — a Report whose genel_degerlendirme was constructed some
    other way must fail loudly, not silently produce a malformed
    result."""
    kwargs = dict(_BASE_REPORT_KWARGS)
    kwargs["genel_degerlendirme"] = "Bu e-posta güvenilir görünüyor."
    report = Report(**kwargs)
    try:
        apply_narrative(report, _DRAFT)
        raise AssertionError("expected ValueError")
    except ValueError as e:
        assert "Olası senaryo" in str(e)


if __name__ == "__main__":
    import traceback

    tests = [(name, obj) for name, obj in list(globals().items())
              if name.startswith("test_") and callable(obj)]
    passed, failed = 0, 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS: {name}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL: {name}: {e}")
            failed += 1
        except Exception:
            print(f"ERROR: {name}")
            traceback.print_exc()
            failed += 1

    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
