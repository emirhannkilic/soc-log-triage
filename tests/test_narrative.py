"""Unit tests for schemas/narrative.py (rapor mimarisi değişikliği, adım 1)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from schemas.narrative import NarrativeDraft

_VALID_KWARGS = dict(
    olasi_senaryo="Alıcı, bankasından geldiğini iddia eden bir e-posta alıyor.",
    mailin_talep_ettigi_eylem="Alıcının bir bağlantıya tıklayıp giriş bilgilerini girmesi isteniyor.",
    olasi_zarar="Girilen kimlik bilgileri saldırgan tarafından ele geçirilebilir.",
)


def test_valid_narrative_draft_constructs():
    draft = NarrativeDraft(**_VALID_KWARGS)
    assert draft.olasi_senaryo == _VALID_KWARGS["olasi_senaryo"]
    assert draft.mailin_talep_ettigi_eylem == _VALID_KWARGS["mailin_talep_ettigi_eylem"]
    assert draft.olasi_zarar == _VALID_KWARGS["olasi_zarar"]


def test_missing_field_rejected():
    kwargs = dict(_VALID_KWARGS)
    del kwargs["olasi_zarar"]
    try:
        NarrativeDraft(**kwargs)
        raise AssertionError("expected ValidationError")
    except Exception as e:
        assert "olasi_zarar" in str(e)


def test_extra_field_rejected():
    # This is the property apply_narrative() (adım 4) relies on for
    # safety: the model cannot smuggle a risk_seviyesi, teknik_bulgular,
    # or any other Report-owned field through NarrativeDraft — extra
    # fields fail construction instead of being silently accepted and
    # then ignored (which would hide a real prompt/schema drift).
    kwargs = dict(_VALID_KWARGS, risk_seviyesi="Phishing")
    try:
        NarrativeDraft(**kwargs)
        raise AssertionError("expected ValidationError")
    except Exception as e:
        assert "extra" in str(e).lower() or "forbidden" in str(e).lower()


def test_wrong_type_rejected():
    kwargs = dict(_VALID_KWARGS, olasi_senaryo=["not", "a", "string"])
    try:
        NarrativeDraft(**kwargs)
        raise AssertionError("expected ValidationError")
    except Exception as e:
        assert "olasi_senaryo" in str(e)


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
