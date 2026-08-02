"""Unit tests for schemas/report.py and templates/report.html.j2 (v3 Adım 5)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jinja2

from schemas.report import Report

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_PATH = PROJECT_ROOT / "templates" / "report.html.j2"

_VALID_KWARGS = dict(
    risk_seviyesi="Phishing",
    sonuc_ve_gerekce="Test gerekçe.",
    genel_degerlendirme="Test değerlendirme.",
    teknik_bulgular=[{"baslik": "SPF", "aciklama": "fail"}],
    phishing_gostergeleri=["SPF fail"],
    onerilen_aksiyon="Sil.",
)


def _env():
    # autoescape=True (not select_autoescape) because the template file is
    # named report.html.j2 — select_autoescape keys off the *last*
    # extension (".j2"), which isn't in its default HTML/XML list, so it
    # would silently disable escaping. The template always renders HTML,
    # so escaping should always be on regardless of filename.
    return jinja2.Environment(
        loader=jinja2.FileSystemLoader(TEMPLATE_PATH.parent),
        autoescape=True,
    )


def test_valid_report_constructs():
    r = Report(**_VALID_KWARGS)
    assert r.risk_seviyesi == "Phishing"


def test_invalid_risk_seviyesi_rejected():
    kwargs = dict(_VALID_KWARGS, risk_seviyesi="Kesinlikle Phishing")
    try:
        Report(**kwargs)
        raise AssertionError("expected ValidationError")
    except Exception as e:
        assert "risk_seviyesi" in str(e)


def test_extra_field_rejected():
    kwargs = dict(_VALID_KWARGS, uydurma_alan="x")
    try:
        Report(**kwargs)
        raise AssertionError("expected ValidationError")
    except Exception as e:
        assert "extra" in str(e).lower() or "forbidden" in str(e).lower()


def test_template_renders_phishing_banner():
    template = _env().get_template(TEMPLATE_PATH.name)
    r = Report(**_VALID_KWARGS)
    html = template.render(**r.model_dump(), subject="Test Subject", date=None)
    assert 'class="risk-banner risk-phishing"' in html
    assert "Test Subject" in html
    assert "SPF fail" in html


def test_template_renders_empty_gostergeler_fallback():
    template = _env().get_template(TEMPLATE_PATH.name)
    kwargs = dict(_VALID_KWARGS, risk_seviyesi="Güvenilir", phishing_gostergeleri=[])
    r = Report(**kwargs)
    html = template.render(**r.model_dump(), subject=None, date=None)
    assert 'class="risk-banner risk-guvenilir"' in html
    assert "Belirgin bir phishing göstergesi tespit edilmedi" in html


def test_template_escapes_html_in_user_content():
    template = _env().get_template(TEMPLATE_PATH.name)
    kwargs = dict(_VALID_KWARGS, sonuc_ve_gerekce="<script>alert(1)</script>")
    r = Report(**kwargs)
    html = template.render(**r.model_dump(), subject=None, date=None)
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


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
