"""
v3 plan Adım 5: renders the 30 hold-out emails through templates/report.html.j2
with NO LLM involved — Report objects are built mechanically from
EmailFacts.flat_signals() + the rule engine's Verdict (src/rules/engine.py),
one teknik_bulgular entry per fired rule. This exists purely to validate
that schemas/report.py + the Jinja2 template render correctly end to end
before Adım 6 puts a real model in the loop; the report text itself is not
meant to read like the teacher's future Turkish prose (see CLAUDE.md
"Teacher generation ayarları" — that's a distinct, more careful step).

Output: data/holdout/reports/candidate_<N>.html (one per hold-out email).
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import jinja2  # noqa: E402

from schemas.facts import EmailFacts  # noqa: E402
from schemas.report import Report, TechnicalFinding  # noqa: E402
from src.rules.engine import RuleMatch, Verdict, evaluate, load_rules  # noqa: E402

CANDIDATES_PATH = PROJECT_ROOT / "data" / "holdout" / "candidates.jsonl"
TEMPLATE_PATH = PROJECT_ROOT / "templates" / "report.html.j2"
OUT_DIR = PROJECT_ROOT / "data" / "holdout" / "reports"

METADATA_KEYS = ("source_label", "_eml_path", "is_spam_not_phishing", "spam_reason")

_SONUC_BY_VERDICT = {
    "Phishing": (
        "Bu e-posta, rule engine tarafından toplam {score} puanla Phishing "
        "olarak sınıflandırıldı. Aşağıdaki teknik bulgular bu kararı destekliyor."
    ),
    "Muhtemel Phishing": (
        "Bu e-posta, rule engine tarafından toplam {score} puanla Muhtemel "
        "Phishing olarak sınıflandırıldı. Kesin bir karar için analist "
        "incelemesi gerekiyor."
    ),
    "Güvenilir": (
        "Bu e-posta, rule engine tarafından toplam {score} puanla Güvenilir "
        "olarak sınıflandırıldı. Aşağıdaki bulgular bu değerlendirmeyi destekliyor."
    ),
}

_ONERI_BY_VERDICT = {
    "Phishing": "E-postayı silin, linklere tıklamayın, ekleri açmayın ve gönderen adresi engelleyin.",
    "Muhtemel Phishing": "E-postadaki linklere/eklere etkileşimde bulunmadan önce SOC analistine iletin.",
    "Güvenilir": "Ek bir aksiyon gerekmiyor.",
}


def _finding_from_match(m: RuleMatch) -> TechnicalFinding:
    sign = "+" if m.weight >= 0 else ""
    return TechnicalFinding(
        baslik=m.signal.replace("_", " "),
        aciklama=f"{m.description} ({sign}{m.weight} puan)",
    )


def build_report(signals: dict, verdict: Verdict) -> Report:
    findings = [_finding_from_match(m) for m in verdict.matches]
    gostergeler = [m.description for m in verdict.matches if m.weight > 0]

    return Report(
        risk_seviyesi=verdict.verdict,
        sonuc_ve_gerekce=_SONUC_BY_VERDICT[verdict.verdict].format(score=verdict.score),
        genel_degerlendirme=(
            f"Toplam {len(verdict.matches)} sinyal değerlendirildi, "
            f"skor {verdict.score} (eşikler: >= "
            f"{load_rules()['thresholds']['phishing']} Phishing, "
            f">= {load_rules()['thresholds']['suspicious']} Muhtemel Phishing)."
        ),
        teknik_bulgular=findings,
        phishing_gostergeleri=gostergeler,
        onerilen_aksiyon=_ONERI_BY_VERDICT[verdict.verdict],
    )


def main():
    rules = load_rules()
    # autoescape=True (not select_autoescape) because the template file is
    # named report.html.j2 — select_autoescape keys off the *last*
    # extension (".j2"), not in its default HTML/XML list, so it would
    # silently disable escaping despite the file always rendering HTML.
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(TEMPLATE_PATH.parent),
        autoescape=True,
    )
    template = env.get_template(TEMPLATE_PATH.name)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(CANDIDATES_PATH, encoding="utf-8") as f:
        candidates = [json.loads(line) for line in f]

    for i, cand in enumerate(candidates, start=1):
        facts_dict = {k: v for k, v in cand.items() if k not in METADATA_KEYS}
        facts = EmailFacts(**facts_dict)
        signals = facts.flat_signals()
        verdict = evaluate(signals, rules)
        report = build_report(signals, verdict)

        html = template.render(
            **report.model_dump(),
            subject=facts.subject,
            date=facts.date,
        )
        out_path = OUT_DIR / f"candidate_{i}.html"
        out_path.write_text(html, encoding="utf-8")

    print(f"Wrote {len(candidates)} reports to {OUT_DIR}")


if __name__ == "__main__":
    main()
