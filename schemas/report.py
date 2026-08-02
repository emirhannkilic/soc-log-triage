"""
Pydantic schema for the LLM-generated report (v3 plan section 6.2).

The teacher model (Adım 6-7) and later the fine-tuned student both produce
this shape and nothing else — extra="forbid" so a model that drifts from
the schema fails validation instead of silently producing an
unrecognized field. risk_seviyesi is never decided by the model; it's
copied verbatim from the rule engine's Verdict.verdict (schemas/report.py
does not import src/rules/engine.py to avoid a schema-depends-on-engine
coupling, but scripts/generate_training_pairs.py and templates/
render_report.py, both later steps, are responsible for enforcing that
match — see CLAUDE.md "Teacher generation ayarları").
"""
from typing import Literal

from pydantic import BaseModel, ConfigDict

RiskSeviyesi = Literal["Phishing", "Muhtemel Phishing", "Güvenilir"]


class TechnicalFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    baslik: str
    aciklama: str


class Report(BaseModel):
    model_config = ConfigDict(extra="forbid")

    risk_seviyesi: RiskSeviyesi
    sonuc_ve_gerekce: str
    genel_degerlendirme: str
    teknik_bulgular: list[TechnicalFinding]
    phishing_gostergeleri: list[str]
    onerilen_aksiyon: str
