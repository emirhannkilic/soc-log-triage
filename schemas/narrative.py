"""
Pydantic schema for Qwen's narrowed contribution to a hybrid report
(PROGRESS.md "rapor mimarisi değişikliği" — narrative-only architecture).

NarrativeDraft is deliberately NOT schemas.report.Report. The model never
sees or produces risk_seviyesi, sonuc_ve_gerekce, teknik_bulgular,
phishing_gostergeleri, or onerilen_aksiyon — those five fields are
authored entirely by src/report/mechanical.py, deterministically, in
both fast and hybrid mode. This schema is the model's ENTIRE output
surface: three fixed-purpose sentence fragments that
src/report/mechanical.py's apply_narrative() slots into
Report.genel_degerlendirme's existing three-sentence template
("Olası senaryo: ... Alıcıdan beklenen eylem: ... Olası zarar: ...").
Splitting the narrative into three separately-typed fields (rather than
having the model write one free-form paragraph) is what makes
apply_narrative() a pure substitution instead of a place the model could
smuggle a differently-shaped sentence, an extra field, or content
belonging to one of the five deterministic fields into the wrong slot.

extra="forbid" matches the convention in schemas/report.py and
schemas/facts.py: a field outside this schema is a bug, not a silent
pass-through.
"""
from pydantic import BaseModel, ConfigDict


class NarrativeDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    olasi_senaryo: str
    mailin_talep_ettigi_eylem: str
    olasi_zarar: str
