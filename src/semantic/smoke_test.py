"""
Ad-hoc single-email smoke test for src/semantic/analyze.py's shadow-mode
extractor (PHISHING_ROUTING_PLAN.md step 6). Not wired into any pipeline —
run manually to see real model output against a real .eml before trusting
the mocked unit tests in tests/test_semantic_analyze.py.

Prints raw model output, then accepted/rejected findings from the
validator, so a bad run is visible at both stages (did the model produce
garbage, or did it produce plausible-looking findings that failed
grounding).

Usage:
    python3 src/semantic/smoke_test.py path/to/mail.eml
"""
import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.parser.parse import parse_eml  # noqa: E402
from src.semantic.analyze import analyze_semantic  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("eml", type=Path)
    args = ap.parse_args()

    if not args.eml.is_file():
        raise SystemExit(f"dosya bulunamadı: {args.eml}")

    print(f"[1/2] Parse ediliyor: {args.eml.name}", file=sys.stderr)
    facts = parse_eml(args.eml)
    print(f"      body_text: {len(facts.body_text)} karakter", file=sys.stderr)

    print("[2/2] Qwen3.5-9B yükleniyor ve semantic extraction çalıştırılıyor "
          "(ilk yükleme birkaç dakika sürebilir) ...", file=sys.stderr)
    t0 = time.time()
    result = analyze_semantic(facts)
    print(f"      {time.time() - t0:.0f} saniye", file=sys.stderr)

    print(f"\n=== KABUL EDİLEN ({len(result.accepted)}) ===")
    for f in result.accepted:
        print(f"  [{f.type.value}] {f.evidence!r} (conf={f.model_confidence:.2f})")
        print(f"    -> {f.explanation}")

    print(f"\n=== REDDEDİLEN ({len(result.rejected)}) ===")
    for vf in result.rejected:
        finding = vf.finding
        label = getattr(finding, "evidence", repr(finding))
        print(f"  [{vf.rejection_reason.value}] {label!r}")


if __name__ == "__main__":
    main()
