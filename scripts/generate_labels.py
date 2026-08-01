"""
Self-labeling: samples N examples each from data/processed/phishing.jsonl and
legitimate.jsonl, runs them through Seneca (GGUF, via llama-simple) with Cihan
abi's exact phishing system prompt, and writes the resulting HTML reports as
LoRA training data ({"messages": [system, user, assistant]}) to
data/processed/training_data.jsonl.

This is a proof-of-concept labeling strategy, not a production one — Seneca
labels its own training data (no stronger teacher model), so it can reinforce
its own mistakes. See CLAUDE.md "Cihan Abi Sistem Promptu" for the tradeoff
rationale (demo/PoC scope, no external API cost/latency, data stays local).

Runs a llama-simple subprocess per example, so this is slow (~60-90s per
example on CPU) — meant for small sample sizes (hundreds, not thousands).
"""
import argparse
import json
import random
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "SenecaLLM_x_Qwen2.5-7B-CyberSecurity-Q4_K_M-GGUF"
    / "senecallm_x_qwen2.5-7b-cybersecurity-q4_k_m.gguf"
)

SYSTEM_PROMPT = """Rol: Kıdemli E-posta Güvenlik Analisti (SOC L3).
Görev: Aşağıdaki header bilgilerini, PDF'teki phishing göstergeleriyle analiz et ve sonucu SADECE TÜRKÇE ve BASİT HTML formatında sun.
Analiz Kriterleri:
Domain: From, Return-Path, Reply-To uyumu. Typosquatting ve domain sahteciliği kontrolü.
Kimlik Doğrulama: SPF/DKIM/DMARC durumları ve Return-Path uyumu.
Yönlendirme: Received header dizilimi, IP-Domain uyumu ve şüpheli kaynaklar.
Header: Message-ID tutarsızlığı ve olağandışı X-Mailer bilgileri.
İçerik: Aciliyet, genel hitap ve şüpheli ek/link talepleri.
Sınıflandırma: [Phishing / Muhtemel Phishing / Güvenilir]
Çıktı Şablonu (HTML):
Risk Seviyesi
Sonuç ve Gerekçe
Genel Değerlendirme
Teknik Bulgular
Phishing Göstergeleri
NOT: Hiçbir İngilizce kelime kullanma. Teknik terimlerin Türkçe karşılıklarını veya yerleşik kullanımlarını tercih et utf-8 encoding i olsun."""

MAX_BODY_CHARS = 2000
MAX_TOKENS = 700


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def build_prompt(sample: dict) -> str:
    headers = sample["headers"]
    body = sample["body"][:MAX_BODY_CHARS]
    system = f"{SYSTEM_PROMPT}\nHeader Bilgisi: {headers}"
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{body}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def run_seneca(prompt: str) -> str | None:
    try:
        result = subprocess.run(
            [
                "llama-simple",
                "-m", str(MODEL_PATH),
                "-n", str(MAX_TOKENS),
                prompt,
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        return None

    output = result.stdout
    marker = "<|im_start|>assistant"
    idx = output.rfind(marker)
    if idx == -1:
        return None
    return output[idx + len(marker):].strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=100,
                         help="samples per class (default 100 -> 200 total)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        raise SystemExit(f"Model not found: {MODEL_PATH}\nRun scripts/download_model.sh gguf first.")

    random.seed(args.seed)

    phishing = load_jsonl(PROCESSED_DIR / "phishing.jsonl")
    legitimate = load_jsonl(PROCESSED_DIR / "legitimate.jsonl")

    phishing_sample = random.sample(phishing, min(args.count, len(phishing)))
    legitimate_sample = random.sample(legitimate, min(args.count, len(legitimate)))

    combined = [(s, "phishing") for s in phishing_sample] + \
               [(s, "legitimate") for s in legitimate_sample]
    random.shuffle(combined)

    out_path = PROCESSED_DIR / "training_data.jsonl"
    total = len(combined)
    written = 0
    failed = 0

    with open(out_path, "w") as out_f:
        for i, (sample, source_label) in enumerate(combined, start=1):
            prompt = build_prompt(sample)
            print(f"[{i}/{total}] source={source_label} sender={sample.get('sender', '')[:60]!r} ...", flush=True)

            output = run_seneca(prompt)
            if not output:
                print(f"  FAILED (no output / timeout), skipping")
                failed += 1
                continue

            record = {
                "messages": [
                    {"role": "system", "content": f"{SYSTEM_PROMPT}\nHeader Bilgisi: {sample['headers']}"},
                    {"role": "user", "content": sample["body"][:MAX_BODY_CHARS]},
                    {"role": "assistant", "content": output},
                ],
                "source_label": source_label,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()
            written += 1
            print(f"  OK ({len(output)} chars)")

    print(f"\nDone. {written} written, {failed} failed. Output: {out_path}")


if __name__ == "__main__":
    main()
