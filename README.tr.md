# soc-log-triage

[English](README.md) | [Türkçe](README.tr.md)

Bir phishing triyaj hattı: bir e-postanın phishing olup olmadığına deterministik
bir parser ve kural motoru karar veriyor, yerel olarak fine-tune edilmiş 7B bir
model ise analiste sunulacak Türkçe raporu yazıyor. Model sınıflandırma yapmıyor.

---

## Temel tasarım kararı

"Phishing tespiti için LLM" demolarının çoğu dil modeline ham bir e-posta verip
karar istiyor. Bu proje bunu bilinçli olarak yapmıyor — çünkü önce o yol
denendi ve ölçülebilir biçimde başarısız oldu.

Önceki bir iterasyonda (v2) 70 gerçek e-posta 4-bit quantize edilmiş 7B bir
modele verilip sınıflandırma istendi. Sonuç: **3 sınıflı bir problemde %35
doğruluk — rastgeleden kötü.** Rakamın kendisinden daha kötüsü hata biçimiydi:
model karar veremediğinde destekleyici kanıt uyduruyordu. Bir örnekte bir maili
"phishing" diye etiketleyip gerekçe olarak "olağandışı X-Mailer header'ı"
gösterdi — o mailde X-Mailer header'ı hiç yoktu.

Kök sebep model boyutu değil. SPF/DKIM/DMARC sonuçları ve domain string
karşılaştırmaları gibi sinyaller **deterministik bir protokolün çıktısı**:
`pass`/`fail`/`none`, iki string'in eşit olup olmaması. Bunları bir dil
modeline verip "ağırlıklı toplamı sen hesapla" demek, model ne kadar büyük
olursa olsun kategorik olarak yanlış araç seçimi. Bu yüzden sorumluluk
ayrıldı:

```
.eml dosyası
    │
    ▼
┌──────────────────────┐
│  feature parser      │  Python email stdlib + BeautifulSoup
│  (deterministik)     │  → facts: dict
└──────────────────────┘
    │
    ▼
┌──────────────────────┐
│  kural motoru        │  skor tabanlı, eşikler config/rules.yaml'da
│  (deterministik)     │  → Phishing | Muhtemel Phishing | Güvenilir
└──────────────────────┘
    │
    ▼  facts + verdict
┌──────────────────────┐
│  LLM                 │  training: Qwen3.5-9B (teacher)
│  (rapor yazıcı)      │  inference: Seneca 7B + LoRA (student)
└──────────────────────┘
    │
    ▼  JSON (asla HTML değil)
┌──────────────────────┐
│  Jinja2 template     │  → HTML rapor
└──────────────────────┘
```

Bundan çıkan ve her yerde uygulanan iki kural:

1. **LLM sınıflandırma yapmıyor.** Kendisine zaten verilmiş bir kararı alıp
   gerekçelendiriyor. Çıktısındaki `risk_seviyesi` alanı kural motorunun
   verdiği kararla birebir aynı olmak zorunda; farklıysa örnek düşürülüyor.
2. **LLM JSON üretiyor, HTML değil.** HTML'i template basıyor. v2'de 70
   çıktının 9'unda hiç ayrıştırılabilir bir sınıflandırma yoktu ve 5 farklı
   format çıkmıştı; bu tasarım o sorun sınıfını tanım gereği ortadan
   kaldırıyor.

Bu bir **demo / proof-of-concept**, production sistemi değil. Modeller 16 GB'lık
fansız bir dizüstünde çalışacak şekilde seçildi. Amaç mümkün olan en iyi
doğruluğu almak değil, **yaklaşımın çalıştığını gösteren dürüst ve ölçülebilir
bir gösterim.**

---

## Sonuçlar

### Kural motoru (asıl sınıflandırmayı yapan bileşen)

Elle etiketlenmiş 30 maillik bir hold-out setinde (15 phishing, 15 legitimate)
kalibre edildi:

| Metrik | Değer | Anlamı |
|---|---|---|
| Recall | %86.7 (13/15) | Üst eşiğin üstünde yakalanan phishing |
| Yanlış-pozitif oranı | %0.0 (0/15) | Yanlışlıkla işaretlenen meşru mail |
| Abstention oranı | %23.3 (7/30) | Orta banda düşüp analiste bırakılan |

**Bu rakamları dikkatli okuyun.** Ağırlıklar ve eşikler *aynı* 30 mail üzerinde
ayarlandı — yani bu bir **kalibrasyon** sonucu, bağımsız bir doğrulama değil.
Örneklem küçük: 0/15 yanlış-pozitif, gerçek yanlış-pozitif oranının sıfır
olduğu anlamına gelmiyor — Wilson %95 güven aralığı üst sınırı kabaca %20
civarına çıkıyor. Tek bir "accuracy" rakamı bilinçli olarak raporlanmıyor:
motor üç sınıf üretirken ground truth ikili; orta bant abstention olarak
raporlanıyor, ki SOC bağlamında bu hata değil doğru davranış.

### LoRA fine-tuning (rapor yazıcı)

Teacher modelin ürettiği 229 training çifti, 206 train / 23 validation,
400 iterasyon (≈1.94 epoch).

**Adapter overfit etti.** Validation loss hiçbir noktada başlangıç değerinin
altına inmedi:

| Iter | 1 | 50 | 100 | 150 | 200 | 250 | 300 | 350 | 400 |
|---|---|---|---|---|---|---|---|---|---|
| Val loss | 1.301 | 1.343 | 1.491 | 1.462 | 1.406 | **1.393** | 1.535 | 1.439 | 1.424 |

Train loss 40. iterasyondan itibaren sürekli 0.000'da kaldı. 206 örnek, ~576
token'lık hedefler, `batch_size=1` ve 23M eğitilebilir parametre ile model
genelleme yerine training setini ezberledi.

Bu bir bulgu olarak kaydediliyor, gizlenmiyor. Validation loss bu projenin
başarı metriği değil — değerlendirme kriterleri şema uyumu, groundedness ve
Türkçe kalite (aşağıya bakın); ezberlemiş bir model bile bir formatı doğru
uygulayabilir. Ayrıca iyileştirme payı baştan dardı: **fine-tune edilmemiş**
baseline zaten %100 şema uyumu veriyordu.

### Baseline (fine-tune edilmemiş Seneca, karşılaştırma için)

| Metrik | Sonuç |
|---|---|
| Şema uyumu | 27/27 = %100 |
| Groundedness (ham) | %67.4 |
| Groundedness (düzeltilmiş) | %85.2 |

Ham ve düzeltilmiş groundedness arasındaki fark modelin değil, kontrol
aracının sözlük sınırlamasının sonucu: "desteksiz" sayılan 41 iddianın 28'i
regex'in hiç tanımadığı ifadeler kullanıyordu. Gerçek halüsinasyon oranı —
aracın anladığı ve facts'in yalanladığı iddialar — ~%15 (13/88).

---

## Değerlendirme kriterleri

LLM sınıflandırma yapmadığı için, sınıflandırma doğruluğu *parser'ın* metriği
ve ayrı raporlanıyor. Model şunlarla değerlendiriliyor:

1. **Şema uyumu** — çıktının ne kadarı rapor şemasına uyan geçerli JSON.
2. **Groundedness** — rapordaki her teknik iddianın `facts`'te karşılığı olup
   olmadığının programatik kontrolü (`src/eval/groundedness.py`).
3. **Türkçe kalite** — elle 1–5 rubric. BLEU/ROUGE kullanılmıyor; onlar bir
   referansla örtüşmeyi ölçer, bir güvenlik raporunun analiste doğru okunup
   okunmadığını değil.
4. **Sınıflandırma doğruluğu** — parser'ın metriği, ayrı raporlanıyor.

---

## Depo yapısı

```
config/
  rules.yaml            skor ağırlıkları ve eşikler (koda gömülü değil)
  lora.yaml             LoRA hiperparametreleri
schemas/
  facts.py              EmailFacts — parser'ın çıktı sözleşmesi
  report.py             Report — LLM'in çıktı sözleşmesi
src/
  parser/               deterministik özellik çıkarımı
    headers.py            SPF/DKIM/DMARC, adres tutarlılığı, marka adları
    urls.py               text/href uyumsuzluğu, IP tabanlı, punycode
    attachments.py        riskli ve çift uzantılar, arşivler
    body.py               gizli metin, sadece görsel gövde, aciliyet kalıpları
  rules/engine.py       ağırlıklı skorlama → verdict
  teacher/              teacher modelle training verisi üretimi
    generate_training_data.py
    prepare_lora_data.py
  eval/
    baseline.py           fine-tune edilmemiş ölçüm
    groundedness.py       iddia-vs-facts doğrulaması
scripts/
  anonymize.py          mailbox sahibinin kimliğini redakte eder
  check_anonymization.py doğrulama geçişi
  select_holdout.py     hold-out örnekleme
templates/
  report.html.j2        Jinja2 → HTML rapor
tests/                  78 birim testi
```

---

## Kurulum

Python 3.14 ve Apple Silicon gerekiyor (MLX çalışma zamanı Apple'a özgü).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Anonimleştirme yapılandırması

Hat tam olarak tek bir kimliği redakte ediyor: mailbox sahibinin kendi adı ve
e-posta adresi. Geri kalan her şey — gönderen domain'leri, IP'ler, üçüncü şahıs
adresleri — bilinçli olarak gerçek bırakılıyor. Gerekçe: domain'leri
anonimleştirmek modele gerçek domain yapısı hakkında hiçbir şey öğretmezken
sıfır gizlilik faydası sağlardı (gönderen ve marka domain'leri zaten kamuya
açık bilgi).

Bu kimlik ortam değişkeninden okunuyor ve asla commit edilmiyor:

```bash
cp .env.anonymize.example .env.anonymize
# sonra .env.anonymize dosyasını gerçek değerlerle doldurun
```

`.env.anonymize` gitignore'da. Dosya yoksa hat güvenli şekilde bozuluyor:
korpusu bozmak yerine hiçbir kişisel ismi redakte etmiyor.

### Çalıştırma

```bash
# birim testleri
for t in tests/test_*.py; do python3 "$t"; done

# bir korpusu parse edip anonimleştir
python3 scripts/parse_and_anonymize.py

# sahibinin kimliğinin sızmadığını doğrula
python3 scripts/check_anonymization.py

# LoRA eğitimi (uzun GPU işlerini uykuya karşı sarmalayın — aşağıya bakın)
caffeinate -dims mlx_lm.lora --config config/lora.yaml 2>&1 | tee logs/train.log
```

---

## Dizüstünde tekrar denemek isteyenler için notlar

Burada saatlere mal olmuş iki operasyonel ders:

**Uykuyu `caffeinate -dims` ile engelleyin, `-i` ile değil.** İki macOS kernel
panic'i (`completeMemory() prepare count underflow` @ `IOGPUMemory.cpp:550`)
training çalıştırmalarını düşürdü. Belirleyici ipucu panic raporundaki zaman
damgalarıydı: 19:05'te uyku, 19:18'de uyanma, 19:39'da panic. Uyku geçişine
"prepare" edilmiş halde giren Metal buffer'ları sürücünün sayacını bozuyor.
Tek başına `caffeinate -i` yetersiz — sadece idle sleep'i tutar. Güç kablosunu
takılı tutun; bataryada macOS bazı uyku politikalarını yine de uygular.

**`Train loss 0.000` mutlaka "model hiçbir şey öğrenmiyor" demek değil.**
mlx_lm bu değeri `{:.3f}` ile biçimlendiriyor, yani gerçek loss 0.0004 ise
ekranda `0.000` yazıyor. `Trained Tokens` ve `Tokens/sec` sayaçları da çöp
değer üretebiliyor (buradaki bir çalıştırma 229 örneklik bir veri setinde 1.06
milyar token raporladı). Güvenilir kontrol: bir checkpoint'i yükleyip `lora_b`
tensörlerine bakmak — bunlar sıfırla başlatılıyor ve ancak gradyan gerçekten
uygulandığında sıfırdan çıkıyorlar.

Bilinmesi gereken bir şey daha: mlx_lm'deki `Peak mem` fiziksel RAM kullanımı
değil. Unified memory üzerinden görülen kümülatif tepe *ayırma* miktarı (swap
dahil) ve hiç düşmüyor. 16 GB'lık bir makinede `42.881 GB` görmek yanlış okuma
değil — ağır swap'in göstergesi.

---

## Bilinen sınırlamalar

- Hold-out seti 30 mail. Ondan türeyen her metrik geniş bir güven aralığı
  taşıyor.
- Kural motoru eşikleri aynı hold-out üzerinde kalibre edildi, dolayısıyla o
  rakamlar bağımsız doğrulama değil kalibrasyon sonucu.
- Phishing korpusunun tahminen ~%43'ü salt ticari spam (marka taklidi, kimlik
  bilgisi talebi veya sahte aciliyet içermeyen) — heuristic bir denetime göre.
  Bu bir tahmin, doğrulanmış ground truth değil.
- 229 training örneğinin 9'u (%3.9) 4096 token sınırını aşıyor ve hedef JSON'u
  kesiliyor.
- Hold-out'ta bilinen 2 kaçırılmış phishing var. Biri Portekizce hukuki-tehdit
  sosyal mühendisliği, gerçek altyapıdan forward edilmiş ve SPF, DKIM, DMARC
  hepsi pass — header sinyalleriyle yakalanamaz.
- Fine-tune edilen adapter overfit etti; sonuçlar bölümüne bakın.

---

## Lisans ve veri

Bu depoya hiçbir e-posta verisi commit edilmiyor. `data/` ve `models/`
gitignore'da. Phishing korpusu kamuya açık
[`rf-peixoto/phishing_pot`](https://github.com/rf-peixoto/phishing_pot)
veri setinden geliyor; meşru mail korpusu kişisel bir mailbox export'u ve
yeniden dağıtılabilir değil.
