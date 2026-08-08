# soc-log-triage

[English](README.md) | [Türkçe](README.tr.md)

Bir phishing triyaj hattı: bir e-postanın phishing olup olmadığına deterministik
bir parser ve kural motoru karar veriyor, opsiyonel bir semantik katman kural
motorunun yapısal olarak göremediği sinyaller için gövde metnini okuyor, ve
yerel bir Qwen3.5-9B model analiste sunulacak rapora en fazla kısa bir senaryo
anlatımı ekliyor. **Model, HER İKİ MODDA DA sınıflandırma yapmıyor.**

---

## Hat (güncel)

```
.eml dosyası ya da yapıştırılmış ham mail metni
    │
    ▼
Router                yapısal: dosya uzantısı ya da ≥3 RFC 5322 header
    │
    ▼
Parser                 email stdlib + BeautifulSoup → EmailFacts
    │
    ▼
Rule Engine             ağırlıklı skor, config/rules.yaml → RuleAssessment
    │
    ▼
Semantic Extractor       (sadece hybrid) Qwen3.5-9B gövde metnini okuyup
(hybrid)                 kural motorunun göremediği sinyalleri çıkarır —
    │                    bulgular tipli, karar DEĞİL
    ▼
Evidence Validator       her bulgu gövdenin BİREBİR bir alt dizesi olmalı;
                         modelin parafraze ettiği ya da uydurduğu her şey
                         karara ulaşamadan reddedilir
    │
    ▼
Deterministic            RuleAssessment + doğrulanmış bulguları
Decision Policy          final_verdict'e birleştirir — sabit, denetlenebilir
                         bir kural seti, model çağrısı DEĞİL
    │
    ▼
Deterministic Report     risk_seviyesi, kategori gerekçesi, teknik bulgular
                         ve önerilen SOC aksiyonu TAMAMEN mekanik üretilir —
                         model bunların HİÇBİRİNE yazma erişimine sahip
                         değil, her iki modda da
    │
    ▼ (SADECE final_verdict != "Güvenilir" ise)
Optional Qwen            üç kısa cümle — olası senaryo, mailin alıcıdan
Narrative                istediği eylem, olası zarar — zaten tamamlanmış
                         raporun tek bir alanına yerleştirilir
    │
    ▼
Jinja2 template           → HTML rapor
```

İki mod, bu hattın farklı kısımlarından geçerek aynı rapor biçimine ulaşır:

| | fast | hybrid |
|---|---|---|
| Çalışan aşamalar | Router → Parser → Rule Engine → Deterministic Report | yukarıdakilerin tamamı |
| Model çağrısı | 0 | en fazla 2 (semantic extraction + narrative) |
| Süre (M2 Air) | ~1 saniye | ~60–270 saniye |
| Model ne zaman atlanır | her zaman | `rule_verdict` zaten "Phishing" (semantic çağrı); `final_verdict` "Güvenilir" (narrative çağrı) |
| Varsayılan | **evet** | sadece opt-in |

`fast`, hem CLI'da (`src/demo.py`) hem web arayüzünde (`src/web.py`)
varsayılan — kullanıcının model yüklenmeden önce açıkça `hybrid` istemesi
gerekiyor.

**"Model sınıflandırma yapmıyor" somut olarak ne demek:** `risk_seviyesi`
(Phishing / Muhtemel Phishing / Güvenilir), kategori gerekçesi, her teknik
bulgu ve önerilen SOC aksiyonu `src/report/mechanical.py` tarafından
üretiliyor — deterministik, şablon tabanlı, her iki modda da birebir aynı.
Qwen'in tek olası katkısı, hybrid modda ve karar "Güvenilir" değilse, üç
cümle parçası (`schemas/narrative.py::NarrativeDraft`) —
`src/report/assemble.py::apply_narrative()` bunu raporun tek bir alanına
(`genel_degerlendirme`) yerleştirip jenerik fallback metnini daha spesifik
bir metinle değiştiriyor. **Bu narrative metni tek başına güvenilir bir
çıktı sayılmıyor** — kararda hiçbir ağırlığı yok, üç sabit alana şema
kısıtlı (kaçak bir kategori ya da karar iddiasına yer yok), ve modelin
çağrısı başarısız olursa ya da şemaya uymayan bir şey üretirse rapor
mekanik fallback metnini korur, retry ya da onarım YOK
(`narrative_status="failed_fallback"`, sonuçta görünür, asla gizlenmez).

---

## Temel tasarım kararı

"Phishing tespiti için LLM" demolarının çoğu dil modeline ham bir e-posta verip
karar istiyor. Bu proje bunu bilinçli olarak yapmıyor — çünkü önce o yol
denendi ve iki farklı aşamada, iki farklı şekilde ölçülebilir biçimde
başarısız oldu.

**Birinci başarısızlık (v2, sınıflandırma):** 70 gerçek e-posta 4-bit
quantize edilmiş 7B bir modele verilip doğrudan sınıflandırma istendi.
**3 sınıflı bir problemde %35 doğruluk — rastgeleden kötü.** Rakamın
kendisinden daha kötüsü hata biçimiydi: model karar veremediğinde
destekleyici kanıt uyduruyordu. Bir örnekte bir maili "phishing" diye
etiketleyip gerekçe olarak "olağandışı X-Mailer header'ı" gösterdi — o
mailde X-Mailer header'ı hiç yoktu.

**İkinci başarısızlık (hybrid v1, kategori gerekçesi):** sınıflandırma
tamamen deterministik kural motoru + karar politikasına taşındıktan sonra
bile, rapor yazıcı hâlâ bir kararın **neden** geçerli olduğuna dair bir
gerekçe kategorisi (kapalı, altı öğelik bir sözlük) seçmesi isteniyordu.
Development-set ölçümü, modelin bu sabit sözlüğü 18 adayın 9'unda terk
ettiğini buldu — bunların %69'u "Güvenilir" kararlardı, temiz bir sonucu
saldırı-şekilli bir kategori listesinden geçirmeye çalışmak modeli tutarlı
şekilde sözlük dışına itiyordu. Çözüm aynı dersin bir katman daha derine
uygulanmasıydı: kategori gerekçesi de artık mekanik üretiliyor
(`src/report/categories.py`), ve modelin kalan tek yüzeyi ihlal edecek bir
sözlüğü olmayan üç narrative cümlesi (yukarıdaki Hat bölümüne bakın).

Kök sebep, her iki seferde de model boyutu değil. SPF/DKIM/DMARC sonuçları
ve domain string karşılaştırmaları gibi sinyaller **deterministik bir
protokolün çıktısı**: `pass`/`fail`/`none`, iki string'in eşit olup
olmaması. Sabit bir kuralın hangi kategori altında tetiklendiği de aynı
derecede deterministik — model görmeden önce kural motoru tarafından zaten
hesaplanmıştı. Bir dil modelinden bu tür bir cevabı yeniden üretmesini
istemek, model ne kadar büyük olursa olsun kategorik olarak yanlış araç
seçimi.

Bundan çıkan ve her iki modda, her yerde uygulanan iki kural:

1. **LLM sınıflandırma yapmıyor.** `risk_seviyesi`, kategori gerekçesi, her
   teknik bulgu ve önerilen aksiyon, herhangi bir model çağrılmadan ÖNCE
   `src/report/mechanical.py` tarafından üretiliyor. Hybrid modda, kabul
   edilen bir semantic bulgu kararı SADECE sabit, denetlenebilir bir karar
   politikası kuralı üzerinden (`src/decision/phishing_policy.py`)
   değiştirebilir — asla rapor yazıcı modelin kendisi üzerinden değil.
2. **LLM serbest metin ya da HTML değil, JSON üretiyor.** Her model çağrısı
   — semantic extraction, narrative üretimi — rapora ulaşmadan önce şema
   doğrulamasından geçiyor (`schemas/semantic.py`, `schemas/narrative.py`).
   Şemaya uymayan her şey onarılmadan düşürülüyor; eski Seneca+LoRA rapor
   yazıcının (aşağıdaki "Önceki iterasyon"a bakın) 70 çıktısının 9'unda hiç
   ayrıştırılabilir bir sınıflandırma yoktu ve 5 farklı format çıkmıştı —
   şema doğrulaması bu sorun sınıfını yamama yerine tanım gereği ortadan
   kaldırıyor.

Bu bir **demo / proof-of-concept**, production sistemi değil. Modeller 16 GB'lık
fansız bir dizüstünde çalışacak şekilde seçildi. Amaç mümkün olan en iyi
doğruluğu almak değil, **yaklaşımın çalıştığını gösteren dürüst ve ölçülebilir
bir gösterim.**

### Önceki iterasyon: Seneca + LoRA rapor yazıcı (v3, yerini aldı)

Yukarıdaki hybrid hattan önce, bu proje yerel bir 7B modeli
(Seneca-Cybersecurity-LLM) LoRA adapter'ıyla fine-tune ederek `facts +
verdict`'ten Türkçe raporu yazdırıyordu — semantic extraction aşaması ve
karar politikası olmadan, kural motorunun kararı nihaiydi, modelin tek işi
metin yazmaktı. O yol (`src/demo.py`'nin `--adapter`/`--constrain`/`--no-llm`
bayrakları, `src/teacher/`) kod olarak hâlâ repoda duruyor ve referans için
aşağıda belgeleniyor, ama **hybrid hatla BAĞLANTILI DEĞİL** — CLI'da
`--hybrid` ve `--adapter` birlikte kullanılamıyor, web arayüzü ise sadece
hybrid yolu sunuyor. Aşağıdaki LoRA sonuçları (overfit, baseline
karşılaştırması) o spesifik fine-tuning denemesinin neden karşılığını
vermediğinin tarihsel kaydı — güncel hat hakkında bir iddia değil.

---

## Sonuçlar

### Hybrid hat: semantic katman fayda sağlıyor mu?

Mevcut, önceden cache'lenmiş 18 hybrid koşusu (gerçek Qwen3.5-9B, önceki bir
reliability ölçümünden) kaynak etiketine (phishing korpusu vs. legitimate
mailbox export'u) göre yeniden analiz edilerek ölçüldü — **yeni bir model
koşusu DEĞİL**, ve bu, `src/semantic/analyze.py`'nin prompt'unun üzerinde
iterasyonla geliştirildiği AYNI 18 maillik development set, o yüzden bu
rakamlar o sette gözlenen davranışı anlatıyor, bağımsız bir benchmark değil:

| Metrik | Değer |
|---|---|
| Semantic katmanın değiştirdiği karar | 1/18 |
| Yanlış yönde yükseltme (legitimate mail "Güvenilir"in ötesine itildi) | bu örneklemde 0/18 |
| Doğru yönde yükseltme (phishing mail "Güvenilir"den çıkarıldı) | 1/18 |
| Semantic/model hata oranı | cache'deki kayıtların 0/18'i temiz işlenmedi anlamına DEĞİL — 18/18'i temiz işlendi |
| Hybrid latency, önceki koşularda kaydedilen | 60–268 sn (medyan 139 sn) |
| Fast mod latency | ~1 sn |

**Semantic extraction, kural motorunun kaçırdığı tek phishing örneğini
yükseltti ve bu örneklemde yeni bir yanlış-yönde yükseltme üretmedi.**
Yükseltilen mailde SPF/DKIM/DMARC hepsi pass'ti (kural motoru tek başına
"Güvenilir" diyordu), ama gövdesinde doğrudan alıntılanmış, doğrulanmış bir
kimlik bilgisi talebi vardı — tam olarak kural motorunun tanım gereği sahip
olduğu kör nokta (aşağıdaki "Bilinen sınırlamalar"a bakın). 18 mailde sıfır
yanlış-yönde yükseltme, BU örneklemin bir tanımı, gerçek oranın bir sınırı
DEĞİL — çok daha büyük, bağımsız çekilmiş bir set olmadan popülasyon
seviyesinde bir yanlış-yükseltme oranı iddia edilemez. **Bu development set
sadece gözlem için kullanıldı, kural motorunu ya da semantic prompt'u
yeniden kalibre etmek için ASLA** — aksi halde hat, ölçüldüğü sete uydurulmuş
(overfit) olurdu, genel probleme değil.

### Kural motoru (asıl sınıflandırmayı yapan bileşen)

Elle etiketlenmiş **80 maillik** bir hold-out setinde (15 phishing, 65
legitimate), 22 ağırlıklı sinyal ve ≥5 / 3–4 / <3 eşikleriyle ölçüldü:

| Metrik | Değer | Anlamı |
|---|---|---|
| Recall | %86.7 (13/15) | Üst eşiğin üstünde yakalanan phishing |
| Yanlış-pozitif oranı | %12.3 (8/65) | Yanlışlıkla işaretlenen meşru mail |
| Abstention oranı | %7.5 (6/80) | Orta banda düşüp analiste bırakılan |

Tek bir "accuracy" rakamı bilinçli olarak raporlanmıyor: motor üç sınıf
üretirken ground truth ikili, ve orta bant abstention — SOC bağlamında hata
değil doğru davranış.

#### Yanlış-pozitif oranı "%0.0"dan %12.3'e nasıl geldi

Bu README'nin önceki sürümü **%0.0 yanlış-pozitif** raporluyordu. O rakam 15
meşru mail üzerinde ölçülmüştü ve yanlıştı — hatalı hesaplanmış değil, o
örneklem boyutunda anlamsız. Sessizce değiştirmek yerine burada anlatılıyor,
çünkü nasıl çürüdüğü daha faydalı bir sonuç.

O zamanki uyarı, Wilson %95 üst sınırının ~%20 olduğuydu. Meşru tarafı 65
elle etiketlenmiş maile çıkarmak gerçek değeri **%26.2** gösterdi — o sınırın
bile üstünde.

Yanlış işaretlenenler sıradan mailler değildi: `google.com`,
`email.openai.com`, `discord.com`, `client.louisvuitton.com`,
`tr-info.adidas.com`. On yedinin on altısında SPF, DKIM ve DMARC hepsi
geçiyordu ve DKIM domain'i uyumluydu. İşaretlenme sebepleri şuydu:
doğrulanmış göndereni ödüllendiren tek sinyal
(`all_auth_pass_and_consistent`, −3) Return-Path'in From ile eşleşmesini de
şart koşuyordu — ve ESP kullanan her gönderen bounce'ları sağlayıcının
domain'i üzerinden yönlendirir. Meşru toplu mail bu bonusu **tanım gereği**
kazanamıyordu.

İki düzeltme oranı %12.3'e indirdi: o bonustan Return-Path şartını kaldırmak,
ve *geçerli* ama yanlış domain'e ait DKIM imzası için yeni bir sinyal eklemek
(üçüncü taraf spoofing — "DKIM eksik veya başarısız" kuralının hiç
kapsamadığı durum).

**İkisi de ayrı bir 60 maillik dev set üzerinde kalibre edildi, hold-out
üzerinde DEĞİL.** Hold-out'a göre ayarlamak onu bir eğitim setine çevirirdi.
Dev set %10.0, hold-out %12.3 çıktı — birbirine yeterince yakın, yani tek bir
örnekleme uydurma değil gerçek bir iyileşme.

#### Rakamlar hâlâ ne, ne değil

Ağırlıklar ve eşikler başlangıçta ilk 30 mail üzerinde ayarlandı, o yüzden
onlar **kalibrasyon** sonucu olmaya devam ediyor. Sonradan eklenen 50 meşru
mail hiç ayar için kullanılmadı — bu da yanlış-pozitif rakamını buradaki en
bağımsız ölçüme yaklaştırıyor. Recall hâlâ ilk 15 phishing maili üzerinde
ölçülüyor ve buna karşılık gelen geniş bir güven aralığı taşıyor.

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
  rules.yaml             skor ağırlıkları ve eşikler (koda gömülü değil)
  lora.yaml              LoRA hiperparametreleri (önceki iterasyon, aşağıya bakın)
schemas/
  facts.py               EmailFacts — parser'ın çıktı sözleşmesi
  report.py              Report — nihai, her zaman eksiksiz rapor sözleşmesi
  semantic.py            SemanticFindingCandidate / ValidatedSemanticFinding
  decision.py            PhishingDecisionContext / FinalDecision
  narrative.py           NarrativeDraft — modelin hybrid moddaki TÜM çıktı yüzeyi
src/
  demo.py                .eml girdi → HTML rapor, tek komut; --hybrid opt-in
  web.py                 aynı hat, tarayıcı arayüzünün arkasında
  web_ui.html            arayüzün kendisi (tek sayfa, build adımı yok)
  router.py              bu girdi hattın işleyebileceği bir şey mi?
  intent.py              router'ın çözemediği düz metin için persona seçici
  workflows/
    phishing.py            analyze_phishing() — fast/hybrid'in yaşadığı TEK yer;
                            CLI ve web ikisi de bunu çağırır, ikinci implementasyon yok
  parser/                 deterministik özellik çıkarımı
    headers.py              SPF/DKIM/DMARC, adres tutarlılığı, marka adları
    urls.py                 text/href uyumsuzluğu, IP tabanlı, punycode
    attachments.py          riskli ve çift uzantılar, arşivler
    body.py                 gizli metin, sadece görsel gövde, gateway banner'ı
  rules/engine.py         ağırlıklı skorlama → RuleAssessment
  semantic/
    analyze.py              Qwen3.5-9B semantic extraction (hybrid mod)
    validate.py              birebir alt dize doğrulaması — gövdede birebir
                             bulunamayan bir bulgu reddedilir
  decision/
    phishing_policy.py       decide() — bir semantic bulgunun kararı
                             değiştirebileceği TEK yer, sabit bir kural seti, model çağrısı değil
  report/
    mechanical.py            risk_seviyesi/kategori gerekçesi/bulgular/aksiyonun
                             TAMAMINI, deterministik, her zaman üretir
    narrative.py             Qwen'in narrative çağrısı — şema-girdi, şema-çıktı
    narrative_prompts.py     PII-minimize prompt kurulumu (ham gövde/konu yok)
    assemble.py              apply_narrative() — narrative'i tek bir alana yerleştirir
    categories.py            sabit kategori sözlüğü, mekanik uygulanır
  llm/service.py          paylaşılan QwenService — process başına tek model
                          yükleme, semantic + narrative çağrıları arasında paylaşılır
  teacher/                training verisi üretimi (önceki iterasyon, aşağıya bakın)
    generate_training_data.py
    prepare_lora_data.py
  eval/
    baseline.py             fine-tune edilmemiş ölçüm (önceki iterasyon)
    finetuned.py             fine-tune sonrası karşılaştırma (önceki iterasyon)
    groundedness.py          iddia-vs-facts doğrulaması
scripts/
  anonymize.py            mailbox sahibinin kimliğini redakte eder
  check_anonymization.py  doğrulama geçişi
  select_holdout.py       hold-out örnekleme
  expand_holdout_legitimate.py    hold-out'u sadece ekleyerek büyütme
  evaluate_hybrid_reliability.py  process-izole hybrid hat reliability ölçümü
  smoke_test_hybrid.py            tek mail, gerçek model smoke test
templates/
  report.html.j2          Jinja2 → HTML rapor
tests/                    500+ birim testi, gerçek model çağrısı yok
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

### Bir e-postayı analiz etme

```bash
# fast mod (varsayılan): parse → kural motoru → deterministik rapor → HTML  (~1 sn)
python3 src/demo.py mail.eml --open

# hybrid mod: + semantic extraction + karar politikası + narrative  (M2 Air'de ~60-270 sn)
python3 src/demo.py mail.eml --hybrid --open
```

`fast`, yeni bir maili, template'i ya da hattı günlük olarak hızlıca kontrol
etmek için doğru mod — hiçbir model yüklenmiyor, karar ve her teknik bulgu
`hybrid`'in üreteceğiyle birebir aynı, sadece narrative cümlesi eksik.
Mailin gövde metni kural motorunun yapısal olarak göremeyeceği bir sinyal
taşıyabilir diye düşünüyorsanız (aşağıdaki "Bilinen sınırlamalar"a bakın) ve
beklemeye değerse `--hybrid` kullanın.

`--hybrid`, hattın aşamalarını terminale ayrı ayrı yazdırır — rule verdict,
semantic bulgular (kabul edilen/reddedilen), final verdict, decision path,
narrative status — böylece modelin bir şeye katkı verip vermediği, yoksa
deterministik katmanların her şeye tek başına karar verip vermediği her
adımda görünür kalır.

**Önceki**, yerini alan Seneca+LoRA rapor yazıcıdan (yukarıdaki "Önceki
iterasyon"a bakın) kalan iki bayrak daha var, `--hybrid` ile birlikte
kullanılamıyorlar:

- `--adapter 0000400` LoRA adapter'ını Seneca'nın üstüne takar. Varsayılan
  kapalı — her iki metrikte de daha kötü ölçüldü.
- `--constrain` üretimi `llguidance` ile rapor şemasına kısıtlar, bozuk
  JSON'u yapısal olarak imkânsız kılar. O da varsayılan kapalı: buradaki
  bütün rakamlar kısıtsız ölçüldü, demo'nun farklı koşulda çalışması ikisini
  de yanlış tanıtırdı. Modelin string içinde ısrarla kaçışsız tırnak ürettiği
  ve JSON'un bir türlü ayrıştırılamadığı maillerde işe yarıyor.

### Tarayıcı arayüzü

```bash
python3 src/web.py          # http://127.0.0.1:8000
```

Ham maili yapıştırın ya da `.eml` dosyasını sürükleyin. Yönlendirme kararını,
kural motorunun verdict'ini (tetiklenen her sinyal ve ağırlığıyla) ve
render edilmiş raporu gösterir. Bir "Hybrid analiz" anahtarı (varsayılan
kapalı, CLI ile aynı fast/hybrid ayrımı) işaretlendiğinde semantic bulguları,
final verdict'i, decision path'i ve narrative status'u ayrı kartlar olarak
gösterir.

Web katmanında hiçbir analiz mantığı yok — her iki mod da CLI'ın çağırdığı
aynı `analyze_phishing()`'i (`src/workflows/phishing.py`) çağırıyor; bu dosya
onun etrafında ince bir FastAPI kabuğu, ikinci bir implementasyon değil.

### Yönlendirme

Router dört makine-okunur sonuç üretir: `phishing_direct`,
`phishing_missing_email`, `needs_clarification` ve `unsupported`. Geçerli
`.eml`/ham mail intent modelini atlayarak doğrudan hatta girer. Güvenilir bir
upstream servis `trusted_route_hint="phishing"` verebilir; bu metadata son
kullanıcının serbest metninden türetilmemelidir.

```bash
python3 src/router.py mail.eml              # hat bunu alabilir mi?
python3 src/router.py --text "$(pbpaste)"   # yapıştırılmış mail
python3 src/router.py --text "SPF nedir?" --classify   # + niyet sınıflandırıcı
```

### Bakım

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

- **Recall 15 phishing maile dayanıyor.** Meşru taraf 65'e çıkarıldı ama
  phishing tarafı çıkarılmadı — büyütmek düşmanca örnekleri elle etiketlemeyi
  gerektiriyor, çünkü korpusun tahminen ~%43'ü salt ticari spam ve kaynak
  klasörü etiket olarak güvenilir değil. %86.7 bu yüzden geniş bir aralık
  taşıyor.
- Ağırlıklar ve eşikler başlangıçta ilk 30 mail üzerinde kalibre edildi, o
  rakamlar kalibrasyon sonucu olmaya devam ediyor. Sonraki düzeltmeler ayrı
  bir dev sette ayarlandı (yukarıya bakın).
- **Kural motoru tek başına zarfı okuyor, mektubu değil.** 22 sinyalinin
  19'u header, URL ya da ek dosyaya bakıyor. Kimlik doğrulaması temiz, linki
  ve eki olmayan bir mail, metni ne kadar açık şekilde dolandırıcı olursa
  olsun `fast` modda motora görünmez. Bilinen iki kaçak tam olarak bu: gerçek
  altyapıdan forward edilmiş, SPF/DKIM/DMARC hepsi pass Portekizce
  hukuki-tehdit sosyal mühendisliği; ve gerçek bir `.edu.tr` hesabından
  gönderilmiş 419 avans-ücreti dolandırıcılığı. **`hybrid` mod tam olarak bu
  boşluğun bir kısmını kapatmak için var** — semantic katman kural motorunun
  göremediği gövde metnini okuyor — ama opt-in, M2 Air'de ~60–270 saniyeye
  mal oluyor, ve yukarıdaki 18 maillik gözlem (1/18 yükseltme, 0/18 yanlış
  yön) boşluğun genel olarak ne kadarını kapattığını sınırlamak için çok
  küçük bir örneklem.
- **Narrative cümlesi tek başına güvenilecek bir iddia değil.** Offline
  teacher/adapter yolunun groundedness metriğinin teknik iddiaları kontrol
  ettiği şekilde (yukarıdaki "Değerlendirme kriterleri"ne bakın)
  `EmailFacts`'e karşı doğrulanmıyor — üç narrative cümlesi için henüz eşdeğer
  bir groundedness kontrolü yok. Kararda hiçbir ağırlığı yok, yeni bir teknik
  bulgu ya da kategori getiremiyor (şemada ikisi için de alan yok), ve
  başarısız ya da şemaya uymayan bir narrative çağrısı raporu engellemek
  yerine mekanik fallback metnini koruyor (görünür şekilde,
  `narrative_status` üzerinden, asla gizlenmeden) — ama cümlelerin kendisi
  bağımsız doğrulanmış bulgular değil, bir senaryo özeti olarak okunmalı.
- Önceki-iterasyon 229 LoRA training örneğinin 9'u (%3.9) 4096 token
  sınırını aşıyor ve hedef JSON'u kesiliyor — yukarıdaki "Önceki iterasyon"a
  bakın; bu, hiç fine-tuning yapmayan güncel hybrid hattı ETKİLEMİYOR.
- Önceki iterasyonun fine-tune edilen adapter'ı overfit etti; sonuçlar
  bölümüne bakın. Güncel hat Qwen3.5-9B'yi hiç değiştirmeden kullanıyor,
  fine-tuning yok.
- **Router yalnızca 1. aşama.** "Bu bir e-posta mı?" sorusunu yapıdan
  cevaplıyor. Arkasındaki niyet sınıflandırıcı (`--classify`) `titus` ya da
  `cybersec_qa` adını verebiliyor, ama ikisi de bu repoda kurulmadı —
  yönlendiriyormuş gibi yapmak yerine bunu söylüyor.

---

## Lisans ve veri

Bu depoya hiçbir e-posta verisi commit edilmiyor. `data/` ve `models/`
gitignore'da. Phishing korpusu kamuya açık
[`rf-peixoto/phishing_pot`](https://github.com/rf-peixoto/phishing_pot)
veri setinden geliyor; meşru mail korpusu kişisel bir mailbox export'u ve
yeniden dağıtılabilir değil.
