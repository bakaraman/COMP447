# Kadir Onboarding — Windows Setup

## Durum

ECT pipeline Colab'da çalışıyor. Batuhan cuma akşamı test etti: T4 + Blackwell üstünde sanity training, FID eval, sample export hepsi OK. Sen de aynı pipeline'ı kendi tarafından doğrulaman lazım ki pzt toplantısında ikimiz de "çalıştırdık" diyelim.

Senin asıl görevin pzt SONRASI: Heun baseline + latency profiling. Şimdilik sadece ortamı kur ve Cell 1-8b'yi çalıştır.

---

## Seçenek A: VS Code + Colab Extension (önerilen)

### 1. VS Code kur (zaten varsa geç)

https://code.visualstudio.com/ → Windows installer

### 2. Colab extension kur

VS Code içinde: Extensions (Ctrl+Shift+X) → "Colab" ara → **Google** yayıncısı olan "Colab" extension → Install

### 3. Repo'yu clone et

PowerShell veya Git Bash aç:

```powershell
cd ~\Downloads
git clone https://github.com/bakaraman/COMP447.git
cd COMP447
```

Git yoksa: https://git-scm.com/download/win → yükle, sonra tekrar dene.

### 4. Notebook'u aç

VS Code'da File → Open Folder → `COMP447` klasörünü seç.

Sol panelde `project/colab_first_run.ipynb` dosyasına tıkla.

### 5. Colab'a bağlan

Notebook sağ üstte **Select Kernel** butonu var:

1. Tıkla → **Colab** seç
2. **Connect to new runtime** veya **Auto Connect**
3. Google hesabına giriş yap (browser açılır)
4. GPU tipi sor: **T4** seç (L4/A100 varsa onlar da olur, daha hızlı)

### 6. Cell'leri sırayla çalıştır

**Cell 1** → Cell 8b'ye kadar sırayla. Her cell'e tıkla, **Shift+Enter** veya ▶ butonu.

Her cell'in üstünde markdown açıklama + "Expected output" notu var. Çıktın beklenenle uyuşuyorsa devam et.

**Cell 9'u ÇALIŞTIRMA.** Compute unit yakar, pzt'ye gerek yok.

### 7. Sorun çıkarsa

Notebook'u kaydet (Ctrl+S), WhatsApp grubuna at. Batuhan veya Claude çıktıyı okuyup fix yazar.

---

## Seçenek B: Browser Colab (extension istemezsen)

### 1. Repo'yu zip olarak indir

https://github.com/bakaraman/COMP447 → yeşil **Code** butonu → **Download ZIP**

ZIP'i Google Drive'a yükle.

### 2. Colab aç

https://colab.research.google.com → yeni notebook aç

Runtime → Change runtime type → **T4 GPU** → Save

### 3. İlk hücrelerde repo'yu aç

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
%cd /content
!cp /content/drive/MyDrive/COMP447-main.zip .
!unzip -q COMP447-main.zip
!mv COMP447-main COMP447
%cd /content/COMP447
!ls
```

Sonra Cell 4'ten itibaren `colab_first_run.ipynb` içindeki komutları sırayla çalıştır.

**Not:** Browser Colab'da `drive.mount()` çalışır (VS Code extension'daki bug yok).

### 4. Cell sırası (browser için)

Cell 1 (GPU check) → atla Cell 2 (drive zaten mount) → Cell 3 yerine yukarıdaki zip unpack → Cell 4'ten devam → Cell 8b'ye kadar.

---

## Windows'a özel dikkat edilecekler

### Git line endings

Windows Git bazen `\r\n` satır sonları yazar, Linux/Colab `\n` bekler. Sorun çıkmaması için:

```powershell
git config --global core.autocrlf input
```

Bu komutu **clone'dan ÖNCE** çalıştır. Zaten clone ettiysen sil tekrar clone et.

### PowerShell vs Git Bash

- **PowerShell:** `cd`, `ls`, `git` çalışır ama Unix komutları (`sed`, `grep`) yok
- **Git Bash:** Unix benzeri ortam, daha uyumlu
- **Tavsiye:** Git Bash kullan, PowerShell ile uğraşma

### Python local'de gerekli mi?

**Hayır.** Tüm Python kodu Colab'da (remote kernel) çalışıyor. Windows'ta Python kurulu olmasına gerek yok. Sadece VS Code + Colab extension + Git yeterli.

### Firewall / proxy

Kurumsal ağdaysan (okul WiFi, VPN):
- `github.com` açık olmalı (clone için)
- `colab.research.google.com` açık olmalı
- `*.googleapis.com` açık olmalı (Colab kernel bağlantısı)

Sorun çıkarsa hotspot dene.

---

## Pzt toplantısına kadar yapılacaklar

| Görev | Kim | Durum |
|---|---|---|
| Cell 1-8b sanity run | Kadir | ⬜ yapılacak |
| Pipeline'ın kendi Colab'ında çalıştığını doğrula | Kadir | ⬜ yapılacak |
| MONDAY_CHECKLIST.md oku | Kadir | ⬜ yapılacak |
| Heun baseline plan gözden geçir | Kadir | ⬜ opsiyonel |
| Toplantı soruları düşün | Kadir | ⬜ opsiyonel |

## Pzt toplantısından SONRA senin görevlerin

1. **Heun sampling** — `project/src/edm/generate.py` ile pretrained checkpoint'tan Heun sampling, farklı step sayılarında (5, 10, 18, 25, 50 step)
2. **Latency profiling** — `project/scripts/measure_latency.py` ile Heun vs ECT latency karşılaştırması (T4 üstünde)
3. **Break-even analysis** — `project/scripts/break_even.py` ile kaç sample üretince ECT'nin tuning maliyeti amorti edilir hesabı
4. **Tuning ablation yardımı** — 500/1000/1500/2000 kimg checkpointlarda 2-step FID ölçümü

---

## Önemli dosyalar

| Dosya | Ne işe yarar |
|---|---|
| `project/colab_first_run.ipynb` | Colab setup + sanity notebook |
| `project/MONDAY_CHECKLIST.md` | Pzt toplantı gündemi + throughput bulguları |
| `project/PLAN.md` | Proje planı + milestone'lar |
| `project/scripts/setup_ect.sh` | ECT + EDM clone + patch (otomatik) |
| `project/scripts/measure_latency.py` | Senin kullanacağın latency script |
| `project/scripts/eval_fid.py` | FID eval wrapper |
| `project/scripts/break_even.py` | Break-even hesaplama |
| `project/src/ect/` | ECT upstream repo (Colab'da clone'lanıyor) |
| `project/src/edm/` | EDM upstream repo (Heun baseline buradan) |
| `README.md` | Genel proje açıklaması |

---

## Sorular veya sorunlar

WhatsApp grubuna yaz. Notebook'u kaydet ve at → Batuhan veya Claude direkt okuyup fix yazar.
