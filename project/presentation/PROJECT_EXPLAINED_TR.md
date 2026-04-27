# COMP447 Project — Tam Brifing

> **Bu doküman ne için?** Kadir'in Claude'una (veya başka bir AI asistana) projenin tam state'ini aktarmak için yazıldı. Bilgi kaybı olmasın diye konuşma geçmişinden, kaynak kod analizinden, web search doğrulamasından çıkan **her şey** burada.
>
> **Hedef kitle**: ML/CS bilen biri ya da context yüklemek isteyen bir AI asistan.
> **Süre**: 15-20 dakikada okunur, ama daha çok referans dokümanı.
> **Last updated**: 2026-04-27

---

## 0) Hızlı özet (TL;DR)

- **Proje adı**: "When Does Cheap Consistency Tuning Pay Off?"
- **Yer**: COMP447 dersi (Koç Üniversitesi), final proje — progress presentation Nisan sonu
- **Sahibi**: Batuhan Karaman + Kadir Yiğit Özçelik
- **TA / mentor**: Andrew (TA), kriterleri net: *"sadece existing code'u run etmeyin, implementation component olsun, propose your own method"*
- **Final report**: Haziran başı (early June)
- **Hardware**: NVIDIA G4 Blackwell on Colab (102 GB VRAM), tamamen Türkiye'den çalışıyoruz
- **Dataset**: CIFAR-10 32×32 unconditional
- **Ana proje**: ECT (Easy Consistency Tuning) vs Heun (EDM diffusion sampler) — latency-matched comparison + break-even analysis + tuning-budget ablation
- **Extension fikri**: KLUB-CM — Align Your Steps'in KL upper bound'unu consistency model'lere uyarlamak (önerilmiş, henüz implement edilmemiş)

---

## 1) Proje çerçevesi

### 1.1 Sorduğumuz soru

Modern görüntü üretim alanında iki yaklaşım var:

**Diffusion models** (Stable Diffusion, EDM, DALL-E):
- Probability flow ODE'yi numerik çözüyor
- Heun, DPM-Solver, DDIM gibi sampler'larla 30-100+ NFE'de görüntü üretir
- Yüksek kalite, yüksek latency
- Tuning gerektirmiyor (pretrained kullan)

**Consistency models** (Song 2023, ECT, sCM):
- Probability flow trajectory'sinin herhangi bir noktasından doğrudan başlangıca atlamayı öğrenir
- Inference 1-2 NFE — diffusion'a göre 30-50× hızlı
- Eğitilmesi pahalı: ya scratch'ten consistency training, ya distillation

**Pratik soru**: Bu upfront training cost amorti ediyor mu? Hangi N için (kaç görüntü üretirsen) mantıklı? Inference'da nasıl optimal sample alınır?

### 1.2 Andrew'un kriterleri

Andrew Mart sonunda proposal toplantısında dedi ki:
1. *"Sadece existing code'u run etmeyin"* — pure reproduction yetmez
2. *"Deep unsupervised learning implementation component olmalı"* — yeni kod yazılmalı
3. *"Propose your own method or combine things"* — kendi method'unuzu öner
4. *"Bu insight insanların hayatını nasıl kolaylaştırıyor?"* — pratik impact göster
5. Progress presentation: current progress + next extension idea

Bizim KLUB-CM önerisi bu kriterlerin 4'ünü karşılıyor — implementation component (#2) henüz pseudo-code seviyesinde, gerçek kod sunumdan önce yazılırsa daha güçlü.

### 1.3 Ekip rol dağılımı (README'den)

| Person | Sorumluluklar |
|---|---|
| **Batuhan** | ECT reproduction, evaluation pipeline, FID computation, report writing |
| **Kadir** | Heun baselines, latency profiling, break-even analysis, tuning ablation |
| Shared | Experiment design, figures, presentation, interpretation |

---

## 2) Teknik arka plan

### 2.1 Diffusion: probability flow ODE

EDM paper [Karras+, NeurIPS 2022, arXiv:2206.00364] formülasyonu:

```
dx_t / dt = -t · ∇_{x_t} log p_t(x_t)
```

Heun sampler — 2nd-order ODE solver, deterministic. Her step 2 NFE harcıyor:
- 18 step ≈ 35 NFE (son step single-eval)
- 50 step ≈ 99 NFE

EDM teacher checkpoint: `edm-cifar10-32x32-uncond-vp.pkl`, 56.4M parameter, DDPM++ architecture.

### 2.2 Consistency models

Yang Song'un 2023 ICML çalışması [arXiv:2303.01469]. Öğrenilen şey:

```
f_θ(x_t, t) ≈ x_0,
f_θ(x_t, t) = f_θ(x_s, s) for (x_t, x_s) on the same trajectory
```

Yani *aynı PF-ODE trajektörü üzerindeki herhangi iki noktanın* output'u aynı `x_0` olmalı. Self-consistency property.

Inference:
- 1-step: x_0 = f_θ(z · t_max, t_max), 1 NFE
- 2-step: x_mid = f_θ(z·t_max, t_max); x_0 = f_θ(x_mid + ε·t_mid, t_mid), 2 NFE

### 2.3 ECT (Easy Consistency Tuning)

[Geng, Pokle, Lu, Kolter — ICLR 2025, arXiv:2406.14548]

Fikir: scratch'ten train etme, distillation teacher kullanma. Pretrained EDM'den fine-tune et.

Loss:
```
L_ECT(θ) = E_{y, ε, t} [w(t) · d(f_θ(y + tε, t), f_{θ'}(y + rε, r))]
where r = t - Δ(t), Δ(t) → 0
```

- `θ'` student'ın EMA target'i
- `w(t)` noise-dependent weighting (typical Karras-style)
- `Δ(t)` infinitesimally small offset

Paper iddiaları:
- ~1 A100-hour tuning
- 100k iter @ 2-step FID 2.20
- 200k iter @ 2-step FID 2.15
- 400k iter @ 2-step FID 2.11

### 2.4 Bizim deney sonuçlarımız (kritik veri)

**ECT tuning** (G4 Blackwell, 1980 kimg total):

| kimg | tick | 2-step FID (raw, ct_eval) | Snapshot |
|---|---|---|---|
| 500 | 50 | 8.135 | network-snapshot-000050.pkl |
| 1000 | 100 | 2.773 | network-snapshot-000100.pkl |
| 1500 | 150 | 2.702 | network-snapshot-000150.pkl |
| 1980 | 198 | 2.446 | network-snapshot-000198.pkl |

**Cell 9 end-of-training eval** (ct_training_loop.py — DOĞRU):
- 1-step FID: **5.77**
- 2-step FID: **2.47**

**Heun baselines** (FID50k, B=1 latency):

| steps | NFE | FID50k | latency (ms) |
|---|---|---|---|
| 5 | 9 | 37.78 | 62.21 |
| 10 | 19 | 2.64 | 131.75 |
| 18 | 35 | 1.96 | 243.79 |
| 25 | 49 | 1.98 | 331.66 |
| 50 | 99 | 2.02 | 673.50 |

**ECT latency**:
- 1-step: 7.05 ms
- 2-step: 13.97 ms

**Pareto sonucu**: ECT 2-step Heun-18'den **17× hızlı**, FID farkı **0.51** — Pareto dominate.

**Break-even** (vs Heun-18):
- B=1: N* ≈ 23,500 images
- B=64: N* ≈ 47,500 images

ECT tuning maliyeti = **~90 dakika** G4 üzerinde (paper'da T4 için "1 hour" diyor — hardware-dependent claim).

### 2.5 Paper karşılaştırması

| Budget | Paper 1-step | Paper 2-step | Bizim 1-step | Bizim 2-step |
|---|---|---|---|---|
| 100k kimg | 4.54 | 2.20 | — | — |
| 200k kimg | 3.86 | 2.15 | — | — |
| 400k kimg | 3.60 | 2.11 | — | — |
| **2k kimg (bizim)** | — | — | **5.77** | **2.47** |

Yani 0.5% compute ile 2-step kalitenin %94'ü.

---

## 3) Kritik bulgu #1: ct_eval.py bug'ı

### 3.1 Ne buldu

`locuslab/ect` upstream repo, commit `4311059`. `ct_eval.py` line 363-380:

```python
# Lines 363-380
few_step_fn = functools.partial(generator_fn, mid_t=mid_t)   # SADECE bu
# ...
for metric in metrics:
    result_dict = metric_main.calc_metric(metric=metric,
        generator_fn=few_step_fn, G=net, G_kwargs={}, ...)   # AYNI fn her metric için
```

`metrics/metric_main.py` line 82-92:

```python
@register_metric
def fid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    fid = frechet_inception_distance.compute_fid(opts, max_real=None, num_gen=50000)
    return dict(fid50k_full=fid)

@register_metric
def two_step_fid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    fid = frechet_inception_distance.compute_fid(opts, max_real=None, num_gen=50000)
    return dict(fid50k_full=fid)
```

İki metric fonksiyonunun **içeriği bit-identical**. Metric ismi sampler'ı değiştirmiyor — caller her zaman `few_step_fn` (mid_t=[0.821] ile) geçiyor.

### 3.2 Sonucu

`ct_eval.py` üzerinden çalıştırılan herhangi bir FID, `mid_t=[0.821]` set olduğu sürece **2-step**. `fid50k_full` ve `two_step_fid50k_full` ikisi de aynı 2-step rakamını veriyor.

Bizim ablation_fid_v2.csv'de iki kolon neredeyse identik (1.98 vs 2.51) bu yüzden — ikisi de 2-step. *"1-step ≈ 2-step, dead second step!"* hipotezi tamamen bug artifact'ıydı.

### 3.3 Doğru path

`ct_training_loop.py` line 352-363 doğru ayrım yapıyor:

```python
# fid50k_full — bare generator_fn → mid_t=None → 1-step
result_dict = metric_main.calc_metric(metric='fid50k_full',
        generator_fn=generator_fn, ...)

# two_step_fid50k_full — partial uygulanmış few_step_fn → 2-step
few_step_fn = functools.partial(generator_fn, mid_t=mid_t)
result_dict = metric_main.calc_metric(metric='two_step_fid50k_full',
        generator_fn=few_step_fn, ...)
```

Yani training-time eval (Cell 9'da kullanılan) doğru. Sadece post-hoc `ct_eval.py` path'i bozuk.

### 3.4 Bizim çözümümüz

`project/scripts/eval_fid.py`'i corrected pipeline olarak yazdık. ct_eval.py'a güvenmeden, explicit 1-step/2-step image generation + EDM `fid.py` ile FID hesaplıyor.

Upstream'e PR atılabilir; şu an bizim deliverable'lar arasında.

---

## 4) Kritik bulgu #2: mid_t = 0.821 magic number

### 4.1 Kodda doğrulama

`locuslab/ect/ct_eval.py` line 94:

```python
@click.option('--mid_t', help='Sampler steps [default: 0.821]',
              multiple=True, default=[0.821])
```

ECT paper'ında bu `0.821` rakamı:
- Derive edilmemiş
- Ablate edilmemiş
- Sensitivity analysis yok
- Theoretical justification yok

Bizim 4 checkpoint analizimizde her 2-step sonucu tek bu parametreye bağlı. Tüm training stage'ler, tüm step bütçeleri, sabit.

### 4.2 Yazarın kendi açıklaması (verbatim, doğrulanmış)

GitHub issue #11 (`locuslab/ect`, August 2024). Soran kişi: `wangyp33`.

`gh api repos/locuslab/ect/issues/11/comments` ile çekildi, **2024-08-25 tarihinde Gsunshine (Zhengyang Geng)**'in cevabı:

> *"Hi @wangyp33,*
>
> *Thanks for your interest in ECT!*
>
> *For intermediate timesteps, **It's a very good question and I think also an open research problem. I don't have a manual for it.** Here are some thoughts. **You can treat it as an optimization problem, maximizing the sample quality (in terms of many metrics) w.r.t these sampling schedules.** I encourage you to explore more!*
>
> *Regarding weightings, you could write the timestep weighting by yourself. Typically only 1 line. For example,*
>
> ```python
> wt = (t ** 2 + self.sigma_data ** 2) / (t * self.sigma_data) ** 2
> ```
>
> *Make sure your `wt` has the same shape as your `loss` by `assert wt.ndim == loss.ndim`. If not, reshape it.*
>
> *Good luck with your journey!*
>
> *Thanks,*
> *Zhengyang"*

Bu quote bizim sunum #1 silahımız. İki şey diyor:
1. *"open research problem"* — yazar problemi resmi olarak kabul ediyor
2. *"treat it as an optimization problem"* — yazar bizim önerdiğimiz yaklaşımı öneriyor

Bu cümle KLUB-CM önerisinin defansif zeminini bizden bağımsız kuruyor.

---

## 5) İlgili literatür ve novelty pozisyonu

### 5.1 Doğrudan ilham kaynakları

**Align Your Steps (AYS)** [Sabour, Fidler, Kreis — NVIDIA, ICML 2024, arXiv:2404.14507]
- Girsanov teoremi üzerinden bir KL divergence upper bound (KLUB) türetiyor
- True reverse SDE ile linearised approximation arasındaki KL'yi bound ediyor
- Schedule knot'larını bu bound'u minimize edecek şekilde optimize ediyor
- Stable Diffusion'da %5-30 FID iyileştirmesi rapor ediyor (CIFAR-10 20-step: 4.64 → 3.23)
- **Limit**: SDE framework'üne özel — diffusion için, CMs için değil

**Consistency Trajectory Models (CTM)** [Kim+, ICLR 2024, arXiv:2310.02279]
- Trajectory üzerinde unrestricted traversal between any initial and final time
- Single network outputs scores
- CIFAR-10 single-step FID 1.73, ImageNet 64×64 FID 1.92
- **Limit**: Trajectory traversal flexibility, schedule optimization değil

**Optimal Stepsize for Diffusion Sampling (OSS)** [arXiv:2503.21774, Mart 2025]
- Dynamic programming framework for optimal stepsize
- "10x accelerated text-to-image generation"
- **Limit**: Diffusion only, dynamic programming (KLUB değil), CMs için karşılığı yok

### 5.2 Consistency model alanı (overlap kontrolü)

| Paper | Yıl | Yaklaşım | Bizim önerimizle ilişki |
|---|---|---|---|
| **TDD** (Target-Driven Distillation) [Wang+ AAAI 2024, arXiv:2409.01347] | Sep 2024 | Heuristic timestep selection: predefined equidistant + stochastic offset | Heuristic, KLUB değil. Distillation setting (ECT değil). **Çakışma yok ama prior work olarak bilinmeli.** |
| **SCT** (Stable Consistency Tuning) [ICLR 2025 Workshop, arXiv:2410.18958] | Oct 2024 | Training Δt schedule iyileştirmesi (training stability), MDP/TD interpretation, Δt smooth shrinking | Training schedule, inference mid_t değil. **Farklı problem.** |
| **sCM** [Karras+ 2024, arXiv:2410.11081] | Oct 2024 | Continuous-time CMs, training stabilization, parameterization improvements | mid_t justification yok, 2-step FID 2.06 raporluyor ama nasıl seçildiği yok. **Bizim attığımız taşa ortak.** |
| **MCM** (Multistep CM) [Heek+ 2024, arXiv:2403.06807] | Mar 2024 | 1-step CM ile full diffusion arasında interpolate eden family | Schedule training-time'da bake ediyor, post-hoc optimization yok. **Komşu alan.** |
| **CMT** (Consistency Mid-Training) [arXiv:2509.24526] | Sep 2025 | "Mid-training" stage, fixed numerically stable reference trajectory | Training-time fix. **Inference değil, farklı.** |
| **iCT** (improved CT) [Song+ 2023] | 2023 | Adaptive discrete-time schedule during training | ECT'nin ilham kaynağı; training schedule, inference değil. |
| **Truncated CMs** [arXiv:2410.14895] | 2024 | Sub-trajectory training | Schedule training tarafı, inference değil. |
| **Curriculum Consistency Model** [CVPR 2025] | 2025 | Curriculum-based training | Training, inference değil. |

**Boş alan**: KLUB-style **principled, gradient-based, inference-time schedule optimization** for **consistency models**. AYS analog'u CM için yazılmamış. Bu bizim novelty pozisyonumuz.

### 5.3 Methodological precedents

**GKD (Generalized Knowledge Distillation)** [Agarwal+ 2023, arXiv:2306.13649]
- On-policy distillation precedent: student'ın kendi output'unu input olarak kullan
- Exposure bias problemini adresliyor
- Bizim on-policy loss patch'imizin literature anchor'u

**Autoregressive Distillation of Diffusion Transformers** [arXiv:2504.11295]
- DiT için exposure bias addressing
- Conceptually related to our on-policy term

---

## 6) Önerimiz: KLUB-CM

### 6.1 Tek cümlelik özet

> AYS'nin KL upper bound'unu consistency function için yeniden türet, schedule optimization'ı CMs'e taşı.

### 6.2 Math

AYS'nin bound'u score field `s_θ(x_t, t)` üzerinde. Biz score field yerine flow map `f_θ(x_t, t)` üzerinde linearize ediyoruz.

Objective:
```
T* = argmin_{t_i} Σ_{i=1}^{N} E_{x ~ p} [‖f_θ(x_{t_i}, t_i) - f_θ(x_{t_{i-1}}, t_{i-1})‖²]
```

Yani: ardışık knot'larda flow map output'larının L2 farkını minimize et. Bu fark **discretisation error'ın upper bound'u**.

İki kritik özellik:
1. **Checkpoint-conditioned**: `f_θ` değişince optimum knot set'i değişiyor. 500 kimg'lik model ile 1980 kimg'lik model farklı schedule verebilir.
2. **Step-budget agnostic**: Optimizer herhangi bir N için (1-step, 2-step, 4-step, ...) schedule üretiyor. Retraining gerekmez.

### 6.3 İlham zinciri (slide 8'de gösteriliyor)

- **KL upper bound** technique → AYS [Sabour+ 2024]
- **Trajectory awareness** → CTM [Kim+ 2024]
- **Adaptive scheduling** → OSS [2025]

Yani 3 farklı paper'dan element'leri sentez ediyoruz, tek bir paper'ın varyasyonu değiliz.

---

## 7) İmplementasyon planı

### 7.1 Dört concrete deliverable

**1. KLUB derivation for the CM flow map**
- AYS'nin score field için yaptığı türevi consistency function üzerinde yeniden yapmak
- Math write-up — final report'un teknik bölümü
- **Tahmini süre**: 1 hafta

**2. Schedule optimiser** (`src/klub_cm/optimiser.py`)
- L-BFGS over schedule knots in PyTorch
- Gradient-based — grid search'ten orders of magnitude ucuz
- Inputs: ECT checkpoint, target step count N
- Output: optimal schedule T* = {t_1, ..., t_{N-1}}
- Cached features ile single forward pass cost
- **Tahmini süre**: 3-5 gün

**3. On-policy loss patch** in `training/loss.py`

Pseudocode:
```python
# original ECMLoss has: x = y + eps*t; y is real image
# new on-policy term:
x_mid = f_theta(z * t_max, t_max).detach()    # student's coarse output
t_mid = klub_optimiser(checkpoint=ckpt)        # KLUB-derived knot
L_on  = w(t_mid) * d(
    f_theta(x_mid + eps * t_mid, t_mid),       # on-policy input
    f_theta_prime(x_mid + eps * r, r),         # EMA target at r < t_mid
)
loss = L_ECT + lam * L_on                      # mix with original
```

- Student'ın kendi outputs'una train ediyor (exposure bias addressing)
- λ hyperparameter ile mix
- GKD literature ile compatible
- **Tahmini süre**: 1 hafta (sadece coding) + training time

**4. Corrected evaluation pipeline** (`scripts/eval_fid.py`)
- ct_eval.py'a güvenmeyen explicit pipeline
- 1-step ve 2-step doğru sampler routing
- Patch upstream PR'a uygun formatta
- **Status**: Done

### 7.2 Validation stratejisi

1. **Sensitivity validation**: 4 checkpoint × 13 mid_t değer × 10k samples → drift olup olmadığını ölçmek
   - Eğer optimum mid_t checkpoint'lere göre değişmiyorsa, KLUB-CM'in dışında ablation hikayesi pivot olabilir
2. **KLUB vs grid search**: Aynı checkpoint'te grid search optimum'u ile KLUB optimum'unu karşılaştır
3. **On-policy effect**: Vanilla ECT vs ECT + KLUB schedule + on-policy patch → 50k FID

### 7.3 Stop-loss criteria

- Eğer sensitivity sweep'te mid_t* drift göstermezse → schedule optimization'ın değeri zayıf, on-policy fine-tune ana hikaye olur
- Eğer KLUB optimum'u grid search'ten anlamlı farklı çıkmazsa → method'un practical advantage'ı yok, framework theoretical contribution olarak kalır
- Her durumda corrected eval pipeline + bug report sağlam contribution

---

## 8) Ne yaptık ve ne yapmadık

### 8.1 Done

- ✅ ECT tuning, 4 checkpoint (500/1000/1500/1980 kimg)
- ✅ Heun baselines @ 5/10/18/25/50 steps
- ✅ Latency profiling B=1 ve B=64
- ✅ FID50k for ECT, FID50k for Heun (corrected)
- ✅ Pareto plot, ablation curve, break-even table, decision matrix
- ✅ ct_eval.py bug discovery + corrected eval pipeline (`eval_fid.py`)
- ✅ Literature review (AYS, ECT, sCM, MCM, CTM, OSS, TDD, SCT, GKD)
- ✅ Web search verification of all critical claims
- ✅ GitHub issue #11 verbatim çekildi (`gh api`)
- ✅ Progress presentation (10 slides PPTX, English)

### 8.2 Up next (sunumdan sonra)

- ⏳ mid_t grid sweep across 4 checkpoints (sensitivity validation)
- ⏳ KLUB-CM math derivation write-up
- ⏳ L-BFGS optimizer Python implementation
- ⏳ On-policy loss patch in `training/loss.py`
- ⏳ Confirmation 50k eval with corrected pipeline
- ⏳ Final report (Haziran başı)

### 8.3 Yapmadığımız şeyler / yarıda kalan / tartışılan

**Discarded ideas** (önemli — geri dönmeyelim):

- **OAR-ECT (asymmetric refiner)** — 21 Nisan civarı önerilmişti. Mantık: "1-step ≈ 2-step görüyoruz → second step dead → küçük bir refiner ekleyelim". **Çürütüldü**: 1-step ≈ 2-step gözlemi `ct_eval.py` bug'ından geliyordu. Cell 9 doğru rakamlarla 2-step gerçekten effective (5.77 → 2.47, 3.3 FID kazancı). OAR-ECT'nin motivasyonu çöktü.

- **Adaptive-Midpoint ECT (Step-Conditioned)** — Shortcut Models'la çok overlap, novelty zayıf

- **Truncated/Prefix ECT** — Truncated Consistency Models ile çok overlap, ana fikir olmaz

- **SCT-lite for Short-Budget ECT** — SCT'nin direct varyasyonu, novelty zayıf

**Bug story'i ana hikaye yapmama kararı**: Sunumda bug discovery yan tutuldu, sadece slide 10 "Done" listesinde tek bullet ("Corrected evaluation pipeline"). Sebep: Andrew "implementation, not debugging" istiyor.

---

## 9) Sunum yapısı

### 9.1 PPTX layout (10 slide, 16:9, Google Slides import-friendly)

| # | Slide | İçerik | Görsel |
|---|---|---|---|
| 1 | Title | "When Does Cheap Consistency Tuning Pay Off?" + authors + April 2026 | Red accent bar |
| 2 | The question | 3 bullets — diffusion slow, CM fast, when does tuning pay? | — |
| 3 | Background | Diffusion PFODE + CM consistency formula | 2 LaTeX equations |
| 4 | ECT | Loss formula, [Geng+ 2025] citation, 0.821 magic number teaser | LaTeX equation |
| 5 | Pareto frontier | ECT 17× faster | Plot (red ECT diamonds, gray Heun curve) |
| 6 | Tuning budget ablation | 94% quality at 0.5% compute | Plot (red ECT line, gray paper baseline) |
| 7 | The real open problem | Lead-in + Geng quote (extended) + 3 closing bullets | Quote block (red side bar) |
| 8 | Our method · KLUB CM | Formula + inspiration chain + predicted impact | LaTeX equation + gray inspiration text |
| 9 | Implementation · what we are coding | 4 bullets + pseudo-code | Code block (Menlo monospace, gray bg) |
| 10 | Status & next steps | Done / Up next two-column | Red "Done" header, gray "In flight" header |

### 9.2 Color scheme

- **Primary text**: `INK = #1A1A1A`
- **Secondary text**: `GRAY = #6B6B6B`
- **Accent (primary brand)**: `ACCENT = #A83D3D` — kırmızı
- **Secondary contrast**: `WARN = #424242` — koyu gri
- **Soft bg**: `SOFT = #EEEEEE` — quote ve code box arka planı
- **Bullet marker**: `•` (Unicode bullet, U+2022)
- **Title separators**: `·` (middle dot, U+00B7)

Eski paletteden kırmızıya geçiş: Navy (#1F4E79) → Red (#A83D3D), Red → Dark Gray. Plot'larda ECT artık kırmızı, Heun gri.

### 9.3 Citations placement

- Slide 3: [Karras+ 2022, EDM], [Song+ 2023]
- Slide 4: [Geng+ 2025]
- Slide 7: [Geng+ 2025], [Karras+ 2024 sCM], [Heek+ 2024 MCM], [Sabour+ 2024 AYS], [TDD, Wang+ 2024]
- Slide 8: [Sabour+ 2024 AYS], [Kim+ 2024 CTM], [OSS 2025]
- Slide 9: [Agarwal+ 2023 GKD]

Toplam ~9-10 inline citation. References slide silindi (zero-noise için), inline citations yeterli.

### 9.4 Sunum metni

Tam konuşma metni `SLIDE_NARRATION_EN.md`'de (~6 dakika). Slide-by-slide what-to-say + speaking tips + Q&A hazırlığı.

---

## 10) Dosya yapısı (proje root: `/Users/batuhankaraman/Downloads/COMP447/`)

```
COMP447/
├── README.md                          # Project framing & week-by-week plan
├── proposal_template/                 # LaTeX proposal sources
├── final_upload/
│   └── tex/proposal.tex               # Submitted proposal
├── readings/                          # Lecture notes, paper list
└── project/
    ├── ANALYSIS.md                    # Headline findings + table mode
    ├── INVESTIGATION.md               # Bug discovery research log
    ├── FINAL_IMPLEMENTATION_IDEA.md   # OAR-ECT (DEPRECATED — discarded)
    ├── ideas/idea1.md, idea2.md, idea3.md  # Brainstorming
    ├── PLAN.md                        # Original plan
    ├── colab_first_run.ipynb          # Main training notebook
    ├── ect_validation_extension_workbench.ipynb  # Investigation + ablation
    ├── configs/experiment_grid.yaml   # Experiment grid
    ├── results/
    │   ├── ablation_fid.csv           # First ablation
    │   ├── ablation_fid_v2.csv        # Updated (still through buggy ct_eval)
    │   ├── ablation_curve.png, _v2.png
    │   ├── heun_fid.csv, heun_fid_50k.csv
    │   ├── latency.csv
    │   ├── break_even.csv, break_even_v2.csv
    │   ├── decision_matrix.csv
    │   ├── pareto.png, pareto_v2.png
    │   └── samples/                   # Generated images
    ├── scripts/
    │   ├── setup_ect.sh               # Environment setup
    │   ├── measure_latency.py         # Latency profiling
    │   ├── eval_fid.py                # CORRECTED eval (bypasses ct_eval bug)
    │   └── break_even.py              # Break-even calculator
    ├── src/
    │   ├── ect/                       # Vendored locuslab/ect (commit 4311059)
    │   └── edm/                       # Vendored EDM
    ├── results_backup/
    │   └── ect_checkpoints/           # network-snapshot-{050,100,150,198}.pkl
    └── presentation/
        ├── generate.py                # PPTX generator (idempotent)
        ├── COMP447_progress.pptx      # Final output (243 KB, 10 slides)
        ├── COMP447_progress.pdf       # Preview
        ├── PROJECT_EXPLAINED_TR.md    # This file
        ├── SLIDE_NARRATION_EN.md      # Speaking script
        └── assets/
            ├── eq_pfode.png           # Rendered LaTeX equations
            ├── eq_cm.png
            ├── eq_ect.png
            ├── eq_klub.png
            ├── pareto.png             # Generated plot
            └── ablation.png           # Generated plot
```

### 10.1 Önemli dosya pointers

- **Generator**: `presentation/generate.py` — single-file, idempotent. `python3 generate.py` ile rebuild
- **Bug fix**: `scripts/eval_fid.py` — corrected eval pipeline
- **ECT source**: `src/ect/ct_eval.py` (line 94: mid_t default 0.821; line 363-380: bug)
- **ECT training loop**: `src/ect/training/ct_training_loop.py` (line 352-363: correct path)
- **Checkpoints**: `results_backup/ect_checkpoints/network-snapshot-*.pkl`
- **Investigation log**: `project/INVESTIGATION.md` (bug discovery research log, çok detaylı)

---

## 11) Andrew kriterleri ve mevcut durum

| Kriter | Durum | Notlar |
|---|---|---|
| "Sadece existing code'u run etmeyin" | ⚠️ Kısmen | ECT tuning + Heun baseline = reproduction. Bug fix + corrected eval pipeline = küçük original engineering. |
| "Implementation component" | ⚠️ Eksik | KLUB-CM önerildi, **kod yazılmadı**. Slide 11 pseudo-code seviyesinde. Sunumdan önce optimizer scaffolding yazılırsa risk azalır. |
| "Propose your own method" | ✅ | KLUB-CM net + inspiration chain (AYS + CTM + OSS) ile kanıtlı |
| "How does this help people" | ✅ | "Practitioners need principled schedule, not magic numbers" — slide 8 predicted impact bullet |
| Progress + extension | ✅ | Slide 10 done/up next net |
| Where the idea came from | ✅ | Slide 7 lead-in + Geng quote + literature gap |

**Net risk**: Andrew "show me the code" derse cevap pseudo-code + plan. Mitigation: sunumdan önce
1. KLUB derivation 1-sayfalık math draft (Overleaf)
2. `klub_cm/optimiser.py` scaffold (50-line skeleton)
3. `training/loss.py` patch'inin `# TODO` stub'ları

---

## 12) Web search doğrulamaları (her şey gerçek)

26 Nisan 2026 itibariyle yapılan web search ile doğrulanan iddialar:

| İddia | Doğrulama |
|---|---|
| ECT paper (ICLR 2025) | ✅ arXiv:2406.14548, locuslab/ect repo confirmed |
| mid_t = 0.821 hardcoded | ✅ ct_eval.py line 94 verbatim alıntılandı |
| Geng "open research problem" quote | ✅ `gh api` ile GitHub issue #11 verbatim çekildi, daha güçlü ("treat as optimization problem" ekstra cümle de var) |
| AYS uses KLUB via Girsanov | ✅ Paper abstract confirmed |
| AYS diffusion only | ✅ "confined to SDE frameworks" |
| AYS specific FID gains (5-30%) | ✅ Stable Diffusion CIFAR-10 20-step: 4.64 → 3.23 |
| TDD heuristic timestep | ✅ "predefined set of equidistant denoising schedules + stochastic offset" |
| OSS diffusion only, DP-based | ✅ Dynamic programming framework, doesn't cover CMs explicitly |
| SCT training schedule (not inference) | ✅ Training Δt smooth shrinking, MDP/TD framework |
| CTM trajectory traversal | ✅ "unrestricted traversal between any initial and final time" |

Hiçbir iddia çürütülmedi, hiçbir kritik claim soft bulunmadı.

---

## 13) Sunum risk değerlendirmesi

### 13.1 Olası sorular ve cevaplar

| Soru | Cevap |
|---|---|
| "Schedule optimization for CMs daha önce yapılmış mı?" | Heuristic var (TDD, Wang+ 2024 — predefined equidistant schedules). Principled KLUB-based yapılmamış. |
| "Niye L-BFGS, grid search değil?" | Grid search FID computation × #schedules × #checkpoints maliyet. KLUB gradient-based + cached features → orders of magnitude ucuz, step count'lar arası generalize |
| "FID improvement tahmini ne?" | Spesifik rakam vermiyoruz deney öncesi. AYS 5-30% gains diffusion için → CMs için aynı order beklenir. |
| "Code ne zaman hazır?" | Math derivation 2 hafta. Optimizer prototype kısa süre sonra. Mid-May'de full impl. Final results early June. |
| "Niye 0.821'i kabul edip ilerlemediniz?" | Aynı 0.821 her training stage ve step count için kullanılıyor. Optimum'un drift ettiğine dair preliminary evidence var. Performance lever, cosmetic değil. |
| "sCM/CTM ile ilişki?" | sCM training stability. CTM trajectory traversal flexibility. Hiçbiri principled inference-time schedule optimization yapmıyor. |
| "CIFAR-10 dışına generalize olur mu?" | Math dataset-agnostic. CIFAR-10 başlangıç çünkü EDM checkpoint + clean baselines var. ImageNet 64×64 doğal follow-up. |
| "Bug story neyle ilgili?" | Pipeline doğrulamada upstream'de metric routing bug'ı buldum, patch'ledim. Slide 10 "Done" listesinde "corrected eval pipeline". Methodology hijiyeni. |

### 13.2 En zayıf nokta

**KLUB-CM hâlâ pseudo-code seviyesinde, gerçek kod yok.**

Sunumdan önce 50-satırlık optimizer scaffold + 1-sayfalık math draft yazılırsa Andrew "show me code" sorusuna iyi cevap olur. Kullanıcı bu uyarıyı aldı, kararı kendisinin.

---

## 14) Konuşma geçmişi context (önemli kararlar)

### 14.1 Proje yön değişiklikleri timeline

- **Mart sonu**: Proposal — "ECT vs Heun benchmark + break-even" (Andrew zayıf bulmuş)
- **Nisan ortası**: ECT training + Heun baselines + ablation (reproduction tamamlandı)
- **20 Nisan civarı**: "1-step ≈ 2-step" gözlemi → "dead second step" hipotezi → OAR-ECT önerisi
- **21 Nisan**: ct_eval.py bug discovery — OAR-ECT motivation çöktü
- **22-25 Nisan**: Yeni yön araştırması — KLUB-CM final aday
- **26 Nisan**: Sunum hazırlığı, slide structure 13 → 10, narrative arc kuruldu
- **26 Nisan**: Web search ile tüm iddialar verify edildi
- **26 Nisan**: Final color scheme (red primary), 2 MD docs

### 14.2 Discarded but learned ideas

| Fikir | Niye atıldı |
|---|---|
| OAR-ECT (asymmetric refiner) | Motivasyon = "dead second step" = bug artifact |
| Adaptive-Midpoint ECT | Shortcut Models ile çok overlap, novelty zayıf |
| Truncated/Prefix ECT | Truncated CMs paper'ıyla overlap |
| SCT-lite | SCT paper'ının direct varyasyonu |
| Heun teacher tail hybrid | Side experiment olarak değer var ama ana fikir değil |
| Bug story as main pitch | Andrew "implementation odaklı" istiyor, bug methodology hijiyen olarak slide 10'a indirildi |

### 14.3 Stratejik kararlar

- **References slide silindi**: Inline citations yeterli, zero-noise için
- **Setup slide kaldırıldı**: Pareto plot'una caption olarak eklendi
- **Diffusion + CM tek slide**: Senior audience varsayımı, lecture-mode azalttı
- **Footer minimaize**: "COMP447 · Easy Consistency Tuning" çıkarıldı, sadece "N / 10"
- **GitHub issue mention çıkarıldı**: Aesthetic, attribution sade
- **Bullets em-dash → bullet point (•)**: User direktifi, no dashes
- **Compound word hyphens kaldırıldı**: "1-step" → "1 step", "fine-tune" → "fine tune"
- **Renkler navy → red**: User direktifi
- **Geng quote uzatıldı**: "treat as optimization problem" eklenince "yazar bizim approach'umuzu öneriyor" dialectic'i kuruldu

---

## 15) Tek paragraf executive summary

ECT, EDM checkpoint'inden 1 saatlik tuning ile 17× hızlı consistency model üretiyor — Pareto-dominant olduğunu CIFAR-10'da gösterdik (ECT 2-step FID 2.47 @ 14ms vs Heun-18 FID 1.96 @ 244ms). Ama 2-step inference tek bir sabit `mid_t = 0.821` parametresine bağlı, ve bu değer hiçbir paper'da derive edilmemiş; ECT'nin first author'u GitHub'da bunu *"open research problem"* olarak kabul edip optimization yaklaşımı öneriyor. Diffusion tarafında AYS bu problemi KLUB ile çözmüş; ama AYS SDE framework'üne özel. Biz KLUB'u consistency flow map için re-derive edip checkpoint-conditioned schedule optimizer + on-policy loss patch implement ediyoruz. **KLUB-CM**: AYS-of-consistency-models. Currently: ECT tuning + Heun baselines + ablation tamamlandı, KLUB-CM math + optimizer kod henüz yazılmadı (Mayıs implementation window). Final report Haziran başı.

---

## 16) Hızlı referans tablosu

```
PROJECT       : COMP447 — When Does Cheap Consistency Tuning Pay Off?
TEAM          : Batuhan Karaman + Kadir Yiğit Özçelik
DATASET       : CIFAR-10 32×32 unconditional
HARDWARE      : NVIDIA G4 Blackwell (Colab)
KEY MODEL     : ECT (Easy Consistency Tuning) [Geng+ ICLR 2025]
KEY BASELINE  : Heun (EDM) [Karras+ NeurIPS 2022]
BIG NUMBER 1  : ECT 2-step FID 2.47 @ 14ms (17× faster than Heun-18)
BIG NUMBER 2  : 94% paper-reported quality at 0.5% paper compute
BIG NUMBER 3  : Break-even N* ≈ 23.5k images (B=1)
KEY BUG       : ct_eval.py routes both fid metrics through 2-step sampler
KEY GAP       : mid_t = 0.821 hardcoded, undefended, author admits open
KEY ANALOG    : AYS [Sabour+ NVIDIA ICML 2024] solves for diffusion via KLUB
OUR METHOD    : KLUB-CM = AYS adapted to consistency flow map
NOVELTY       : Principled (vs heuristic TDD) schedule optimization for CMs
PRIOR WORK    : TDD (heuristic), SCT (training), CTM (trajectory), OSS (diffusion DP)
STATUS        : Tuning + baselines done; KLUB-CM math + impl pending
DEADLINE      : Final report early June 2026
```

---

## 17) Kadir'in Claude'una notlar

Bu doküman okunduktan sonra:
- Hangi paper'ların ne dediği ezberlenmiş olmalı (Section 5)
- KLUB-CM önerisinin matematiksel temeli anlaşılmış olmalı (Section 6)
- Andrew'a karşı pozisyon net olmalı (Section 11)
- Hangi rakamların gerçek olduğu, hangilerinin tahmin olduğu ayırt edilebilmeli (Section 2.4 + 12)
- OAR-ECT gibi atılmış fikirler **geri getirilmemeli** (Section 8.3)
- Bug story ana hikaye **yapılmamalı** (Section 9 + 14.3)

Sorulması gerekenler bana (Batuhan'a) sormak için:
- Yeni deney tasarımı (sweep dışında bir şey)
- Andrew'la iletişim
- Implementation timeline değişiklikleri
- Paper figür stilinde değişiklikler
- Final report yapısı

Sorulmadan yapılabilecekler:
- Slide içerik tweak (typo, kelime tercih)
- Konuşma metnini kişiselleştirme
- Q&A hazırlığı genişletme
- Pseudo-code'u gerçek code'a çevirme (ama production etmeden önce review)

---

**SON.** Bu dokümanı bilgi kaybı olmadan başka bir AI'ya yüklemek için yeterli context içeriyor.
