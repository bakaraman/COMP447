Genel değerlendirmem şu: sizin profiliniz için en iyi proje bandı **küçük/orta ölçekli discrete latent modeling (VQ-VAE türevleri)** ile **küçük ölçekli self-supervised learning** arasında. Bunlar hem dersin ana omurgasına çok net oturuyor, hem iki kişilik ekip için gerçekçi, hem de proposal/sunum/final report’ta temiz bir hikâye kurduruyor. Buna karşılık **full diffusion, video generation, ImageNet-ölçekli autoregressive modeller ve “çok havalı ama çok ağır” yeni paper’ların tam reprodüksiyonu** sizin kısıtlara göre kötü bahis. Ayrıca senin paylaştığın metinde **progress report** için hem **May 3** hem **May 11** geçiyor; internette bulabildiğim resmi sayfa 2024 sürümüydü ve orada deliverables ile bölüm içi deadline’lar tutarlıydı. 2026 resmi sayfasını doğrulayamadığım için ben planlama tarafında **deliverables satırını esas alır**, ama hocadan bunu ayrıca teyit ederdim. Resmi proje sayfası, proje türleri ve proposal yapısı açısından senin verdiğin özetle uyumlu; MLRC tarafı da yalnızca reproduction değil, **generalisability / novel insights beyond the original paper** türü extension’ları açıkça teşvik ediyor. ([aykuterdem.github.io][1])

## Aday proje fikirleri

### En uygun

**1) SQ-VAE under low-data / low-codebook budget**

* **Hangi paper’dan çıktığı:** *SQ-VAE: Variational Bayes on Discrete Representation with Self-annealed Stochastic Quantization* (ICML 2022). Resmi PyTorch repo var. ([ar5iv][2])
* **Neyin reproduce edileceği:** SQ-VAE’yi küçük bir image benchmark’ta yeniden kurup, aynı encoder/decoder ile **vanilla VQ-VAE** karşılaştırması yapmak. VQ-VAE temel paper’ı discrete latent + autoregressive prior çizgisini net veriyor. ([arXiv][3])
* **Neyin extend edileceği:** Paper’ın çıkış problemi doğrudan **codebook collapse**. En mantıklı extension, SQ-VAE’nin faydasını özellikle **low-data** (%10/%25/%50 veri) ve/veya **small codebook** rejiminde test etmek. İsterseniz latent prior için küçük bir **PixelCNN** de eklenebilir; paper zaten örnek üretim için PixelCNN prior kullandığını söylüyor. ([ar5iv][2])
* **Veri / metrik / compute tahmini:** Ana veri olarak **CIFAR-10** çok güvenli; 60k adet 32×32 görüntü var. Alternatif/ikincil veri olarak **MedMNIST** çok hafif ve standartlaştırılmış; 28×28 taban boyutu, sabit split’ler ve çok sayıda alt veri kümesi sunuyor. Metrikler: reconstruction loss, codebook utilization/perplexity, gerekirse CIFAR-10 üzerinde FID / sample quality. Compute: düşük-orta. ([U of T Computer Science][4])
* **Neden uygun:** VQ-VAE hattı dersin merkezinde; recent ama aşırı yeni değil; resmi PyTorch repo var; deney planı ve baseline çok net. Proposal yazması da kolay: “codebook collapse → self-annealing → low-data stress test.” ([GitHub][5])
* **Neden riskli:** Büyük ölçeğe çıkarsanız paper’ın kendisi hiyerarşik yapı ve PixelSNAIL/PixelCNN benzeri daha ağır prior’lara gidiyor; yani scope kaçarsa proje büyür. Bu yüzden küçük veri + küçük prior sınırını baştan koymak lazım. ([ar5iv][2])

**2) RQ-VAE-lite: residual quantization gerçekten küçük ölçekte işe yarıyor mu?**

* **Hangi paper’dan çıktığı:** *Autoregressive Image Generation using Residual Quantization* (CVPR 2022). Resmi repo var; ayrıca daha basit bir PyTorch implementation da bulunuyor. ([arXiv][6])
* **Neyin reproduce edileceği:** Full RQ-Transformer değil; **stage-1 RQ-VAE tokenizer** kısmı ve mümkünse bunun üstüne çok küçük bir latent prior. Paper’ın ana fikri, fixed codebook size altında residual quantization ile daha iyi rate-distortion elde etmek. ([arXiv][6])
* **Neyin extend edileceği:** En mantıklı extension, **residual depth**, **shared vs separate codebooks** veya **fixed total code budget** altında VQ-VAE vs RQ-VAE karşılaştırması. İsterseniz bunu **EuroSAT-RGB** gibi ders dışı ama küçük/orta bir domain’e de taşıyabilirsiniz. EuroSAT 27k örnekli, 10 sınıflı ve RGB/multispectral varyantları olan temiz bir benchmark. ([TensorFlow][7])
* **Veri / metrik / compute tahmini:** CIFAR-10 en güvenlisi; EuroSAT ikinci seçenek. Metrikler: reconstruction, codebook usage, latent prior NLL, opsiyonel FID. Compute: stage-1 için orta; full paper için yüksek. ([U of T Computer Science][4])
* **Neden uygun:** Discrete latent + autoregressive prior = dersle çok doğal eşleşme. Ayrıca assignment’lara çok yakın. Hikâye de net: “vanilla VQ-VAE’ye göre residual quantization küçük bütçede ne kazandırıyor?” ([arXiv][6])
* **Neden riskli:** Resmi repo Linux/CUDA odaklı ve orijinal paper checkpoint’leri ciddi ölçekte; full reproduction’a kayarsanız proje hızla büyür. Bu proje ancak scope’u bilinçli biçimde küçültürseniz iyi olur. ([GitHub][8])

**3) VICReg vs SimSiam in low-batch / low-label regimes**

* **Hangi paper’dan çıktığı:** *Exploring Simple Siamese Representation Learning* (SimSiam, CVPR 2021) ve *VICReg* (ICLR 2022). İkisinin de resmi repo’su var. ([CVF Open Access][9])
* **Neyin reproduce edileceği:** Küçük bir backbone ile **linear probe** ve/veya **kNN** değerlendirmesi. SimSiam paper’ı özellikle stop-gradient’in kritik olduğunu ve makul batch size’larda da çalışabildiğini gösteriyor; VICReg ise collapse’ı explicit variance regularization ile ele alıyor. ([CVF Open Access][9])
* **Neyin extend edileceği:** İki iyi extension var:

  1. **small-batch / low-label stress test**,
  2. SimSiam’a **VICReg variance term** benzeri bir regularizer eklemek.
* **Veri / metrik / compute tahmini:** **STL-10** çok uygun; 96×96 çözünürlük, 100k unlabeled görüntü ve düşük sayıda labeled örnek içeriyor. Alternatif olarak MedMNIST de olur. Compute: düşük-orta. Metrikler: linear probe accuracy, kNN accuracy, collapse diagnostic. ([Computer Science][10])
* **Neden uygun:** Kodlama en kolay projelerden biri; report hikâyesi de çok net: “implicit anti-collapse vs explicit anti-collapse.” İki kişi için çok dengeli. ([CVF Open Access][9])
* **Neden riskli:** En büyük risk, bunun **generative** taraftan çok **self-supervised representation learning** tarafına kayması. Ders kapsamına giriyor, ama proposal’da bunu özellikle vurgulamak gerekir. ([aykuterdem.github.io][1])

**4) SimMIM-lite on STL-10 or EuroSAT**

* **Hangi paper’dan çıktığı:** *SimMIM: A Simple Framework for Masked Image Modeling* (CVPR 2022), resmi repo var. ([arXiv][11])
* **Neyin reproduce edileceği:** Raw-pixel regression, random masking ve hafif prediction head ile masked image modeling. Paper özellikle “basit tasarım bile güçlü performans verebilir” tezini savunuyor. ([arXiv][11])
* **Neyin extend edileceği:** **mask ratio**, **patch size**, **decoder head** hafifliği veya domain transfer (ör. STL-10 → EuroSAT).
* **Veri / metrik / compute tahmini:** STL-10 ya da EuroSAT mantıklı. Metrikler: linear probe, reconstruction loss, training speed. Compute: orta-altı. ([Computer Science][10])
* **Neden uygun:** Method çok sade; basit ablation’larla güçlü bir final report çıkar.
* **Neden riskli:** Resmi repo ImageNet-ölçekli ve CUDA odaklı; gerçekçi proje için daha hafif bir yeniden implementasyon gerekir. Bu da “official repo’yu birebir çalıştırma” değil, “paper reproduction at smaller scale” anlamına gelir. ([GitHub][12])

### Orta riskli

**5) MAE-lite on CIFAR-10 / STL-10**

* **Hangi paper’dan çıktığı:** *Masked Autoencoders Are Scalable Vision Learners* (2021), resmi PyTorch repo var. ([arXiv][13])
* **Neyin reproduce edileceği:** Asymmetric encoder-decoder ve yüksek mask ratio fikri. Paper’da %75 masking ve 3×+ training speedup vurgulanıyor. ([arXiv][13])
* **Neyin extend edileceği:** Küçük görüntülerde **patch size / mask ratio / decoder width** ablation’ı.
* **Veri / metrik / compute tahmini:** CIFAR-10 veya STL-10. Compute: orta.
* **Neden uygun:** MAE çok bilinen ve anlatması kolay bir paper; hocanın da “clean extension” diyeceği türden.
* **Neden riskli:** Tiny images üzerinde ViT-tabanlı masked reconstruction bazen beklenenden nazlı olabilir; ayrıca resmi repo net biçimde GPU/CUDA tabanlı. ([GitHub][14])

**6) TarFlow-inspired low-compute flow study**

* **Hangi paper’dan çıktığı:** *Normalizing Flows are Capable Generative Models* (ICML 2025 oral), resmi repo var. Paper NFs’in yeniden ciddi adaylar olduğunu savunuyor; Gaussian noise augmentation, post-training denoising ve guidance öneriyor. ([OpenReview][15])
* **Neyin reproduce edileceği:** Dürüst cevap: **full TarFlow değil**. Onun yerine küçük bir **RealNVP / MAF baseline** kurup TarFlow’daki 1-2 reçeteyi küçük ölçekte test etmek.
* **Neyin extend edileceği:** **noise augmentation** ve/veya basit bir **post-hoc denoising** stratejisinin küçük image benchmark’ta sample quality ve likelihood’e etkisi.
* **Veri / metrik / compute tahmini:** CIFAR-10 ile yapılabilir. Compute: orta. Metrikler: NLL/bpd + sample quality. ([U of T Computer Science][4])
* **Neden uygun:** Flow konusu derse çok doğal uyuyor; exact likelihood vermesi raporu teknik olarak güçlü yapar.
* **Neden riskli:** Resmi repo doğrudan ImageNet64/AFHQ gibi veri hazırlığı ve `torchrun --nproc_per_node=8` örnekleri içeriyor; yani paper’ı gerçekten birebir çoğaltmak sizin setup için gerçekçi değil. Bu yüzden ancak “TarFlow-inspired small-scale study” olarak öneriyorum. ([GitHub][16])

**7) DiffAugment for data-efficient GANs on tiny subsets**

* **Hangi paper’dan çıktığı:** *Differentiable Augmentation for Data-Efficient GAN Training* (NeurIPS 2020), resmi repo hem PyTorch hem TensorFlow içeriyor. Paper limited-data koşulunda GAN performansının ciddi bozulduğunu ve DiffAugment’in bunu azalttığını söylüyor. ([arXiv][17])
* **Neyin reproduce edileceği:** CIFAR-10’un %10 / %20 / %50 altkümelerinde GAN + DiffAugment karşılaştırması.
* **Neyin extend edileceği:** augmentation policy ablation veya EuroSAT gibi doğal olmayan image domain’e taşıma.
* **Veri / metrik / compute tahmini:** CIFAR-10 veya EuroSAT. Compute: orta. Metrikler: FID, IS, training stability. ([U of T Computer Science][4])
* **Neden uygun:** Veri azlığı hikâyesi çok net; ablation yapmak kolay.
* **Neden riskli:** GAN tarafı doğası gereği daha huysuz; özellikle StyleGAN2 hattına kayarsanız proje gereksiz yere engineering’e dönüşebilir. ([GitHub][18])

**8) MaskGIT-lite with a frozen tokenizer**

* **Hangi paper’dan çıktığı:** *MaskGIT: Masked Generative Image Transformer* (CVPR 2022), resmi repo var. ([arXiv][19])
* **Neyin reproduce edileceği:** İdeal scope, frozen/small bir tokenizer üstünde second-stage masked token generation ve decoding schedule incelemesi.
* **Neyin extend edileceği:** confidence schedule, mask schedule veya iteration count ablation.
* **Veri / metrik / compute tahmini:** CIFAR-10 tokenized images ile ancak “lite” sürümde yapılır. Compute: orta-yüksek.
* **Neden uygun:** Autoregressive image modeling çizgisine çok yakışır; sunumda da “parallel-ish iterative decoding” hikâyesi güzel görünür.
* **Neden riskli:** Bu, pratikte en büyük engineering riski olan fikirlerden biri. Resmi repo **JAX**, ayrıca repo **archived/read-only** ve training kısmında açıkça **“Coming Soon”** yazıyor. Yani kağıt üstünde güzel, proje olarak riskli. ([GitHub][20])

## Özellikle eleyeceğim işler

Ben sizin yerinizde şunları **baştan elerdim**:

* **Full ARPG / Randomized Parallel Decoding**: paper çok yeni, ImageNet-256 üzerinde 32 step ve 30× speedup gibi iddialarla geliyor; bu seviyeyi iki kişi ve düşük/orta compute ile sağlıklı çoğaltmak gerçekçi değil. ([arXiv][21])
* **Full TarFlow reproduction**: resmi repo örnekleri 8 process, ImageNet64/AFHQ gibi daha ağır akışlara gidiyor. ([GitHub][16])
* **Full MaskGIT / ImageNet-scale token generation**: resmi kod JAX, repo archived, training yolu eksik. ([GitHub][20])
* **Ağır diffusion / video generation / DiT-style projeler**: dersle uyumlu olsa da sizin “kısa sürede, düşük/orta compute, stabil eğitim” kriterlerinize ters.

## Objektif karşılaştırma tablosu

Aşağıdaki skorlar benim sentezim; 5 en iyi.

| Proje                          | Ders uyumu | Uygulanabilirlik (2 kişi) | Compute uygunluğu | Kod erişilebilirliği | Sunum/rapor hikâyesi | Genel not |
| ------------------------------ | ---------: | ------------------------: | ----------------: | -------------------: | -------------------: | --------: |
| SQ-VAE low-data / low-codebook |          5 |                         5 |                 5 |                    5 |                    5 |   **5.0** |
| RQ-VAE-lite                    |          5 |                         4 |                 4 |                    4 |                    5 |   **4.4** |
| VICReg vs SimSiam              |          4 |                         5 |                 5 |                    5 |                    5 |   **4.8** |
| SimMIM-lite                    |          4 |                         4 |                 4 |                    4 |                    4 |   **4.0** |
| MAE-lite                       |          4 |                         4 |                 4 |                    4 |                    4 |   **4.0** |
| TarFlow-inspired flow study    |          5 |                         3 |                 4 |                    3 |                    4 |   **3.8** |
| DiffAugment low-data GAN       |          4 |                         3 |                 3 |                    4 |                    4 |   **3.6** |
| MaskGIT-lite                   |          5 |                         2 |                 2 |                    2 |                    4 |   **3.0** |

## En iyi 3

**1) SQ-VAE under low-data / low-codebook budget**
Bence en dengeli proje bu. Discrete latent modeling çizgisinde kalıyor, VQ-VAE tabanlı olduğu için dersle çok iyi örtüşüyor, resmi PyTorch repo’su var, problemi net: **codebook collapse**. Extension da doğal: düşük veri ve küçük codebook koşullarında gerçekten daha iyi mi? Bu, hem teknik hem anlatısal olarak çok temiz. ([ar5iv][2])

**2) VICReg vs SimSiam in low-batch / low-label regimes**
Bu, “en güvenli teslim edilebilir” proje. Çalışması yüksek olasılıkla kolay, ablation yapmak kolay, rapor yazmak kolay. Dersin SSL kısmına çok net oturuyor. Tek eksiği, generative modeling’den biraz uzaklaşması. ([CVF Open Access][9])

**3) RQ-VAE-lite vs VQ-VAE**
Bunun artısı, discrete latent + autoregressive prior çizgisinde çok “course-native” olması. Eksisi ise SQ-VAE’ye göre biraz daha ağır ve scope kaçırmaya daha açık olması. Yine de iyi yönetilirse çok güzel proje olur. ([arXiv][6])

## Ben olsam bunu seçerdim

**Ben olsam `SQ-VAE under low-data / low-codebook budget` seçerdim.**

Neden?
Çünkü bu fikir aynı anda şu dört şeyi sağlıyor:

1. **Derse çok net uyuyor**: VQ-VAE/discrete latent çizgisi dersin göbeğinde.
2. **Recent paper extension**: 2022 paper, ama hâlâ yeterince güncel ve “future work/open question” tarafı canlı.
3. **Düşük/orta compute ile gerçekçi**: CIFAR-10 ve hatta MedMNIST üzerinde rahat ölçeklenir.
4. **Rapor hikâyesi çok net**: problem = codebook collapse; yöntem = self-annealed stochastic quantization; extension = low-data/small-codebook stress test. ([ar5iv][2])

## Proposal-ready taslak

### Provisional title

**Reproducing and Stress-Testing SQ-VAE in Low-Compute Regimes: Codebook Utilization, Reconstruction Quality, and Small-Scale Generation**

### Research question

**Does SQ-VAE improve codebook utilization and downstream reconstruction / generation quality over vanilla VQ-VAE under low-data and small-codebook regimes, without requiring delicate training heuristics?**
Bu soru doğrudan paper’ın motivasyonundan çıkıyor: VQ-VAE’de codebook collapse ve heuristic-heavy training. ([ar5iv][2])

### Key papers

* **SQ-VAE: Variational Bayes on Discrete Representation with Self-annealed Stochastic Quantization** — ana paper. ([arXiv][22])
* **Neural Discrete Representation Learning (VQ-VAE)** — temel baseline paper. ([arXiv][3])
* **Conditional Image Generation with PixelCNN Decoders** — küçük latent prior ekleyecekseniz en doğal okuma. ([arXiv][23])
* **Official SQ-VAE repo** — ana kod referansı. ([GitHub][5])

### Proposed baseline

* **Vanilla VQ-VAE** with the same encoder/decoder family and comparable codebook budget.
* İsterseniz ikinci, daha zayıf bir baseline olarak plain **Gaussian VAE** de eklenebilir; ama zorunlu değil. En temiz baseline VQ-VAE. ([arXiv][3])

### Proposed extension

Ana extension’ı tek cümlede böyle kurardım:
**Evaluate whether SQ-VAE’s self-annealed quantization is especially beneficial when the problem is made harder by low data and limited codebook capacity.**
Pratikte bu şu iki ablation’dan oluşabilir:

* **Data regime ablation:** %10 / %25 / %50 / %100 train data
* **Codebook budget ablation:** küçük / orta / büyük codebook veya sabit toplam budget altında latent-grid trade-off
  Bu extension paper’daki codebook-usage motivasyonundan doğal çıkıyor. ([ar5iv][2])

### Dataset

* **Main dataset:** **CIFAR-10** — küçük, hızlı, karşılaştırması kolay. 60k adet 32×32 image içeriyor. ([U of T Computer Science][4])
* **Optional second dataset:** **MedMNIST** (ör. PathMNIST veya BloodMNIST) — çok hafif, standardized, ek domain testi için iyi. ([medmnist.com][24])

### Metrics

* **Reconstruction:** MSE / L1, mümkünse LPIPS
* **Codebook diagnostics:** codebook utilization, perplexity, dead code ratio
* **Generation:** küçük bir latent prior kullanırsanız CIFAR-10 üzerinde FID veya en azından qualitative sample grid
* **Training behavior:** convergence stability, epoch time
  SQ-VAE paper’ı reconstructive ve prior-based generation tarafını zaten kullanıyor; PixelCNN prior özellikle küçük ölçekte mantıklı. ([ar5iv][2])

### Rough timeline

**Hafta 1:**

* VQ-VAE baseline
* CIFAR-10 pipeline
* reconstruction ve codebook usage logging

**Hafta 2:**

* SQ-VAE reproduction
* baseline ile ilk karşılaştırmalar
* dead code / perplexity analizleri

**Hafta 3:**

* low-data ve codebook-budget ablation
* en az bir güçlü tablo + bir qualitative figure

**Hafta 4:**

* opsiyonel küçük PixelCNN prior
* final plots
* proposal/progress presentation slide’ları

### Proposal’da hangi başlık altında ne yazılır

**Introduction (1/4 page)**

* Deep unsupervised learning’de discrete latent representation neden önemli
* VQ-VAE’nin gücü ama codebook collapse problemi
* Sizin sorunuz: low-data / low-budget koşulunda SQ-VAE gerçekten fark yaratıyor mu? ([arXiv][3])

**Related Work (1/2 page)**

* VQ-VAE
* SQ-VAE
* kısa bir paragraf latent prior olarak PixelCNN
* opsiyonel: RQ-VAE’yi “adjacent extension” diye anabilirsiniz, ama projeye sokmak zorunda değilsiniz. ([arXiv][3])

**Method (1/4 page)**

* baseline VQ-VAE
* SQ-VAE’nin stochastic quantization + self-annealing fikri
* sizin extension: low-data ve codebook-budget stress test
* deneyde sabit tutacağınız şeyler: encoder/decoder family, training budget, optimizer

**Experimental Evaluation (1/8 page)**

* Dataset: CIFAR-10, opsiyonel MedMNIST
* Metrikler: reconstruction, codebook usage, sample quality
* Adil karşılaştırma protokolü: aynı epoch, aynı seed sayısı, aynı latent budget

**Work Plan (1/8 page)**

* Week-by-week plan yukarıdaki gibi
* Risk yönetimi: prior yetişmezse reconstruction + codebook study yine yeterli deliverable üretir

**Abstract + References (1/4 page)**

* Abstract’ta contribution’ı çok mütevazı ama net yazın:
  “We reproduce SQ-VAE at small scale and evaluate whether its self-annealed quantization is particularly beneficial under low-data and limited-codebook settings.”

Bu isterseniz bir sonraki adımda doğrudan **2 sayfalık proposal iskeletine** de çevrilebilir.

[1]: https://aykuterdem.github.io/classes/comp547.s24/project.html "https://aykuterdem.github.io/classes/comp547.s24/project.html"
[2]: https://ar5iv.labs.arxiv.org/html/2205.07547 "https://ar5iv.labs.arxiv.org/html/2205.07547"
[3]: https://arxiv.org/abs/1711.00937 "https://arxiv.org/abs/1711.00937"
[4]: https://www.cs.toronto.edu/~kriz/cifar.html "https://www.cs.toronto.edu/~kriz/cifar.html"
[5]: https://github.com/sony/sqvae "https://github.com/sony/sqvae"
[6]: https://arxiv.org/abs/2203.01941 "https://arxiv.org/abs/2203.01941"
[7]: https://www.tensorflow.org/datasets/catalog/eurosat "https://www.tensorflow.org/datasets/catalog/eurosat"
[8]: https://github.com/kakaobrain/rq-vae-transformer "https://github.com/kakaobrain/rq-vae-transformer"
[9]: https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_Exploring_Simple_Siamese_Representation_Learning_CVPR_2021_paper.pdf "https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_Exploring_Simple_Siamese_Representation_Learning_CVPR_2021_paper.pdf"
[10]: https://cs.stanford.edu/~acoates/stl10/ "https://cs.stanford.edu/~acoates/stl10/"
[11]: https://arxiv.org/abs/2111.09886 "https://arxiv.org/abs/2111.09886"
[12]: https://github.com/microsoft/SimMIM "https://github.com/microsoft/SimMIM"
[13]: https://arxiv.org/abs/2111.06377 "https://arxiv.org/abs/2111.06377"
[14]: https://github.com/facebookresearch/mae "https://github.com/facebookresearch/mae"
[15]: https://openreview.net/forum?id=2uheUFcFsM "https://openreview.net/forum?id=2uheUFcFsM"
[16]: https://github.com/apple/ml-tarflow "https://github.com/apple/ml-tarflow"
[17]: https://arxiv.org/abs/2006.10738 "https://arxiv.org/abs/2006.10738"
[18]: https://github.com/mit-han-lab/data-efficient-gans "https://github.com/mit-han-lab/data-efficient-gans"
[19]: https://arxiv.org/abs/2202.04200 "https://arxiv.org/abs/2202.04200"
[20]: https://github.com/google-research/maskgit "https://github.com/google-research/maskgit"
[21]: https://arxiv.org/abs/2503.10568 "https://arxiv.org/abs/2503.10568"
[22]: https://arxiv.org/abs/2205.07547 "https://arxiv.org/abs/2205.07547"
[23]: https://arxiv.org/abs/1606.05328?utm_source=chatgpt.com "Conditional Image Generation with PixelCNN Decoders"
[24]: https://medmnist.com/ "https://medmnist.com/"
