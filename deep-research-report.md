# COMP547 Deep Unsupervised Learning Projesi için düşük/orta compute odaklı proje konu havuzu, risk analizi ve proposal taslağı

Paylaştığınız proje yönergesi (grup projeleri, “reproduction/extension” teşviki, LaTeX şablonu, ara/final sunum+rapor, baseline/ablation/limitations beklentisi) tipik bir “küçük ölçekli ama research‑oriented” generative modeling dersi projesi formatına çok net oturuyor. citeturn3view0turn4view0turn23view0turn24view0 En gerçekçi strateji: **resmi/sağlam kod tabanı olan** (tercihen PyTorch), **küçük veriyle anlamlı metrik üreten** ve **net bir “reproduce + drop‑in extension” hikâyesi** kurabileceğiniz bir paper seçmek.

## Ders kapsamı ve proje beklentisinin pratik yorumu

Ders sayfasındaki (Spring 2024) resmi proje açıklaması; projenin %36 olduğunu, grupların 2–3 kişi olacağını ve proje tiplerini açıkça listeliyor: (i) yeni task/dataset üzerinde generative model uygulama, (ii) yeni (non‑trivial) unsupervised method tasarımı, (iii) yakın dönem bir çalışmaya anlamlı extension, (iv) published work reproduction (ve özellikle **entity["organization","ML Reproducibility Challenge","annual reproducibility venue"]** tarzı çalışma teşviki). citeturn3view0turn6view0turn6view2

Bu format, “sadece fikir” yerine şu somut parçaları zorunlu kılıyor: problem/motivasyon, güçlü related work, metodoloji netliği, deney düzeni (dataset+metrik), nicel+nitel sonuçlar, baseline kıyasları, ablation ve limitation/future direction. Sunum rubric’i bunu puanlayarak teyit ediyor. citeturn4view0

Dersin assignment eksenleri (Spring 2024 resmi sayfada) proje için “doğal” yolları da gösteriyor:  
- **Autoregressive (PixelCNN türevleri)** citeturn23view0  
- **Normalizing flows (RealNVP) + VAE ailesi** citeturn23view0  
- **GAN + diffusion** citeturn23view0  
Bu yüzden “assignment’larda zaten görülen temel bileşenleri” alıp, **yakın dönem paper’lara** bağlayan bir reproduction+extension seçimi, hocanın beklentisiyle genelde daha uyumlu olur. citeturn23view0turn3view0

## Takvim ve teslim formatı: doğrulananlar, çelişkiler ve planlama gerçeği

Paylaştığınız Spring 2026 metninde deliverable’lar (proposal, progress presentation/report, final presentation/report) tarihleri net; ayrıca final report’ta “no late submissions” vurgusu var. Bu yapı Spring 2024 resmi projeyle birebir aynı iskelete sahip (LaTeX template ile PDF teslim; ara rapor 6 sayfa; final rapor 8 sayfa; sunum rubric’i). citeturn3view0turn4view0

Spring 2026 metninizde dikkat çeken **tarih çelişkisi**:  
- “Project progress reports: **May 3, 2026**” diyorsunuz, ama aşağıda “Progress Report Due: **May 11, 2026 (11:59pm)**” yazıyor (aynı deliverable için iki farklı tarih). Bu tür çelişkiler Spring 2024 sayfasında yok; orada progress report tek tarih (May 5, 2024) olarak geçiyor. citeturn3view0  
Bu nedenle, planlamayı “May 3 erken tarih”e göre yapmak (risk azaltır), ama resmi duyuru/Blackboard ile doğrulamak kritik.

Ayrıca dersin “Presentations” kısmı, paper sunumları için **dersten 3–4 gün önce hocayla görüşme** ve “bir gece önce teslim” gibi pratik süre kısıtları koyuyor. Bu tip ara deadline’lar proje yoğunluğuna binince zaman yönetimi daha da önemli hale geliyor. citeturn24view0

MLRC tarafında da (ders projesiyle aynı olmasa bile) “kodun çalıştırılabilirliği, README, exact run steps” vurgusu var; hatta MLRC kaynakları “reproducible code” için **ML Code Completeness Checklist** gibi çerçevelere yönlendiriyor. citeturn6view2turn33view0 Bu yaklaşımı ders projesine taşımak, rapor/sunum kalitesini doğrudan yükseltir.

## Aday proje fikirleri: düşük/orta compute odaklı, reproduction + anlamlı extension

Aşağıdaki fikirlerde ortak prensip: **küçük/orta dataset (ör. CIFAR‑10, STL‑10, CelebA, Imagenette)** üzerinde **ölçek küçültülmüş** deney; metrikleri “paper ölçeğiyle aynı çıkarmaya çalışmak” yerine **trendleri ve ablation’ı doğru göstermeye** odaklanmak. (CIFAR‑10: 60k 32×32 görüntü; CelebA: 200k+ yüz; STL‑10: unlabeled ağırlıklı SSL için tasarlanmış.) citeturn9search0turn9search1turn9search2turn9search3

### Fikir A: MaskGIT sampling scheduler’larının sistematik analizi ve Halton Scheduler “drop‑in” extension

**Kısa başlık:** MaskGIT decoding’de “mask schedule / token unmasking” tasarımı

**Hangi paper’dan çıktığı:** *MaskGIT: Masked Generative Image Transformer* (CVPR 2022) citeturn13view0turn7search0  
Paper’ın kendi içinde “masking design”ın kaliteyi ciddi etkilediğini ve bu tasarımın gelecekte daha fazla çalışılmasının ilginç olduğunu açıkça işaret etmesi, extension için çok temiz bir kapı açıyor. citeturn14view0

**Neyi reproduce edeceksiniz:**  
- Küçük ölçekte (örn. CIFAR‑10 veya Imagenette‑160px) MaskGIT benzeri masked‑token generation pipeline’ını kurup, en azından **cosine vs linear** gibi scheduler varyantlarının FID/KID ve örnek çeşitliliğine etkisini göstermek. MaskGIT’in iki aşamalı tokenization + transformer yaklaşımı (VQ based tokenizer + masked modeling) zaten paper’ın ana iskeleti. citeturn13view0turn14view1

**Neyi extend edeceksiniz:**  
- *Halton Scheduler for Masked Generative Image Transformer* (ICLR 2025) ile gelen **Halton scheduler’ı** “drop‑in replacement” olarak ekleyip (paper’a göre retraining gerektirmeden) aynı model üzerinde örnek kalitesi/çeşitlilik/sampling hataları açısından kıyas yapmak. citeturn28view1turn28view0  
- Ek mini‑extension: “iteration sayısı” (T) ve “mask ratio schedule concavity”nin (cosine/square/linear) etkileşimini küçük ölçekte taramak (MaskGIT paper’ında da T için “sweet spot” gözlemi var). citeturn14view0

**Veri / metrik / compute tahmini:**  
- Veri: CIFAR‑10 (hızlı prototip) veya Imagenette (daha “gerçekçi” görsel çeşit). citeturn9search0turn9search3  
- Metrikler: FID/KID, (opsiyonel) Inception Score; ayrıca **sampling hızı** (token/iterasyon) ve “aynı compute ile kalite” grafiği.  
- Compute: *Orta*. Tam ImageNet ölçeği gerektirmeden, küçük çözünürlükte makul; Halton scheduler test kısmı özellikle uygun. Halton repo’su pretrained model indirme/Colab demo gibi pratik girişler sağlıyor. citeturn28view0

**Neden uygun:**  
- Ders kapsamına “discrete latents + transformers + generative decoding” şeklinde çok net oturuyor. citeturn23view0turn3view0  
- Extension “scheduler” olduğu için: (i) contribution net, (ii) ablation doğal, (iii) rapor/sunum hikâyesi çok temiz (problem → scheduler tasarımları → metrikler → sonuç). citeturn4view0turn28view1  
- Paper’ın kendi “future work” işaretine doğrudan bağlanıyor. citeturn14view0

**Neden riskli:**  
- İki aşama (tokenizer + transformer) kurulum/ayarlama gerektirebilir; küçük ölçekte “good looking sample” almak bazen hyperparametre hassasiyeti ister.  
- MaskGIT’in “tokenizer stage”ini VQGAN düzeyinde kurmak, eğer sıfırdan eğitilecekse zaman alabilir; bu yüzden pretrained tokenizer veya hazır repo kullanımı planlanmalı. citeturn13view0turn28view0

**Linkler (paper / repo / veri):**
```text
MaskGIT (CVPR 2022, PDF): https://openaccess.thecvf.com/content/CVPR2022/papers/Chang_MaskGIT_Masked_Generative_Image_Transformer_CVPR_2022_paper.pdf
MaskGIT (arXiv): https://arxiv.org/abs/2202.04200
MaskGIT official JAX repo: https://github.com/google-research/maskgit

Halton Scheduler (OpenReview): https://openreview.net/forum?id=RDVrlWAb7K
Halton Scheduler (arXiv): https://arxiv.org/abs/2503.17076
Halton-MaskGIT official PyTorch repo: https://github.com/valeoai/Halton-MaskGIT

CIFAR-10 official: https://www.cs.toronto.edu/~kriz/cifar.html
Imagenette official: https://github.com/fastai/imagenette
```

image_group{"layout":"carousel","aspect_ratio":"16:9","query":["MaskGIT pipeline overview figure","VQGAN tokenizer architecture diagram","normalizing flow model diagram RealNVP coupling layer","Flow Matching generative modeling diagram"],"num_per_query":1}

### Fikir B: VQGAN tokenizer kalitesi ile MaskGIT/AR kalite-hız tradeoff’u

**Kısa başlık:** Tokenizer (VQGAN) kalitesi → generative transformer kalitesi

**Hangi paper’dan çıktığı:** *Taming Transformers for High‑Resolution Image Synthesis* (VQGAN, CVPR 2021) + MaskGIT (CVPR 2022) citeturn7search10turn7search2turn13view0

**Neyi reproduce edeceksiniz:**  
- Küçük bir görüntü setinde VQGAN tokenizer eğitimi (ya da pretrained tokenizer kullanımı) ve reconstruction metrikleri (örn. rFID/LPIPS/PSNR gibi). VQGAN’ın “tokenization stage + transformer stage” yaklaşımı paper’ın özünde var. citeturn7search6turn7search15

**Neyi extend edeceksiniz:**  
- Aynı MaskGIT/AR modelini, **farklı tokenizer ayarlarıyla** (codebook size, latent grid resolution, perceptual/adversarial loss ağırlıkları) çalıştırıp “tokenizer distortion”ın downstream generative kaliteye etkisini incelemek. Bu pratikte hocanın sevdiği türden “net ablation + net lesson” üretir.

**Veri / metrik / compute tahmini:**  
- Veri: CIFAR‑10 (32×32) veya CelebA’nın düşük çözünürlüklü sürümü (64×64). citeturn9search0turn9search1  
- Metrikler: reconstruction (PSNR/SSIM/LPIPS) + generation (FID/KID).  
- Compute: *Orta‑üst*. Tokenizer eğitimi + generative stage toplam yük getirebilir.

**Neden uygun:**  
- Dersteki “discrete latent models / VQ” eksenine çok doğal bağ. citeturn23view0turn3view0  
- “Neden ikinci aşama modelim kötü?” sorusuna bilimsel bir cevap üretir.

**Neden riskli:**  
- İki aşama da eğitim gerektirebilir; zaman kısıtında risk artar (özellikle GPU erişimi sınırlıysa).

**Linkler:**
```text
VQGAN / Taming Transformers (arXiv): https://arxiv.org/abs/2012.09841
VQGAN official repo: https://github.com/compvis/taming-transformers

MaskGIT (CVPR 2022, PDF): https://openaccess.thecvf.com/content/CVPR2022/papers/Chang_MaskGIT_Masked_Generative_Image_Transformer_CVPR_2022_paper.pdf

CelebA official: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
```

### Fikir C: Flow Matching ile CIFAR‑10 üzerinde diffusion‑benzeri path’ler ve sampler adım sayısı

**Kısa başlık:** Flow Matching ile “daha stabil diffusion‑vari training” + hızlı sampling ablation’ı

**Hangi paper’dan çıktığı:** *Flow Matching for Generative Modeling* (2022) + resmi PyTorch codebase citeturn31view1turn31view0

**Neyi reproduce edeceksiniz:**  
- Resmi `flow_matching` kütüphanesinin örneklerinden yararlanarak CIFAR‑10 üzerinde bir FM modeli eğitimi; temel metriklerle (FID/KID) “çalışır baseline” elde etmek. Repo, CIFAR‑10 üzerinde “full training examples” verdiğini ve modellerin “single command” ile train edilebildiğini söylüyor. citeturn31view0

**Neyi extend edeceksiniz:**  
- **Path seçimi / scheduler** (diffusion path vs alternatif path’ler) ve **ODE solver / step sayısı** üzerinden kalite‑hız tradeoff’larını sistematik taramak. Paper da FM’in diffusion path’lerini kapsayan bir çerçeve olduğunu; farklı path’lerin hem eğitim hem örnekleme verimliliğine etki edebileceğini tartışıyor. citeturn10search4turn31view0

**Veri / metrik / compute tahmini:**  
- Veri: CIFAR‑10. citeturn9search0  
- Metrikler: FID/KID + sampling wall‑clock time.  
- Compute: *Orta*. Diffusion’a benzer ama bazı kurulumlar daha doğrudan olabilir.

**Neden uygun:**  
- “Diffusion vs Flow” çizgisini ders kapsamı içinde modern biçimde yakalar (unsupervised generative). citeturn3view0turn23view0  
- Ablation ekseni çok net: path/solver/steps.

**Neden riskli:**  
- FM literatürü hızlı değişiyor; iyi bir sonuç almak için doğru default’lara sadık kalmak önemli.  
- Yazımda (method kısmında) biraz daha matematiksel netlik gerekebilir.

**Linkler:**
```text
Flow Matching paper (arXiv): https://arxiv.org/abs/2210.02747
facebookresearch/flow_matching repo: https://github.com/facebookresearch/flow_matching
CIFAR-10 official: https://www.cs.toronto.edu/~kriz/cifar.html
```

### Fikir D: OpenAI improved DDPM “küçük ölçekte” reproduction + guidance/variance öğrenme ablation’ı

**Kısa başlık:** Improved DDPM: variance learning ve sampling hız/kalite

**Hangi paper’dan çıktığı:** *Improved Denoising Diffusion Probabilistic Models* (2021) + OpenAI kodu citeturn11search2turn11search1

**Neyi reproduce edeceksiniz:**  
- Küçük bir çözünürlükte (CIFAR‑10) DDPM eğitimini çalıştırıp temel sample kalitesi üretmek; paper’ın vurguladığı “log‑likelihood + sample quality” ölçüm mantığını küçük ölçekte taklit etmek. citeturn11search2turn9search0

**Neyi extend edeceksiniz:**  
- (i) variance öğrenme açık/kapalı, (ii) sampling step sayısı (DDIM‑style hızlı sampling vs klasik), (iii) (opsiyonel) classifier‑free guidance tarzı conditioning stratejileriyle kalite/çeşitlilik tradeoff’u. citeturn11search2turn11search1

**Veri / metrik / compute tahmini:**  
- Veri: CIFAR‑10. citeturn9search0  
- Metrikler: FID/KID, precision‑recall (distribution coverage), sampling süresi.  
- Compute: *Orta‑üst*. Diffusion training yine de GPU ister; fakat 32×32’de yönetilebilir.

**Neden uygun:**  
- Dersin GAN+diffusion modülüyle birebir uyumlu. citeturn23view0  
- Baseline/ablation yazımı kolay; metrikler net.

**Neden riskli:**  
- Training süresi ve hyperparametre hassasiyeti (özellikle sınırlı GPU ile).  
- “Çok ağır diffusion istemiyoruz” hedefinize yaklaşabilir; bu yüzden ölçeği iyi sınırlamak şart.

**Linkler:**
```text
Improved DDPM (arXiv): https://arxiv.org/abs/2102.09672
OpenAI guided-diffusion repo: https://github.com/openai/guided-diffusion
CIFAR-10 official: https://www.cs.toronto.edu/~kriz/cifar.html
```

### Fikir E: Normalizing flows’ta modern “recipe”lerin küçük ölçekte test edilmesi

**Kısa başlık:** TarFlow “recipe” ablation’ı (noise augmentation, denoising, guidance) küçük ölçekte

**Hangi paper’dan çıktığı:** *Normalizing Flows are Capable Generative Models* (TarFlow; ICML 2025 oral) + Apple kodu citeturn20search6turn20search0turn2search4

**Neyi reproduce edeceksiniz:**  
- Resmi repo üzerinde, küçük dataset/çözünürlükte (örn. CIFAR‑10 veya ImageNet‑64 benzeri) training’i çalıştırıp “NF ile sample üretimi + likelihood” hattının çalıştığını göstermek. Paper, TarFlow’un image patch’leri üzerinde transformer‑vari MAF fikriyle çalıştığını ve bazı “recipe”lerin sample kalitesini arttırdığını anlatıyor. citeturn20search6turn20search3

**Neyi extend edeceksiniz:**  
- “Recipe” parçalarını tek tek açıp kapatarak: Gaussian noise augmentation / post‑training denoising / guidance etkisini ölçmek. Bu, paper’daki iddiaları “ucuz ablation” ile test etmenin temiz bir yolu.

**Veri / metrik / compute tahmini:**  
- Metrikler: bits/dim (likelihood), FID/KID.  
- Compute: *Orta‑üst*. Flow modelleri inference/training tarafında bazen ağır olabilir; repo issue’larında sampling hızına dair tartışmalar var (hız kısıtı). citeturn20search4turn20search6

**Neden uygun:**  
- Dersin normalizing flow modülüne çok güçlü bağ. citeturn23view0turn3view0  
- “Flow’lar geri mi geldi?” gibi güncel ve net bir hikâye.

**Neden riskli:**  
- Kod tabanı modern ama yine de güçlü compute isteyebilir; iki kişilik ekip için “ölçek küçültme” şart.  
- Eğer hedefiniz “stabil ve kısa sürede” ise, bu fikir MaskGIT‑scheduler fikrine göre daha riskli.

**Linkler:**
```text
TarFlow paper (arXiv): https://arxiv.org/abs/2412.06329
TarFlow official code: https://github.com/apple/ml-tarflow
Apple research page: https://machinelearning.apple.com/research/normalizing-flows
```

### Fikir F: Visual tokenizer’da codebook collapse / utilization problemi: IBQ vs klasik VQ (küçük ölçekli karşılaştırma)

**Kısa başlık:** IBQ ile codebook utilization stabil mi? Küçük ölçekli test

**Hangi paper’dan çıktığı:** *Taming Scalable Visual Tokenizer for Autoregressive Image Generation* (IBQ) + resmi repo citeturn8search3turn8search7  
Paper, klasik VQ’da codebook instabilitesi ve utilization düşüşü → collapse riskine odaklanıp IBQ öneriyor. citeturn8search3turn8search2

**Neyi reproduce edeceksiniz:**  
- Küçük bir codebook ve küçük veri üzerinde “utilization” metrikleri (perplexity/active codes) ve reconstruction kalitesi ölçümü.

**Neyi extend edeceksiniz:**  
- IBQ’yu “küçük codebook” rejiminde, klasik EMA‑VQ ve/veya basit straight‑through VQ ile kıyaslayıp “collapse eğilimi”ni raporlamak.

**Veri / metrik / compute tahmini:**  
- Veri: CIFAR‑10 veya küçük bir yüz veri seti. citeturn9search0turn9search1  
- Metrikler: reconstruction (PSNR/SSIM), utilization (aktif kod yüzdesi, perplexity), downstream generative kalite (opsiyonel).  
- Compute: *Orta*.

**Neden uygun:**  
- “Discrete tokenization” dersin güncel ekseni; ablation çok net.

**Neden riskli:**  
- IBQ’nun resmi hedefi büyük ölçek; küçük ölçekte davranışları daha “noisy” olabilir.  
- Repo büyük bir ekosistemin parçası olabilir; minimal koşacak şekilde sadeleştirme gerekebilir. citeturn8search7turn8search3

**Linkler:**
```text
IBQ paper (arXiv): https://arxiv.org/abs/2412.02692
IBQ code (SEED-Voken): https://github.com/TencentARC/SEED-Voken
```

### Fikir G: StyleGAN2‑ADA ile “limited data” rejiminde stabil GAN eğitimi + augmentation ablation

**Kısa başlık:** Limited data GAN training: ADA gerçekten fark yaratıyor mu?

**Hangi paper’dan çıktığı:** *Training Generative Adversarial Networks with Limited Data* (StyleGAN2‑ADA) + resmi PyTorch repo citeturn19search1turn19search0

**Neyi reproduce edeceksiniz:**  
- Küçük bir datasette (binler mertebesi) StyleGAN2‑ADA training/fine‑tune ile görsel örnekler + FID raporu.

**Neyi extend edeceksiniz:**  
- ADA şiddeti / augmentation türleri / dataset büyüklüğü eksenlerinde ablation; “overfitting” belirtilerini (train vs eval quality gap) görselleştirme.

**Veri / metrik / compute tahmini:**  
- Metrikler: FID/KID; ayrıca training stability ve mode collapse belirtileri.  
- Compute: *Orta‑üst*. GAN training yine hassas olabilir (ama ADA amacı zaten bu hassasiyeti azaltmak). citeturn19search1turn19search0

**Neden uygun:**  
- “Limited data” koşulu, iki kişilik ekip için pratik; dataset toplamak kolay.

**Neden riskli:**  
- Training hâlâ GPU ister ve konfigürasyon ayrıntıları zaman alabilir; ayrıca GAN’lar bazen “nazlıdır”.

**Linkler:**
```text
StyleGAN2-ADA paper (arXiv): https://arxiv.org/abs/2006.06676
StyleGAN2-ADA official PyTorch repo: https://github.com/nvlabs/stylegan2-ada-pytorch
```

### Fikir H: VICReg ile düşük maliyetli self-supervised representation learning + collapse/ablation analizi

**Kısa başlık:** VICReg’de collapse’a karşı regularization dengeleri

**Hangi paper’dan çıktığı:** *VICReg: Variance‑Invariance‑Covariance Regularization for Self‑Supervised Learning* + resmi repo citeturn12search4turn12search12

**Neyi reproduce edeceksiniz:**  
- Küçük datasette (STL‑10 özellikle SSL için tasarlanmış) self‑supervised pretrain + linear probe / k‑NN değerlendirmesi. citeturn9search2turn12search12

**Neyi extend edeceksiniz:**  
- VICReg loss ağırlıkları (variance/covariance) üzerinde ablation; embedding dağılımlarını ve “collapse” göstergelerini raporlamak.

**Veri / metrik / compute tahmini:**  
- Veri: STL‑10. citeturn9search2  
- Metrikler: linear probe accuracy, k‑NN accuracy, embedding covariance heatmap gibi nitel analiz.  
- Compute: *Düşük‑orta* (özellikle küçük backbone ile).

**Neden uygun:**  
- Ders kapsamındaki “self‑supervised learning” bölümüne doğrudan uyum. citeturn26view0turn23view0  
- Writing/presentation hikâyesi net (collapse problemi → VICReg çözümü → ablation).

**Neden riskli:**  
- Generative sample kalitesi gibi “görsel wow” üretmez; ama rapor açısından çok sağlam olabilir.

**Linkler:**
```text
VICReg paper (arXiv): https://arxiv.org/abs/2105.04906
VICReg official code: https://github.com/facebookresearch/vicreg
STL-10 official: https://cs.stanford.edu/~acoates/stl10/
```

### Fikir I: MAE masking ratio ablation’ı ve küçük ölçekte downstream etkisi

**Kısa başlık:** MAE’de masking ratio / decoder kapasitesi → temsil kalitesi

**Hangi paper’dan çıktığı:** *Masked Autoencoders Are Scalable Vision Learners* + resmi repo citeturn12search3turn12search7

**Neyi reproduce edeceksiniz:**  
- Küçük veri üzerinde MAE pretrain + linear probe; masking ratio değişince sonuç trendi. Paper, yüksek mask oranının (örn. 75%) “anlamlı bir self‑supervisory task” olduğunu vurguluyor. citeturn12search3

**Neyi extend edeceksiniz:**  
- Aynı compute bütçesinde farklı mask oranları ve decoder derinliğiyle “efficiency vs representation” tradeoff’u.

**Veri / metrik / compute tahmini:**  
- Veri: CIFAR‑10/Imagenette gibi; görüntü çözünürlüğü küçük tutulursa compute makul. citeturn9search0turn9search3  
- Metrikler: linear probe accuracy, reconstruction görselleri (nitel).  
- Compute: *Orta*.

**Neden uygun:**  
- Dersin self‑supervised kısmına güçlü bağ; ayrıca “masked modeling” hattı MaskGIT ile kavramsal akraba.

**Neden riskli:**  
- ViT tabanlı eğitim, CPU/Mac üzerinde yavaş olabilir; GPU/Colab planı gerekebilir (MLRC kaynakları da Colab gibi seçenekleri öneriyor). citeturn6view2turn12search7

**Linkler:**
```text
MAE paper (arXiv): https://arxiv.org/abs/2111.06377
MAE official code: https://github.com/facebookresearch/mae
```

## Objektif karşılaştırma tablosu ve risk sınıflandırması

Aşağıdaki tabloda 1–5 arası puanlar **pratik yapılabilirlik odağında** (5 = daha iyi). “Compute” sütununda 5 = daha az compute.

| Fikir | Dersle uyum | 2 kişiyle yapılabilirlik | Compute (az=5) | Mac/Win uyumu | Resmi/sağlam repo | Prototype hızı | Net “hikâye” |
|---|---:|---:|---:|---:|---:|---:|---:|
| A MaskGIT scheduler + Halton | 5 | 5 | 4 | 5 | 5 | 4 | 5 |
| B VQGAN tokenizer ablation | 5 | 3 | 3 | 5 | 4 | 3 | 4 |
| C Flow Matching CIFAR | 5 | 4 | 3 | 5 | 5 | 4 | 4 |
| D Improved DDPM ablation | 5 | 3 | 2 | 5 | 5 | 3 | 4 |
| E TarFlow “recipe” ablation | 5 | 2 | 2 | 4 | 5 | 2 | 4 |
| F IBQ tokenizer collapse analizi | 4 | 3 | 3 | 4 | 4 | 2 | 4 |
| G StyleGAN2‑ADA limited data | 4 | 3 | 2 | 4 | 5 | 3 | 4 |
| H VICReg SSL ablation | 4 | 5 | 4 | 5 | 5 | 5 | 4 |
| I MAE masking ablation | 4 | 4 | 3 | 5 | 5 | 3 | 4 |

Bu puanlama, dersin proje rubric’inde öne çıkan “net problem/method/experiments/results” beklentisini doğrudan hedefliyor. citeturn4view0turn3view0

**En uygun (düşük/orta risk, en dengeli):**  
- **Fikir A (MaskGIT + Halton scheduler)** citeturn13view0turn28view1  
- **Fikir C (Flow Matching CIFAR)** citeturn31view0turn10search4  
- **Fikir H (VICReg SSL ablation)** citeturn12search12turn12search4  

**Orta riskli (iyi ama zaman/compute yönetimi şart):**  
- Fikir B (VQGAN tokenizer ablation) citeturn7search6turn7search2  
- Fikir D (Improved DDPM) citeturn11search2turn11search1  
- Fikir G (StyleGAN2‑ADA) citeturn19search1turn19search0  
- Fikir I (MAE) citeturn12search7turn12search3  

**Gereksiz zor (hedef kısıtlarınıza göre risk yüksek):**  
- Fikir E (TarFlow) ve kısmen Fikir F (IBQ) — ikisi de ölçek küçültülse bile kod/compute karmaşıklığı nedeniyle “takvim baskısı” altında risk büyütebilir. citeturn20search6turn20search0turn8search3turn8search7  

## En iyi üç aday, gerekçeli seçim ve final öneri

### En iyi üç

**Birinci:** MaskGIT scheduler + Halton Scheduler (Fikir A)  
MaskGIT paper’ında “mask schedule / masking design”ın kritik olduğu ve burada “future work” alanı olduğu açıkça belirtiliyor; Halton scheduler ise tam bu noktayı hedefleyen, retraining gerektirmeyen “drop‑in” bir improvement olarak sunuluyor. citeturn14view0turn28view1turn28view0 Bu kombinasyon, iki kişilik ekip için hem yapılabilir hem de contribution çok net.

**İkinci:** Flow Matching CIFAR‑10 (Fikir C)  
Resmi PyTorch codebase’in CIFAR‑10 örnekleri vermesi ve “tek komutla train edilebilir” iddiası, kısa sürede sağlam baseline alabilmenizi kolaylaştırır. citeturn31view0 Ayrıca path/solver/steps gibi ablation eksenleri rapor için altın değerinde.

**Üçüncü:** VICReg (Fikir H)  
Compute düşük; resmi repo var; metrikler çok net (linear probe/kNN). citeturn12search12turn9search2 Eğer generative modellerde “örnek kalitesi almak” riskli gelirse, bu fikir güvenli bir “B planı”dır.

### Ben olsam bunu seçerdim

**Ben olsam Fikir A’yı seçerdim:** *MaskGIT scheduling design’i reproduce + Halton Scheduler’ı drop‑in extension olarak test etme.*  
Sebep: Dersin generative‑transformer ekseninde, **en net “problem → ablation → sonuç”** hikâyesini ve en düşük “engineering sürprizi” riskini veriyor. citeturn4view0turn13view0turn28view1

## Seçilen proje için proposal taslağı

Aşağıdaki taslak, paylaştığınız “~2 sayfa proposal” bölüm dağılımına (intro/related/method/eval/work plan/abstract+refs) göre proposal‑ready şekilde yazıldı. (Spring 2024 proje sayfasındaki aynı şablon yapısını referans aldım.) citeturn3view0

### Geçici başlık

**“MaskGIT Decoding Schedulers at Small Scale: Reproducing Mask Scheduling Effects and Extending with Halton Sampling”**

### Araştırma sorusu

Masked token image generation’da, **token unmasking schedule** seçimi (1) sample kalitesini (FID/KID), (2) çeşitliliği (precision/recall proxy’leri), (3) inference verimini (iteration/latency) nasıl etkiler?  
Ayrıca Halton scheduler gibi “spatially uniform” seçim stratejileri, küçük ölçekte bile confidence‑based schedule’a kıyasla ölçülebilir iyileşme sağlar mı? citeturn14view0turn28view1

### Temel paper’lar ve okunacaklar

- **MaskGIT**: *Masked Generative Image Transformer* (CVPR 2022). “Mask scheduling”in kritik etkisi ve “future work” sinyali. citeturn13view0turn14view0  
- **Halton Scheduler**: *Halton Scheduler for Masked Generative Image Transformer* (ICLR 2025). “Noise injection olmadan, drop‑in scheduler” iddiası. citeturn28view1turn28view0  
- (Arka plan) VQ‑tabanlı tokenizer yaklaşımı: *Taming Transformers / VQGAN* (CVPR 2021). citeturn7search6turn7search2  

### Önerilen baseline

**Baseline‑1 (MaskGIT cosine schedule):** MaskGIT paper’ındaki “cosine schedule”ı temel alıp küçük çözünürlükte train/infer; FID/KID raporla. citeturn14view1turn14view0  

**Baseline‑2 (linear / square schedule):** Aynı model ve tokenizer ile schedule değiştir; “schedule concavity” farklarını göster.

Pratik not: Tokenizer eğitimi en pahalı kısım olursa, pretrained VQ tokenizer kullanımı (Halton repo’nun indirme akışı gibi) planlanacak. citeturn28view0turn13view0

### Önerilen extension

**Extension‑1: Halton Scheduler drop‑in**  
- Aynı trained model üzerinde sampling prosedürünü Halton scheduler ile değiştirip FID/KID ve nitel örnekleri kıyasla. Paper, retraining gerekmeden “drop‑in replacement” olabileceğini söylüyor. citeturn28view1turn27search10  

**Extension‑2: Iteration sayısı ve “sweet spot” analizi**  
- T ∈ {4, 8, 12, 16} gibi küçük bir grid ile: kalite‑hız tradeoff grafiği. MaskGIT paper’ında T arttıkça her zaman iyileşme olmayabileceğine dair gözlem var. citeturn14view0  

### Dataset seçimi

**Ana seçenek:** CIFAR‑10 (hızlı prototip, hızlı ablation) citeturn9search0turn23view0  
**Opsiyonel ikinci seçenek:** Imagenette (daha “ImageNet benzeri” görünüm; ama yine küçük) citeturn9search3  

Seçim gerekçesi: küçük ölçekte FID/KID çalıştırmak, iki kişilik ekip ve düşük/orta compute hedefi için en gerçekçi. citeturn9search0turn33view0

### Metrikler ve değerlendirme protokolü

- **FID** (ana metrik) + **KID** (opsiyonel daha stabil küçük örneklemde)  
- Sampling verimi: **iteration sayısı**, **saniye başına örnek** (tek GPU/Colab ölçümü)  
- Nitel analiz: class‑conditional örnekler, failure case’ler, scheduler’a duyarlı artefact’lar

Sunum rubric’inin “quantitative + qualitative comparison, ablation, limitations” beklentisini doğrudan hedefler. citeturn4view0

### Compute planı

- Hedef: tek GPU (örn. Colab) üzerinde küçük çözünürlük/mini model ile çalışır pipeline. MLRC kaynakları Colab’ı bir compute opsiyonu olarak listeliyor; bu ders projesi için de pratik bir kaçış yolu. citeturn6view2turn33view0  
- “Önce çalışır baseline, sonra ablation” yaklaşımı: scheduler deneyleri inference‑ağırlıklı tutulursa eğitim yükü azalır. citeturn28view1turn28view0

### Kabaca zaman çizelgesi

Aşağıdaki planı Spring 2026 deliverable’larınıza göre tasarladım (proposal → progress presentation/report → final). Spring 2026 metninizde progress report tarih çelişkisi olduğundan, “erken tarih”e göre güvenli plan öneriyorum. (Spring 2024 sayfasında deliverable yapısı benzer.) citeturn3view0turn24view0

- **Proposal günü (Mart sonu)**  
  - Seçim kesinleştirme: CIFAR‑10 + MaskGIT scheduler study + Halton extension  
  - Repo seçimi ve minimal “runs on my machine” denemesi (inference demo). citeturn28view0turn9search0  

- **Nisan başı–ortası**  
  - Baseline training (küçük model) + ilk FID/KID ölçümü  
  - Schedule ablation (cosine vs linear vs square)  

- **Progress presentation’a kadar (Nisan sonu)**  
  - Halton scheduler entegrasyonu ve kıyas grafikleri  
  - 1–2 failure case analizi + “limitations” taslağı  

- **Progress report’a kadar (Mayıs başı)**  
  - Deneyleri stabilize etme, tablolama  
  - Method şeması + reproducibility (seed, config, exact commands)  

- **Final dönem (Mayıs sonu–Haziran başı)**  
  - Ek ablation: iteration sayısı T sweep  
  - Son metin: related work genişletme + sonuçların yorumlanması + future directions

### Proposal’da hangi başlık altında ne yazılacağı

**Introduction (1/4 sayfa):**  
- Generative transformers’da raster‑order AR’nin verimsizliği; MaskGIT’in masked‑token iterative decoding yaklaşımı. citeturn13view0  
- Araştırma odağı: “scheduler choice”ın kritik olması; bu nedenle scheduler analizi. citeturn14view0  

**Related Work (1/2 sayfa):**  
- MaskGIT + masked modeling çerçevesi. citeturn13view0  
- Halton scheduler katkısı ve drop‑in iddiası. citeturn28view1  
- Tokenization arka planı (VQGAN). citeturn7search6turn7search2  

**Method (1/4 sayfa):**  
- Pipeline: tokenizer → discrete tokens → masked transformer → iterative sampling  
- Scheduler tanımı: cosine/linear/square ve Halton‑based token selection. citeturn14view1turn28view1  

**Experimental evaluation (1/8 sayfa):**  
- Dataset: CIFAR‑10 (opsiyonel Imagenette) citeturn9search0turn9search3  
- Metrikler: FID/KID + sampling time

**Work plan (1/8 sayfa):**  
- Baseline → scheduler ablation → Halton extension → raporlama

**Abstract + References (1/4 sayfa):**  
- 4–5 cümlelik hedef + katkı + beklenen çıktı; referans listesi (MaskGIT, Halton, VQGAN, dataset).

### Proje “reproducibility” paketini baştan garantiye alma

Ders projesi MLRC’ye submit edilmese bile MLRC’nin önerdiği “reproducible code + README + exact run steps” yaklaşımı burada çok işe yarar. citeturn6view2turn33view0 Minimum checklist: `requirements/environment.yml`, `train.py`, `eval.py`, config dosyaları, seed’ler, kısa “reproduce table” komutları.

**Proposal‑ready link paketi**
```text
MaskGIT (CVPR 2022 PDF): https://openaccess.thecvf.com/content/CVPR2022/papers/Chang_MaskGIT_Masked_Generative_Image_Transformer_CVPR_2022_paper.pdf
Halton Scheduler (ICLR 2025 OpenReview): https://openreview.net/forum?id=RDVrlWAb7K
Halton-MaskGIT repo: https://github.com/valeoai/Halton-MaskGIT
VQGAN repo: https://github.com/compvis/taming-transformers
CIFAR-10: https://www.cs.toronto.edu/~kriz/cifar.html
Imagenette: https://github.com/fastai/imagenette

MLRC resources (reproducible code pointers): https://reproml.org/challenge_resources/
ML Code Completeness Checklist: https://github.com/paperswithcode/releasing-research-code
```