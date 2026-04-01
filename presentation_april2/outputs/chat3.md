Ham araştırma dosyasını burada bıraktım: [SQ-GAN Private Investigator dossier](sandbox:/mnt/data/sq_gan_private_investigator_dossier.md)

Not: ortak isimler yüzünden özellikle Jian Wang, Xin Lan ve Yuxin Tian için yalnızca güvenle eşleştirebildiğim profilleri kullandım; doğrulanamayan alanları dosyada **bulunamadı** diye işaretledim.

## En net önerim: **Jian Wang (王坚)** üzerine odaklan

Sunum için en güçlü “hikâye” onda. Çünkü 2025 paper döneminde Sichuan University’de Jiancheng Lv danışmanlığında doktora öğrencisi olarak görünüyor; resmi merkez öğrenci listesinde doktora tez başlığı doğrudan **“küçük örneklemli görüntü üretimi için adversarial ağ yöntemleri”** olarak geçiyor. Ayrıca yayın çizgisi de çok temiz: 2020’de generative/painting, 2021’de FusionGAN, 2022’de CR-GAN, 2024’te skull-to-face ve MS3D, 2025’te SQ-GAN. Bu, “bu paper bir anda çıkmadı; yıllardır biriken doktora hattının sonucu” demeni sağlar. ([OpenReview][1])

Yedek aday olarak **Jizhe Zhou (周吉喆)** çok iyi. Kamuya açık profili daha zengin: kişisel sitesi var, tüm dereceleri Macau’dan, CUHK araştırma stajı var, IEEE TETCI associate editor, üretim + forensics + reasoning hattı var. Ama SQ-GAN’a girişinin Wang kadar “doğrudan doktora izi” şeklinde olmadığı görülüyor. ([center.dicalab.cn][2])

## Yazarlar – kısa ama sunum-odaklı dosya

### 1) Jian Wang / 王坚

2025 paper döneminde Sichuan University College of Computer Science’ta doktora öğrencisi. OpenReview profili, SCU’de 2017–2020 arasında yüksek lisans, 2021’den itibaren doktora yaptığını ve danışmanının Jiancheng Lv olduğunu gösteriyor. Kişisel sayfa, Google Scholar, DBLP, ORCID, OpenReview ve GitHub eşleşmeleri güvenli biçimde bulunabiliyor; Google Scholar’da yaklaşık 87 atıf ve h-index 5 görünüyor. Resmî merkezin daha güncel öğrenci listesi ise onu mezun doktora öğrencileri arasında veriyor; yani paper zamanı PhD student, daha sonra mezun olmuş görünüyor. ([OpenReview][1])

Onu SQ-GAN için çok güçlü yapan şey tez başlığı: **小样本生图像生成的对抗网络方法**. Yayın çizgisi de buna uyuyor: 2020’de *An Abstract Painting Generation Method Based on Deep Generative Model*, 2021’de *Using FusionGAN to Create Abstract Paintings*, 2022’de *CR-GAN: Automatic Craniofacial Reconstruction for Personal Identification*, 2024’te *From Skulls to Faces* ve aynı yıl *MS3D*, 2025’te ise *SQ-GAN*. Benim çıkarımım: SQ-GAN’daki ana teknik omurga büyük olasılıkla Wang’ın doktora hattından geliyor; özellikle limited-data GAN eğitimine ilişkin deney tasarımı, style-space manipülasyonu ve ablation’ların önemli kısmında belirleyici olmuş olabilir. Bu son cümle yorumdur, doğrudan yazmıyor; ama tez başlığı + yayın sırası bunu kuvvetle destekliyor. ([center.dicalab.cn][3])

### 2) Xin Lan / 兰鑫

Xin Lan için en güvenli eşleşme, DICALab’ın yüksek lisans üye sayfasındaki **兰鑫** kaydı. Burada 2022 girişli yüksek lisans öğrencisi olarak görünüyor; araştırma alanları “generative models, multimodal learning, transfer learning”. Kişisel sayfa olarak `whalelan.space` veriliyor; OpenReview profil URL’si var; ancak DBLP sayfası temiz bir yazar sayfası değil, açıkça bir **disambiguation page**. Merkezin CVPR 2025 haberinde de Lan Xin’in 2022 yüksek lisans öğrencisi olduğu ve ilgili işlerinin ICML 2024 ile CVPR 2025’te kabul aldığı yazıyor. Dosyadaki “bulunamadı” işaretlerinin önemli kısmı bu isim çakışmasından kaynaklanıyor. ([center.dicalab.cn][4])

Güvenle atayabildiğim yayın hattı üçlü: *From Skulls to Faces* (MMM 2024), *MS3D* (ICML 2024), *SQ-GAN* (CVPR 2025). Yani Lan, bu limited-data GAN hattına doğrudan 2024’te giriyor ve 2025’te co-first author seviyesine çıkıyor. Bu da “yüksek lisans öğrencisinden arka arkaya ICML + CVPR” gibi ilginç bir anekdot sunuyor. Scholar metrikleri ve sosyal profiller için güvenli eşleşme bulamadım; dosyada bu alanlar bu yüzden açıkça eksik bırakıldı. ([DBLP][5])

### 3) Jizhe Zhou / 周吉喆 / Ji-Zhe Zhou

Jizhe Zhou’nun kimliği en temiz eşleşen profillerden biri. Resmî genç öğretim üyesi sayfasında **周吉喆**, e-posta `jzzhou@scu.edu.cn`, araştırma alanları “image generation, anti-image generation, neural-network reasoning” olarak veriliyor. OpenReview ve kişisel sayfasına göre Sichuan University’de associate professor; lisans, yüksek lisans ve doktoranın tamamını University of Macau’da yapmış, doktorayı 2021’de tamamlamış, danışmanı Chi-Man Pun, ayrıca CUHK’de research intern olmuş. ORCID ve DBLP eşleşmesi de bulunuyor. ([center.dicalab.cn][2])

Jizhe’nin profiline asıl renk katan şey üretim ile forensics’i aynı anda çalışması. Kendi sitesinde IMDL-BenCo, IML-ViT, SparseViT, ForensicHub gibi projeleri yönettiğini yazıyor; ayrıca 2026’dan itibaren IEEE TETCI associate editor olduğunu hem sitesinde hem IEEE sayfasında görüyoruz. Google Scholar arama sonucu yaklaşık 805 atıf gösteriyor; h-index değerini doğru profile güvenle çekemedim, bu yüzden dosyada “bulunamadı” olarak bıraktım. SQ-GAN bağlamında en makul yorum şu: Jizhe, ekibe 2025 aşamasında katılan ve özellikle semantic guidance, CLIP/foundation-model tarafı ve codebook/discrete representation fikrini zenginleştiren öğretim üyesi olabilir. Bu son cümle yine yorumdur; ama profilindeki generation + forensics + reasoning kombinasyonu paper’ın semantik hizalama tarafıyla iyi örtüşüyor. ([Ji-Zhe Zhou (周吉喆)][6])

### 4) Yuxin Tian / 田煜鑫

Yuxin Tian’ın OpenReview profili çok bilgilendirici: 2016–2020 arasında SCU’de lisans, 2020–2026 arası SCU’de doktora, 2022–2023’te Alibaba DAMO Academy’de intern, 2024’ten itibaren de Ant Group’ta intern görünüyor. Merkezin öğrenci listesinde **田煜鑫** mezun doktora öğrencileri arasında ve tez başlığı **“multi-source noise altında robust learning”**. Scholar arama sonucu 141 atıf ve h-index 6 gösteriyor. Scholar’daki profil açıklamasında da “Ph.D. in CS, inclusionAI, Ant Group” ifadesi geçiyor; bu yüzden güncel endüstri bağlantısı güçlü ama kalıcı unvanını güvenle söylemek zor. ([OpenReview][7])

Yuxin’in yayın çizgisi GAN’e dar bir açıdan değil, “veri kısıtı / gürültü / verimli öğrenme” açısından geliyor. OpenReview’de ve ilgili kayıtlarda *MS3D*, *SQ-GAN*, federated learning/noisy labels işleri ve vision-language pretraining üstüne verimli fine-tuning çalışmaları görünüyor. Bu yüzden SQ-GAN’daki muhtemel katkısı, limited-data training ile robust/efficient learning sezgisini birleştirmek, deneysel tasarım ve belki de CLIP/feature alignment tarafını desteklemek olabilir. Yine bu bir yorum; fakat önceki işlerinin dağılımı bunu makul kılıyor. ([OpenReview][7])

### 5) Jiancheng Lv / 吕建成

Jiancheng Lv bu paper’ın en açık “PI/corresponding author” figürü. Resmî SCU sayfasında profesör / doktora danışmanı olarak görünüyor; başka resmî sayfalar ve merkez tanıtımı onu aynı zamanda Sichuan University Computer Science College dean’i ve DICALab/ERC direktörü olarak konumluyor. Eğitim çizgisi de net: 2003’te UESTC’den bilgisayar uygulamaları yüksek lisansı, 2006’da UESTC’den bilgisayar bilimi doktorası, 2007–2008 arasında NUS’ta postdoc, 2008’den beri Sichuan University. Resmî biyografisinde ayrıca akademiye dönmeden önce iki yıl ZTE’de çalıştığı yazıyor. Google Scholar arama sonucu yaklaşık 12.749 atıf ve h-index 48 gösteriyor. ([SCU Computer Science][8])

Jiancheng Lv’nin SQ-GAN’daki yeri büyük olasılıkla stratejik: laboratuvar hattını kurmak, Wang–Lan–Tian çizgisini uzun vadeli bir limited-data GAN programına çevirmek, altyapı ve fonları sağlamak ve paper’ı daha geniş “machine learning for constrained data” araştırma gündemine yerleştirmek. Bu yine yorumdur; fakat 2024’te MS3D, 2025’te SQ-GAN, eşzamanlı olarak da çok sayıda efficient learning, multimodal ve industrial AI projesi yürütmesi bu okumanın güçlü olduğunu gösteriyor. ([Proceedings of Machine Learning Research][9])

## Lab ve grup resmi

**Machine Intelligence Lab** eski çekirdek yapı gibi görünüyor: 2008’de kurulmuş, ilgi alanları machine intelligence, neural networks, machine learning, data mining, data fusion, computer vision ve medical image processing. Direktör Yi Zhang; people sayfasında 10 öğretim üyesi ve ayrıca konuk profesörler listeleniyor. ([machineilab.org][10])

**DICALab / Engineering Research Center of Machine Learning and Industry Intelligence** ise daha yeni ve daha kurumsal çatı. Resmî tanıtım sayfasına göre Sichuan University ev sahipliğinde, China Nuclear Power Research and Design Institute ve dönüşüm/uygulama şirketleriyle birlikte kurulmuş; dört resmî yönü var: industrial perception/computing, industrial analysis/prediction, intelligent decision optimization ve nuclear-industry intelligent technology integration. Direktör Jiancheng Lv; akademik liderler arasında Yi Zhang, Lei Zhang ve Xi Peng var. Bu yüzden benim okuma biçimim şu: MachineILab eski “araştırma laboratuvarı”, DICALab ise bunun üstüne inşa edilmiş, bakanlık onaylı, uygulama ve sanayi bağlantısı daha belirgin merkez. Bu son cümle bir çıkarım, ama kişi örtüşmeleri ve kurumsal roller bunu kuvvetle destekliyor. ([center.dicalab.cn][11])

Bu merkez bağlamı paper’ı anlamayı kolaylaştırıyor. Çünkü resmî araştırma yönleri doğrudan “bol veriyle internet-scale pretraining” değil; endüstriyel analiz, karar verme ve nükleer uygulamalara kadar giden, çoğu zaman veri erişimi pahalı veya sınırlı olan ortamlar. Dolayısıyla limited-data GAN training bu grup için teorik bir hobi değil, laboratuvarın kurumsal yönelimiyle uyumlu bir problem. Ayrıca öğrenci/mezun sayfalarında Huawei, DAMO Academy, Ant Group, Tencent yarışmaları ve benzeri endüstri izleri de görülüyor; bunlar resmî sponsor sayfası değil ama güçlü endüstri teması olduğunu gösteriyor. ([center.dicalab.cn][11])

## SQ-GAN araştırma bağlamı

Bu paper aynı ekibin **MS3D** paper’ından sonra geliyor. MS3D, ICML 2024’te limited-data GAN training için bir regularization yöntemi öneriyor; SQ-GAN ise 2025’te aynı sorunu başka bir açıdan, style-space quantization üzerinden ele alıyor. Takvim açısından bakınca bu bence açıkça **sıralı** bir araştırma hattı: MS3D önce konferans paper’ı olarak ortaya çıkıyor, ardından SQ-GAN CVPR 2025’e gidiyor. ArXiv tarihleri ilk bakışta bunu biraz bulanıklaştırsa da konferans kronolojisi MS3D → SQ-GAN zincirini destekliyor. ([Proceedings of Machine Learning Research][9])

Teknik olarak paper’ın kilit iddiası şu: limited-data durumda latent uzaydaki komşu noktalar bile çok farklı gerçeklik düzeyinde görüntüler üretebiliyor; bu da consistency regularization’ı zayıflatıyor. Bunun üzerine SQ-GAN, StyleGAN-benzeri style space’i ayrık bir codebook ile quantize ediyor, sonra bu quantized space üzerinde consistency regularization kuruyor. Üstelik codebook başlangıcını sadece rasgele yapmıyor; CLIP’ten gelen semantik bilgi ve optimal transport hizalamasıyla güçlendiriyor. Yani paper’ı “VQ-VAE fikrini GAN’in style space’ine, hem de limited-data CR problemi için uyarlama” diye özetlemek bence çok doğru. ([CVF Open Access][12])

Burada baseline farkını da kısa anlatabilirsin. **StyleGAN2-ADA**, discriminator overfitting’ini azaltmak için adaptive augmentation ile limited-data setting’de çok güçlü bir temel veriyor. **DiffAugment** yine differentiable augmentations üzerinden çalışıyor. **CR / ICR** ise augmentation sonrası tahminlerin tutarlılığını regularize ediyor. SQ-GAN’ın farkı, görüntü düzeyinde augmentation eklemekten çok, latent/style uzayının geometrisini ve semantik yapısını düzenleyerek CR’yi “daha anlamlı” hale getirmesi. ([NeurIPS Proceedings][13])

“Bu fikir daha önce de var mıydı?” sorusuna dengeli cevap: evet, discrete latent / learnable discrete GAN fikirleri tamamen yeni değil; örneğin *Discrete and Efficient Latent Distributions for GANs* gibi işler var. VQ-VAE hattı da codebook/discrete representation fikrini çoktan kurmuştu. Ama SQ-GAN’ın özgün yanı, bu discrete/codebook fikrini **style-space + limited-data + consistency regularization** üçlüsünde kullanması. Benim okuma biçimimle novelty tam burada. ([OpenReview][14])

## Motivasyon analizi

Az veriyle GAN eğitimi neden önemli? Sunumda bunu dört başlıkta söylemek güçlü olur: tıp/forensics gibi veri toplaması zor alanlar, telif ve sanat alanında kısıtlı koleksiyonlar, gizlilik nedeniyle büyük veri havuzu kurulamayan uygulamalar ve nadir sınıflar/olaylar. Bu son madde özellikle SCU grubunun craniofacial reconstruction, boneprint ve industrial AI işlerine çok iyi oturuyor. Bu bağlantı benim yorumum, fakat grup yayınları ve merkez yönleri bunu destekliyor. ([DBLP][15])

Çin’in **PIPL** yasasının 1 Kasım 2021’de yürürlüğe girdiği doğru; kişisel verinin toplanması ve işlenmesine daha sıkı kurallar getiriyor. Doğrudan “SQ-GAN bunu yüzden yazıldı” diyecek kanıt bulamadım. Ama özellikle yüz, sağlık ve hassas verilerde büyük veri seti kurmanın zorlaşması limited-data yöntemleri için dolaylı bir motivasyon oluşturmuş olabilir; bunu ancak **ihtiyatlı bir bağlamsal yorum** olarak söylemek doğru olur. ([China Briefing][16])

## Zaman çizelgesi, funding, public review

Takvim çok temiz: CVPR 2025 paper registration deadline Kasım 2024 başı, submission deadline Kasım 2024 ortası, kararlar 26 Şubat 2025’te açıklandı. Merkezin haber sayfası paper kabulünü 11 Mart 2025’te duyuruyor; arXiv v1 ise 31 Mart 2025 tarihli. Bu da paper’ın CVPR kararından sonra arXiv’e çıktığını gösteriyor. ([CVPR][17])

Acknowledgements kısmı da çok işlevsel bir slide malzemesi: Central Universities fonu (1082204112364), NSFC National Major Scientific Instruments and Equipments Development Project (62427820) ve Sichuan Province Creative Research Groups fonu (2024NSFTD0035). Yani paper yalnızca “öğrenci projesi” değil, kurumsal olarak finanse edilen bir araştırma hattının parçası. Public CVPR review thread’ine erişemedim; görünen OpenReview kaydı public review’den çok CoRR/arXiv archive girişi gibi duruyor. ([CVF Open Access][12])

## 8–10 dakikalık sunum için en güçlü 10 bilgi

1. Jian Wang’ın doktora tez başlığı neredeyse SQ-GAN’ın problem cümlesiyle aynı: “small-sample image generation.” ([center.dicalab.cn][3])
2. Aynı çekirdek ekip 2024’te ICML’de **MS3D**, 2025’te CVPR’de **SQ-GAN** yayımlıyor; bu tek seferlik bir paper değil, bir araştırma hattı. ([Proceedings of Machine Learning Research][9])
3. Xin Lan yüksek lisans öğrencisiyken ICML 2024 ve CVPR 2025 çizgisine giriyor; bu hızlı yükseliş sunumda dikkat çeker. ([center.dicalab.cn][4])
4. Jizhe Zhou’nun profili “image generation + anti-image generation/forensics + reasoning”; paper’daki semantic/codebook yönüyle çok uyumlu. ([center.dicalab.cn][2])
5. Jiancheng Lv önce ZTE’de çalışmış, sonra NUS postdoc sonrası SCU’ye geçmiş; endüstri + akademi karışımı bir PI. ([SCU Faculty Homepages][18])
6. DICALab’ın resmî araştırma yönleri industrial intelligence ve hatta nuclear-industry applications içeriyor; veri kıtlığı bu grup için gerçek bir problem. ([center.dicalab.cn][11])
7. SQ-GAN, StyleGAN style space’ini codebook ile quantize edip CR’yi güçlendiriyor; yani katkı augmentation’dan çok latent geometry tarafında. ([CVF Open Access][12])
8. Paper, codebook initialization için CLIP ve optimal transport kullanıyor; bu, foundation-model alignment’ın paper’daki en dikkat çekici kısmı. ([CVF Open Access][12])
9. Paper üç ayrı kurumsal fonla desteklenmiş; bu da ekibin bu konuda ciddi ve süreklilik taşıyan bir programı olduğunu düşündürüyor. ([CVF Open Access][12])
10. Public CVPR review bulunmuyor; görünen OpenReview kaydı büyük olasılıkla CoRR/arXiv kaydı. Bu küçük ama hoş bir “dedektiflik” detayı. ([OpenReview][19])

## Önerilen slide sırası

1. Başlık: `🕵️: Batuhan` + paper adı.
2. “Bu ekip kim?” — SCU, DICALab, MachineILab org haritası.
3. Neden **Jian Wang**? — tez başlığı ve rol.
4. Wang’ın zaman çizelgesi: 2020 → 2025 yayın yolu.
5. Diğer yazarlar: Lan, Jizhe, Yuxin, Jiancheng — her biri 1 cümlelik rol.
6. MS3D → SQ-GAN teknik evrim.
7. Lab, funding ve endüstri/uygulama bağlamı.
8. Sonuç: “Bu paper bir doktora hattının ve kurumsal araştırma programının ürünü.”

## Söyleyebileceğin 5 ilginç gözlem

Jian Wang için “paper aslında tezinden çıkmış gibi duruyor” cümlesi çok güçlü.

Xin Lan için “yüksek lisans öğrencisinden ICML + CVPR co-first author” ayrıntısı sınıfta dikkat çeker.

Jizhe Zhou’nun generation ve forensics’i aynı anda çalışması, “neden CLIP ve semantics?” sorusuna insan hikâyesi üzerinden cevap veriyor. ([center.dicalab.cn][2])

MS3D ile SQ-GAN arasındaki geçiş, aynı problemi önce gradient-flow/reg olarak, sonra style-space/discretization olarak ele alan iki ardışık hamle gibi okunabilir. Bu benim yorumum, ama paper kronolojisi bunu destekliyor. ([Proceedings of Machine Learning Research][9])

Lab’ın resmî olarak industrial intelligence ve nuclear applications vurgulaması, “neden limited data?” sorusunu soyut değil kurumsal hale getiriyor. ([center.dicalab.cn][11])

## Yazarlara atılabilecek kısa mail taslağı

```text
Subject: Question about your CVPR 2025 paper for Prof. Aykut Erdem’s seminar

Dear [Author Name],

I am a student in Prof. Aykut Erdem’s COMP547/447 Deep Unsupervised Learning seminar at Koç University, and I will present your CVPR 2025 paper “Style Quantization for Data-Efficient GAN Training.”

The seminar page is:
https://aykuterdem.github.io/classes/comp547.s25

I am preparing a short “private investigator” segment about the authors’ research background, and I wanted to ask one brief question: which previous project or practical application most directly motivated SQ-GAN?

Any short comment would be greatly appreciated. Thank you for your time and for your paper.

Best regards,
Batuhan
```

En pratik kullanım yolu şu: sunumda **tek ana karakter olarak Jian Wang’ı** seç, ama bir slaytta “neden bu ekip birlikte mantıklı?” sorusunu Jizhe + Yuxin + Jiancheng üzerinden tamamla.

[1]: https://openreview.net/profile?id=~Jian_Wang12 "https://openreview.net/profile?id=~Jian_Wang12"
[2]: https://center.dicalab.cn/%E9%9D%92%E5%B9%B4%E6%95%99%E5%B8%88%E5%9B%A2%E9%98%9F%E4%BB%8B%E7%BB%8D/ "https://center.dicalab.cn/%E9%9D%92%E5%B9%B4%E6%95%99%E5%B8%88%E5%9B%A2%E9%98%9F%E4%BB%8B%E7%BB%8D/"
[3]: https://center.dicalab.cn/%E5%AD%A6%E7%94%9F%E5%90%8D%E5%8D%95/ "https://center.dicalab.cn/%E5%AD%A6%E7%94%9F%E5%90%8D%E5%8D%95/"
[4]: https://center.dicalab.cn/2024/08/16/%E7%A1%95%E5%A3%AB%E6%88%90%E5%91%98%E4%BB%8B%E7%BB%8D%EF%BC%88%E9%83%A8%E5%88%86%EF%BC%89/ "https://center.dicalab.cn/2024/08/16/%E7%A1%95%E5%A3%AB%E6%88%90%E5%91%98%E4%BB%8B%E7%BB%8D%EF%BC%88%E9%83%A8%E5%88%86%EF%BC%89/"
[5]: https://dblp.org/pid/144/7703 "https://dblp.org/pid/144/7703"
[6]: https://knightzjz.github.io/ "https://knightzjz.github.io/"
[7]: https://openreview.net/profile?id=~Yuxin_Tian3 "https://openreview.net/profile?id=~Yuxin_Tian3"
[8]: https://cs.scu.edu.cn/info/1288/13627.htm "https://cs.scu.edu.cn/info/1288/13627.htm"
[9]: https://proceedings.mlr.press/v235/wang24af.html "https://proceedings.mlr.press/v235/wang24af.html"
[10]: https://machineilab.org/ "https://machineilab.org/"
[11]: https://center.dicalab.cn/%E4%B8%AD%E5%BF%83%E7%AE%80%E4%BB%8B/ "https://center.dicalab.cn/%E4%B8%AD%E5%BF%83%E7%AE%80%E4%BB%8B/"
[12]: https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_Style_Quantization_for_Data-Efficient_GAN_Training_CVPR_2025_paper.pdf "https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_Style_Quantization_for_Data-Efficient_GAN_Training_CVPR_2025_paper.pdf"
[13]: https://proceedings.neurips.cc/paper_files/paper/2020/hash/8d30aa96e72440759f74bd2306c1fa3d-Abstract.html "https://proceedings.neurips.cc/paper_files/paper/2020/hash/8d30aa96e72440759f74bd2306c1fa3d-Abstract.html"
[14]: https://openreview.net/forum?id=eyyS-zovT9m "https://openreview.net/forum?id=eyyS-zovT9m"
[15]: https://dblp.org/pid/39/449-124.html "https://dblp.org/pid/39/449-124.html"
[16]: https://www.china-briefing.com/news/wp-content/uploads/2021/08/Personal-Information-Protection-Law-of-the-Peoples-Republic-of-China.pdf "https://www.china-briefing.com/news/wp-content/uploads/2021/08/Personal-Information-Protection-Law-of-the-Peoples-Republic-of-China.pdf"
[17]: https://cvpr.thecvf.com/Conferences/2025/CallForPapers "https://cvpr.thecvf.com/Conferences/2025/CallForPapers"
[18]: https://faculty.scu.edu.cn/lvjiancheng/zh_CN/index.htm "https://faculty.scu.edu.cn/lvjiancheng/zh_CN/index.htm"
[19]: https://openreview.net/search?content=authors&group=all&sort=cdate%3Adesc&source=forum&term=~Yuxin_Tian3 "https://openreview.net/search?content=authors&group=all&sort=cdate%3Adesc&source=forum&term=~Yuxin_Tian3"
