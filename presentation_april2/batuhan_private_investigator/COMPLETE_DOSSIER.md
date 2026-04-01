# Private Investigator Dossier: SQ-GAN (CVPR 2025)

Paper: Style Quantization for Data-Efficient GAN Training
Authors: Jian Wang, Xin Lan, Jizhe Zhou, Yuxin Tian, Jiancheng Lv
All from: Sichuan University, College of Computer Science, Chengdu, China

---

## The Big Picture

This is a single-lab paper. Five people, one university, two labs (MachineILab + DICALab), three academic generations. Not a one-off paper — it is the latest output of a multi-year program on data-efficient GANs that already produced MS3D at ICML 2024.

The academic hierarchy:
- Jiancheng Lv (Dean, PI, last author) — runs everything
- Jizhe Zhou (Assoc. Prof, direct supervisor) — generation + forensics
- Jian Wang (PhD student, co-first author) — main technical driver
- Xin Lan (MSc student, co-first author, corresponding) — co-developer
- Yuxin Tian (PhD student, co-author) — lab collaborator

---

## 1. JIAN WANG — First Author (recommended focus for presentation)

**Why him:** His PhD thesis title is almost word-for-word the paper's problem statement.

### Identity
- Chinese name: 王坚 (from lab page) or 王健 (from chat1) — extremely common name
- Position: PhD student (2021–present), may have recently graduated
- All degrees from SCU: BSc (2013-2017), MSc (2017-2020), PhD (2021-present)
- Advisor: Prof. Jiancheng Lv
- PhD thesis title: 小样本生图像生成的对抗网络方法 = "adversarial network methods for small-sample image generation"

### Profiles
- Email: jianwang.scu@gmail.com (found in MS3D paper)
- GitHub: https://github.com/jianwang-scu
- OpenReview: https://openreview.net/profile?id=~Jian_Wang12
- DBLP: https://dblp.org/pid/39/449-124.html
- Google Scholar: ~87 citations (snippet)
- Photo: https://center.dicalab.cn/wp-content/uploads/2024/08/WechatIMG568.jpg
- Personal website: not found
- LinkedIn/Twitter: not found

### Publication timeline (the story arc)
| Year | Venue | Paper | Relevance |
|------|-------|-------|-----------|
| 2020 | — | Abstract painting generation (deep generative model) | Early GAN work |
| 2021 | — | FusionGAN (abstract paintings) | Generative models |
| 2022 | — | CR-GAN (craniofacial reconstruction) | GAN for limited medical data |
| 2024 | IEEE TNNLS | SLIT: Selective image translation | GAN, spatial disentanglement |
| 2024 | **ICML** | **MS3D**: discriminator regularization for limited-data GANs | **First author**, direct predecessor |
| 2024 | IEEE TSMC | UNITE: multitask learning | Not GAN |
| 2025 | **CVPR** | **SQ-GAN**: style quantization for data-efficient GANs | **Co-first author**, this paper |
| 2025 | Info. Fusion | Neural architecture representation | Not GAN |
| 2025 | KBS | ICCR-Diff: craniofacial reconstruction with diffusion | Generative, medical |

**The arc:** painting -> faces -> image translation -> limited-data GAN discriminator (MS3D) -> limited-data GAN generator (SQ-GAN). This is a PhD built on one problem.

### Competition awards (from lab page)
- Ascend AI Innovation Contest
- National/CCF-related competitions (details unclear)

---

## 2. XIN LAN — Co-First Author, Corresponding Author

### Identity
- Chinese name: 兰鑫
- Position: MSc student (2022-2025), NOT faculty despite being corresponding author
- BSc: Software Engineering, SCU (2018-2022)
- MSc: Computer Science, SCU (2022-2025)
- Advisor: Prof. Jiancheng Lv
- Personal site: https://whalelan.space (could not be verified)
- Email: 2971529737@qq.com (from lab page, may have typo)
- Photo: https://center.dicalab.cn/wp-content/uploads/2024/08/WechatIMG598.jpg

### Publications
| Year | Venue | Paper |
|------|-------|-------|
| 2024 | MMM | From Skulls to Faces (craniofacial reconstruction) |
| 2024 | **ICML** | MS3D (co-author) |
| 2025 | **CVPR** | SQ-GAN (co-first, corresponding) |

**Remarkable fact:** MSc student who co-first-authored at ICML and CVPR in consecutive years. This is unusual and worth mentioning in the presentation.

**Why corresponding author as MSc student?** Likely handled paper submission, reviewer correspondence, and revision process. Shows significant responsibility for a master's student.

---

## 3. JIZHE ZHOU — Associate Professor, Direct Supervisor

### Identity
- Chinese name: 周吉喆
- Position: Tenured Associate Professor (2025-), previously Associate Researcher (2021-2025)
- Provincial "Thousand Talents" designation
- Homepage: https://knightzjz.github.io/
- Email: jzzhou@scu.edu.cn
- Photo: https://center.dicalab.cn/wp-content/uploads/2024/08/WechatIMG1146.jpeg

### Education (all from University of Macau — unusual)
| Degree | Year | Institution | Notes |
|--------|------|-------------|-------|
| BSc | 2010-2013 (or 2013) | University of Macau | Software Engineering |
| MSc | 2013-2018 | University of Macau | Software Engineering |
| PhD | 2018-2021 | University of Macau | Early graduation. Advisor: Prof. Chi-Man Pun (World's Top 2% Scientists) |

### International experience (most traveled in the team)
- Research assistant, CUHK (2016-2018) — image manipulation detection
- Kakao Track, Yonsei University, South Korea (2019-2020)
- Joined SCU in 2021

### Key profiles
- Google Scholar: https://scholar.google.com/citations?user=-cNWmJMAAAAJ (~324-805 citations depending on source)
- DBLP: https://dblp.org/pid/172/4712.html
- GitHub: https://github.com/Knightzjz (personal), https://github.com/scu-zjz (org)
- OpenReview: https://openreview.net/profile?id=~Ji-Zhe_Zhou1
- ORCID: 0000-0002-2447-1806
- LinkedIn: https://www.linkedin.com/in/jizhe-zhou/
- Twitter: https://twitter.com/knightzjz

### Research focus: "content generation & anti-generation"
He works both sides: making images AND detecting fake images.

| Year | Venue | Paper | Type |
|------|-------|-------|------|
| 2025 | **CVPR** | SQ-GAN | Generation |
| 2025 | **NeurIPS** | ForensicHub: fake image detection benchmark | Forensics |
| 2026 | **AAAI** | FGM-HD: fractal generative models | Generation |
| 2024 | **NeurIPS (Spotlight)** | IMDL-BenCo: manipulation detection benchmark | Forensics |
| 2024 | IEEE TNNLS | SLIT: selective image translation | GAN |
| 2023 | **ICCV (Oral)** | NCL-IML: contrastive learning for manipulation localization | Forensics |
| 2023 | IEEE TNNLS | DHI-GAN: dental human identification | GAN |
| 2024 | **ACM MM** | Neural Boneprint: person ID from bones | Generative/forensic |

### Service
- IEEE TETCI Associate Editor (2026-)
- Area Chair: PRCV 2025, ICME 2026, ICMR 2026
- Senior PC: IJCAI 2025 & 2026
- PC: CVPR, ICML, NeurIPS, ICLR, ICCV, ECCV, AAAI

### Awards
- 1st Prize, Tencent AI Arena Competition (1/2000+ teams, 2024)
- 2nd Prize, Alibaba Tianchi AI Challenge (2/1574 teams, 2024)

### Key observation for presentation
Zhou was **NOT on MS3D (ICML 2024)** but appears on SQ-GAN. He joined the project at a later stage. His forensics expertise likely contributed to the discriminator robustness analysis and the semantic/CLIP alignment aspects of the codebook.

---

## 4. YUXIN TIAN — Senior PhD Student

### Identity
- Chinese name: 田煜鑫
- Position: PhD student (2020-2026), now at Ant Group (2026-)
- BSc: SCU (2015/2016-2019/2020)
- PhD: SCU (2020-2025/2026), advisors: Jiancheng Lv + Xi Peng
- PhD thesis: "multi-source noise altinda robust learning"
- Photo: https://center.dicalab.cn/wp-content/uploads/2024/08/WechatIMG478.jpg

### Industry
- Alibaba DAMO Academy intern (2022-2023)
- Ant Group (inclusionAI) intern then researcher (2024-2026+)

### Profiles
- Google Scholar: https://scholar.google.com/citations?user=n36mg0QAAAAJ (~141 citations, h-index 6)
- OpenReview: https://openreview.net/profile?id=~Yuxin_Tian3
- Email: cs.yuxintian@outlook.com

### Not primarily a GAN researcher
Main focus: continual learning, MoE, LLMs, federated learning. Also has CVPR 2025 paper "Ferret" (continual learning). Co-authored MS3D and SQ-GAN as a lab collaborator with shared infrastructure and robust learning expertise.

---

## 5. JIANCHENG LV — Dean, Principal Investigator

### Identity
- Chinese name: 吕建成
- Position: Full Professor, Dean of CS + Dean of Intelligence Science, SCU
- National Science Fund for Distinguished Young Scholars
- State Council Expert for Special Allowance
- Email: lvjiancheng@scu.edu.cn
- Faculty page: https://faculty.scu.edu.cn/lvjiancheng/zh_CN/index.htm

### Education
| Degree | Year | Institution | Notes |
|--------|------|-------------|-------|
| BSc | 1999 | UESTC | Computer Science |
| MSc | 2003 | UESTC | Computer Application |
| PhD | 2006 | UESTC | CS, advisor: Zhang Yi (IEEE Fellow) |
| Postdoc | 2007-2008 | NUS Singapore | ECE, with Kok Kiong Tan |
| ~2 years | pre-2008 | ZTE Corporation | R&D and management |

### Profiles
- Google Scholar: https://scholar.google.com/citations?user=0TCaWKwAAAAJ (~12,000 citations, h-index ~48)
- DBLP: https://dblp.org/pid/68/2367-1.html
- OpenReview: https://openreview.net/profile?id=~Jiancheng_Lv2
- ResearchGate: 375+ publications

### Awards
- Second Prize, National Natural Science Award of China (2020)
- First Prize, Natural Science Award, Ministry of Education (2012)
- First Prize, Sichuan Science and Technology Progress Award (2011)
- CCF Excellent Doctoral Thesis Award (2007)

### Co-authored monograph
"Subspace Learning of Neural Networks" with Zhang Yi (CRC Press, 2011)

### Notable alumni
Dayiheng Liu (刘大一恒) — now at Alibaba DAMO Academy, core contributor to Qwen LLM

---

## 6. THE TWO LABS

### Machine Intelligence Lab (machineilab.org)
- Founded: October 10, 2008
- Director: Zhang Yi (IEEE Fellow, Lv's PhD advisor)
- Focus: neural networks, ML, medical image processing, data fusion, CV
- ~11 faculty, 20+ PhD students
- Guest professors include Pheng Ann Heng (CUHK) and Zhihua Zhou (Nanjing Univ.)

### DICALab (center.dicalab.cn)
- Director: Jiancheng Lv
- Focus: NLP/NLG, medical imaging, industrial fault detection, image generation
- Engineering Research Center of Machine Learning and Industry Intelligence (Ministry of Education designation)
- 4 research directions: industrial perception, industrial analysis, intelligent decision optimization, **nuclear-industry intelligent technology**

### Relationship
Genealogical: Zhang Yi (founder) -> Lv (PhD student, now Dean) -> current students. MachineILab is the parent, DICALab is Lv's independent expansion. Same college, shared intellectual lineage.

### Industry connections
DICALab alumni at Alibaba DAMO. Tian interned at Alibaba + Ant Group. Zhou won prizes at Alibaba Tianchi + Tencent AI Arena. Strong Chinese tech industry ties.

---

## 7. MS3D to SQ-GAN: THE TWO-PAPER ARC

### MS3D (ICML 2024)
- Authors: Jian Wang, Xin Lan, Yuxin Tian, Jiancheng Lv (note: NO Jizhe Zhou)
- arXiv: August 20, 2024
- Problem: under limited data, discriminator gradient field becomes aggregated
- Solution: RG (renormalization group) flow-based regularization to maintain consistent gradients
- Fixes the **discriminator side**

### SQ-GAN (CVPR 2025)
- Authors: + Jizhe Zhou added
- arXiv: March 31, 2025
- Submission: ~November 14, 2024 (CVPR deadline)
- Decision: February 26, 2025
- Problem: sparse latent space undermines consistency regularization
- Solution: quantize style space with codebook, align with CLIP via optimal transport
- Fixes the **generator side**

### The strategy
Same problem (limited-data GANs), two complementary attacks. MS3D fixed the discriminator, SQ-GAN fixed the generator. Likely developed in parallel during 2024.

---

## 8. PAPER FUNDING & ACKNOWLEDGEMENTS

From the paper:
- Fundamental Research Funds for the Central Universities (Grant 1082204112364)
- NSFC National Major Scientific Instruments and Equipments Development Project (Grant 62427820)
- Science Fund for Creative Research Groups of Sichuan Province (No. 2024NSFTD0035)

Not a student side project — institutionally funded research program.

Public CVPR reviews: not available (CVPR does not release reviews).

---

## 9. WHY LIMITED-DATA GANs? MOTIVATION

- **Medical imaging:** rare diseases, pathological slides, specialized scans — data is inherently scarce. The lab does craniofacial reconstruction, boneprint ID, dental GAN.
- **Privacy:** China's PIPL (November 2021) restricts collection of biometric/medical data. Makes large datasets harder to build.
- **Nuclear industry:** DICALab's official direction includes nuclear-industry AI. Data in nuclear settings is classified/limited.
- **Art/culture:** rare images, historical documents, limited collections.
- **The lab's own need:** their industrial intelligence focus means they work in domains where data is expensive by default.

---

## 10. TOP ANECDOTES FOR THE PRESENTATION

1. **Wang's thesis title = paper's problem statement.** His entire PhD is about this exact problem. SQ-GAN is not a side project; it is his thesis.

2. **Xin Lan: MSc student with ICML + CVPR co-first in consecutive years.** Unusual achievement for a master's student. And she's the corresponding author.

3. **Zhou was NOT on MS3D but joined for SQ-GAN.** The forensics expert entered at the right moment — her "anti-generation" expertise strengthened the semantic/codebook side.

4. **Three academic generations in one paper.** Zhang Yi (IEEE Fellow, founded MachineILab) -> Lv (PhD student, now Dean, runs DICALab) -> Wang/Lan/Tian (current students).

5. **Nuclear industry AI.** DICALab officially works on nuclear-industry intelligent technology. Limited data is their daily reality, not an academic exercise.

6. **Two papers, one strategy.** MS3D fixed the discriminator. SQ-GAN fixed the generator. Same team, consecutive top venues, two sides of the same coin.

7. **Lv worked at ZTE before academia.** Industry experience before becoming Dean — bridges practical and theoretical.

---

## 11. RECOMMENDED SLIDE STRUCTURE

1. **🕵️ Batuhan: The Team** — org chart, two labs, three generations
2. **🕵️ Batuhan: Jian Wang** — thesis title, PhD arc, photo
3. **🕵️ Batuhan: Wang's Publication Timeline** — painting to SQ-GAN (visual timeline)
4. **🕵️ Batuhan: MS3D -> SQ-GAN** — discriminator then generator, two-paper strategy
5. **🕵️ Batuhan: The Supporting Cast** — Lan (MSc + CVPR!), Zhou (forensics), Tian (industry), Lv (Dean)
6. **🕵️ Batuhan: Why This Problem?** — medical, nuclear, PIPL, lab's own need
7. **🕵️ Batuhan: Summary** — thesis = paper, systematic research, not an accident

---

## 12. EMAIL

**To:** jianwang.scu@gmail.com (Jian Wang, first author)
**CC:** jzzhou@scu.edu.cn (Jizhe Zhou, has public email)

**Subject:** Question about SQ-GAN for Prof. Aykut Erdem's seminar at Koc University

Dear Jian Wang,

I am a student in Prof. Aykut Erdem's COMP547 Deep Unsupervised Learning seminar at Koc University, Istanbul. Your CVPR 2025 paper "Style Quantization for Data-Efficient GAN Training" was assigned for our class discussion.

Seminar page: https://aykuterdem.github.io/classes/comp547.s25

I am preparing a "Private Investigator" segment about the authors. Two short questions:

1. I noticed your PhD thesis title is very close to SQ-GAN's problem statement. Was data-efficient GAN training your planned thesis direction from the start, or did it develop after your earlier work on painting generation and craniofacial reconstruction?

2. MS3D addressed the discriminator and SQ-GAN addressed the generator. Were these planned as a pair, or did SQ-GAN come from observing what MS3D could not fix?

Any brief reply would be appreciated. Thank you for your time.

Best regards,
Batuhan Karaman
Koc University, Department of Computer Engineering
