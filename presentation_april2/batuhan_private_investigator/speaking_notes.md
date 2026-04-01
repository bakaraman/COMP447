# Speaking Notes — Private Investigator: Batuhan Karaman

## Slide 1 — Title

Hi everyone, I am Batuhan and my role is Private Investigator. I researched the people behind this paper and tried to understand how and why they ended up writing it.

## Slide 2 — The Team

All five authors work at Sichuan University in Chengdu. They belong to two labs in the same department. The first one is MachineILab, founded in 2008 by Zhang Yi, who is an IEEE Fellow. The second is DICALab, run by Jiancheng Lv. Lv was actually Zhang Yi's PhD student and is now the Dean of the CS department. So you have three academic generations here: the founder, his student who became dean, and then the current students who wrote this paper. This is not something they put together last minute. It comes from a research program that has been going on for years.

## Slide 3 — Jian Wang

The first author is Jian Wang. He is a PhD student and he did all his degrees at Sichuan. The interesting part is his thesis title. It translates to "adversarial network methods for small-sample image generation." If you compare that to the title of SQ-GAN, they say almost the same thing. So this paper is not a side project for him. It is literally his PhD thesis work.

## Slide 4 — Research Arc

If you look at what Wang published over the years, the path is very clear. He started with generating abstract paintings in 2020. Then he moved to reconstructing faces from skulls in 2022. Then image translation in 2024. That same year he published MS3D at ICML, which was about fixing the discriminator when you have limited data. And now SQ-GAN at CVPR, which fixes the generator side. So it is one person gradually narrowing down to one specific problem over five years.

## Slide 5 — Two Papers

This is worth pointing out. The same core team published two papers at two consecutive top venues. MS3D came out at ICML 2024 and fixed the discriminator. SQ-GAN came out at CVPR 2025 and fixed the generator. Together they cover both sides of the same problem. One thing I noticed is that Jizhe Zhou was not on MS3D but appears on SQ-GAN. He joined later. The two papers were probably being developed at the same time during 2024. MS3D went on arXiv in August and SQ-GAN was submitted to CVPR in November.

## Slide 6 — Supporting Cast

Xin Lan is a master's student but she is co-first author on both MS3D and SQ-GAN. Getting co-first author at ICML and CVPR back to back as a master's student is not common. She also handled the reviewer correspondence as corresponding author.

Jizhe Zhou is an associate professor. He did all his degrees at the University of Macau and spent time at CUHK in Hong Kong and Yonsei University in Korea. He is the most internationally experienced person on the team. His research area is what he calls "content generation and anti-generation," so he both builds generative models and builds systems that detect fakes. He was not on the MS3D paper, he joined at the SQ-GAN stage.

Yuxin Tian is a PhD student who now works at Ant Group. His main research is on continual learning and large language models, not GANs. He is more of a lab collaborator on this paper.

## Slide 7 — Jiancheng Lv

Jiancheng Lv is the person who runs everything. He is a full professor and the Dean of the CS department. He has about 12,000 citations and an h-index around 48. Before academia he worked at ZTE, a telecom company. His own PhD advisor was Zhang Yi, the IEEE Fellow who founded MachineILab. One of Lv's former students, Dayiheng Liu, is now at Alibaba DAMO Academy and is one of the core people behind the Qwen language model. In this paper Lv's role is strategic. He sets the research direction, provides funding, and makes sure the group has a long-term plan.

## Slide 8 — Motivation

I wanted to understand why this specific lab cares about training GANs with limited data. It turns out it is not just an academic interest. DICALab officially lists nuclear-industry AI as one of its research directions. In that domain data is scarce, sometimes expensive, sometimes classified. The lab also works on craniofacial reconstruction and dental identification, and medical data is inherently limited. On top of that, China passed the PIPL law in 2021, which makes it harder to collect biometric and medical data. So for this group, limited-data generation is a practical problem they face in their own applications. The paper is also backed by three separate institutional grants, so this is a funded research program, not a student side project.

## Slide 9 — Summary

To wrap up: SQ-GAN is the product of Jian Wang's PhD thesis. His thesis title and the paper's problem statement say almost the same thing. The team published two papers in a row, MS3D for the discriminator and SQ-GAN for the generator, attacking the same problem from both ends. Xin Lan is a master's student who co-first authored at ICML and CVPR in consecutive years, which is unusual. Jizhe Zhou brought in forensics expertise when he joined for SQ-GAN. And the lab's focus on industrial and medical applications means limited-data training is something they actually need, not just something they study. Three academic generations, two labs, one sustained program.
