# Slide Narration Script — Technical, Conversational

> Total speaking time: ~6 minutes
> Style: technical but accessible — assumes the audience knows basic ML, doesn't assume they've read the diffusion / consistency model literature
> Use the slide titles as anchors; the script below is what to say while each slide is on screen

---

## Slide 1 — Title

> "Hello. I am Kadir, my partner is Batuhan. This is our progress report for COMP447. Our main question is simple: **when is it worth spending time to tune a consistency model, instead of using a diffusion model directly?** While we worked on this, we found a bigger open question. That open question is the main story today. Let me show you."

*~15 seconds*

---

## Slide 2 — The question

> "Quick framing. Modern image generators come from two camps — and they make opposite trade-offs.
>
> The first camp is diffusion models. They give state of the art image quality. But the price is high — each image needs 30 to 100+ neural function evaluations, called NFEs. So generation is slow, even if the model itself is ready to use.
>
> The second camp is consistency models. They flip the trade-off. Sampling drops to just 1 or 2 steps — sometimes 30 to 50 times faster. But this speed is not free. The upfront tuning is non trivial. You either train one from zero, or copy knowledge from a diffusion model. Either way, you pay compute before the first image.
>
> So if you actually want to use one of these models, the question has two parts: **when is the cost of consistency tuning worth paying? And once you've paid, what's the right way to sample?**"

*~40 seconds*

---

## Slide 3 — Background

> "Now the math behind it. Two equations on screen.
>
> The top one is the probability flow ODE. This is the equation that diffusion models actually solve — it tells you how to move noise toward a clean image, step by step. Karras and team published this clean form in the EDM paper, NeurIPS 2022. To solve it on a computer, the standard tool is Heun's method, a second order ODE solver. Heun does two forward passes per step. So 18 steps cost 35 NFEs in total.
>
> The bottom one is the consistency model, from Yang Song's 2023 paper. Instead of solving an ODE step by step, the model learns one function. You give it a noisy image and a noise level, and it returns the clean image directly. The key property is in the second part of the equation: two points on the same noise trajectory must give the same output. So one call gives you a 1 step image, two calls give you a 2 step image. That's the speed gain."

*~42 seconds*

---

## Slide 4 — Easy Consistency Tuning

> "The consistency model we use in this project is **Easy Consistency Tuning**, or ECT for short. Published at ICLR 2025 by Geng and team. The interesting part is what's on the screen: instead of distilling from a teacher model, ECT starts from a pretrained diffusion checkpoint and fine tunes it directly. The loss enforces self consistency between two noise levels that are very close to each other. Cheap, simple, effective — the paper reports FID 2.11 on CIFAR 10 with about one A100 hour of tuning.
>
> Now please look at the second bullet. To run a 2 step inference, you have to pick a midpoint — the noise level where you make the second function call. The ECT paper sets this midpoint to 0.821, and uses the same value everywhere. **No derivation. No ablation. No justification.** A magic number. Please hold onto this — we come back to it in three slides."

*~45 seconds*

---

## Slide 5 — Pareto frontier

> "Now the experiment. Everything matched: CIFAR 10, the same EDM checkpoint, the same GPU, the same evaluation protocol. Two branches. On one side, we fine tune with ECT for 1980 kimg and save 4 checkpoints along the way. On the other side, we run Heun on the original diffusion model at 5, 10, 18, 25, and 50 steps.*
>
> *Quick tour of the plot. The x axis is wall clock latency per image, in milliseconds. The y axis is FID — lower is better. Both axes are log scale. The grey line is Heun, with one point per step count. The red diamonds are our two ECT operating points: 1 step and 2 step.*
>
> *Look at the 2 step diamond. **ECT 2 step reaches FID 2.47 in 14 milliseconds. The best Heun configuration, Heun 18, reaches FID 1.96 in 244 milliseconds.** So we are giving up 0.5 FID, and getting 17 times faster generation in return. And this is on a very short tuning budget. ECT Pareto dominates Heun at every latency point we measured."

*~50 seconds*

---

## Slide 6 — Tuning budget ablation

> "Next question — do we actually need the full training run? To find out, we measured 2 step FID at 4 stages: 500, 1000, 1500, and 1980 kimg.
>
> The plot shows what happened. Big drop in the first half: from FID 8 at 500 kimg down to FID 2.7 at 1000 kimg. After that, the curve flattens out fast. **By 1980 kimg we are at FID 2.45 — that is 94 percent of the paper reported quality, using only 0.5 percent of the paper's training budget.** Most of the gain happens early.*
>
> *Two takeaways. First, diminishing returns past 1000 kimg are clear — so early stopping is a realistic recipe, you don't have to train all the way. Second, the bullet at the bottom: break even against Heun 18 lands at around 23.5 thousand generated images. Once you generate more than that, the upfront tuning has already paid for itself."

*~40 seconds*

---

## Slide 7 — The real open problem  *(climax — slow down)*

> "This is the slide where the project changed direction.
>
> While we were comparing 2 step behaviour across our 4 checkpoints, we noticed something. Every single number we reported depended on one hidden parameter — the midpoint at 0.821. Same value at 500 kimg, same value at 1980 kimg, same value for every step budget. Nobody had asked whether this is the right value.
>
> *(pause briefly, then introduce the quote)*
>
> It turns out somebody actually asked. On the ECT GitHub page, a user opened an issue: how do you choose the midpoint? The author of ECT — Zhengyang Geng — answered. His reply is on the screen, word for word:
>
> *'It's a very good question and I think also an open research problem. I don't have a manual for it. **You can treat it as an optimization problem, maximizing sample quality with respect to these sampling schedules.**'*
>
> *(2 second pause)*
>
> So we have two things from the author himself. **One**: the author calls this an open research problem. **Two**: he is literally telling us to treat it as an optimization problem. That is exactly what we are going to do.
>
> The wider literature agrees. The bottom bullets list it: ECT, sCM, Multistep CM — none of them derive the midpoint. Target Driven Distillation, a 2024 paper, picks timesteps **heuristically** from a fixed set. But nobody has done a **principled, KL bound based** schedule optimization for consistency models. NVIDIA already solved this exact problem for diffusion models, in Align Your Steps, using a KL upper bound. But their derivation works on the score field — and consistency models don't have a score field, they have a flow map. So their math does not transfer.
>
> **That gap is what our method fills.**"

*~75 seconds — pause for 2 seconds after the quote*

---

## Slide 8 — Our method · KLUB CM schedule

> "Our method is called **KLUB CM**. The idea is short: take the KL upper bound technique from Align Your Steps, and re-derive it for the consistency function — the flow map — instead of the score field.
>
> The equation on screen is what the optimizer actually solves. We are picking a set of schedule knots — call them t_1 to t_N — that **minimize the squared distance** between two consecutive applications of the consistency function. This distance is an upper bound on the error you introduce when you approximate the full trajectory using only these N knots. Same logic as AYS, just applied to the flow map instead of the score field.
>
> Where the inspiration comes from is on the small line under the equation. Three sources: the KL upper bound technique itself from AYS, trajectory awareness from Consistency Trajectory Models, and the idea of adaptive schedules from a 2025 paper on optimal step sizes for diffusion.
>
> Two reasons this matters. **First**, the schedule is checkpoint conditioned — meaning the optimizer gives you a different schedule for a 500 kimg model and a 1980 kimg model. It adapts to model quality, instead of staying frozen at one magic number. **Second**, the predicted impact: this should push our ECT points further to the left on the Pareto plot, and it generalizes to step counts we never trained for, with no extra training."

*~55 seconds*

---

## Slide 9 — Implementation · what we are coding

> "Four pieces, listed on the slide.
>
> **One**: the KLUB math. AYS uses the score function. We redo it for the flow map.
>
> **Two**: the schedule optimizer in PyTorch. We use L-BFGS over the knots — gradient based, much cheaper than a grid search.
>
> **Three**: a patch to the ECT loss. The pseudocode at the bottom shows it. The student first generates its own image at the midpoint. Then we train it to be consistent with the EMA target at that midpoint. The student learns from its own outputs — this is called **on policy training**. Idea from the GKD paper, 2023.
>
> **Four**: a corrected evaluation pipeline. The upstream script was running both 1 step and 2 step FID through the same sampler. Ours sends each one to the right sampler. This part is already done."

*~42 seconds*

---

## Slide 10 — Status & next steps

> "Final slide. Where we stand right now.
>
> The left column, in red, is what is already done. ECT tuning with 4 checkpoints saved. Heun baselines from 5 to 50 steps. Latency profiling at batch size 1 and batch size 64. And the corrected evaluation pipeline that I mentioned a minute ago.
>
> The right column is what comes next. First, a midpoint sweep across all 4 checkpoints — this is the empirical check that the optimal midpoint really drifts with model quality, before we commit to the KLUB math. Second, the KLUB derivation and the schedule optimizer. Third, the on policy fine tune is a stretch goal — we will get to it if the schedule on its own is not enough. The final report is due early June.
>
> The bottom line lists the main risk: GPU time on Colab. We are managing it by running the sweep on small 10 thousand sample screens, and by keeping the checkpoint footprint small.
>
> That is everything from our side. Happy to take questions."

*~30 seconds*

---

## Speaking tips

| Section | Pace | Key emphasis |
|---|---|---|
| Slides 1–4 (setup) | Brisk, conversational | Plant the **0.821 magic number** seed |
| Slides 5–6 (results) | Confident, point at the numbers | Land **17× faster, 0.5 FID gap** |
| Slide 7 (climax) | Slow, deliberate. Read the quote. **Pause 2 seconds.** | Land **"the author of the model is telling us this is unsolved AND suggesting our exact approach"** |
| Slide 8 (method) | Steady. Point at the equation. | Land the **predicted impact** sentence |
| Slide 9 (implementation) | Steady. Point at the pseudocode. | Show real code is being written |
| Slide 10 (status) | Brief, calm | Open the floor |

---

## If you only remember three things

1. **0.821 is hardcoded** in every consistency model paper, with no derivation.
2. **The author of ECT publicly calls this an open research problem and suggests an optimization approach.**
3. **We adapt NVIDIA's Align Your Steps KLUB to the consistency flow map** — first principled solution in this space.

---

## Q&A — likely questions and short answers

| Question | Short answer |
|---|---|
| Has anyone done schedule optimization for consistency models? | Heuristic approaches exist (TDD, Wang et al. 2024 — predefined equidistant schedules). Nobody has done principled KLUB-based optimization. That's our angle. |
| Why not grid search instead of L-BFGS? | Grid search costs FID computations across many schedules per checkpoint. KLUB optimization is gradient-based on cached features — orders of magnitude cheaper, and it generalizes across step counts without re-search. |
| What FID improvement do you expect? | We won't claim a number until the experiment runs. AYS reports 5–30% FID gains for diffusion at low NFE; we expect a similar order of magnitude for consistency models. |
| When will the code be ready? | Math derivation in two weeks. Optimizer prototype shortly after. Full implementation by mid-May, final results target early June. |
| Why didn't you just use 0.821 and move on? | Because the same 0.821 is used regardless of training stage and step count. Our preliminary observations suggest the optimum drifts with checkpoint quality. Fixing this is a real performance lever, not cosmetic tuning. |
| What's the relationship to sCM and CTM? | sCM works on training-time stability of continuous-time CMs. CTM gives flexibility in trajectory traversal. Neither does principled inference-time schedule optimization, which is what we propose. |
| Can the method generalize beyond CIFAR-10? | The math is dataset-agnostic. We start with CIFAR-10 because we have an EDM checkpoint and clean baselines; ImageNet 64×64 is a natural follow-up if time permits. |
