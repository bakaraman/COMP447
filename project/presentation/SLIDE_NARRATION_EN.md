# Slide Narration Script — Technical, Conversational

> Total speaking time: ~6 minutes
> Style: technical but accessible — assumes the audience knows basic ML, doesn't assume they've read the diffusion / consistency model literature
> Use the slide titles as anchors; the script below is what to say while each slide is on screen

---

## Slide 1 — Title

> "Hi, I'm Batuhan, and my partner is Kadir Yiğit. This is the progress report for our COMP447 project. The high-level question is simple: **when does paying upfront for consistency tuning actually beat the alternative of just running a diffusion solver?** And the answer pointed us to a deeper open question that we want to take on. Let me walk you through it."

*~15 seconds*

---

## Slide 2 — The question

> "Quick framing. Modern image generators come from two camps.
>
> Diffusion models reach state-of-the-art quality, but inference takes 30 to 100+ neural function evaluations per image. They're slow.
>
> Consistency models compress sampling into one or two NFEs — orders of magnitude faster. But you don't get them for free; you have to either train them from scratch or distill from a diffusion teacher. Both are compute-heavy.
>
> So the practical question for someone deploying these systems is: **is the upfront training worth it, and once you've done it, what's the right way to sample?**"

*~30 seconds*

---

## Slide 3 — Background

> "Two equations to anchor the math.
>
> The top one is the probability flow ODE — the deterministic backbone of diffusion models, formalized by Karras and team in EDM at NeurIPS 2022. To solve it numerically you typically use Heun's method, a 2nd-order ODE solver. Heun spends two forward passes per step — so 18 steps adds up to 35 NFEs.
>
> The bottom one is the consistency model formulation from Yang Song's 2023 paper. The model learns a single function that jumps directly from any point on the trajectory to the data manifold. So instead of solving an ODE step by step, you just call the function once or twice."

*~30 seconds*

---

## Slide 4 — Easy Consistency Tuning

> "The specific consistency model we work with is **Easy Consistency Tuning** — ECT — published at ICLR 2025. The reason ECT is interesting: instead of distilling from a teacher, you take a pretrained diffusion model and fine-tune it directly with this self-consistency loss. They report FID 2.11 on CIFAR-10 with about one A100-hour of tuning.
>
> One detail to keep in mind. Two-step inference requires picking an intermediate timestep — the place between full-noise and clean-data where you make the second function call. The paper hardcodes this at 0.821. **No derivation, no ablation, no justification.** A magic number. Hold onto that."

*~40 seconds*

---

## Slide 5 — Pareto frontier

> "On to the experiment. CIFAR-10. Same EDM checkpoint, same GPU, same evaluation protocol. Two branches: ECT fine-tuning for 1980 kimg with four saved checkpoints, and Heun on the original model at 5 to 50 steps.
>
> Pareto plot on screen. X-axis is wall-clock latency, Y-axis is FID, both log scale.
>
> The gray curve is Heun. The red diamonds are our ECT runs. **ECT 2-step gets FID 2.47 in 14 milliseconds. Heun-18 — the best Heun configuration — gets FID 1.96 in 244 milliseconds.** That's 17× faster, and the FID gap is 0.5. ECT Pareto-dominates Heun across every operating point we measured, even on this short tuning budget."

*~45 seconds*

---

## Slide 6 — Tuning budget ablation

> "Quick check on whether the full training was necessary. We measured FID at 500, 1000, 1500, and 1980 kimg.
>
> Big drop early — 500 to 1000 takes us from FID 8 down to 2.7. After that, gains are small. **By 1980 kimg we're at FID 2.45, which is 94% of paper-reported quality at half a percent of the paper's training budget.**
>
> Diminishing returns past 1000 kimg are clear — early stopping is realistic. Break-even versus Heun-18 sits around 23.5k generated images. After that, the upfront tuning has paid for itself."

*~30 seconds*

---

## Slide 7 — The real open problem  *(climax — slow down)*

> "This is where the project shifted.
>
> While we were characterizing 2-step behavior across the four checkpoints, we noticed every single result depended on that one parameter — the midpoint at 0.821. Same value, fixed across every training stage, every step budget.
>
> *(read the quote on screen, slow)*
>
> Someone on GitHub asked the original ECT author exactly this question — how to choose the midpoint. The author's answer is on screen verbatim:
>
> *'It's a very good question and I think also an open research problem. I don't have a manual for it. **You can treat it as an optimization problem, maximizing sample quality with respect to these sampling schedules.**'*
>
> So we have two things. **One**: the author of the model himself calls this an open research problem. **Two**: he literally suggests treating it as an optimization problem. That's the move we're making.
>
> The literature backs this up. ECT, sCM, Multistep CM — none of them derive the midpoint. Target-Driven Distillation, a 2024 paper, picks timesteps **heuristically** from a predefined set. But nobody has done **principled, KL-bound-based** schedule optimization for consistency models. NVIDIA solved that for diffusion in Align Your Steps using a KLUB derivation — but their derivation is for the score field, not the consistency flow map. Different mathematical object, derivation doesn't transfer.
>
> **That's the gap our method fills.**"

*~70 seconds — pause for 2 seconds after the quote*

---

## Slide 8 — Our method · KLUB CM schedule

> "Our proposal: **KLUB CM**. Adapt the Align Your Steps KL upper bound to consistency models — re-derive it for the consistency function instead of the score field.
>
> Equation on screen. We pick schedule knots that minimize the L2 distance between consecutive applications of the consistency function. That distance upper-bounds the discretization error introduced by approximating the trajectory at our chosen knots — same logic as AYS, just applied to the flow map.
>
> Three sources of inspiration: the KL upper bound technique from AYS, trajectory awareness from Consistency Trajectory Models, and adaptive scheduling from a 2025 paper on optimal stepsizes for diffusion.
>
> Two payoffs. **First**, the schedule is checkpoint-conditioned — it adapts to model quality instead of being a fixed magic number. **Second**, predicted impact: this should push our ECT operating points further left on the Pareto frontier, and it generalizes to step counts we haven't trained for, without retraining."

*~50 seconds*

---

## Slide 9 — Implementation · what we are coding

> "Concretely, four pieces.
>
> **One**: derive the KL upper bound for the consistency setting. AYS's derivation uses the score function — we redo it for the flow map. Math write-up.
>
> **Two**: schedule optimizer in PyTorch. L-BFGS over the schedule knots — gradient-based, orders of magnitude cheaper than grid-searching across schedules.
>
> **Three**: a patch to ECT's loss function. Pseudocode on screen: the student produces its own intermediate output `x_mid`, then we enforce consistency at the optimized midpoint with the EMA target. This is on-policy training — the student sees its own outputs at inference-relevant noise levels. Idea borrowed from the GKD distillation paper, Agarwal et al. 2023.
>
> **Four**: a corrected evaluation pipeline so 1-step and 2-step are routed through the right sampler — needed because we found the upstream evaluation script was conflating them."

*~50 seconds*

---

## Slide 10 — Status & next steps

> "Where we stand. On the left, what's done: ECT tuning with four checkpoints, Heun baselines from 5 to 50 steps, latency profiling at B=1 and B=64, and the corrected evaluation pipeline.
>
> On the right, up next: a midpoint sensitivity sweep across the four checkpoints — we want to verify the optimum drifts before committing the math. Then the KLUB derivation and the L-BFGS optimizer. The on-policy fine-tune is a stretch goal. Final report early June.
>
> Main risk is GPU time on Colab; mitigated by running screening sweeps at 10k samples and keeping the checkpoint footprint small.
>
> Happy to take questions."

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
