## Bottom-line diagnosis

Your measurements point to a sharper research question than “ECT vs Heun”:

> **Why does your tuned consistency model behave like a very fast one-shot projector, but not like a useful two-step generator?**

The key empirical seed is not only “ECT is faster.” It is:

* **ECT 1-step ≈ ECT 2-step** after 1000 kimg: 2.797 vs 2.803, 2.736 vs 2.672, 2.460 vs 2.510.
* **Heun 18-step remains the quality anchor**: FID 1.960, but 17× slower than ECT 2-step at batch 1.
* **ECT training saturates early**: most improvement happens by 1000 kimg; another 980 kimg only improves 1-step from 2.797 to 2.460.
* **The amortization story is fragile**, especially at batch 64: the tune cost only pays off after ~285k generated images.

So I would not pitch “more ECT ablations” or “run DPM-Solver++/UniPC.” Those are useful baselines, but not a graduate-level implementation. DPM-Solver, DPM-Solver++, UniPC, and Restart Sampling already occupy the “better fast sampler for the original diffusion model” lane. DPM-Solver targets high-quality generation in roughly 10–20 NFEs; UniPC adds a corrector without extra model evaluations; Restart alternates forward noising and reverse ODE steps. ([arXiv][1])

The best proposals below are designed around your actual anomaly: **the second ECT step is nearly dead weight.**

---

# Prior-work boundary you need to respect

ECT itself frames diffusion pretraining as a loose consistency condition and then tightens that condition during tuning; it claims a 2-step CIFAR-10 FID of 2.73 within 1 A100 hour, starting from pretrained diffusion models. ([arXiv][2]) EDM is the correct quality baseline because the original EDM paper reports unconditional CIFAR-10 FID around 1.97 at 35 NFEs, matching your Heun 18-step measurement almost exactly. ([arXiv][3])

The obvious “make it multistep” idea is already covered. Multistep Consistency Models split the diffusion process into predefined segments and explicitly show 2–8 step CMs as a speed/quality interpolation between one-step CMs and diffusion models. ([arXiv][4]) SCT builds on ECT, frames consistency tuning as TD learning, adds variance-reduced targets, smoother schedules, multistep settings, and edge-skipping inference. ([arXiv][5]) Adaptive Discretization for Consistency Models, arXiv:2510.17266, directly attacks the timestep/discretization schedule problem with adaptive local/global consistency constraints. ([arXiv][6]) CTM and Shortcut Models also generalize CMs by allowing traversal to arbitrary final times or conditioning on the desired step size. ([arXiv][7])

That means the strongest ideas are not “multistep CM” in general. They are **targeted implementations that explain or fix why your ECT checkpoint’s second step contributes almost nothing.**

---

# Idea 1 — On-policy low-noise ECT refinement loss

## Core proposal

Train ECT’s second step on the **distribution it actually sees at inference time**.

Vanilla ECT trains low-noise denoising mostly on real images corrupted with low noise. But your 2-step sampler does this at inference:

[
x_1 = F_\theta(\sigma_{\max} z,\sigma_{\max}),
]

[
\tilde{x}_1 = x_1 + \sigma_r \epsilon,
]

[
x_2 = F_\theta(\tilde{x}_1,\sigma_r).
]

The low-noise input is therefore **not** (x_0+\sigma_r\epsilon) for real CIFAR image (x_0). It is a noisy version of the model’s own one-step sample. If (x_1) has artifacts, color shifts, or off-manifold structure, the second step may never have learned how to correct those.

Implement an auxiliary loss after the base ECT run reaches roughly 1000 kimg:

[
x_c = \operatorname{sg}\left(F_{\bar{\theta}}(\sigma_{\max} z,\sigma_{\max})\right),
]

[
y = x_c + \sigma_r \epsilon,
]

[
x_T = \operatorname{sg}\left(\mathrm{HeunTail}_{\phi}(y;\sigma_r \rightarrow 0, K)\right),
]

[
\mathcal{L}_{\text{on-policy}}
==============================

\rho_c\left(F_\theta(y,\sigma_r)-x_T\right),
]

where:

* (F_\theta) is the ECT consistency model.
* (F_{\bar{\theta}}) is the EMA ECT model used to generate on-policy states.
* (D_\phi) is the frozen original EDM denoiser.
* (\mathrm{HeunTail}_{\phi}) runs only a tiny low-noise EDM Heun solve, for example (K=2) or (K=3) Heun steps from (\sigma_r) to 0.
* (\rho_c) is Pseudo-Huber or the same robust distance used in iCT/ECT-style training.

Full training loss:

[
\mathcal{L}
===========

\mathcal{L}*{\text{ECT}}
+
\lambda(t)\mathcal{L}*{\text{on-policy}},
]

with (\lambda(t)) ramped from 0 to 0.2–1.0 over the first 50–100 kimg of this refinement stage.

### Pseudocode

```python
# frozen:
#   edm: pretrained EDM denoiser
#   ema_ect: EMA copy of ECT checkpoint
# trainable:
#   ect: current ECT model

for real_x in dataloader:
    # Standard ECT loss on real data.
    loss_ect = ect_consistency_loss(ect, ema_ect, real_x)

    # On-policy second-step refinement loss.
    z = torch.randn_like(real_x)
    x_c = ema_ect(z * sigma_max, sigma_max).detach()

    sigma_r = sample_log_uniform(0.4, 1.5)   # centered around current 2-step sigma
    eps = torch.randn_like(x_c)
    y = x_c + sigma_r * eps

    with torch.no_grad():
        x_teacher = edm_heun_tail(
            edm,
            y,
            sigma_start=sigma_r,
            sigma_end=0.0,
            num_steps=2,
        )

    x_student = ect(y, sigma_r)
    loss_on_policy = pseudo_huber(x_student - x_teacher)

    loss = loss_ect + lambda_refine * loss_on_policy
    loss.backward()
    optimizer.step()
    update_ema(ect, ema_ect)
```

## Motivation from your data

Your 2-step ECT adds almost exactly one extra forward pass of latency but does not improve FID. At 1980 kimg, 1-step FID is 2.460 and 2-step FID is 2.510. That is not “multistep refinement”; it is an expensive identity-like postprocess.

This idea directly targets that failure. It asks:

> Does the second step fail because ECT has not learned low-noise correction on its own generated samples?

If yes, the exact same 2-step latency should become more valuable.

## What to implement vs reuse

**Reuse**

* Existing ECT `ct_train.py` loop.
* Existing EDM checkpoint.
* Existing CIFAR-10 data pipeline.
* Existing ECT and Heun samplers.
* Existing FID evaluation scripts.

**Implement from scratch**

* `on_policy_refinement_loss.py`, roughly 100–150 LOC.
* A low-noise EDM tail function that starts from arbitrary (y,\sigma_r), roughly 80–120 LOC if adapted from EDM’s sampler.
* A training flag such as `--on_policy_refine=1`.
* Logging for:

  * standard ECT loss,
  * on-policy loss,
  * 1-step FID,
  * 2-step FID,
  * “2-step gain” = FID1 − FID2.

No architecture change is required, but this is still a real training-loop/loss implementation.

## Expected outcome

**If it works:** ECT 2-step should move from FID 2.51 toward roughly 2.1–2.3 without changing inference latency. The ideal plot point is:

| Method                       | FID target | Batch-1 latency |
| ---------------------------- | ---------: | --------------: |
| ECT 2-step baseline          |       2.51 |        13.97 ms |
| On-policy refined ECT 2-step |    2.1–2.3 |        13.97 ms |
| Heun 18-step                 |       1.96 |       243.79 ms |

This would create a genuinely new Pareto point: not as good as Heun 18, but much closer in quality at ~17× lower latency.

**If it fails:** The result is still scientifically useful. It would imply the 2-step failure is not merely a train/inference distribution mismatch at low noise. The first-step ECT sample may already have errors that low-noise score correction cannot fix.

## Compute estimate

* Start from your 1000 or 1980 kimg checkpoint.
* Refinement: 250–500 kimg.
* Standard ECT cost on G4: 2.69 sec/kimg, so 500 kimg is ~22.4 minutes before extra teacher cost.
* With low-noise EDM teacher tail, expect ~2–4× overhead depending on (K), so roughly 45–90 minutes.
* FID50k evaluation: same as existing ECT eval.
* Total: one full experiment should fit in ~2 hours.

## Novelty argument

This is close to consistency distillation, but not the same as vanilla CD/ECD. ECT’s ECD variant uses a pretrained diffusion model as a teacher and also explores data-free synthetic training, but the paper does not specifically target **second-step failure caused by generated low-noise states**. ([arXiv][2]) SCT uses score-identity variance reduction and extends ECT to multistep/edge-skipping, but it is not framed as an on-policy repair of ECT’s dead second step. ([arXiv][5]) DMD2 explicitly discusses training/inference mismatch for multistep generators, but in a distribution-matching distillation setting, not a cheap ECT second-step refinement loss. ([arXiv][8])

This is not guaranteed publication-level novelty, but it is a strong course-project implementation: it changes the loss and training distribution in a way directly motivated by your anomaly.

## Main risks

* The EDM low-noise teacher may not correct ECT artifacts if the first-step sample is semantically wrong.
* The auxiliary loss may damage the 1-step model by overfitting low-noise behavior.
* Teacher tail targets may be biased toward EDM’s trajectory and reduce ECT’s diversity.
* If (\lambda) is too high, the model may forget the original consistency objective.

---

# Idea 2 — ECT-first, EDM-tail hybrid sampler

## Core proposal

Use ECT as a learned long-jump initializer, then use the original EDM model only for a tiny low-noise tail.

Instead of comparing:

[
\text{ECT: } \sigma_{\max} \rightarrow 0
]

against:

[
\text{Heun: } \sigma_{\max} \rightarrow \cdots \rightarrow 0,
]

implement:

[
\text{Hybrid: } \sigma_{\max}
\xrightarrow{\text{ECT one-step}}
0
\xrightarrow{\text{restart noise } \sigma_r}
\sigma_r
\xrightarrow{\text{EDM Heun tail}}
0.
]

Algorithm:

[
x_c = F_\theta(\sigma_{\max}z,\sigma_{\max}),
]

[
y = x_c + \sigma_r \epsilon,
]

[
x_{\text{out}} = \mathrm{HeunTail}_{\phi}(y; \sigma_r \rightarrow 0, K).
]

### Pseudocode

```python
def ect_edm_tail_sampler(
    ect,
    edm,
    batch_size,
    sigma_max=80.0,
    sigma_restart=0.821,
    tail_steps=2,
    rho=7,
):
    z = torch.randn(batch_size, 3, 32, 32, device="cuda")
    x = ect(z * sigma_max, sigma_max)

    # Restart at low noise.
    eps = torch.randn_like(x)
    x = x + sigma_restart * eps

    # Karras-style low-noise schedule from sigma_restart to 0.
    sigmas = karras_schedule(
        sigma_min=0.002,
        sigma_max=sigma_restart,
        num_steps=tail_steps,
        rho=rho,
    )
    sigmas = list(reversed(sigmas)) + [0.0]

    for sigma_cur, sigma_next in zip(sigmas[:-1], sigmas[1:]):
        x = edm_heun_step(edm, x, sigma_cur, sigma_next)

    return x
```

Run a small grid:

[
\sigma_r \in {0.25, 0.5, 0.821, 1.5, 2.5},
]

[
K \in {1,2,3,5}.
]

Evaluate FID10k first, then FID50k for the best 2–3 settings.

## Motivation from your data

Your ECT second step is not useful, but Heun’s later denoising path is. This replaces the ineffective second ECT call with a score-based low-noise corrector from the original EDM.

The reason this is promising is that ECT already does the expensive high-noise global transport in one call. Heun’s 18-step sampler spends 35 NFEs solving the whole path. The hybrid asks whether only the **last few low-noise EDM evaluations** are enough to recover much of the FID gap.

## What to implement vs reuse

**Reuse**

* ECT checkpoint.
* EDM checkpoint.
* EDM denoiser and preconditioning.
* CIFAR FID pipeline.

**Implement from scratch**

* `hybrid_sampler.py`, roughly 150–250 LOC.
* A sampler that can load both ECT and EDM simultaneously.
* A modified EDM Heun function that accepts arbitrary initial (x,\sigma), rather than always starting at (\sigma_{\max}z).
* Grid evaluation script for ((\sigma_r,K)).
* Latency measurement for hybrid configs.

This satisfies the TA’s “change the sampler” requirement without needing risky training.

## Expected outcome

Approximate latency, using your measured per-network-call timing:

| Hybrid config         | Approx NFEs | Batch-1 latency guess | Expected FID hypothesis |
| --------------------- | ----------: | --------------------: | ----------------------: |
| ECT only              |           1 |                7.0 ms |                    2.46 |
| ECT + 1-step EDM tail |        ~2–3 |              14–21 ms |                 2.3–2.5 |
| ECT + 2-step EDM tail |          ~4 |                 28 ms |                2.1–2.35 |
| ECT + 3-step EDM tail |          ~6 |                 42 ms |                2.0–2.25 |
| Heun 5-step           |           9 |                 62 ms |                   37.78 |
| Heun 10-step          |          19 |                132 ms |                    2.64 |
| Heun 18-step          |          35 |                244 ms |                    1.96 |

The most valuable success case is **FID near Heun 10 or better at below Heun 5 latency**. That would be a very clean Pareto plot.

## Compute estimate

* No training.
* FID10k sweep over 20 configs: perhaps 1–2 GPU hours depending on cleanup and FID overhead.
* FID50k for best 3 configs: likely 30–60 minutes.
* Latency grid: <10 minutes.

## Novelty argument

Restart Sampling already alternates forward noising and reverse ODE solving, but it does so within the diffusion model’s own sampling process. ([arXiv][9]) This proposal uses a consistency model as a learned one-step global transport map, then uses the original diffusion model only as a local score corrector. Multistep CMs and SCT focus on consistency-model-only multistep inference; this explicitly mixes two models with different roles. ([arXiv][4])

This is probably not a publishable algorithm by itself, but it is a strong project implementation because it creates a new sampler and directly tests whether ECT’s value is as a **coarse solver** rather than a full replacement for diffusion sampling.

## Main risks

* Adding low noise to an ECT sample may not put it on a valid EDM reverse trajectory.
* Low-noise EDM steps may fix texture but not semantic/class-level mistakes.
* Hybrid uses two checkpoints, which complicates the deployment story.
* If the best FID is still around 2.5, the result becomes “ECT is not a useful initializer for EDM tail correction.”

---

# Idea 3 — Distill the Heun/Hybrid tail into a tiny residual corrector

## Core proposal

Train a small post-ECT residual network that approximates the missing quality correction:

[
x_c = F_\theta(\sigma_{\max}z,\sigma_{\max}),
]

[
x_T = \text{teacher output},
]

[
R_\psi(x_c,z) \approx x_T - x_c,
]

[
x_{\text{out}} = x_c + \alpha R_\psi(x_c,z).
]

The teacher can be either:

1. **Full Heun 18-step** from the same initial noise (z), or
2. **The best hybrid sampler from Idea 2**, which is more aligned with (x_c).

I would use the hybrid teacher first, because full Heun 18 and ECT one-step samples from the same seed may not be pixel-aligned enough for residual regression. The hybrid teacher starts from (x_c), so the target residual is more local.

### Pseudocode

```python
# Offline data generation.
for seed_batch in seeds:
    z = torch.randn(batch, 3, 32, 32, device="cuda")

    with torch.no_grad():
        x_c = ect(z * sigma_max, sigma_max)
        x_T = ect_edm_tail_sampler(
            ect=ect,
            edm=edm,
            z=z,
            sigma_restart=best_sigma_r,
            tail_steps=best_K,
        )

    save_pair(z.cpu(), x_c.cpu(), x_T.cpu())

# Train tiny residual corrector.
for z, x_c, x_T in pair_loader:
    inp = torch.cat([x_c, z / z.std()], dim=1)
    delta_target = x_T - x_c

    delta_pred = R_psi(inp)
    loss = pseudo_huber(delta_pred - delta_target)

    # Optional: keep correction small.
    loss += gamma * (delta_pred ** 2).mean()

    loss.backward()
    opt.step()

# Inference.
x_c = ect(z * sigma_max, sigma_max)
x = x_c + alpha * R_psi(torch.cat([x_c, z_norm], dim=1))
```

Architecture:

```text
Input: 6×32×32  [ECT sample, original latent noise]
Conv 3×3, 64
GroupNorm + SiLU
Residual block × 4
Conv 3×3, 64
SiLU
Conv 3×3, 3
Output: residual image correction
```

This is deliberately tiny: (<1)M parameters is enough for CIFAR-10.

## Motivation from your data

Heun 18-step improves FID from ECT’s 2.51-ish range to 1.96, but the latency jump is enormous. Your data suggests the missing quality is not worth 35 NFEs for many deployment regimes. The residual corrector tries to compress the difference into a cheap learned module.

This also addresses the diminishing-return observation. Rather than spending another 19 hours approximating the full ECT paper schedule, train a small correction module for exactly the quality gap you measured.

## What to implement vs reuse

**Reuse**

* Existing ECT model.
* Existing EDM/Heun sampler or hybrid sampler.
* Existing FID pipeline.

**Implement from scratch**

* Synthetic pair dataset generator: 100–150 LOC.
* `ResidualCorrector` module: 80–120 LOC.
* Training loop: 150–250 LOC.
* Inference wrapper: 50 LOC.
* Evaluation script comparing:

  * ECT 1-step,
  * ECT 2-step,
  * ECT + residual corrector,
  * hybrid sampler,
  * Heun 10/18.

## Expected outcome

**If it works:** The best case is a point near ECT latency but with hybrid-like FID:

| Method                   |    Latency | FID hypothesis |
| ------------------------ | ---------: | -------------: |
| ECT 1-step               |     7.0 ms |           2.46 |
| ECT + residual corrector | 7.5–9.0 ms |        2.2–2.4 |
| ECT + EDM tail           |   20–40 ms |        2.0–2.3 |
| Heun 18                  |     244 ms |           1.96 |

This is the most deployment-friendly idea: one ECT call plus one tiny network.

**If it fails:** The likely failure is regression blur. If (x_T) and (x_c) are not pixel-aligned, the residual network learns an average correction that improves MSE but worsens FID.

## Compute estimate

* Generate 50k teacher pairs:

  * If teacher is Heun 18: about 17 minutes using your batch-64 latency.
  * If teacher is hybrid tail: likely less.
* Train residual corrector: 20–60 minutes.
* FID50k eval: same as current pipeline.
* Total: ~1–2 hours.

## Novelty argument

Progressive Distillation distills a deterministic diffusion sampler into a faster diffusion model by repeatedly halving sampling steps. ([arXiv][10]) DMD/DMD2, SiD, and EM Distillation train one-step generators or distribution-matching students, often with substantially more machinery. ([arXiv][11])

This proposal is narrower: freeze the ECT generator and train only a tiny post-hoc residual corrector to approximate the measured Heun/hybrid quality gap. That is not the standard diffusion-distillation setup.

## Main risks

* The corrector may reduce diversity by pulling samples toward teacher-average residuals.
* Pixel-space residual loss may not correlate with FID.
* If teacher and ECT samples are not aligned, the task is ill-posed.
* A tiny model may only fix color/texture, not structural errors.

---

# Idea 4 — Two-target segmented ECT: make the first step stop trying to be the final step

## Core proposal

Your current ECT model learns:

[
F_\theta(x_t,t) \rightarrow x_0.
]

So in 2-step sampling, the first call already tries to produce (x_0). The second call receives a nearly clean image plus small noise and mostly repeats the projection. That can explain why 1-step and 2-step FID are nearly identical.

Instead, train a two-target consistency model:

[
C_\theta(x_t,t,s) \rightarrow x_s,
]

but only for two target times:

[
s \in {\tau,0}.
]

High-noise segment:

[
C_\theta(x_t,t,\tau) \rightarrow x_\tau,
\quad t>\tau.
]

Low-noise segment:

[
C_\theta(x_t,t,0) \rightarrow x_0,
\quad t\le \tau.
]

For real CIFAR sample (x_0) and shared noise (\epsilon):

[
x_t = x_0 + t\epsilon,
]

[
x_\tau = x_0 + \tau \epsilon.
]

Loss:

[
\mathcal{L}_{\text{high}}
=========================

\rho\left(C_\theta(x_t,t,\tau)-\operatorname{sg}(x_\tau)\right),
]

[
\mathcal{L}_{\text{low}}
========================

\rho\left(C_\theta(x_t,t,0)-x_0\right),
]

[
\mathcal{L}
===========

\mathcal{L}*{\text{high}}
+
\mathcal{L}*{\text{low}}
+
\lambda \mathcal{L}_{\text{ECT-consistency}}.
]

Inference:

[
y_\tau = C_\theta(\sigma_{\max}z,\sigma_{\max},\tau),
]

[
\hat{x}*0 = C*\theta(y_\tau,\tau,0).
]

The first network call is no longer asked to solve the whole problem. It learns a **trajectory-preserving jump** to an intermediate noise level. The second step then has a meaningful denoising task.

## Implementation variants

### Variant A: target-time conditioning

Add a small embedding for target time (s) to the existing SongUNet time embedding:

[
e = e_t + e_s.
]

This is closest to CTM/Shortcut-style conditioning but restricted to two targets.

### Variant B: two output heads

Keep the same trunk, but add two final heads:

```text
head_tau: predicts x_tau
head_0:   predicts x_0
```

This is simpler to implement and easier to debug.

## Motivation from your data

This directly attacks the observation:

> ECT 2-step has no quality gain because the model’s first call already collapses the trajectory to the endpoint.

If the first step instead maps to an intermediate state, the second step should become necessary and measurable. Your current “second step adds no quality” becomes the central hypothesis.

## What to implement vs reuse

**Reuse**

* ECT checkpoint initialization.
* SongUNet trunk.
* CIFAR data loader.
* EMA and optimizer setup.
* FID pipeline.

**Implement from scratch**

* Modified preconditioning wrapper supporting `target_sigma`.
* Either:

  * target-time embedding, or
  * two output heads.
* New training loss that samples (t) above/below (\tau).
* New sampler for:

  * one-step direct endpoint mode,
  * two-step segmented mode.

Expected code size: 300–500 LOC, depending on how invasive the architecture modification is.

## Expected outcome

**If it works:** 2-step FID separates from 1-step FID. For example:

| Method                        | Expected behavior          |
| ----------------------------- | -------------------------- |
| Standard ECT 1-step           | FID ~2.46                  |
| Standard ECT 2-step           | FID ~2.51                  |
| Segmented ECT 1-step endpoint | maybe worse or unavailable |
| Segmented ECT 2-step          | possibly 2.0–2.3           |

The project story would be strong:

> Standard ECT makes two-step sampling redundant under a short tuning budget; a trajectory-preserving two-target objective restores a real speed/quality tradeoff.

## Compute estimate

* Training from your 1980 kimg ECT checkpoint: 500–1500 kimg.
* G4 base time: 22–67 minutes.
* With modified loss and two heads: likely <90 minutes.
* Debugging architecture/preconditioning: nontrivial, maybe several days.
* FID50k per final checkpoint: same as current.

## Novelty argument

This is the closest to existing literature and needs careful framing. Multistep CMs already split the path into segments. ([arXiv][4]) CTM learns arbitrary traversal between initial and final times and reports strong one-step CIFAR-10 FID. ([arXiv][7]) Shortcut Models condition on step size to allow one-step or multistep sampling from a single network. ([arXiv][12])

The defensible novelty is not “segmented consistency models exist.” They do. The project-level novelty is:

> A minimal two-target ECT modification, initialized from the same EDM/ECT checkpoint, designed specifically to test why cheap ECT’s second step is useless under a 2000-kimg budget.

This is scientifically interesting, but the literature-overlap risk is high.

## Main risks

* The TA may view it as too close to Multistep CMs/CTM unless you frame it carefully.
* Predicting (x_\tau) from very high noise may be harder than expected.
* The two-head model may split capacity and hurt both endpoints.
* It may require more tuning than your 5-week budget allows.

---

# Idea 5 — Consistency-residual adaptive compute sampler

## Core proposal

Do not spend extra compute on every sample. Spend it only when ECT looks unreliable.

Define a cheap self-consistency uncertainty score. Generate the one-step ECT sample:

[
x_c = F_\theta(\sigma_{\max}z,\sigma_{\max}).
]

Probe its low-noise stability:

[
r(z)
====

\frac{1}{K}
\sum_{k=1}^{K}
\left|
F_\theta(x_c+\sigma_r \epsilon_k,\sigma_r)-x_c
\right|_2^2.
]

Large (r(z)) means the ECT model is not stable under the same low-noise perturbation used by its 2-step sampler.

Then route samples:

[
x_{\text{out}} =
\begin{cases}
x_c, & r(z) < q_p, \
\mathrm{EDMTail}(x_c+\sigma_r\epsilon,\sigma_r), & r(z)\ge q_p,
\end{cases}
]

where (q_p) is a threshold that sends only the top (p%) most uncertain samples to the expensive corrector.

### Pseudocode

```python
def adaptive_ect_sampler(ect, edm, z, p=0.25, sigma_r=0.821):
    x_c = ect(z * sigma_max, sigma_max)

    # Self-consistency probe.
    eps_probe = torch.randn_like(x_c)
    x_probe = ect(x_c + sigma_r * eps_probe, sigma_r)
    r = ((x_probe - x_c) ** 2).mean(dim=(1, 2, 3))

    # Batch-level threshold.
    thresh = torch.quantile(r, 1.0 - p)
    mask = r >= thresh

    x_out = x_c.clone()

    if mask.any():
        eps_tail = torch.randn_like(x_c[mask])
        y = x_c[mask] + sigma_r * eps_tail
        x_out[mask] = edm_heun_tail(edm, y, sigma_r, 0.0, num_steps=2)

    return x_out
```

## Motivation from your data

ECT 2-step doubles latency but gives no FID gain. That suggests applying the same extra computation to every sample is wasteful. Your break-even analysis also shows that fixed-cost ECT tuning and fixed-cost ECT 2-step inference are hard to justify at batch 64.

Adaptive compute gives a new curve:

[
p = 0, 0.1, 0.25, 0.5, 1.0
]

where (p=0) is ECT 1-step and (p=1) is the full hybrid tail from Idea 2. The interesting question is whether FID improves sharply for small (p).

## What to implement vs reuse

**Reuse**

* ECT checkpoint.
* EDM checkpoint.
* Hybrid tail code from Idea 2.
* FID and latency scripts.

**Implement from scratch**

* Consistency-residual scoring function.
* Batch-wise routing sampler.
* Threshold calibration over a held-out seed set.
* Logging:

  * average routed fraction,
  * latency,
  * FID,
  * residual distribution.

Expected code size: 150–250 LOC if built on Idea 2.

## Expected outcome

Best-case result:

| Routed fraction |      Latency hypothesis | FID hypothesis |
| --------------: | ----------------------: | -------------: |
|              0% |                    7 ms |           2.46 |
|             10% |                15–18 ms |           2.35 |
|             25% |                19–25 ms |        2.2–2.3 |
|             50% |                30–40 ms |       2.1–2.25 |
|            100% | 28–50 ms depending tail |        2.0–2.3 |

The best result would be a curve that beats fixed ECT 2-step and approaches hybrid quality at lower average latency.

## Compute estimate

* No training.
* Calibration: generate 10k samples with residual scores and maybe FID10k.
* Final FID50k for 3–5 routing fractions.
* Total: 1–2 GPU hours.

## Novelty argument

Adaptive timestep and solver-selection work exists, including differentiable solver search and adaptive discretization methods. ([arXiv][13]) But this is a different adaptive axis: per-sample routing between a consistency projector and a diffusion tail, using the consistency model’s own low-noise instability as the routing statistic.

This is more novel as a **diagnostic sampler** than as a guaranteed performance win.

## Main risks

* FID is distribution-level; a per-sample residual may not correlate with contribution to FID.
* The consistency residual may identify unusual but valid samples, not bad samples.
* The probe itself costs an extra ECT call, which hurts batch-1 latency.
* If routing requires an EDM call for scoring, the method becomes self-defeating.

---

# Idea 6 — Learn a tiny low-NFE solver after the ECT jump

## Core proposal

Instead of hand-designing the EDM tail, learn the tail solver coefficients and timesteps while keeping both ECT and EDM frozen.

Let:

[
x_0 = F_\theta(\sigma_{\max}z,\sigma_{\max}) + \sigma_0\epsilon.
]

Define the EDM ODE direction:

[
d_\phi(x,\sigma)
================

\frac{x-D_\phi(x,\sigma)}{\sigma}.
]

Use a small multistep solver:

[
x_{i+1}
=======

x_i
+
\Delta\sigma_i
\left(
a_i d_i + b_i d_{i-1}
\right),
]

where (a_i,b_i,\sigma_i) are learned scalar parameters.

Teacher target:

[
x_T = \mathrm{Heun18}*{\phi}(\sigma*{\max}z).
]

Loss:

[
\mathcal{L}_{\text{solver}}
===========================

\left|
x_m - \operatorname{sg}(x_T)
\right|_2^2
+
\eta \sum_i (a_i-a_i^{\text{Heun}})^2.
]

The learned parameters are tiny: maybe 10–30 scalars total.

### Pseudocode

```python
# Freeze ECT and EDM.
params = {
    "raw_sigmas": torch.nn.Parameter(init_sigmas),
    "a": torch.nn.Parameter(init_a),
    "b": torch.nn.Parameter(init_b),
}

for z in train_seed_loader:
    with torch.no_grad():
        x_teacher = edm_heun_full(edm, z, steps=18)
        x_init = ect(z * sigma_max, sigma_max)

    sigmas = monotone_sigmas(params["raw_sigmas"])

    x = x_init + sigmas[0] * torch.randn_like(x_init)
    d_prev = None

    for i in range(num_tail_steps):
        d = edm_direction(edm, x, sigmas[i])
        if d_prev is None:
            update_dir = params["a"][i] * d
        else:
            update_dir = params["a"][i] * d + params["b"][i] * d_prev

        x = x + (sigmas[i+1] - sigmas[i]) * update_dir
        d_prev = d

    loss = ((x - x_teacher) ** 2).mean()
    loss += regularize_coefficients(params)
    loss.backward()
    opt.step()
```

## Motivation from your data

Your Heun curve is not monotone: Heun 18 gives FID 1.960, but Heun 25 and 50 are slightly worse. That suggests the schedule/solver matters more than raw step count. Heun 10 is already close to ECT in FID but much slower; a custom low-NFE tail could create a useful middle point.

## What to implement vs reuse

**Reuse**

* ECT and EDM checkpoints.
* Existing Heun code for teacher generation.
* FID pipeline.

**Implement from scratch**

* Differentiable solver parameterization.
* Monotone sigma schedule parameterization.
* Teacher-pair loader.
* Optimization loop over scalar solver parameters.
* Final sampler using learned coefficients.

Expected code size: 250–400 LOC.

## Expected outcome

Best case:

| Method             | Approx latency | FID hypothesis |
| ------------------ | -------------: | -------------: |
| ECT 1-step         |           7 ms |           2.46 |
| Learned 2-NFE tail |       20–30 ms |       2.2–2.35 |
| Learned 4-NFE tail |       35–55 ms |       2.0–2.25 |
| Heun 10            |         132 ms |           2.64 |
| Heun 18            |         244 ms |           1.96 |

The most impressive result would be a custom 3–5 NFE tail that beats Heun 10 FID at less than half Heun 5 latency.

## Compute estimate

* Generate 5k–20k teacher pairs with Heun 18: 2–7 minutes for 20k using your batch-64 Heun 18 latency, plus overhead.
* Optimize scalar solver: <30 minutes.
* FID50k for final sampler: 10–30 minutes depending tail.
* Total: under 2 hours.

## Novelty argument

Differentiable Solver Search already learns solver coefficients and timesteps for fast diffusion sampling. ([arXiv][13]) The novelty here is narrower: the solver is not starting from Gaussian noise. It is a learned **post-ECT low-noise solver**, where ECT handles the long jump and the learned solver handles the residual low-noise dynamics.

This is a good implementation idea, but the novelty claim is weaker than Idea 1 or Idea 3 because the solver-search literature is close.

## Main risks

* MSE to Heun 18 may not optimize FID.
* Backpropagating through frozen EDM calls may be memory-heavy if not carefully batched.
* Learned coefficients may overfit the training seed set.
* It may perform no better than the hand-designed hybrid sampler.

---

# Final ranking: strongest project to weakest

## 1. **On-policy low-noise ECT refinement loss**

This is the best overall project. It is a real training-loop/loss change, it directly explains your most surprising measurement, and it keeps the final inference cost identical to ECT 2-step. Even a negative result is meaningful because it tests whether the dead second step is caused by train/inference mismatch.

**Why it will satisfy the TA:** new loss, new training distribution, new teacher-tail code, direct connection to ECT mechanics.

---

## 2. **ECT-first, EDM-tail hybrid sampler**

This is the safest implementation win. It is easy to build, easy to explain, and likely to produce a new FID-latency point. It directly responds to “when does cheap tuning pay off?” by treating ECT as a coarse solver rather than a full sampler.

**Weakness:** no training. It is sampler-level, which the TA explicitly allowed, but it may feel less “deep” than Idea 1 unless you analyze the ODE mechanics carefully.

---

## 3. **Residual corrector distilled from Heun/hybrid tail**

This is strong if you want an architecture component. It could produce the most practically attractive result: near-ECT latency with better FID. It is also very feasible on CIFAR-10.

**Main risk:** paired residual regression may blur or fail if teacher and student samples are not aligned.

---

## 4. **Learned low-NFE post-ECT solver**

This is elegant and compute-light. It gives a clean story about learning a custom sampler under a fixed NFE budget. But differentiable solver search is already a known direction, so the novelty must be framed as “heterogeneous ECT-initialized solver,” not generic solver search.

---

## 5. **Consistency-residual adaptive compute sampler**

This is the most “systems meets generative modeling” idea. It might expose a useful hidden structure: some ECT samples may need correction and others may not. But FID may not reward per-sample routing, and the uncertainty score may not correlate with sample quality.

---

## 6. **Two-target segmented ECT**

This is the most conceptually direct attack on the 1-step≈2-step anomaly, but it is also closest to Multistep CMs, CTM, and Shortcut Models. It could be the most research-interesting if executed well, but it has the highest literature-overlap and implementation risk.

---

# Most interesting research idea, regardless of risk

**Two-target segmented ECT** is the most interesting scientifically.

It asks whether cheap ECT collapses the probability-flow trajectory too aggressively. Standard ECT says: “map every point to the endpoint.” Your data suggests that, under a short budget, this produces a one-step projector whose second step cannot improve anything. A two-target model asks: “What if the first step is trained not to finish the job?”

That is a deeper question than “can we get a better FID?” It probes the representation learned by ECT.

---

# Best idea for a strong final report

Use a two-stage plan:

## Main implementation

**Idea 1: On-policy low-noise ECT refinement.**

This is the best match to the TA’s request: new training objective, new loop, grounded in your data.

## Backup / parallel implementation

**Idea 2: ECT-first, EDM-tail hybrid sampler.**

This gives you a guaranteed new sampler and likely useful Pareto points even if training is unstable.

These two also combine naturally:

1. First implement the hybrid sampler.
2. Use its low-noise tail as the teacher for on-policy ECT refinement.
3. Final report compares:

   * ECT 1-step,
   * ECT 2-step,
   * ECT + EDM tail,
   * on-policy refined ECT 2-step,
   * Heun 10/18.

That gives a coherent story rather than six disconnected experiments.

---

# The likely blind spot

The blind spot is assuming ECT is “a faster sampler.” Your measurements suggest something more specific:

> **Your short-budget ECT checkpoint is a fast learned projection, not a useful iterative solver.**

That distinction matters. If ECT is a projector, then adding more ECT steps will not help. The right extension is not “try 3-step ECT” or “search a better ECT step count.” The right extension is to either:

1. **train the second step on the states it actually sees**, or
2. **replace the second step with a different local corrector**, or
3. **change the objective so the first step preserves an intermediate trajectory state instead of collapsing to the endpoint.**

That is the proposal theme I would take to the TA.

[1]: https://arxiv.org/abs/2206.00927?utm_source=chatgpt.com "DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps"
[2]: https://arxiv.org/html/2406.14548v2 "Consistency Models Made Easy"
[3]: https://arxiv.org/abs/2206.00364?utm_source=chatgpt.com "Elucidating the Design Space of Diffusion-Based Generative Models"
[4]: https://arxiv.org/html/2403.06807v2 "Multistep Consistency Models"
[5]: https://arxiv.org/html/2410.18958v2 "Stable Consistency Tuning: Understanding and Improving Consistency Models"
[6]: https://arxiv.org/html/2510.17266v1 "Adaptive Discretization for Consistency Models"
[7]: https://arxiv.org/abs/2310.02279?utm_source=chatgpt.com "Consistency Trajectory Models: Learning Probability Flow ODE Trajectory of Diffusion"
[8]: https://arxiv.org/abs/2405.14867?utm_source=chatgpt.com "Improved Distribution Matching Distillation for Fast Image Synthesis"
[9]: https://arxiv.org/abs/2306.14878?utm_source=chatgpt.com "Restart Sampling for Improving Generative Processes"
[10]: https://arxiv.org/abs/2202.00512?utm_source=chatgpt.com "Progressive Distillation for Fast Sampling of Diffusion Models"
[11]: https://arxiv.org/abs/2311.18828?utm_source=chatgpt.com "One-step Diffusion with Distribution Matching Distillation"
[12]: https://arxiv.org/abs/2410.12557?utm_source=chatgpt.com "One Step Diffusion via Shortcut Models"
[13]: https://arxiv.org/abs/2505.21114?utm_source=chatgpt.com "Differentiable Solver Search for Fast Diffusion Sampling"
