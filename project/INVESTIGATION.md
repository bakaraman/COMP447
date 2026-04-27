# Investigation: Finding the Real Bottleneck

Purpose: follow the "boil the ocean, no guesswork" principle. Before proposing a project, verify what is broken, what is measured correctly, and where the genuine open problem lies. This document is the research log.

## TL;DR

1. **Upstream bug confirmed**: `locuslab/ect` `ct_eval.py` evaluates **2-step FID** for every metric name — `fid50k_full` and `two_step_fid50k_full` both come out identical. Bug is in upstream `main` (commit `4311059`), not in our vendor copy.
2. **Our "dead second step" anomaly was an artifact.** The 1-step ≈ 2-step results (2.46 vs 2.51) came from this bug. Our Cell 9 end-of-training numbers (1-step 5.77, 2-step 2.47) were run through the training loop (`ct_training_loop.py`), which is correct. The paper's Table 1 (1-step 3.60, 2-step 2.11 at 400k iters) is consistent with our correct training-loop numbers.
3. **Real open problem**: the ECT author himself (issue #11) calls **optimal intermediate-timestep selection for multi-step CM sampling** an open research problem. `mid_t = 0.821` is a hardcoded magic number with no justification in the paper, in `sCM`, or in `Multistep Consistency Models`. `Align Your Steps` (NVIDIA, 2024) solves this for diffusion models but is not adapted to consistency models.
4. **Our concrete proposal**: corrected evaluation pipeline + principled schedule search + on-policy schedule-aware fine-tuning. Minimal credible experiment runs on our existing four checkpoints (no extra training), which is the "find evidence before changing the objective" step.

---

## 1. The `ct_eval.py` bug

### 1.1 What we expected

Two distinct metrics:
- `fid50k_full` → 1-step sampling (`mid_t=None`, so `t_steps=[80,0]`)
- `two_step_fid50k_full` → 2-step sampling (`mid_t=[0.821]`, so `t_steps=[80, 0.821, 0]`)

### 1.2 What actually happens

In [upstream ct_eval.py](https://github.com/locuslab/ect/blob/main/ct_eval.py) (our vendor is identical, see commit `4311059`):

```python
# Lines 363-380
few_step_fn = functools.partial(generator_fn, mid_t=mid_t)   # <-- only this variant built
# ...
for metric in metrics:
    result_dict = metric_main.calc_metric(metric=metric,
        generator_fn=few_step_fn, G=net, G_kwargs={}, ...)   # <-- SAME fn for every metric
```

And [metric_main.py lines 82-92](https://github.com/locuslab/ect/blob/main/metrics/metric_main.py):

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

Both metric functions are **bit-identical**. Neither inspects its name to decide the sampler. The sampler is fixed by the caller, and the caller only ever builds `few_step_fn` with `mid_t=[0.821]`. Consequence: any FID reported by `ct_eval.py` is 2-step, regardless of the metric name written into the JSONL.

### 1.3 Where training-time eval is correct

[`ct_training_loop.py` lines 352-363](project/src/ect/training/ct_training_loop.py#L352-L363) builds two separate `generator_fn` callables — bare `generator_fn` for `fid50k_full` (1-step, `mid_t=None`) and `few_step_fn = partial(generator_fn, mid_t=mid_t)` for `two_step_fid50k_full` (2-step). So end-of-training eval inside a tuning run is fine. This is the path used by the paper's Table 1. The post-hoc path (`eval_ecm.sh` → `ct_eval.py`) is the one that silently collapses to 2-step only.

### 1.4 Why we hit it

We ran a clean per-checkpoint ablation via `ct_eval.py` in Cell 10 / Cell 15 of the Colab notebook. Both `fid50k_full` and `two_step_fid50k_full` came back as 2-step FID, so the "ECT 1-step ≈ ECT 2-step" pattern across 500/1000/1500/1980 kimg is a tautology, not a finding.

### 1.5 Sanity check with the paper

Table 1 of [Consistency Models Made Easy (ICLR 2025)](https://arxiv.org/abs/2406.14548):

| Iterations | 1-step FID | 2-step FID |
|---:|---:|---:|
| 100k | 4.54 | 2.20 |
| 200k | 3.86 | 2.15 |
| 400k | 3.60 | 2.11 |

Our Cell 9 end-of-training eval at 1980 kimg (≈15.5k iters, much less than the paper's 100k): 1-step 5.77, 2-step 2.47. Consistent shape — 1-step quality is substantially worse than 2-step quality, and both improve with more training. The "flat 1-step vs 2-step" story only appears when we route through `ct_eval.py`.

### 1.6 Upstream issue trail

- [Issue #19](https://github.com/locuslab/ect/issues/19) "how to sample the provided checkpoint?" — user yuanzhi-zhu reports `eval_ecm.sh` gives garbled output. No author response since 2025-01-01. Consistent with evaluation-pipeline problems being under-tested upstream.
- [Issue #11](https://github.com/locuslab/ect/issues/11) "How to control the sampling steps through intermediate t?" — the ECT author (Zhengyang Geng, `@Gsunshine`) answers verbatim: *"It's a very good question and I think also an open research problem. I don't have a manual for it. ... I encourage you to explore more!"* This is a direct admission that optimal intermediate-timestep selection is an acknowledged open problem.
- No existing issue or PR flags the `ct_eval.py` metric-name bug. We are the first to document it.

---

## 2. Literature gap: schedule selection for consistency models

### 2.1 What exists

- [Align Your Steps (Sabour et al., NVIDIA, 2024)](https://arxiv.org/abs/2404.14507): principled schedule optimization by minimizing a KL-divergence upper bound (KLUB) between the true generative SDE and its linearization at chosen timesteps. Designed for DDIM / DPM-Solver / SDE-DPM-Solver; ships plug-and-play schedules for Stable Diffusion. Does **not** target consistency models.
- [Optimal Stepsize for Diffusion Sampling (2025)](https://arxiv.org/abs/2503.21774): Wasserstein-bounded adaptive timestep selection for diffusion samplers. Again diffusion-specific.
- [Multistep Consistency Models (Heek et al., 2024)](https://arxiv.org/abs/2403.06807): introduces a family that interpolates between 1-step CM and full diffusion. Trains the model for a chosen number of steps; still bakes the schedule into training rather than optimizing it post-hoc, and does not publish a search-over-schedules study.
- [sCM — Simplifying, Stabilizing & Scaling Continuous-Time CMs (Karras et al., 2024)](https://arxiv.org/abs/2410.11081): reports 2-step FID 2.06 on CIFAR-10 but does not explain how the 2-step schedule is chosen.
- [How to build a consistency model via self-distillation (2025)](https://arxiv.org/abs/2505.18825): alternative training algorithm, schedule not the focus.

### 2.2 The gap

No paper does for consistency models what AYS does for diffusion models. On CIFAR-10 at 2 steps, the ECT / sCM line uses a single hand-picked `mid_t`, and neither the magnitude nor the training dependence of the FID gap between that choice and an optimized one is characterized.

### 2.3 Why this is an implementation-worthy question

- Adapting KLUB to the CM setting is non-trivial: CMs are not solving an SDE, they are approximating a flow map. The AYS derivation has to be re-done with the flow-map generator as the object being linearized.
- The naive alternative — black-box evolutionary search of the schedule (mentioned in older CM literature) — is expensive per FID and does not generalize across checkpoints.
- A learned, cheap schedule selector conditioned on the checkpoint would be a new artifact. This is real implementation work, not a spreadsheet of FID numbers.

---

## 3. Second real bottleneck: exposure bias in CM multistep sampling

### 3.1 Where it comes from

[ECMLoss in training/loss.py](project/src/ect/training/loss.py) takes a real image `y`, adds noise `eps*t`, and asks the network to be self-consistent between `(y+eps*t, t)` and `(y+eps*r, r)` for two correlated noise levels. The model is **never** trained on its own outputs during the ECT tuning loop.

At inference time, the 2-step sampler feeds the model:
1. First call: `latent * 80` at `t=80`. Output: `x_mid = model(latent*80, 80)`.
2. Second call: `x_mid + eps * mid_t` at `t=mid_t`. Output: `x_final = model(x_mid + eps*mid_t, mid_t)`.

The second input is **not drawn from the training marginal** `{y_real + eps*mid_t : y_real ~ data}`. It is drawn from `{x_mid + eps*mid_t : x_mid ~ model}`. The usual train-test distribution shift / exposure-bias problem.

### 3.2 Why this compounds with schedule selection

The exposure-bias gap itself depends on `mid_t`:
- Large `mid_t` (close to `t_max`): second step sees mostly noise, so `x_mid` quality barely matters, exposure bias shrinks.
- Small `mid_t` (close to 0): second step sees mostly signal, so any defect in `x_mid` is visible to the model — and since it never trained on these defects, it cannot fix them.

So the optimal `mid_t` is jointly determined by "how much refinement headroom does a second model call give me" and "how far from the training distribution does the input of that second call land." Neither term is static across training — both depend on checkpoint quality. This is exactly the kind of problem that benefits from a learned or checkpoint-conditioned schedule.

### 3.3 Why this is an established research direction

The broader generative-model literature on exposure bias in distillation is very active — e.g. [GKD / on-policy distillation](https://arxiv.org/abs/2306.13649), [Autoregressive Distillation of Diffusion Transformers](https://arxiv.org/abs/2504.11295), and the 2025 survey on on-policy self-distillation. None of them have been applied to CMs' multistep schedule mismatch.

---

## 4. What we will do about it

### 4.1 Fix the measurement pipeline (one-time, mandatory)

- Patch `ct_eval.py` to build *two* generator functions and route each metric to the right one — conditioned on the metric name.
- Patch `metric_main.py` so `two_step_fid50k_full` returns its key under `two_step_fid50k_full=fid`, not `fid50k_full=fid`, so the JSONL becomes unambiguous.
- Re-run the full ablation on our four saved checkpoints (`ect_checkpoints/network-snapshot-{000050,000100,000150,000198}.pkl`) with the corrected script. This produces the first clean 1-step-vs-2-step trajectory for ECT under our tuning budget.

This alone is a publishable methodological contribution (bug report + fix + corrected curve) and it's prerequisite for everything else.

### 4.2 Experiment A: does 0.821 hold up?

- For each of the four checkpoints, sweep `mid_t` ∈ {0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.821, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0}, where `0.0` = 1-step.
- Use 10k samples per (checkpoint, mid_t) pair for fast screening; re-run the three best `mid_t` per checkpoint at 50k for publication-grade FID.
- Deliverables:
  - Curve of `FID(mid_t | checkpoint)` for each checkpoint.
  - Table: `mid_t*(checkpoint)` — the optimum per checkpoint — and its FID vs. the fixed-0.821 baseline.
  - Plot: `mid_t*` vs training budget (does it drift as the model gets stronger?).
- Cost estimate: 4 checkpoints × 13 schedules × 10k samples ≈ 520k samples end-to-end. Each ECT sample is ~7 ms at batch 1; at batch 64 we're in the single-GPU-hour range on G4.

Expected outcome space:
- **If `mid_t*` is roughly constant across checkpoints and close to 0.821**: the hand-picked default is fine, schedule optimization is not the bottleneck, we move focus to exposure bias only.
- **If `mid_t*` drifts with checkpoint quality**: schedule selection is a real, measurable lever. Justifies a schedule predictor.
- **If `mid_t*` varies but the 2-step–vs–1-step gap is tiny at the best `mid_t*` anyway**: the bottleneck is fundamentally in the first-step quality, and adding more steps will not help. This would redirect us to a different project (e.g. curriculum/teacher-schedule work on the first step).

All three outcomes give us a concrete story.

### 4.3 Experiment B: adapt AYS to CM sampling

Only if Experiment A shows `mid_t*` drift.

- Rederive the KLUB (or an analogue) using the CM flow-map generator as the object linearized, not the score field. Write the derivation in the report.
- Implement a lightweight schedule optimizer: given a checkpoint and a target step count, solve for the KLUB-minimizing `{mid_t_1, ..., mid_t_{N-1}}`.
- Compare: fixed `0.821` vs. our schedule vs. grid-search optimum from Experiment A. KLUB schedule should approach grid-search quality at a fraction of the search cost.

### 4.4 Experiment C: on-policy schedule-aware fine-tuning

- Add a second term to `ECMLoss` that samples `x_mid = model(y*t_max, t_max).detach()`, then asks the model to be consistent between `(x_mid + eps*mid_t, mid_t)` and `(x_mid + eps*r, r)` for `r < mid_t`. The student sees its own outputs at inference-relevant noise levels.
- Use `mid_t*` from Experiment A (or schedule from Experiment B) as the distribution of noise levels for the on-policy branch.
- Train from the 1980 kimg checkpoint for an additional short budget (e.g. +500 kimg).
- Measure: FID(1-step), FID(2-step at `mid_t*`), FID(2-step at 0.821). All numbers via the corrected eval.

Expected outcome: 2-step FID improves without giving up 1-step FID. If 1-step regresses, the on-policy term is hurting the endpoint — investigate loss weighting / schedule of `mid_t` samples during training.

### 4.5 Stop-loss criteria

- If the corrected ablation (4.1) already contradicts some core claim in our TA-meeting slides, we rewrite those slides honestly. That is more important than keeping the narrative intact.
- If Experiments A/B/C each individually fail, the final report writes up "what a realistic tuning-budget ECT can and cannot do, given a corrected evaluation" — still a real contribution, and we cite our corrected bug report as the primary artifact.

---

## 5. What we have vs. what we still need

Have:
- Four ECT checkpoints at 500 / 1000 / 1500 / 1980 kimg (`results_backup/ect_checkpoints/`).
- EDM Heun baselines (10k and 50k).
- Latency curves.
- Working Colab G4 environment, ECT repo, CIFAR-10 zip.

Need (all cheap on our G4 box):
- Patched `ct_eval.py` + `metric_main.py`.
- `mid_t` sweep notebook.
- KLUB-style optimizer script (Experiment B; only if A shows drift).
- On-policy loss patch in `training/loss.py` (Experiment C; optional based on A/B outcomes).

Timeline estimate (AI-assisted):
- 4.1 (patch + re-run ablation): 1-2 days.
- 4.2 (Experiment A sweep): 1-2 days.
- 4.3 (Experiment B if triggered): 3-5 days.
- 4.4 (Experiment C if triggered): 5-7 days.

This fits in a May implementation window comfortably, leaves room for the June 7 final report, and gives a strong April 28/30 progress-presentation story regardless of which branches of the experiment tree fire.

---

## 6. Artifacts to produce

- This `INVESTIGATION.md` (the research log).
- `PATCHES/ct_eval_fix.py` + `PATCHES/metric_main_fix.py` (the code fixes).
- `notebooks/schedule_sweep.ipynb` (Experiment A).
- `results/corrected_ablation.csv`, `results/mid_t_sweep.csv` (the data).
- Updated `ANALYSIS.md` section replacing the "dead second step" framing with the corrected story.
- A short write-up "`ct_eval.py` always reports 2-step FID" suitable for submission as a GitHub issue / PR to `locuslab/ect`.

---

## 7. Open questions for the next session

- Is anybody else's ECT-based work downstream of this bug? (Worth a Google Scholar scan on follow-up papers citing `locuslab/ect` and using `ct_eval.py`.)
- Does the `amp` branch of `locuslab/ect` have the same bug? (Very likely yes; it inherits `ct_eval.py`.)
- Is there a cheaper surrogate for FID (e.g. FD-DINOv2) we can use during the `mid_t` sweep to reduce the 50k-sample cost?
- Do our four existing checkpoints sweep enough training budget to see drift in `mid_t*`, or do we need to add 2500 / 3500 / 5000 kimg checkpoints to cover the range?
