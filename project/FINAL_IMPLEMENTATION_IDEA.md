# Final Implementation Idea

## Title

On-Policy Asymmetric Refiner for Easy Consistency Tuning

Short name: `OAR-ECT`

## Core claim

Our measurements suggest that, under a short ECT tuning budget, the second ECT sampling step is nearly useless:

- 1000 kimg: 1-step FID 2.797, 2-step FID 2.803
- 1500 kimg: 1-step FID 2.736, 2-step FID 2.672
- 1980 kimg: 1-step FID 2.460, 2-step FID 2.510

At the same time, the original EDM model with Heun still improves strongly with more steps:

- Heun 10-step: FID 2.637
- Heun 18-step: FID 1.960

This points to a sharper question than "ECT vs Heun":

> Why does the tuned consistency model behave like a fast one-shot projector, but not like a useful two-step generator?

Our hypothesis is that the second ECT step is weak for two reasons:

1. It reuses the same network and objective as the first step, so it has no distinct refinement role.
2. At inference time, the second step sees noisy versions of ECT's own outputs, but training mostly teaches low-noise denoising on real images plus noise.

`OAR-ECT` attacks both issues directly.

## Method

### Step 1: Keep ECT as the coarse projector

We keep the existing ECT checkpoint as the fast coarse generator:

`x_coarse = ECT(z * sigma_max, sigma_max)`

This is already the strongest thing ECT does well: one-shot global denoising.

### Step 2: Replace the dead second step with a dedicated refiner

Instead of calling the same ECT network again, we add a small trainable refinement network:

`x_refined = x_coarse + Refiner(x_coarse + sigma_r * eps, x_coarse, sigma_r)`

The refiner is intentionally asymmetric:

- first step = large frozen or lightly-tuned ECT model
- second step = small trainable residual network

This makes the second pass a real correction stage rather than a repeated endpoint projection.

## Training target

The refiner is trained on the distribution it will actually see at inference time.

For each noise seed:

1. Generate a coarse sample with ECT.
2. Add back low noise at `sigma_r`.
3. Use a short EDM Heun tail from `sigma_r -> 0` as teacher target.
4. Train the refiner to map the noisy coarse sample to that teacher-corrected output.

This is the key difference from vanilla ECT-style low-noise training.

The target is not:

- a real image plus noise
- a full 18-step teacher trajectory

The target is:

- a short low-noise teacher correction applied to ECT's own coarse output

That keeps the task local and realistic.

## Why this idea fits our data

This is not a generic "interesting" extension. It comes directly from our numbers.

### Observation 1

ECT 1-step and 2-step are nearly identical.

Interpretation:

The second step is not acting like a meaningful refinement step.

### Observation 2

Heun keeps getting better when it spends more steps.

Interpretation:

The missing quality is still present in the teacher's low-noise refinement behavior.

### Observation 3

ECT is about 17x faster than Heun 18-step on batch size 1.

Interpretation:

There is a lot of room for adding a tiny second-stage module without losing ECT's speed advantage.

## Why this is strong against related work

This idea is close to the consistency-model literature in spirit, but not redundant with it.

- `Consistency Models` says extra steps can improve quality, but it does not propose a dedicated asymmetric second-step refiner.
- `Multistep Consistency Models` makes the whole model multistep-capable, but does not isolate our specific "dead second step" failure mode.
- `Truncated Consistency Models` argues that full-trajectory training can waste capacity, which supports our diagnosis, but it does not attach a lightweight correction head to a frozen ECT model.
- `Dual-End Consistency Model` argues trajectory selection matters, which also supports our diagnosis, but again does not propose this asymmetric refiner design.
- `Autoregressive Distillation of Diffusion Transformers` is useful conceptually because it explicitly talks about exposure bias in few-step distillation, which is close to our train/inference mismatch argument.

So the novelty is not "we invented refinement."

The novelty is:

> We take a real ECT anomaly from our own measurements, interpret it as a dead-step failure caused by symmetry plus train/inference mismatch, and implement a small on-policy correction module specifically for that second step.

## Implementation scope

This is deep enough for the course and still realistic for five weeks.

### New components

- `training/refiner.py`
  - small residual CNN or mini U-Net
- `training/refiner_loss.py`
  - on-policy teacher-tail loss
- `train_refiner.py`
  - second-stage training script starting from an existing ECT checkpoint
- `eval_refiner.py`
  - FID + latency evaluation for `ECT + Refiner`
- `samplers/oar_sampler.py`
  - one ECT jump + one refiner correction

### Reused components

- existing ECT checkpoint
- existing EDM checkpoint
- existing CIFAR-10 pipeline
- existing FID evaluation setup
- existing latency scripts

## Expected outcome

We are not claiming this should beat Heun 18-step in raw FID.

The realistic success criterion is:

- improve over current ECT 2-step
- keep latency close to ECT 1-step / well below ECT 2-step + Heun territory
- create a new Pareto point between ECT and Heun

Reasonable target:

| Method | FID50k | Batch-1 latency |
|---|---:|---:|
| ECT 1-step | 2.460 | 7.046 ms |
| ECT 2-step | 2.510 | 13.965 ms |
| OAR-ECT | 2.1-2.3 | 9-12 ms |
| Heun 18-step | 1.960 | 243.786 ms |

If we land there, the story is strong:

> Heun remains the quality anchor, ECT remains the speed anchor, and our method turns the previously wasted second-step budget into a useful quality correction.

## Failure story

Even failure is informative here.

If OAR-ECT does not improve over ECT 1-step or 2-step, that means the problem is probably deeper than symmetry or on-policy mismatch. In that case, the result supports a stronger conclusion:

> Short-budget ECT may fundamentally collapse into a one-shot endpoint projector, and meaningful few-step gains require changing the consistency objective itself rather than adding a cheap correction stage.

That is still a valid research outcome.

## Future work

If the base version works, there are two natural extensions.

### 1. ImageNet 64x64 generalization

Andrew is right that CIFAR-10 is simple. The strongest follow-up is to test whether the same dead-second-step pattern appears on ImageNet 64x64.

If yes, our explanation is stronger.
If no, then the anomaly is dataset-dependent, which is also interesting.

### 2. Prefix handoff hybrid

If the refiner works only partially, the next step is a more ambitious hybrid:

- ECT handles the high-noise jump
- EDM Heun handles a short low-noise tail

That would test whether the second stage should belong to a small learned refiner at all, or whether it should be handed back to the original diffusion solver.

## Why this should be our main pitch

This idea is the best balance of:

- directly motivated by our own measurements
- genuinely new implementation work
- manageable within course constraints
- clear success and failure interpretation
- easy to explain in a progress presentation

In one sentence:

> We propose to replace ECT's ineffective second step with a small on-policy refinement network trained specifically on ECT's own coarse outputs, so the second step becomes a true learned correction stage rather than a repeated endpoint projection.

## Key references

- ECT: https://arxiv.org/abs/2406.14548
- Consistency Models: https://arxiv.org/abs/2303.01469
- Multistep Consistency Models: https://arxiv.org/abs/2403.06807
- Consistency Trajectory Models: https://arxiv.org/abs/2310.02279
- Truncated Consistency Models: https://arxiv.org/abs/2410.14895
- Dual-End Consistency Model: https://arxiv.org/abs/2602.10764
- One Step Diffusion via Shortcut Models: https://arxiv.org/abs/2410.12557
- Autoregressive Distillation of Diffusion Transformers: https://arxiv.org/abs/2504.11295
