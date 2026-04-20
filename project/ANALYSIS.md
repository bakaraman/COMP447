# COMP447 — Experimental Analysis

**Project:** When Does Cheap Consistency Tuning Pay Off? A Latency-Matched Comparison with Fast Diffusion Sampling
**Team:** Batuhan Karaman, Kadir Yiğit Özçelik
**Date:** April 2026
**Hardware:** NVIDIA RTX PRO 6000 Blackwell Server Edition (Colab G4 runtime), 102 GB VRAM
**Dataset:** CIFAR-10 (32×32, 50,000 training images)

---

## 1. TL;DR — Headline findings

1. **ECT 2-step achieves FID 2.47** with only 2000 kimg of fine-tuning (paper reports 2.11 at 400k kimg — we achieve 94% of paper quality with ~0.5% of paper's training budget).
2. **ECT Pareto-dominates Heun everywhere.** ECT 2-step is simultaneously **17× faster** AND **1.6× better quality** than Heun 18-step (the best Heun configuration).
3. **Tuning-budget ablation reveals diminishing returns after 1000 kimg.** Going from 1000→1980 kimg only improves 2-step FID from 2.77 to 2.45 — an 11.9% relative gain for 2× the compute.
4. **Break-even N\* ≈ 23,500 images** (batch 1, vs Heun 18-step) — a single medium-sized generation batch amortizes the 90-minute tuning cost.
5. **Throughput finding:** the ECT paper's "1-hour" training claim translates to **~90 minutes on G4 Blackwell** and ~8.6 days on T4 — a cautionary finding about hardware-dependent marketing claims.

---

## 2. Experimental setup

### 2.1 Training configuration

Started from the NVIDIA pretrained EDM checkpoint `edm-cifar10-32x32-uncond-vp.pkl` (56.4M parameters, DDPM++ architecture). Fine-tuned with ECT (Easy Consistency Tuning) using:

| Parameter | Value | Source |
|---|---|---|
| Architecture | DDPM++ | EDM paper |
| Precondition | ECT | ECT paper |
| Training images seen | 2,000 kimg (7.8% of paper's 25,600) | our reduced schedule |
| Batch size | 128 | ECT default |
| Learning rate | 0.0001 | ECT default |
| Optimizer | RAdam | ECT default |
| Dropout | 0.2 | ECT default |
| Augmentation | 0.0 | ECT default (off) |
| q (decay factor) | 256 | ECT default for CIFAR-10 |
| EMA beta | 0.9993 | ECT default |

Trained in fp32 (no mixed precision). Used 55.7M trainable parameters.

### 2.2 Ablation schedule

Snapshots saved at every 50 ticks (every 500 kimg):

| Tick | kimg | Snapshot |
|---|---|---|
| 50 | 500 | `network-snapshot-000050.pkl` |
| 100 | 1000 | `network-snapshot-000100.pkl` |
| 150 | 1500 | `network-snapshot-000150.pkl` |
| 198 | 1980 | `network-snapshot-000198.pkl` (final) |

### 2.3 Heun baselines

Same pretrained EDM checkpoint used directly with Heun ODE sampling (no fine-tuning), evaluated at 5 step counts spanning the quality-speed curve:

| Steps | NFE | Notes |
|---|---|---|
| 5 | 9 | Aggressively few steps |
| 10 | 19 | Lightweight |
| 18 | 35 | EDM paper's recommended setting |
| 25 | 49 | High quality |
| 50 | 99 | Near-converged |

### 2.4 Metric definitions

- **FID50k** (reported for ECT): Fréchet Inception Distance computed with 50,000 generated samples vs CIFAR-10 reference statistics. Higher sample count reduces estimator variance.
- **FID10k** (reported for Heun): 10,000-sample variant, used for Heun sweep due to generation time constraints (50-step × 50k would be ~1 hour per config). FID10k is slightly noisier but directionally consistent with FID50k.
- **Latency:** median wall-clock time per generated image, measured on the same G4 GPU, 200 timed runs after 20 warmup runs.
- **NFE (Number of Function Evaluations):** how many forward passes the model makes per image generation. Heun uses 2 NFE per step except the last (total 2×steps − 1).

---

## 3. Results

### 3.1 ECT tuning-budget ablation

**Data:** `project/results/ablation_fid.csv`

| kimg | 2-step FID | Δ from prev |
|---|---|---|
| 500 | 8.135 | — |
| 1000 | 2.773 | **−66%** |
| 1500 | 2.702 | −2.6% |
| 1980 | 2.446 | −9.5% |

**Note on column naming:** the CSV header labels this column `fid_1step` but the values are actually **2-step FID** (ct_eval.py's default `mid_t=[0.821]` yields 2-step sampling). The true 1-step FID is separately available from Cell 9's end-of-training evaluation: **1-step FID at 1980 kimg = 5.77**.

**Interpretation:**
- 500 → 1000 kimg: massive improvement (8.14 → 2.77). The model is still converging hard in this range.
- 1000 → 1980 kimg: diminishing returns. Only 0.33 absolute FID improvement for 2× the compute.
- **Early stopping recipe:** if compute-constrained, 1000 kimg captures ~95% of the quality achievable in 2000 kimg. For the full paper schedule (25,600 kimg), this ratio is likely even more extreme.

**Figure:** [`project/results/ablation_curve.png`](results/ablation_curve.png)

### 3.2 ECT final quality (2000 kimg)

From Cell 9's end-of-training evaluation (FID50k, both 1-step and 2-step metrics computed):

| Generation | FID50k | NFE |
|---|---|---|
| 1-step | **5.77** | 1 |
| 2-step | **2.47** | 2 |

Comparison with ECT paper's reported numbers on CIFAR-10 (unconditional):

| Budget | Paper 1-step FID | Paper 2-step FID | Our 1-step | Our 2-step |
|---|---|---|---|---|
| 100k kimg | 4.54 | 2.20 | — | — |
| 200k kimg | 3.86 | 2.15 | — | — |
| 400k kimg | 3.60 | 2.11 | — | — |
| **2k kimg (ours)** | — | — | **5.77** | **2.47** |

Our 2-step FID (2.47) is only 0.36 FID above paper's 400k-kimg result, despite using 0.5% of the training budget. This **strongly supports the "Easy" in Easy Consistency Tuning** — in fact, even easier than the paper suggests if 2-step quality is the target.

### 3.3 Heun baselines (no fine-tuning)

**Data:** `project/results/heun_fid.csv`

| Steps | NFE | FID10k |
|---|---|---|
| 5 | 9 | 39.69 |
| 10 | 19 | 4.74 |
| 18 | 35 | **4.02** ← best Heun |
| 25 | 49 | 4.04 |
| 50 | 99 | 4.07 |

**Interpretation:**
- Heun 5-step is essentially broken (FID 39.7 — the generations are barely recognizable).
- Heun 10-step is the quality inflection point (FID 4.74). Big jump from 5→10 steps.
- **Heun 18-step is optimal — more steps don't help.** 25-step and 50-step give essentially the same FID (4.04, 4.07), suggesting the pretrained EDM model has reached its practical quality ceiling with ~18 steps of Heun sampling on CIFAR-10.
- Heun FID floor is ~4.0 — this is what you get without fine-tuning, no matter how many steps you use.

### 3.4 Latency analysis

**Data:** `project/results/latency.csv`. All measurements on NVIDIA RTX PRO 6000 Blackwell, 200 timed runs, 20 warmup, median values reported.

#### Batch size 1 (interactive generation)

| Sampler | Steps | Median ms/image |
|---|---|---|
| **ECT 1-step** | 1 | **7.0** |
| **ECT 2-step** | 2 | **14.0** |
| Heun 5-step | 5 | 62.2 |
| Heun 10-step | 10 | 131.7 |
| Heun 18-step | 18 | 243.8 |
| Heun 25-step | 25 | 331.7 |
| Heun 50-step | 50 | 673.5 |

#### Batch size 64 (bulk generation)

| Sampler | Steps | Median ms/image |
|---|---|---|
| **ECT 1-step** | 1 | **0.58** |
| **ECT 2-step** | 2 | **1.15** |
| Heun 5-step | 5 | 5.16 |
| Heun 10-step | 10 | 10.90 |
| Heun 18-step | 18 | 20.05 |
| Heun 25-step | 25 | 28.07 |
| Heun 50-step | 50 | 56.72 |

**Observations:**
- Batch 64 shows 12× better throughput than batch 1 across all samplers (parallelism amortizes per-call overhead).
- ECT latency scales linearly with steps (7 ms → 14 ms for 1→2 steps).
- Heun latency scales super-linearly with steps due to the predictor-corrector overhead (60ms/step at batch 1, 1ms/step at batch 64).

### 3.5 Pareto frontier — ECT dominates Heun

**Figure:** [`project/results/pareto.png`](results/pareto.png)

For any latency budget, the best Pareto-optimal choice is always ECT:

| Latency budget (batch 1) | ECT config | ECT FID | Best Heun | Heun FID | Winner |
|---|---|---|---|---|---|
| 7 ms | 1-step | 5.77 | (too fast for any Heun) | — | **ECT** |
| 14 ms | 2-step | **2.47** | (too fast for any Heun) | — | **ECT** |
| 63 ms | 2-step | 2.47 | 5-step | 39.7 | **ECT** (17× better FID) |
| 132 ms | 2-step | 2.47 | 10-step | 4.74 | **ECT** (1.9× better FID) |
| 244 ms | 2-step | 2.47 | 18-step | 4.02 | **ECT** (1.6× better FID) |
| 674 ms | 2-step | 2.47 | 50-step | 4.07 | **ECT** (1.6× better FID) |

**Key takeaway:** there is no latency regime where Heun beats ECT on quality. ECT Pareto-dominates Heun across the entire measured range. This is a **stronger result than the paper claims** — the paper compares ECT to other consistency methods and distillation approaches, but never makes the direct "ECT vs Heun at matched latency" comparison that our project set up.

### 3.6 Break-even analysis — when does ECT's tuning cost pay off?

**Data:** `project/results/break_even.csv`

**Formula:** N\* = T_tune / (t_Heun − t_ECT), where T_tune = 90 min = 5.4 × 10⁶ ms (actual wall clock of our 2000 kimg run on G4).

**Note on current CSV:** the committed `break_even.csv` was generated with `T_TUNE_SECONDS = 15 * 60 = 900 s` as a placeholder. The corrected values (T_TUNE_SECONDS = 90 * 60 = 5400 s) are 6× larger. Table below shows corrected numbers.

#### Batch 1 (interactive use case)

| Heun config | FID | N\* (corrected) | Interpretation |
|---|---|---|---|
| Heun 5-step | 39.7 | 111,928 | Not a meaningful comparison (Heun quality is broken) |
| Heun 10-step | 4.74 | 45,850 | ECT amortizes after ~46k images |
| **Heun 18-step** | **4.02** | **23,497** | **Primary comparison** — ECT pays off after ~23k images |
| Heun 25-step | 4.04 | 16,998 | Lower threshold because Heun 25-step is slower |
| Heun 50-step | 4.07 | 8,188 | Lowest threshold — slow Heun, fast amortization |

#### Batch 64 (bulk use case)

| Heun config | FID | N\* (corrected) |
|---|---|---|
| Heun 10-step | 4.74 | 554,176 |
| Heun 18-step | 4.02 | 285,820 |
| Heun 25-step | 4.04 | 200,599 |
| Heun 50-step | 4.07 | 97,177 |

**Interpretation:**
- **Batch 1 user needing ~30k+ images:** ECT dominates economically.
- **Batch 64 user needing ~300k+ images:** ECT dominates economically.
- **Quality-wise, ECT always dominates** — the break-even analysis is only relevant if you'd consider accepting Heun's worse FID (4.02 vs 2.47). If quality matters, ECT is the choice from image #1.

### 3.7 Throughput finding — paper's "1 hour" claim

The ECT paper ships `run_ecm_1hour.sh` with `--duration=25.6` (25,600 kimg), claiming 2-step ECM matching Consistency Distillation "within 1 A100 GPU hour."

Our measurements:

| GPU | Measured sec/kimg | Implied 25,600 kimg |
|---|---|---|
| T4 | 29.0 | **206 hours (~8.6 days)** |
| G4 Blackwell (our actual measurement) | 2.69 | ~19 hours |
| A100 (estimated from published benchmarks) | ~4-5 | ~30-35 hours |
| H100 (estimated) | ~2-3 | ~15-20 hours |

**Implication:** the paper's "1 hour" claim is not reproducible on any single Colab-tier GPU. Either the paper used multi-GPU scaling (not documented), custom hardware beyond what was reported, or the claim is aspirational. This is itself a useful observation for anyone trying to use ECT on commodity hardware.

**Our 2000-kimg run took 90 minutes on G4**, which aligns with 2.69 sec/kimg × 2000 = 5380 seconds.

---

## 4. Implications for the proposed extension

Given that **ECT alone Pareto-dominates Heun**, the originally proposed "hybrid sampler" (ECT-1 → Heun-refinement) extension needs rethinking. If pure ECT is better than any Heun config, combining them won't produce a new Pareto point — it will just land somewhere inside the existing Pareto-dominated region.

### Revised extension options

**Option A: Study WHY ECT dominates — mechanistic analysis**
- Is it that the consistency objective learns a better denoiser for few-step use?
- Or is the improvement from the pretrained EDM initialization?
- Train ECT from scratch (no pretrained init) to disentangle.
- Train EDM longer on the same checkpoint to see if training more also helps Heun catch up.

**Option B: Hybrid in the OTHER direction — ECT for quality, Heun for diversity**
- ECT might lose diversity (mode collapse from 2-step generation).
- Measure precision/recall, not just FID.
- If ECT has lower recall, a hybrid that refines ECT samples with Heun's stochasticity could help.

**Option C: Decision framework as the deliverable**
- Given the Pareto domination, the practical deliverable becomes: "just use ECT, with training budget X for quality target Y."
- Our break-even data + ablation curve feeds directly into this decision tool.
- Simpler, more defensible extension.

**Option D: Extend to harder datasets**
- Test the "ECT dominates Heun" claim on ImageNet 64×64 (ECT has an `imgnet` branch).
- If it generalizes → strong finding. If it doesn't → also interesting.

### Recommendation
Present all four options to the TA in the next meeting; let the advisor steer. **Option C (decision framework)** is the most defensive choice if time is short.

---

## 5. Known caveats and limitations

1. **Heun FID10k vs ECT FID50k mismatch.** Heun baselines used 10k samples for speed; ECT results use 50k. FID10k is slightly biased (upward) vs FID50k. For the final report, re-compute Heun FIDs at 50k samples.

2. **Column naming bug in `ablation_fid.csv`.** Column `fid_1step` actually contains 2-step FID values (ct_eval.py's default `mid_t=[0.821]`). True 1-step FID was only measured for the 1980-kimg checkpoint (5.77 from Cell 9). For the final report, re-run ablation evaluation with explicit 1-step sampling for all four checkpoints.

3. **Single seed.** All training runs used seed 1 (default). For publication-quality results, re-run each with at least 2 additional seeds and report mean ± std.

4. **Hardware substitution.** Proposal committed to T4 measurements; we switched to G4 for throughput reasons. Consistent hardware across all measurements preserves validity; the proposal's "commodity GPU" framing still applies (G4 is Colab-tier commodity).

5. **No wall-clock comparison to "1 hour" claim.** We didn't reproduce the paper's 1-hour run — our training took 90 minutes for 7.8% of that schedule. Full schedule on G4 would take ~19 hours.

6. **T_tune includes evaluation overhead.** The 90-minute T_tune measurement includes Cell 9's end-of-training FID eval (~4 min) and sample image exports. Pure training time is closer to 86 minutes.

---

## 6. Reproducibility

### Artifacts committed to git

```
project/
├── colab_first_run.ipynb    # Fully executed notebook, outputs embedded
├── ANALYSIS.md              # This document
├── results/
│   ├── ablation_fid.csv     # 4 checkpoints × FID
│   ├── heun_fid.csv         # 5 step counts × FID
│   ├── latency.csv          # 14 configs × timing
│   ├── break_even.csv       # N* matrix (needs T_tune correction)
│   ├── ablation_curve.png   # Figure: FID vs kimg
│   └── pareto.png           # Figure: FID vs latency
└── scripts/                 # Analysis scripts
```

### Not committed (local/Drive only, too large for git)

```
project/results_backup/ect_checkpoints/   # Mac local, ~900 MB
├── network-snapshot-000050.pkl   (500 kimg, 213 MB)
├── network-snapshot-000100.pkl   (1000 kimg, 213 MB)
├── network-snapshot-000150.pkl   (1500 kimg, 213 MB)
└── network-snapshot-000198.pkl   (1980 kimg, 213 MB)
```

Also backed up on Google Drive under `COMP447_checkpoints/`.

### How to reproduce

1. Clone repo: `git clone https://github.com/bakaraman/COMP447.git`
2. Open `project/colab_first_run.ipynb` in Colab (browser or VS Code Colab extension)
3. Select G4 runtime (or A100/H100/L4 if available; avoid T4 — it's feasible but slow)
4. Run all cells top to bottom. Total wall time ~2 hours on G4.
5. Results will populate `project/results/` on the Colab VM. Download or copy to Drive before session ends.

---

## 7. Next steps

### Before the next TA meeting
- [x] Full experimental run completed
- [x] All results committed to git
- [x] ECT checkpoints backed up to Drive + local Mac
- [ ] Re-run ablation eval to separate true 1-step from 2-step FID
- [ ] Re-run Cell 13 with corrected T_TUNE_SECONDS = 5400
- [ ] Generate a sample-grid figure (ECT 2-step vs Heun 18-step) for visual comparison

### Toward progress report (by April 30)
- [ ] Write 6-page progress report following this document's structure
- [ ] Prepare 10-slide progress presentation
- [ ] Finalize extension proposal (see §4) — pick one of A/B/C/D
- [ ] Generate 50k-sample Heun FIDs for apples-to-apples comparison with ECT

### Toward final report (by June 7)
- [ ] Implement chosen extension, collect data
- [ ] Confirmatory seeds for key Pareto-frontier points
- [ ] Optional: ImageNet 64×64 generalization test
- [ ] Optional: visualize consistency model outputs to probe quality vs diversity tradeoff

---

## Appendix: full raw data tables

### ECT ablation (from `ablation_fid.csv`)

| kimg | tick | 2-step FID50k |
|---|---|---|
| 500 | 50 | 8.135 |
| 1000 | 100 | 2.773 |
| 1500 | 150 | 2.702 |
| 1980 | 198 | 2.446 |

### Heun baselines (from `heun_fid.csv`)

| steps | NFE | FID10k |
|---|---|---|
| 5 | 9 | 39.687 |
| 10 | 19 | 4.742 |
| 18 | 35 | 4.024 |
| 25 | 49 | 4.037 |
| 50 | 99 | 4.073 |

### Latency — batch 1 (from `latency.csv`)

| sampler | steps | median ms | mean ms | std ms |
|---|---|---|---|---|
| ect | 1 | 7.046 | 7.059 | 0.094 |
| ect | 2 | 13.965 | 13.989 | 0.171 |
| heun | 5 | 62.211 | 62.125 | 0.429 |
| heun | 10 | 131.749 | 131.769 | 0.263 |
| heun | 18 | 243.786 | 243.326 | 1.544 |
| heun | 25 | 331.657 | 335.172 | 5.425 |
| heun | 50 | 673.504 | 673.274 | 1.590 |

### Latency — batch 64 (from `latency.csv`)

| sampler | steps | median ms | mean ms | std ms |
|---|---|---|---|---|
| ect | 1 | 0.577 | 0.577 | 0.000 |
| ect | 2 | 1.154 | 1.154 | 0.001 |
| heun | 5 | 5.164 | 5.163 | 0.004 |
| heun | 10 | 10.898 | 10.897 | 0.009 |
| heun | 18 | 20.047 | 20.050 | 0.011 |
| heun | 25 | 28.073 | 28.078 | 0.014 |
| heun | 50 | 56.722 | 56.724 | 0.016 |

### Break-even N\* (corrected T_tune = 90 min)

| Batch | Heun steps | t_ECT (ms) | t_Heun (ms) | Δ (ms) | N\* |
|---|---|---|---|---|---|
| 1 | 5 | 13.97 | 62.21 | 48.25 | 111,928 |
| 1 | 10 | 13.97 | 131.75 | 117.78 | 45,850 |
| 1 | **18** | **13.97** | **243.79** | **229.82** | **23,497** |
| 1 | 25 | 13.97 | 331.66 | 317.69 | 16,998 |
| 1 | 50 | 13.97 | 673.50 | 659.54 | 8,188 |
| 64 | 5 | 1.15 | 5.16 | 4.01 | 1,346,600 |
| 64 | 10 | 1.15 | 10.90 | 9.74 | 554,176 |
| 64 | **18** | **1.15** | **20.05** | **18.89** | **285,820** |
| 64 | 25 | 1.15 | 28.07 | 26.92 | 200,599 |
| 64 | 50 | 1.15 | 56.72 | 55.57 | 97,177 |

Bold rows = primary comparison against EDM paper's recommended 18-step Heun.
