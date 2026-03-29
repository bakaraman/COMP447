# Project Plan: When Does Cheap Consistency Tuning Pay Off?

Paper: Easy Consistency Tuning (ECT), Geng et al., ICLR 2025
Repo: https://github.com/locuslab/ect

## What we are doing

Comparing ECT (1-2 step consistency sampling) against Heun (20-80 step diffusion sampling) on CIFAR-10, under matched wall-clock latency budgets on a T4 GPU. We also compute break-even generation counts and test early stopping.

## Milestones

### Phase 1: Setup (by April 5)
- [ ] Clone ECT repo into project/src/ect
- [ ] Set up Colab notebook with T4 runtime
- [ ] Download pretrained EDM checkpoint for CIFAR-10
- [ ] Verify repo runs: train for a few steps, generate a few images
- [ ] Set up FID evaluation pipeline

### Phase 2: Reproduction (by April 12)
- [ ] Run ECT tuning on CIFAR-10 (full schedule)
- [ ] Generate 50k samples at 2 steps
- [ ] Compute FID, check it falls in expected range
- [ ] Implement Heun sampling from same EDM checkpoint at various step counts
- [ ] Measure per-image latency at batch size 1 and 64

### Phase 3: Main comparison (by April 26)
- [ ] Pick 3-4 latency targets from pilot runs
- [ ] For each target: ECT 1-step, ECT 2-step, Heun at matched step count
- [ ] Record FID, NFE, latency (batch 1 + 64) for each
- [ ] Compute break-even N* for batch 1 and batch 64
- [ ] Plot FID vs latency (Pareto frontier)
- [ ] Generate sample grids for visual comparison

### Phase 4: Progress report (by April 30)
- [ ] Prepare progress presentation slides (10 min)
- [ ] Write 6-page progress report with preliminary results

### Phase 5: Tuning ablation (by May 17)
- [ ] Save ECT checkpoints at 25%, 50%, 75%, 100% of schedule
- [ ] Evaluate 2-step FID at each checkpoint
- [ ] Plot FID vs tuning budget curve
- [ ] Run confirmatory seeds for Pareto-frontier configs

### Phase 6: Final deliverables (by June 7)
- [ ] Write 8-page final report
- [ ] Prepare final presentation slides
- [ ] Clean up code and results for reproducibility

## Division of work

Batuhan: ECT reproduction, evaluation pipeline, FID computation, report writing
Kadir Yigit: Heun baselines, latency profiling, break-even analysis, tuning ablation
Shared: experiment design, figures, presentations

## Key files

- project/src/ect/ — cloned ECT repo
- project/configs/ — our experiment configs
- project/scripts/ — evaluation and profiling scripts
- project/results/ — saved metrics, plots, samples
- final_upload/ — proposal PDF and LaTeX source

## Metrics we report

- FID (primary quality)
- NFE (algorithmic cost)
- Wall-clock latency, batch 1 (interactive)
- Wall-clock latency, batch 64 (offline)
- Break-even N* (amortization)
- Total GPU-hours (reproducibility)
