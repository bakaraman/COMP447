# COMP447 Project Workspace

## Project

**Title:** When Does Cheap Consistency Tuning Pay Off?  
**Short version:** We compare Easy Consistency Tuning (ECT) against Heun sampling under matched latency budgets on a T4 GPU, then measure when the upfront tuning cost of ECT is actually worth paying.

This repository is the working space for the course project, proposal source, scripts, notes, and readings. It is **not** the upstream ECT repository itself. The external ECT codebase is expected to live under `project/src/ect/` after setup.

## Core idea

We are studying a practical question:

- If you want faster image generation, is it better to:
  - spend compute upfront to tune a diffusion model into a consistency model and then sample in `1-2` steps, or
  - skip tuning and use a fast multi-step diffusion sampler like Heun directly?

Our project answers that question empirically on **CIFAR-10** using a **single T4 GPU**, with a focus on:

- matched wall-clock latency
- FID-quality tradeoffs
- break-even generation count
- tuning-budget ablation

## Main contribution

This is **not meant to be just a reproduction**.

The intended contribution is:

- a **latency-matched comparison** between `ECT` and `Heun`
- a **break-even analysis** showing when ECT's upfront tuning cost is amortized
- a **tuning-budget ablation** asking whether partial tuning or early stopping is already enough

In one line:

> We are comparing two practical speedup strategies for diffusion-style generation and asking when consistency tuning actually pays off on commodity hardware.

## Current experiment plan

All methods start from the same pretrained **EDM** checkpoint on **CIFAR-10**.

### Branch A: ECT

- Fine-tune the pretrained checkpoint using ECT
- Evaluate `1-step` and `2-step` generation
- Measure quality and latency

### Branch B: Heun baseline

- Use the original pretrained checkpoint directly
- Run Heun sampling at several step counts
- Find the step counts that match the latency budgets of ECT

### Metrics

- `FID`
- `NFE`
- wall-clock latency at batch size `1`
- wall-clock latency at batch size `64`
- total GPU-hours
- break-even count `N*`

### Extra analysis

- Save ECT checkpoints at `25%`, `50%`, `75%`, and `100%` of the tuning schedule
- Evaluate `2-step` FID at each checkpoint
- Plot quality versus tuning budget

## What needs to happen by the first TA meeting

The first progress meeting is expected to be with **Andrew** on **Monday, April 13, 2026 at 15:30**.

By then, the goal is not to finish the project. The goal is to make sure the project is no longer vague.

Minimum things that should be ready:

- the project framing in one clean paragraph
- the main experiment design
- the exact metrics
- a clear explanation of why this is more than reproduction
- ECT setup started or at least environment/checkpoint plan confirmed
- Heun baseline plan confirmed

Suggested meeting summary:

> We are not doing a pure reproduction. We are comparing ECT and Heun under matched latency budgets on a T4, then analyzing break-even cost and tuning-budget ablations to understand when consistency tuning is practically worthwhile.

## Week-by-week plan

### Week of April 11-13

- finalize framing for the Andrew meeting
- confirm that `break-even + tuning-budget ablation` is strong enough as the extension angle
- start ECT repo setup
- confirm pretrained checkpoint and runtime environment

### Week of April 14-19

- get the ECT pipeline running end-to-end
- run a tiny sanity-check experiment
- implement or wire up Heun evaluation from the same checkpoint
- begin latency measurement pipeline

### Week of April 20-26

- run pilot timings
- choose `3-4` matched latency targets
- collect first real results for ECT and Heun
- generate first draft tables and plots

### Week of April 27-30

- prepare the progress presentation
- prepare the progress report draft
- make sure there is at least:
  - one results table
  - one latency-quality figure
  - one visual sample figure

### Early May

- finish and polish the progress report
- use the safer earlier deadline in planning until the exact course deadline is confirmed

### Mid May

- run tuning-budget ablations
- test confirmatory seeds for the strongest configurations
- clean up plots and analysis

### Late May to early June

- stabilize final experiments
- write the final report
- prepare the final presentation
- clean the repo for reproducibility

## Repository map

### Project-critical

- [`project/PLAN.md`](project/PLAN.md): main project plan
- [`project/configs/experiment_grid.yaml`](project/configs/experiment_grid.yaml): experiment grid and metrics
- [`project/scripts/setup_ect.sh`](project/scripts/setup_ect.sh): setup helper for the external ECT repo
- [`project/scripts/break_even.py`](project/scripts/break_even.py): break-even analysis
- [`project/scripts/measure_latency.py`](project/scripts/measure_latency.py): latency script skeleton
- [`project/scripts/eval_fid.py`](project/scripts/eval_fid.py): FID evaluation skeleton
- [`final_upload/tex/proposal.tex`](final_upload/tex/proposal.tex): proposal source
- [`final_upload/proposal_karaman_ozcelik.pdf`](final_upload/proposal_karaman_ozcelik.pdf): submitted proposal PDF

### Useful support material

- `readings/`: lecture notes, rubric, paper list, course material
- `proposal_template/`: original LaTeX template files

### Important note

At the moment, this repo contains the **project wrapper and planning scaffold**, not the full experimental implementation.

In particular:

- `project/src/ect/` is currently expected to hold the external ECT repo after setup
- `measure_latency.py` is not fully wired yet
- `eval_fid.py` is not fully wired yet

That means the project idea is well defined, but the experiment pipeline is still under construction.

## Subject to change

These points are still open and may change after the TA meeting or early pilot runs:

- the exact matched latency targets
- the exact Heun step counts used in the final comparison
- whether full ECT tuning is feasible under Colab/T4 limits
- whether we use one or two seeds for each stage
- the exact deadline to target for the progress report if course communication remains inconsistent
- whether the ablation uses only `2-step` evaluation or also includes `1-step`

## Risks

- The biggest practical risk is setup friction, not writing.
- The repo still needs the external ECT code and checkpoint flow to be properly connected.
- If full tuning turns out to be too heavy, we may need to prioritize reduced-schedule tuning plus strong analysis.

## Owners

- **Batuhan:** ECT reproduction, evaluation pipeline, FID computation, report writing
- **Kadir Yigit:** Heun baselines, latency profiling, break-even analysis, tuning ablation
- **Shared:** experiment design, figures, presentation, interpretation

## Quick status

Right now the project is in a good conceptual state, but not yet in a fully runnable state.

That is fine for the current stage, as long as the next step is execution rather than more topic shopping.
