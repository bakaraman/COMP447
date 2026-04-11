# Monday Checklist — Andrew Meeting, April 13, 2026, 15:30

## Status: pipeline alive, throughput measured, plan revised

As of April 11 (Friday), the ECT training + evaluation pipeline is running end-to-end on a Colab GPU. The Monday meeting no longer needs to justify setup — it needs to present measurements and a revised plan.

---

## What works (evidence collected)

- **ECT repo cloned + patched** for torch 2.10 compatibility
  - Upstream bug: `InfiniteSampler.__init__(dataset)` breaks on torch >= 2.2
  - Fix in `project/scripts/setup_ect.sh` is now permanent and idempotent
- **EDM repo cloned** for Heun baseline
- **CIFAR-10 prepared** in EDM format (`datasets/cifar10-32x32.zip`, 159 MB)
- **Pretrained EDM checkpoint** (`edm-cifar10-32x32-uncond-vp.pkl`) downloads and loads
- **End-to-end training loop validated:** 20 kimg sanity run completed on both T4 and Blackwell
  - Loss finite and stable (15–17 band)
  - Clean exit
  - Checkpoint saved
- **FID evaluation working:**
  - 1-step FID50k computed end-to-end
  - 2-step FID50k computed end-to-end
- **Sample image export working**

## Throughput measurements (the headline finding)

Measured `sec/kimg` on two GPUs at batch=64, fp32, CIFAR-10 32×32, DDPM++ 55.7M parameters:

| GPU | Measured sec/kimg | Implied full `run_ecm_1hour.sh` (25,600 kimg) |
|---|---|---|
| Tesla T4 (16 GB) | **29.0** | **~206 hours (~8.6 days)** |
| RTX PRO 6000 Blackwell (102 GB) | **2.68** | ~19 hours |
| A100 (est. from published benchmarks) | ~4-5 | ~30-35 hours |
| H100 (est.) | ~2-3 | ~15-20 hours |

**Implication for the project:** the ECT paper's `run_ecm_1hour.sh` script is **not a 1-hour job on any single Colab GPU**. The "1 hour" name presumably refers to the paper's datacenter hardware or multi-GPU scaling. On a single T4 (our proposal's target hardware), the full schedule would take over a week.

## Headline 20-kimg FID numbers (sanity only, not comparable to paper)

From the Blackwell sanity run:

- 1-step FID50k: **386.46**
- 2-step FID50k: **147.90**

These are deliberately undertrained numbers (0.08% of the paper's tuning budget) and are **not a meaningful quality measurement**. They only prove the evaluation plumbing works. Paper reports ECT 2-step ≈ 2-3 after full tuning.

---

## Revised plan for the Monday meeting

### 1. Honest framing

> We have a working ECT training and evaluation pipeline on Colab. We measured ECT's actual throughput on T4 and found the "1-hour" training script would take over a week on T4. We are revising the tuning budget downward to a schedule that fits Colab's compute constraints while still producing a meaningful ECT vs Heun comparison.

### 2. Revised tuning schedule

Instead of the paper's 25,600 kimg (infeasible), we use a **medium schedule of 2,000 kimg** (~7.8% of paper) as the main tuning budget. This gives:

- Real (not random-init) ECT checkpoint
- Meaningful 2-step FID (expected ~30-80, vs 148 at 20 kimg and ~2-3 fully trained)
- ~15 min on Blackwell / ~30 min on L4 / ~1.5 hours on T4
- Leaves room in the compute budget for tuning-budget ablation checkpoints at 500 / 1000 / 2000 kimg

### 3. Latency measurement commitment (unchanged)

- **All latency measurements will still be on T4**, not Blackwell/L4/A100
- This is the proposal's stated hardware and cannot change
- Training GPU is separate from inference GPU — we can use fast GPUs for tuning and switch back to T4 for the latency-matched comparison

### 4. Tuning-budget ablation (unchanged intent, different absolute numbers)

Instead of checkpoints at 25/50/75/100% of a 25,600-kimg schedule, we use checkpoints at:
- 500 kimg (25% of medium schedule)
- 1,000 kimg (50%)
- 1,500 kimg (75%)
- 2,000 kimg (100%)

We report FID vs kimg to show the early-stopping story, even if the absolute FID is worse than paper.

### 5. Compute budget

Rough estimates per run on Blackwell (~50 compute units/hour):

| Run | kimg | Time | Units |
|---|---|---|---|
| Medium tuning (main) | 2,000 | ~15 min | ~12 |
| Ablation 500-kimg checkpoint | 500 | ~4 min | ~3 |
| Ablation 1,000-kimg checkpoint | 1,000 | ~8 min | ~7 |
| Ablation 1,500-kimg checkpoint | 1,500 | ~11 min | ~9 |
| Latency measurements on T4 (post-training) | — | ~20 min | ~1 |
| Heun baseline sweeps on T4 | — | ~30 min | ~1 |
| **Total estimate** | — | ~1.5 hours | **~35 units** |

Colab Pro budget is ~100 units/month — this fits comfortably if we stay disciplined.

---

## Questions for Andrew

1. **Is 2,000-kimg tuning enough as the main ECT tuning budget, given the paper uses 25,600?**
   Our framing: we're not reproducing the paper's FID. We're measuring the latency–quality tradeoff at budgets that fit commodity GPU time, which is itself an interesting finding ("cheap ECT is cheaper than advertised but not as good").

2. **Should the Heun baseline come from the same pretrained EDM checkpoint throughout?**
   Current plan: yes, `edm-cifar10-32x32-uncond-vp.pkl` is the shared starting point for both ECT tuning and Heun evaluation.

3. **Does the throughput finding itself belong in the report as a claim?**
   Our view: yes, as an auxiliary observation. "ECT's 1-hour marketing is optimistic for commodity GPUs" is a useful note for anyone trying to reproduce.

4. **Is the tuning-budget ablation still compelling at a 2,000-kimg max instead of the paper's 25,600?**
   The story changes slightly: instead of "how close can you get to the paper with less tuning", it becomes "how quickly does the curve bend toward usable quality".

5. **Progress report deadline** — should we plan around April 30 as the safer date if the course communication stays inconsistent?

---

## What changed vs the original plan

| Item | Original plan | Revised plan |
|---|---|---|
| ECT tuning budget | full paper schedule (25,600 kimg) | medium schedule (2,000 kimg) |
| ECT training GPU | T4 | Blackwell / L4 / A100 (whichever is available) |
| Latency measurement GPU | T4 | T4 (unchanged) |
| Ablation checkpoints | 25%, 50%, 75%, 100% of 25,600 | 25%, 50%, 75%, 100% of 2,000 |
| FID target | match paper (2-3) | best achievable at 2,000 kimg (likely 20-60) |
| Break-even analysis | unchanged | unchanged |
| Heun baseline plan | unchanged | unchanged |

---

## Not part of Monday (deferred)

- Actual 2,000-kimg medium run (Cell 9 in the notebook) — will run after meeting confirms the revised plan
- FID vs tuning-budget curve — needs the ablation checkpoints
- Latency-matched grid on T4 — needs the tuned checkpoints
- Final figures and tables — post-Phase 2

---

## Files to show in the meeting

- `project/colab_first_run.ipynb` — the notebook that ran on Blackwell
- `project/MONDAY_CHECKLIST.md` — this document
- `project/PLAN.md` — the unchanged project plan (note: milestones may need date updates)
- `final_upload/proposal_karaman_ozcelik.pdf` — the submitted proposal
