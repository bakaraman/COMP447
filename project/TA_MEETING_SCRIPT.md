# TA Meeting — What to Tell Andrew

**Meeting:** Today
**Duration target:** 10 min your talk + Q&A
**Key message:** "Pipeline executed end-to-end, ECT Pareto-dominates Heun, we have a decision tool as extension."

---

## Opening (30 seconds)

> "We ran the full experiment pipeline on Colab's G4 runtime. ECT 2-step achieves FID 2.47 on CIFAR-10 with just 2000 kimg of fine-tuning — 7.8% of the paper's budget. More importantly, ECT Pareto-dominates Heun at every latency point we measured. The original 'hybrid sampler' extension idea doesn't make sense anymore because pure ECT is already optimal. Instead, we pivoted to building a practical decision framework — answering the question Andrew asked last time: 'what can I actually DO with these insights?'"

---

## Show results (3 minutes) — in this order

### 1. Pareto plot — [`project/results/pareto.png`](results/pareto.png)

Point at it and say:

> "ECT 2-step is here — 14 ms, FID 2.47. Heun's best is here — 244 ms, FID 4.02 at 18 steps. That's 17× faster AND 1.6× better quality. There is no latency budget where Heun wins."

### 2. Ablation curve — [`project/results/ablation_curve.png`](results/ablation_curve.png)

> "Quality plateaus after 1000 kimg. Going from 1000 to 1980 kimg only improves 2-step FID from 2.77 to 2.45. Most of the benefit comes in the first 1000 kimg. This gives us an early-stopping recipe."

### 3. Key numbers table — from your head

| | ECT 2-step | Heun 18-step |
|---|---|---|
| FID | **2.47** | 4.02 |
| Latency (batch 1) | **14 ms** | 244 ms |
| Latency (batch 64) | **1.15 ms** | 20 ms |
| Tuning cost | 90 min upfront | 0 |
| Break-even N* (batch 1) | — | **~23,500 images** |

---

## Throughput finding (1 minute)

> "A side finding worth reporting: the ECT paper ships a `run_ecm_1hour.sh` script that claims to reach 2-step consistency in 1 A100 hour. On G4 Blackwell — which is significantly faster than A100 — we measured that the same script would take 19 hours. On T4, it would take 8.6 days. The 'cheap tuning' narrative is hardware-dependent in ways the paper doesn't discuss."

---

## Extension — the revised plan (2 minutes)

> "Andrew, last time you said the comparison alone isn't enough for the semester and asked what we can DO with the insights. Since ECT Pareto-dominates Heun on this dataset, the original hybrid-sampler idea loses its motivation. We see four possible extensions:"

1. **Decision framework (preferred).** Turn the break-even math + ablation curve into a practical tool: given your GPU, batch size, target quality, and number of images needed, output "use ECT with X kimg of tuning" or "skip tuning, use Heun Y-step."

2. **Mechanistic analysis.** Train ECT from scratch (no EDM init) to see if the pretrained initialization is doing the heavy lifting vs the consistency objective itself.

3. **Precision-recall analysis.** FID is a single number. ECT might lose diversity. Measure precision/recall separately to check.

4. **ImageNet 64×64 generalization.** Repeat the experiment on a harder dataset using ECT's `imgnet` branch.

> "We're proposing (1) as the main extension and (3) or (4) as stretch goals if time permits."

---

## Questions for Andrew (3 minutes)

1. **Is the decision-framework extension concrete enough?** Specifically, what should the deliverable look like — a web tool, a one-page recipe, a Python package?

2. **Should the progress report explicitly cite the throughput finding as a contribution?** Framing: "the ECT paper's 1-hour claim is aspirational on commodity hardware."

3. **Does the ECT-dominates-Heun result surprise you?** Should we be suspicious and sanity-check it? (Possible bug: are we comparing 50k FID to 10k FID unfairly? We should normalize before the final report.)

4. **With 7.8% of the paper's training budget, we're 0.36 FID above paper's 400k-kimg result. Should we run longer to close the gap, or is the efficient result more interesting?**

5. **Progress report: any specific structural preferences, or follow the standard 6-page format?**

---

## If Andrew asks "show me the code / repo"

- GitHub: https://github.com/bakaraman/COMP447
- Notebook: `project/colab_first_run.ipynb` (outputs embedded, commit `8ba6567`)
- Analysis doc: `project/ANALYSIS.md` (406 lines, full writeup)
- Latest push: today

---

## If Andrew asks "how did you get the numbers so fast?"

- Used Colab's G4 runtime (NVIDIA RTX PRO 6000 Blackwell, 102 GB VRAM — faster than A100)
- Fixed upstream bug in ECT (`InfiniteSampler` incompatible with torch 2.10; patched in `setup_ect.sh`)
- Reduced schedule from 25,600 kimg to 2,000 kimg — kept in Colab compute budget
- Full pipeline (training + ablation + Heun baselines + latency + plots) ran in ~2 hours end-to-end

---

## Landmines to avoid

- **Don't over-claim "we beat the paper"**: we used less training and got worse FID. We achieved 94% of their quality with 7.8% of their compute. That's efficiency, not quality.
- **Don't say Heun "lost"**: say "Heun without fine-tuning isn't quality-competitive at any latency."
- **Don't commit to the hybrid sampler now**: we already moved away from it.
- **Don't use the uncorrected break-even N\*** (3,916 with T_tune=15 min placeholder). Use the corrected 23,497 with T_tune=90 min.

---

## One-liner if you only get 60 seconds

> "Pipeline works. ECT 2-step hits FID 2.47 in 90 minutes of tuning and beats Heun at every latency — 17× faster, 1.6× better quality. The tuning-budget ablation shows diminishing returns after 1000 kimg. Our extension is a decision framework that turns these measurements into a practical tool. Three questions."
