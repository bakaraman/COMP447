# Research-Driven Proposal for a Novel ECT Project on CIFAR-10

## What the measurements actually imply

The main phenomenon in your results is not simply “ECT is faster than Heun.” It is that, in your low-budget fine-tuning regime, the student behaves like a **one-shot projector**: after about 1000 kimg, the second ECT sampling step does almost nothing, even though the original consistency-program motivation is that extra steps should buy back quality. That is unusual relative to the literature. **Consistency Models** (arXiv:2303.01469) were explicitly designed so that multistep sampling trades extra compute for better quality; **Improved Techniques for Training Consistency Models** (arXiv:2310.14189) report CIFAR-10 improvement from 2.51 FID in one step to 2.24 in two steps; **Multistep Consistency Models** (arXiv:2403.06807) make the speed-quality interpolation itself the central object; and **Simplifying, Stabilizing and Scaling Continuous-Time Consistency Models** (arXiv:2410.11081) achieves strong two-step results with a continuous-time formulation. Even the public ECT repo presents CIFAR-10 tables where 2-step ECM is substantially better than 1-step ECM. Your measurements do **not** match that pattern. That mismatch is the research seed. citeturn34view0turn35view0turn12view0turn28view0turn30view0turn22search4

A second implication is that your baseline frontier is already closer to the student than a naive “diffusion is slow” story suggests. In your runs, Heun-10 is already near final ECT quality, and Heun-18 still wins on FID. On top of that, **DPM-Solver-v3: Improved Diffusion ODE Solver with Empirical Model Statistics** (arXiv:2310.13268) reports 2.51 FID on unconditional CIFAR-10 at 10 NFE, which is essentially at your final ECT level without any fine-tuning, while **Align Your Steps: Optimizing Sampling Schedules in Diffusion Models** (arXiv:2404.14507) shows that few-step performance is highly schedule-sensitive rather than purely solver-limited. So a project that only re-runs Heun at different settings is too weak, but a project that **modifies the student so it targets the real few-step frontier** is well motivated. citeturn8search4turn19search6turn21view0turn29view0turn29view1

My bottom-line read is blunt: the best extension is probably **not** “another sampler comparison.” It is an implementation that either **forces the second student step to do useful work**, or **deliberately partitions the trajectory between a learned jump and a classical solver tail**, because your measurements are pointing to a finite-step objective mismatch rather than a mere hyperparameter issue. Close recent work reinforces that diagnosis: **Truncated Consistency Models** (arXiv:2410.14895) argues that asking a CM to model the full trajectory can waste capacity on denoising subproblems that are not the true one-step objective, **Align Your Tangent** (arXiv:2510.00658) argues that bad tangents slow and destabilize CM training, and **Dual-End Consistency Model** (arXiv:2602.10764) argues that trajectory selection itself is a core failure mode in CM training. citeturn40search2turn40search3turn27view0turn26view0

## Asymmetric tiny refiner for a meaningful second step

**Technical description.** The cleanest project is to make the second step asymmetric on purpose. Keep your existing ECT-style student \(f_\theta(x_\sigma,\sigma)\) as the first-pass predictor of \(x_0\), but replace the second full CM pass with a **small residual refiner** \(r_\psi\) that only learns the correction the first pass failed to make:
\[
\hat x_0^{(1)} = f_\theta(x_{\sigma_1}, \sigma_1), \qquad
\Delta^{(2)} = r_\psi(\hat x_0^{(1)}+\sigma_2\epsilon,\ \hat x_0^{(1)},\ \sigma_2), \qquad
\hat x_0^{(2)} = \hat x_0^{(1)} + \Delta^{(2)}.
\]
Train with a gain-aware objective:
\[
L = D(\hat x_0^{(1)}, x_0^\phi) + \lambda D(\hat x_0^{(2)}, x_0^\phi)
+ \mu \max\!\Big(0,\ m - \big[D(\hat x_0^{(1)},x_0^\phi)-D(\hat x_0^{(2)},x_0^\phi)\big]\Big),
\]
where \(x_0^\phi\) is a teacher target from the frozen diffusion model or a solver target, \(D\) is Pseudo-Huber or L2, and \(m>0\) is a margin that explicitly rewards the second step only when it actually improves the sample. The key is that the second module is **not another copy of the same function**. It is a dedicated “error corrector.” That is exactly what your measurements say is missing. The implementation can be a small U-Net or 6–10 residual blocks at 32×32, conditioned on \(\sigma_2\) and optionally on \(\hat x_0^{(1)}-x_{\sigma_2}\). citeturn35view0turn12view0turn30view0

**Why your data points here.** Your empirical shock is that ECT 1-step and 2-step FID are basically tied from 1000 kimg onward. That means either the second step is redundant, or the model class/training target gives the second step no role. Published CM work usually does see benefit from extra steps, so the natural experiment is to ask whether a **different second-step parameterization** revives that benefit. This idea is not generic; it is a direct response to your very specific 1-step \(\approx\) 2-step collapse. citeturn34view0turn35view0turn12view0

**What you implement vs reuse.** Reuse your existing ECT training pipeline, dataset loader, checkpoint handling, and first-pass SongUNet. Freeze or lightly fine-tune \(f_\theta\). Add one new module `tiny_refiner.py`, modify the training loop to compute both \(\hat x_0^{(1)}\) and \(\hat x_0^{(2)}\), and add a custom evaluator that reports 1-step and “1+refiner” FID. The new code is real algorithmic work: new architecture, new loss, new sampler path. You are not just calling an existing script with another flag.

**Expected outcome.** If it works, the most plausible target is not “beat Heun-18 in FID.” It is “turn the currently useless second step into a real Pareto move.” Concretely, I would expect a result around the same 1-step latency ballpark plus a modest increment, with 2-step FID improving by perhaps 0.2–0.4 relative to your current ECT-2, enough to create a visibly better speed-quality point. If it fails, that failure is still informative: it would suggest the issue is not second-step parameterization but a deeper mismatch in the ECT objective itself.

**Compute cost.** This is one of the cheapest proposals. A small refiner on CIFAR-10 should fit comfortably under your 3-hour single-run budget; with your measured throughput, a 1000–1500 kimg run is very realistic. Inference cost should be much closer to ECT 1-step than to ECT 2-step, because the second pass is now tiny rather than a full second SongUNet.

**Novelty.** I did not find this exact asymmetric “full first jump + tiny dedicated residual second step” in ECT, EDM, original CM, iCT, multistep CM, sCM, or DE-CM. **Multistep Consistency Models** increases step budget, but it does not make the second step an explicitly distinct lightweight corrector. **Align Your Tangent** changes the training loss geometry; it does not create an asymmetric step-specific refiner. **Dual-End Consistency Model** changes trajectory selection and boundary regularization; it also does not propose this architecture. citeturn12view0turn27view0turn26view0

**Risk and failure modes.** The obvious failure is that the residual to be corrected is not simple enough for a tiny module; then the refiner either learns almost nothing or overfits to artifacts. The second risk is target choice: if the teacher target for the second step is noisy or solver-dependent, training may become unstable. But among all ideas here, this one gives the cleanest “May implementation in five weeks” story.

## Prefix consistency tuning with a diffusion tail handoff

**Technical description.** The strongest conceptual idea is to stop pretending the student should own the whole path. Train a **prefix-only student** that only handles the hard, high-noise part of the trajectory, then hand off to a frozen classical solver for the low-noise tail. Let \(\sigma_c\) be a cutoff. Sample \(\sigma \sim p(\sigma \mid \sigma \ge \sigma_c)\), build \(x_\sigma=x_0+\sigma\epsilon\), and train the student with a cutoff-state matching loss:
\[
L_{\text{prefix}} =
\lambda_1 D(f_\theta(x_\sigma,\sigma), x_0^\phi)
+ \lambda_2 \left\|f_\theta(x_\sigma,\sigma)+\sigma_c\epsilon - x_{\sigma_c}^\phi \right\|_2^2,
\]
where \(x_{\sigma_c}^\phi\) is the teacher’s state after integrating from \(\sigma\) down to \(\sigma_c\). At inference:
```text
ξ ~ N(0,I)
x_{σmax} = σmax ξ
x̂0 = fθ(x_{σmax}, σmax)
x̂_{σc} = x̂0 + σc ξ
return TailSamplerφ(x̂_{σc}; σc → 0, Ktail steps)
```
The idea is that the student learns the **one big jump** the teacher is bad at compressing to 5–10 steps, while the teacher retains the precise low-noise cleanup it already does better than your ECT student. This is the most direct response to your observation that the student’s second step is wasted but the teacher’s extra steps still help a lot. citeturn29view0turn19search6turn8search4

**Why your data points here.** Your data shows three things simultaneously: Heun-5 is catastrophic, Heun-10 is already close to final ECT, and Heun-18 still beats ECT on FID. That pattern suggests the expensive part is not “all steps equally”; it is likely the **coarse high-noise jump plus low-noise polishing**. Your current ECT result hints that the student learned some coarse jump quickly but never learned a useful refinement phase. So let the student do the coarse jump and let the solver keep the polish.

**What you implement vs reuse.** Reuse the frozen EDM teacher and your existing ECT codebase. New code sits in two places: a modified training objective that samples only \(\sigma \ge \sigma_c\) and matches the cutoff state, and a hybrid sampler that reconstructs \(x_{\sigma_c}\) from the student prediction and then launches a short Heun/DPM-Solver-v3/Restart tail. That is new algorithmic code in the loss and the sampler.

**Expected outcome.** If this works, it has the best chance to produce a point that is genuinely new on your plot: something like “much closer to Heun-18 quality than ECT-2, but with latency far below Heun-18.” I would not promise dominance, but this is the idea with the clearest path to a new frontier rather than a small local improvement. If it fails, the likely failure mode is **handoff mismatch**: the tail solver may not recover well from states that are student-generated rather than teacher-generated.

**Compute cost.** Moderate. More expensive than the tiny refiner, because the training target includes cutoff-state matching and the sampler evaluation includes a teacher tail. Still feasible, because CIFAR-10 is tiny and the student only learns a truncated band of sigmas rather than the full range.

**Novelty.** This idea is close to recent work, so the novelty argument must be stated honestly. **Truncated Consistency Models** (arXiv:2410.14895) already argues that full-trajectory CM training can waste capacity and proposes truncated-time training. **Dual-End Consistency Model** (arXiv:2602.10764) also treats trajectory selection as central. So this idea is **not novel** if you pitch it merely as “truncate the CM time range.” It becomes novel if you pitch it as a **hybrid deployment algorithm** with explicit cutoff-state matching and a frozen solver tail, because TCM is still a standalone CM approach, while this proposal is explicitly about **which part of the path should belong to the student and which should belong to the teacher** for your ECT-vs-Heun question. citeturn40search2turn40search3turn26view0

**Risk and failure modes.** The main risk is that the reconstructed \(x_{\sigma_c}\) is off-manifold in a way the teacher tail cannot repair. The second risk is TA perception: if you frame it carelessly, it can sound too close to TCM. The framing matters. The contribution is not “truncate time.” It is “learn a prefix because your data says the full-path student is using the second step badly.”

## Fast-teacher consistency tuning

**Technical description.** Right now the practical comparison is “ECT student versus Heun on the original diffusion model.” A sharper project is to distill **the best training-free fast teacher you can practically run**, not just the raw diffusion endpoint. Let \(S_\phi^{\text{fast}}\) be a solver such as DPM-Solver-v3, UniPC, or Restart with a small NFE budget. Train the student against the fast teacher’s terminal output:
\[
x_{0,\text{fast}}^\phi = S_\phi^{\text{fast}}(x_\sigma; \tau_{\text{fast}}),
\qquad
L_{\text{FT}} = D(f_\theta(x_\sigma,\sigma), \text{sg}(x_{0,\text{fast}}^\phi)) + \beta L_{\text{ECT}}.
\]
A two-step variant lets the first student pass mimic the fast teacher’s coarse solution and the second pass mimic its tail-corrected solution. The conceptual move is simple: if your real applied question is “when does tuning beat a strong fast sampler,” then the student should be trained to approximate that strong fast sampler, not just the original diffusion fixed point.

**Why your data points here.** This idea is justified by the fact that Heun is not actually the only or even the best low-NFE training-free baseline anymore. **DPM-Solver-v3** reports 2.51 FID on unconditional CIFAR-10 at 10 NFE, which is right at your final ECT level, and **Restart Sampling** and **UniPC** were explicitly developed because naive ODE solvers plateau too early in the few-step regime. If your class project ignores those solvers, you risk answering the wrong version of your own research question. citeturn8search4turn29view0turn29view1

**What you implement vs reuse.** Reuse the existing EDM checkpoint and most of the ECT training code. New work is in integrating a fast teacher path into training targets and optionally caching teacher outputs for a subset of sigmas. If you choose DPM-Solver-v3 as teacher, you also need to integrate its solver routine or a faithful minimal reimplementation in your training pipeline. That is legitimate algorithmic implementation work.

**Expected outcome.** If it works, the likely result is not magical 1-step FID. The likely result is that a 1-step or 2-step student closes more of the gap to a **modern** fast-solver baseline than your current ECT does. On a plot, the success story is: the student no longer merely beats Heun-per-latency; it starts to approximate the best practical training-free sampler in your regime.

**Compute cost.** Moderate to high, depending on whether teacher outputs are online or cached. For CIFAR-10 it is still doable. A practical compromise is to precompute teacher targets for a limited sigma grid and interpolate targets during training.

**Novelty.** I did not find an exact “solver-aware ECT” in the cited consistency literature. Existing CM and ECT formulations distill from diffusion trajectories or consistency relations, not from the output of a modern **solver-specific** fast teacher chosen to match the applied comparator. That said, this idea is closest in spirit to general distillation work such as **On Distillation of Guided Diffusion Models** (arXiv:2210.03142) and **Diff-Instruct** (arXiv:2305.18455), so it is novel mainly in the **target definition** and the very specific ECT-vs-fast-solver framing, not in the broad concept of distillation itself. citeturn15search0turn17view0

**Risk and failure modes.** The biggest risk is that solver-specific teacher outputs are too tied to one schedule and do not generalize across the student’s input distribution. The second risk is engineering overhead: DPM-Solver-v3 integration is not hard in principle, but it is more code than a pure refiner idea.

## Consistency-defect gated adaptive compute

**Technical description.** Your current comparison assumes fixed compute per sample. That is probably suboptimal. Define a **self-defect** score for the 1-step student:
\[
\delta(x_{\sigma_1}) =
\left\| f_\theta(x_{\sigma_1},\sigma_1) - f_\theta\!\big(f_\theta(x_{\sigma_1},\sigma_1)+\sigma_2\epsilon,\sigma_2\big)\right\|_2.
\]
Then build a three-level sampler:
```text
y1 = fθ(xσ1, σ1)
if δ(y1) < τ1: return y1
elif δ(y1) < τ2: return tiny_refiner(y1)
else: return diffusion_tail(y1)
```
A better implementation is to train a small predictor \(g_\psi\) on top of student features to estimate whether “extra compute will produce a meaningful teacher-proxy improvement.” The target label can be generated offline by checking whether a second step or tail actually reduces a surrogate error on a held-out calibration set.

**Why your data points here.** This is motivated by your break-even calculation more than by your FIDs. You already showed that the amortization point depends hugely on deployment volume and batch size. That means inference compute is not just an evaluation axis; it is the **economic axis** of the whole question. If many samples are “easy,” a variable-compute student could shift the average latency materially below any fixed 2-step policy while keeping most of the quality benefit.

**What you implement vs reuse.** Reuse the 1-step student and whichever improvement mechanism you choose, ideally the asymmetric tiny refiner or the hybrid diffusion tail. New code lives in a calibrator script, a gating head or defect heuristic, and a variable-compute sampler/evaluator that reports **average NFE**, average latency, and FID under the mixed policy.

**Expected outcome.** If it works, the plot changes from a few static points to a **curve**: same base model, but a family of policies indexed by \((\tau_1,\tau_2)\). That is scientifically interesting because it makes “when does tuning pay off?” sample-dependent rather than globally fixed. If it fails, the likely reason is that the self-defect is not predictive of final sample quality, which is itself a publishable negative result in miniature.

**Compute cost.** Moderate. Calibration adds some overhead, but training a gate is cheap. Runtime overhead is tiny if you use the raw defect instead of a learned gate.

**Novelty.** Adaptive-step and adaptive-region methods exist in diffusion more broadly, but I did not find a CM-specific **self-defect gate that arbitrates among 1-step CM, cheap neural refinement, and teacher tail sampling** in the consistency literature you asked me to trace through early 2026. So the novelty claim is reasonable, but weaker than for the top two ideas because “adaptive compute” is a general theme across efficient generation.

**Risk and failure modes.** The most likely failure is poor correlation between \(\delta\) and perceptual/sample-distribution error. Also, variable-compute FID experiments are harder to explain cleanly in a class report than a fixed 2-step method.

## Sigma-banded adapter tuning to attack the real amortization problem

**Technical description.** Your experiments show that “cheap” consistency tuning is not actually cheap on the hardware you will use. So one extension should attack the tuning cost directly. Freeze the pretrained diffusion backbone and insert trainable low-rank or FiLM-style adapters that are activated only in specific sigma bands:
\[
h_\ell' = h_\ell + g_b(\log \sigma)\, B_\ell(A_\ell h_\ell), \qquad b \in \{\text{high},\text{mid},\text{low}\},
\]
where \(A_\ell, B_\ell\) are low-rank trainable matrices or 1×1 bottlenecks and \(g_b\) is a learned gate over sigma bands. Train only the adapters under the ECT loss or one of the improved losses above. The sigma bands are the key novelty: you are not just doing generic LoRA; you are testing whether the student really needs *different* small corrections in high-noise versus low-noise regions.

**Why your data points here.** Your own break-even numbers and observed hardware dependence are telling you that tuning cost itself is one of the main scientific variables. A method that preserves most of final ECT quality while cutting tuning cost by 3–10× would change the answer to your original question more than a tiny FID improvement would. This is especially true because your measurements suggest diminishing returns after about 1000 kimg.

**What you implement vs reuse.** Reuse the full pretrained SongUNet and all training infrastructure. Insert adapters into the highest-leverage residual blocks only, expose sigma-band gating, and train only those parameters. Add an ablation over which bands are active. This is not just a hyperparameter sweep; it is a new parameterization of the tuning problem.

**Expected outcome.** The success case is not “best FID.” It is “almost the same FID, much cheaper tuning, dramatically better amortization.” On your project’s own terms, that would be a meaningful result. The failure case is also informative: if adapter-only tuning cannot recover ECT, then the student really does need broad weight movement.

**Compute cost.** Lowest training cost of any idea except maybe the defect gate. This is the easiest path to many seeds and multiple ablations under your five-week budget.

**Novelty.** Here the honesty matters again. There is relevant prior art. **Latent Consistency Models** (arXiv:2310.04378) and especially **LCM-LoRA** (arXiv:2311.05556) already show that consistency-style acceleration can be transported through low-rank modules in latent Stable Diffusion. So “adapterized consistency acceleration” is not new in the broad sense. The narrower claim that still looks defensible is: I did not find **sigma-banded adapter tuning for pixel-space ECT on a pretrained EDM checkpoint**, nor a framing where the scientific target is amortization cost on small hardware rather than SDXL-style deployment. That makes this a weaker novelty pitch than the top two, but a very strong practicality pitch. citeturn36view0turn36view1

**Risk and failure modes.** The adapter capacity may simply be too small. Also, because this is closer to existing LCM-LoRA intuition, a TA could read it as “good engineering, less exciting science” unless you frame the sigma-band decomposition and the break-even motivation very clearly.

## Bilevel schedule and handoff search

**Technical description.** The final idea is the most “sampler algorithm” of the set. Parameterize a hybrid schedule \(\tau = (\sigma_c, \sigma_1,\dots,\sigma_K)\), where \(\sigma_c\) is the student-to-teacher handoff point and \(\sigma_1>\cdots>\sigma_K\) are the teacher tail nodes. Optimize
\[
\min_{\tau}\; J(\tau)
=
\underbrace{\mathbb{E}_\xi \left\|S_{\theta,\phi}^{\tau}(\xi)-S_\phi^{35}(\xi)\right\|_2^2}_{\text{teacher-surrogate discrepancy}}
+
\alpha \cdot \text{latency}(\tau),
\]
or, if you can afford it, replace the surrogate with a 5k-sample FD-DINO/FID proxy. Use SPSA, CMA-ES, or a constrained coordinate search over the monotone schedule simplex. The point is not to “try a few timesteps.” The point is to implement a real optimizer over the student-teacher decomposition.

**Why your data points here.** The Heun 5-step to 10-step cliff in your runs screams schedule sensitivity. That is exactly the regime where **Align Your Steps** and **Learning Fast Samplers for Diffusion Models by Differentiating Through Sample Quality** argue that fixed hand-crafted schedules are leaving performance on the table. If your project remains locked to the default EDM schedule, it may never reveal the true hybrid frontier. citeturn19search6turn21view0

**What you implement vs reuse.** Reuse whichever sampler family you choose, ideally the prefix-handoff or fast-teacher variant. New code is an optimizer over monotone schedules and a proxy evaluator. The implementation is real, but it sits more on the sampler-design side than the model-design side.

**Expected outcome.** Best-case, you find that a carefully chosen cutoff and low-noise grid gives a nontrivial gain over obvious schedules at fixed latency. Worst-case, you get only a small improvement, but you still learn whether your current frontier is schedule-limited.

**Compute cost.** Potentially expensive if you use FID-in-the-loop. The feasible version is to optimize against a surrogate discrepancy to a strong teacher sampler, then validate only a handful of schedules with full FID.

**Novelty.** This is the weakest novelty case by itself, because **AYS** and **DDSS** already establish schedule optimization and sampler search as real research directions. Your differentiator would be the **joint optimization of a CM handoff point plus a teacher tail schedule on the exact same pretrained EDM checkpoint**. That is specific and useful, but less fundamental than the architecture/loss ideas above. citeturn19search6turn21view0

**Risk and failure modes.** The obvious risk is TA rejection: this is the idea most likely to be interpreted as “still just running code in a new way.” I would only keep it as a secondary ablation or as a component inside the prefix-handoff project, not as the headline project by itself.

## Ranking and recommendation

**Most likely to produce a strong course project:**  
First, **Asymmetric tiny refiner**. It is the best blend of novelty, direct traceability to your own measurements, low implementation risk, and clean success/fail interpretation. The premise is simple and strong: your current second step doubles latency and buys nothing, so redesign the second step itself. That is easy to explain to the TA and easy to implement in five weeks.

Second, **Prefix consistency tuning with a diffusion tail handoff**. This is the most conceptually powerful idea, and it is the one I would personally bet has the highest upside on the FID-versus-latency plot. But it is also closer to recent trajectory-selection papers such as **Truncated Consistency Models** and **Dual-End Consistency Model**, so the framing must be much sharper. The novelty is in the **hybrid deployment boundary**, not merely in truncation. citeturn40search2turn40search3turn26view0

Third, **Fast-teacher consistency tuning**. This is the idea that best protects you from a blind spot in your current baseline choice: Heun is not the whole training-free story, and DPM-Solver-v3 already narrows the gap materially. A student distilled toward a modern fast solver is a much stronger answer to the actual question “when does tuning pay off?” than a student distilled toward a generic diffusion endpoint. citeturn8search4turn29view0turn29view1

Fourth, **Sigma-banded adapter tuning**. If the TA is receptive to a training-loop/parameterization contribution motivated by your amortization findings, this is genuinely practical and very feasible. I rank it below the top three only because the novelty story is weaker due to **LCM-LoRA**-style prior work. citeturn36view0turn36view1

Fifth, **Consistency-defect gated adaptive compute**. This is interesting and could produce a compelling systems-style story, but it is harder to evaluate and easier to get ambiguous results from.

Sixth, **Bilevel schedule and handoff search**. Useful as a subroutine or ablation, weak as the main project.

**Most interesting research regardless of completion risk:**  
That is **Prefix consistency tuning with a diffusion tail handoff**. The deepest question your data raises is whether a tiny student should ever be expected to replace the full solver uniformly across the entire PF-ODE trajectory. Your results suggest maybe not. If that hypothesis is right, then the “right” fast generator is not a standalone CM or a standalone solver, but a **trajectory-partitioned system**. That is a real research question, not just a course-project patch.

**The blind spot I think you may not have articulated yet:**  
Your real anomaly is not “ECT sometimes loses to Heun.” It is this: **your student stopped benefiting from extra steps long before the teacher solver stopped benefiting from extra steps**. That points to a finite-step supervision problem. Separately, if you continue to use only Heun as the training-free comparator, you may end up overestimating how much the tuned student is really buying, because the modern few-step solver frontier already includes DPM-Solver-v3, UniPC, Restart, AYS-style schedule optimization, and related methods. If you want the strongest final report, define your extension around that sharper statement: *the small student has collapsed into a one-step projector, and the right project is to give the unused second-step budget an explicit job*. citeturn8search4turn19search6turn29view0turn29view1turn12view0turn28view0

**If I had to recommend one project today:** choose **Asymmetric tiny refiner** as the main implementation, and keep **Prefix handoff tuning** as the “ambitious follow-on” or backup if the refiner stabilizes quickly. That pairing is coherent: the first asks whether the second step can be made useful at all; the second asks whether the second step should belong to the student in the first place.