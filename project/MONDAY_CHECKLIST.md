# Monday Checklist

## Goal

By the first Andrew meeting, the project should feel concrete, not fuzzy.

That means we do **not** need final results yet, but we **do** need:

- a precise experimental framing
- a clear explanation of the extension angle
- the actual upstream repos in place
- a runnable setup path for ECT and EDM
- a concrete plan for the first pilot measurements

## What we already fixed

- ECT repo cloned into `project/src/ect/`
- EDM repo cloned into `project/src/edm/`
- local latency script now supports:
  - `ECT` few-step timing
  - `EDM / Heun-style` timing

## What to do before Monday

### 1. Confirm environment path

Choose the runtime for actual experiments:

- preferred: Colab T4
- fallback: any CUDA Linux box

Minimum environment packages from ECT:

- `torch`
- `numpy`
- `scipy`
- `click`
- `pillow`
- `requests`
- `psutil`
- `tqdm`
- `imageio`
- `imageio-ffmpeg`
- `pyspng`
- `diffusers`
- `accelerate`

### 2. Prepare CIFAR-10 in EDM / ECT format

Need:

- raw `cifar-10-python.tar.gz`
- converted dataset zip: `cifar10-32x32.zip`

Reference conversion command from the upstream repos:

```bash
cd project/src/ect
python3 dataset_tool.py --source=/path/to/cifar-10-python.tar.gz --dest=datasets/cifar10-32x32.zip
```

### 3. Dry-run ECT training config

Target command family:

```bash
cd project/src/ect
bash run_ecm_1hour.sh 1 29500 --desc monday_sanity
```

This may still need local adjustments depending on the runtime.

### 4. Confirm ECT evaluation path

Official evaluation command family:

```bash
cd project/src/ect
bash eval_ecm.sh 1 29501 --resume /path/to/network-snapshot.pkl
```

Local wrapper alternative:

```bash
python3 project/scripts/eval_fid.py ect \
  --checkpoint /path/to/network-snapshot.pkl \
  --data /path/to/cifar10-32x32.zip
```

### 5. Confirm Heun / EDM baseline path

For the baseline, use the official EDM sampler on the same pretrained EDM checkpoint family.

Reference command family:

```bash
cd project/src/edm
python3 generate.py --outdir=out --steps=18 \
  --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/baseline/baseline-cifar10-32x32-uncond-vp.pkl
```

And for FID:

```bash
cd project/src/edm
torchrun --standalone --nproc_per_node=1 fid.py calc \
  --images=fid-tmp \
  --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz
```

Local wrapper alternative:

```bash
python3 project/scripts/eval_fid.py images --images /path/to/generated_images
```

### 6. Run the first latency pilots

After the environment works, measure:

- ECT `1-step`, batch `1`
- ECT `2-step`, batch `1`
- Heun baseline, a few step counts, batch `1`
- then repeat for batch `64`

Local wrapper command:

```bash
python3 project/scripts/measure_latency.py \
  --checkpoint /path/to/checkpoint.pkl \
  --sampler ect \
  --steps 2 \
  --batch_size 1 \
  --num_runs 100 \
  --warmup 20
```

## What to tell Andrew

Short version:

> We are not doing pure reproduction. We compare ECT against Heun under matched latency budgets on a T4, then measure break-even generation count and tuning-budget ablations to study when consistency tuning is practically worthwhile.

## Questions to ask Andrew

- Is `latency-matched comparison + break-even + tuning-budget ablation` strong enough as the project extension?
- Should the Heun baseline use the official EDM deterministic sampler as the main baseline throughout?
- For the progress report, should we plan around the earlier date if the course communication stays inconsistent?

## Subject to change

- exact Heun step counts
- exact latency targets
- whether full ECT tuning fits in Colab/T4 comfortably
- whether we keep only `2-step` tuning ablation or include `1-step` there too
