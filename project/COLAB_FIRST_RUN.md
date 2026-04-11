# Colab First Run

This is the shortest path to get the project running in a browser Colab notebook.

## Before you open Colab

If the repo is **not** on GitHub yet, the fastest path is:

1. Zip the repo locally
2. Upload the zip to Google Drive
3. Unzip it inside Colab

Local command:

```bash
cd /Users/batuhankaraman/Downloads
zip -r COMP447.zip COMP447
```

Then upload `COMP447.zip` to `MyDrive`.

## In Colab

Set runtime to:

- `T4 GPU`

## Cell 1: Mount Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

## Cell 2: Copy and unzip the repo

If you uploaded `COMP447.zip` to Drive:

```bash
%cd /content
!cp /content/drive/MyDrive/COMP447.zip .
!rm -rf /content/COMP447
!unzip -q COMP447.zip
%cd /content/COMP447
!ls
```

Expected: you should see folders like `project`, `final_upload`, `readings`, `proposal_template`.

## Cell 3: Check GPU

```python
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
```

Expected: `cuda available: True` and ideally a `T4`.

## Cell 4: Install missing packages

```bash
%cd /content/COMP447
!pip install -q psutil click requests pillow numpy scipy tqdm imageio imageio-ffmpeg pyspng diffusers==0.26.3 accelerate==0.27.2
```

## Cell 5: Verify the upstream repos are present

We already cloned them locally, so they should come from the zip.

```bash
%cd /content/COMP447
!ls project/src
!ls project/src/ect | head
!ls project/src/edm | head
```

Expected: both `ect` and `edm` should exist.

## Cell 6: Prepare CIFAR-10

```bash
%cd /content/COMP447/project/src/ect
!mkdir -p datasets
!wget -nc -O /content/cifar-10-python.tar.gz https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
!python3 dataset_tool.py --source=/content/cifar-10-python.tar.gz --dest=datasets/cifar10-32x32.zip
!ls -lh datasets
```

Expected: `datasets/cifar10-32x32.zip`

## Cell 7: Quick import smoke test

```bash
%cd /content/COMP447/project/src/ect
!python3 - <<'PY'
import psutil
import torch
print("psutil ok")
print("torch ok")
print("cuda:", torch.cuda.is_available())
PY
```

## Cell 8: Dry-run the ECT command

This does **not** fully train. It just makes sure the command line is sane enough to proceed.

```bash
%cd /content/COMP447/project/src/ect
!python3 ct_train.py \
  --outdir ct-runs-dry \
  --data datasets/cifar10-32x32.zip \
  --cond=0 \
  --arch=ddpmpp \
  --metrics=fid50k_full \
  --transfer=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-vp.pkl \
  --duration=0.001 \
  --batch=8 \
  --dry_run
```

If this works, the setup is basically alive.

## Cell 9: First real sanity run

After the dry run succeeds:

```bash
%cd /content/COMP447/project/src/ect
!bash run_ecm_1hour.sh 1 29500 --desc monday_sanity
```

This may take time. If it crashes, send the full output.

## What to send back

After each stage, the most useful outputs to send are:

- Cell 3 GPU output
- Cell 5 repo listing output
- Cell 6 dataset creation output
- Cell 8 dry-run output
- Cell 9 error or first training logs

## Immediate goal

For now, success means:

- Colab sees the repo
- CUDA is available
- CIFAR-10 is converted
- ECT dry-run works

That is already enough progress for the Monday meeting.
