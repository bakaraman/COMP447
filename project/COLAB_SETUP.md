# Colab Setup

## Short answer

Yes, we can run the project on Colab.

The cleanest setup is:

- keep the code in this repo
- move the repo to Colab by either:
  - cloning from GitHub
  - or uploading a zip / copying from Google Drive
- keep **results and checkpoints** on Google Drive so they do not disappear when the Colab session resets

## Recommended workflow

### Best option: GitHub for code, Drive for outputs

This is the least messy setup.

Why:

- code version stays clean
- both teammates can use the same repo
- Colab sessions can be recreated fast
- large outputs do not bloat the code repo

Flow:

1. Put the repo on GitHub, ideally private
2. In Colab, clone it
3. Mount Google Drive
4. Save checkpoints and outputs into Drive

## Option A: GitHub clone in Colab

### 1. Open Colab with GPU

In Colab:

- `Runtime` -> `Change runtime type`
- choose `T4 GPU`

### 2. Mount Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 3. Clone the repo

```bash
%cd /content
!git clone <YOUR_REPO_URL> COMP447
%cd /content/COMP447
```

If the repo is private, either:

- use a GitHub token
- or just upload a zip instead

### 4. Install packages

Colab already has PyTorch, so start simple and install the missing pieces first.

```bash
!pip install -q psutil click requests pillow numpy scipy tqdm imageio imageio-ffmpeg pyspng diffusers==0.26.3 accelerate==0.27.2
```

### 5. Verify runtime

```python
import torch
print(torch.__version__)
print(torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
```

### 6. Prepare CIFAR-10

```bash
%cd /content/COMP447/project/src/ect
!mkdir -p datasets
!wget -O /content/cifar-10-python.tar.gz https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
!python3 dataset_tool.py --source=/content/cifar-10-python.tar.gz --dest=datasets/cifar10-32x32.zip
```

### 7. Run the first sanity check

```bash
%cd /content/COMP447/project/src/ect
!bash run_ecm_1hour.sh 1 29500 --desc monday_sanity
```

### 8. Save outputs to Drive

If you want persistent results:

```bash
!mkdir -p /content/drive/MyDrive/COMP447_runs
!cp -r /content/COMP447/project/src/ect/ct-runs /content/drive/MyDrive/COMP447_runs/
```

Better yet, once things work, directly point outputs into Drive-oriented paths.

## Option B: No GitHub, move repo by zip

If you do not want to push to GitHub right now, this is the fastest fallback.

### Local machine

Zip the repo:

```bash
cd /Users/batuhankaraman/Downloads
zip -r COMP447.zip COMP447
```

Upload `COMP447.zip` to Google Drive.

### In Colab

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
%cd /content
!cp /content/drive/MyDrive/COMP447.zip .
!unzip -q COMP447.zip
%cd /content/COMP447
```

Then continue with the same install + dataset + run steps above.

## Option C: Connect Colab to local runtime

This exists in principle, but for this project it is **not the right move**.

Why not:

- local runtime uses **your own machine**
- your machine is **Apple Silicon**, not NVIDIA CUDA
- your project comparison is supposed to be on **T4-like hardware**
- latency numbers from local runtime would not match the project story

So yes, local runtime is possible in some setups, but it defeats the main purpose here.

For this project:

- use **Colab GPU** for the real experiments
- use your local machine for editing, planning, and light smoke tests

## Practical recommendation for your team

If the goal is to move fast before Monday:

- tonight: keep editing locally in this repo
- before the first real run: either
  - push repo to GitHub and clone in Colab
  - or zip the repo and unzip it in Colab
- store outputs on Google Drive

## What is realistic before Monday

Before Monday, the target is:

- Colab runtime opens
- repo is available in Colab
- packages install cleanly
- CIFAR-10 converts to `cifar10-32x32.zip`
- one ECT sanity command is ready to launch

That is already enough to make the project feel real in the meeting.

## Subject to change

- package versions may need small fixes depending on the exact Colab image
- if the ECT training command is too heavy, we may first do a lighter dry run
- if GitHub private auth is annoying, zip-to-Drive is a perfectly fine short-term fallback
