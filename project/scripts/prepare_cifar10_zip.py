"""
Prepare a local CIFAR-10 image zip for diagnostics.

The on-policy gap diagnostic needs real CIFAR images. Relying on torchvision's
original Toronto URL is brittle in Colab; it failed with HTTP 503 during our
run. This helper uses Hugging Face's uoft-cs/cifar10 dataset and writes a plain
image zip that our diagnostic can read without any further network dependency.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import zipfile
from io import BytesIO
from pathlib import Path


def ensure_datasets_importable() -> None:
    try:
        import datasets  # noqa: F401
    except Exception:
        print("Installing Hugging Face datasets...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "datasets", "pillow"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default="/content/drive/MyDrive/COMP447/cifar10-32x32.zip",
        help="Output zip path.",
    )
    parser.add_argument("--num", type=int, default=50000, help="Number of training images to write.")
    args = parser.parse_args()

    out = Path(args.out).resolve()
    if out.exists():
        print(f"Dataset already exists: {out} ({out.stat().st_size / 1e6:.1f} MB)")
        return

    out.parent.mkdir(parents=True, exist_ok=True)
    ensure_datasets_importable()

    from datasets import load_dataset

    print("Loading uoft-cs/cifar10 train split from Hugging Face...")
    ds = load_dataset("uoft-cs/cifar10", split="train")
    n = min(args.num, len(ds))
    labels = []

    print(f"Writing {n} images to {out}")
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_STORED) as z:
        for i in range(n):
            row = ds[i]
            img = row.get("img", row.get("image"))
            if img is None:
                raise KeyError(f"Could not find image column in row keys: {list(row.keys())}")
            label = int(row["label"])
            fname = f"{i // 1000:05d}/img{i:08d}.png"
            buf = BytesIO()
            img.convert("RGB").save(buf, format="PNG")
            z.writestr(fname, buf.getvalue())
            labels.append([fname, label])
            if (i + 1) % 5000 == 0 or i + 1 == n:
                print(f"  wrote {i + 1}/{n}")
        z.writestr("dataset.json", json.dumps({"labels": labels}))

    print(f"Saved dataset: {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
