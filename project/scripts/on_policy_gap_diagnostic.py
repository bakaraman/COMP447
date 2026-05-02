"""
Measure whether ECT sees a different distribution at training vs inference.

ECT trains on noisy real images:             y + eps * t
2-step inference denoises noisy model states: x_1 + eps * t

This diagnostic measures both endpoint residuals for each checkpoint and mid_t:

  R_data(t)  = E[ mean((f(y + eps*t, t)   - y)^2) ]
  R_model(t) = E[ mean((f(x1 + eps*t, t) - x1)^2) ]
  gap_ratio  = R_model(t) / R_data(t)

where x1 = f(z*80, 80) is the model's own 1-step sample.

If gap_ratio changes with training stage and is related to the FID-optimal
mid_t, then the project has a more structural bottleneck than "0.821 is a bad
constant": the training objective and inference path are not sampling from the
same support.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import pickle
import sys
import tarfile
import zipfile
from pathlib import Path

import torch
import numpy as np
from PIL import Image


DEFAULT_MID_TS = [0.1, 0.3, 0.5, 0.7, 0.821, 1.0, 1.5, 2.5]
DEFAULT_SNAPS = ["050", "100", "150", "198"]


def add_paths(repo_root: Path) -> None:
    scripts_root = repo_root / "project" / "scripts"
    ect_root = repo_root / "project" / "src" / "ect"
    for p in [str(scripts_root), str(ect_root)]:
        if p not in sys.path:
            sys.path.insert(0, p)


def is_image_name(name: str) -> bool:
    return name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))


def normalize_uint8_images(images: np.ndarray) -> torch.Tensor:
    if images.ndim == 4 and images.shape[-1] == 3:
        images = images.transpose(0, 3, 1, 2)
    tensor = torch.from_numpy(images.astype(np.float32))
    return tensor / 127.5 - 1.0


def load_from_cifar_tar(path: Path, num: int) -> torch.Tensor:
    images = []
    with tarfile.open(path, "r:gz") as tar:
        for batch in range(1, 6):
            member = tar.getmember(f"cifar-10-batches-py/data_batch_{batch}")
            with tar.extractfile(member) as f:
                data = pickle.load(f, encoding="latin1")
            batch_images = data["data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
            images.append(batch_images)
            if sum(x.shape[0] for x in images) >= num:
                break
    arr = np.concatenate(images, axis=0)[:num]
    return normalize_uint8_images(arr)


def load_from_image_zip(path: Path, num: int) -> torch.Tensor:
    images = []
    with zipfile.ZipFile(path, "r") as z:
        names = [name for name in sorted(z.namelist()) if is_image_name(name)]
        if not names:
            raise RuntimeError(f"No image files found in zip: {path}")
        for name in names[:num]:
            with z.open(name) as f:
                img = Image.open(f).convert("RGB")
                images.append(np.array(img))
    arr = np.stack(images, axis=0)
    return normalize_uint8_images(arr)


def find_dataset(repo_root: Path, explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit).expanduser()
        if path.exists():
            return path.resolve()
        raise FileNotFoundError(f"Explicit dataset path does not exist: {path}")

    candidates = [
        repo_root / "project" / "src" / "ect" / "datasets" / "cifar10-32x32.zip",
        repo_root / "project" / "src" / "edm" / "datasets" / "cifar10-32x32.zip",
        repo_root / "project" / "data_cache" / "cifar10-32x32.zip",
        repo_root / "project" / "data_cache" / "cifar-10-python.tar.gz",
        Path("/content/drive/MyDrive/COMP447/cifar10-32x32.zip"),
        Path("/content/drive/MyDrive/COMP447/cifar-10-python.tar.gz"),
        Path("/content/drive/MyDrive/COMP447_checkpoints/cifar10-32x32.zip"),
        Path("/content/drive/MyDrive/COMP447_checkpoints/cifar-10-python.tar.gz"),
    ]
    for path in candidates:
        if path.exists():
            return path.resolve()

    # Last resort: Drive glob. This avoids a fragile internet download, which
    # failed with HTTP 503 in Colab during this project.
    for root in [Path("/content/drive/MyDrive"), repo_root]:
        if not root.exists():
            continue
        for pattern in ["**/cifar10-32x32.zip", "**/cifar-10-python.tar.gz"]:
            hits = list(root.glob(pattern))
            if hits:
                return hits[0].resolve()

    raise FileNotFoundError(
        "Could not find CIFAR-10 data. Put either cifar10-32x32.zip or "
        "cifar-10-python.tar.gz in Drive, or pass --dataset /path/to/file."
    )


def load_real_images(repo_root: Path, dataset: str | None, num: int, seed: int) -> torch.Tensor:
    path = find_dataset(repo_root, dataset)
    print(f"Using dataset: {path}")
    if path.name == "cifar-10-python.tar.gz":
        images = load_from_cifar_tar(path, num=max(num, 50000))
    elif path.suffix == ".zip":
        images = load_from_image_zip(path, num=max(num, 50000))
    else:
        raise ValueError(f"Unsupported dataset format: {path}")

    if images.shape[0] < num:
        raise RuntimeError(f"Dataset has only {images.shape[0]} images, need {num}")

    generator = torch.Generator()
    generator.manual_seed(seed)
    perm = torch.randperm(images.shape[0], generator=generator)[:num]
    return images[perm].contiguous()


def iter_batches(images: torch.Tensor, batch_size: int, device: str):
    for start in range(0, images.shape[0], batch_size):
        yield images[start : start + batch_size].to(device)


def checkpoint_path(checkpoint_dir: Path, snap: str) -> Path:
    path = checkpoint_dir / f"network-snapshot-000{snap}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    return path


@torch.no_grad()
def one_step_sample(net, batch_size: int, device: str, generator: torch.Generator):
    import measure_latency

    z = torch.randn(
        batch_size,
        net.img_channels,
        net.img_resolution,
        net.img_resolution,
        device=device,
        generator=generator,
    )
    labels = measure_latency._make_class_labels(net, batch_size, device)
    return measure_latency.ect_sampler(net, z, labels, steps=1)


@torch.no_grad()
def residual_for_mid_t(
    net,
    images: torch.Tensor,
    mid_t: float,
    num: int,
    batch_size: int,
    device: str,
    seed: int,
) -> dict[str, float]:
    import measure_latency

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    t_scalar = float(mid_t)

    data_sse = 0.0
    model_sse = 0.0
    data_l1 = 0.0
    model_l1 = 0.0
    n_values = 0

    for y in iter_batches(images, batch_size, device):
        b = y.shape[0]
        labels = measure_latency._make_class_labels(net, b, device)
        t = torch.full([b, 1, 1, 1], t_scalar, device=device)
        eps = torch.randn(y.shape, device=device, generator=gen)

        x1 = one_step_sample(net, b, device, gen).to(torch.float32)

        data_out = net(y + eps * t, t, labels).to(torch.float32)
        model_out = net(x1 + eps * t, t, labels).to(torch.float32)

        data_diff = data_out - y
        model_diff = model_out - x1
        data_sse += float((data_diff ** 2).sum().detach().cpu())
        model_sse += float((model_diff ** 2).sum().detach().cpu())
        data_l1 += float(data_diff.abs().sum().detach().cpu())
        model_l1 += float(model_diff.abs().sum().detach().cpu())
        n_values += y.numel()

    data_mse = data_sse / n_values
    model_mse = model_sse / n_values
    data_l1_mean = data_l1 / n_values
    model_l1_mean = model_l1 / n_values
    return {
        "data_mse": data_mse,
        "model_mse": model_mse,
        "gap_delta": model_mse - data_mse,
        "gap_ratio": model_mse / data_mse if data_mse > 0 else math.nan,
        "data_l1": data_l1_mean,
        "model_l1": model_l1_mean,
        "l1_ratio": model_l1_mean / data_l1_mean if data_l1_mean > 0 else math.nan,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "snap_id",
                "kimg",
                "mid_t",
                "num_images",
                "data_mse",
                "model_mse",
                "gap_delta",
                "gap_ratio",
                "data_l1",
                "model_l1",
                "l1_ratio",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_results(repo_root: Path, csv_path: Path, output_png: Path) -> None:
    try:
        import pandas as pd
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"WARN: plotting skipped: {exc}")
        return

    df = pd.read_csv(csv_path)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), sharex=True)
    palette = {500: "#fee5d9", 1000: "#fcae91", 1500: "#fb6a4a", 1980: "#a83d3d"}

    for kimg, g in df.groupby("kimg"):
        g = g.sort_values("mid_t")
        color = palette.get(int(kimg), "gray")
        axes[0].plot(g["mid_t"], g["gap_ratio"], "o-", color=color, lw=2, label=f"{int(kimg)} kimg")
        axes[1].plot(g["mid_t"], g["gap_delta"], "o-", color=color, lw=2, label=f"{int(kimg)} kimg")

    for ax in axes:
        ax.axvline(0.821, color="gray", ls="--", lw=1)
        ax.set_xlabel(r"$t_{\mathrm{mid}}$")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(r"$R_{model}(t) / R_{data}(t)$")
    axes[1].set_ylabel(r"$R_{model}(t) - R_{data}(t)$")
    axes[0].set_title("On-policy gap ratio")
    axes[1].set_title("On-policy gap delta")
    axes[0].legend(loc="best", fontsize=8)
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=150)
    print(f"Saved plot: {output_png}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--checkpoint_dir", default="/content/drive/MyDrive/COMP447_checkpoints")
    parser.add_argument("--output_csv", default="project/results/on_policy_gap/gap.csv")
    parser.add_argument("--output_png", default="project/results/on_policy_gap/gap.png")
    parser.add_argument("--snaps", nargs="*", default=DEFAULT_SNAPS)
    parser.add_argument("--mid_ts", type=float, nargs="*", default=DEFAULT_MID_TS)
    parser.add_argument("--num", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dataset", default=None, help="Path to cifar10-32x32.zip or cifar-10-python.tar.gz")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    csv_path = (repo_root / args.output_csv).resolve()
    png_path = (repo_root / args.output_png).resolve()
    add_paths(repo_root)

    import measure_latency

    rows: list[dict[str, object]] = []
    real_images = load_real_images(repo_root=repo_root, dataset=args.dataset, num=args.num, seed=args.seed)

    for snap in args.snaps:
        ckpt = checkpoint_path(checkpoint_dir, snap)
        print(f"\n=== Loading checkpoint {snap}: {ckpt} ===")
        net = measure_latency._load_network(str(ckpt), str(repo_root / "project" / "src" / "ect"), args.device)

        for mid_t in args.mid_ts:
            print(f"[{snap}] mid_t={mid_t} diagnostic on {args.num} CIFAR images")
            metrics = residual_for_mid_t(
                net=net,
                images=real_images,
                mid_t=mid_t,
                num=args.num,
                batch_size=args.batch_size,
                device=args.device,
                seed=args.seed + int(float(mid_t) * 1000) + int(snap),
            )
            row = {
                "snap_id": snap,
                "kimg": int(snap) * 10,
                "mid_t": mid_t,
                "num_images": args.num,
                **{k: round(v, 8) for k, v in metrics.items()},
            }
            rows.append(row)
            write_csv(csv_path, rows)
            print(row)

        del net
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    print(f"\nSaved CSV: {csv_path}")
    plot_results(repo_root=repo_root, csv_path=csv_path, output_png=png_path)


if __name__ == "__main__":
    main()
