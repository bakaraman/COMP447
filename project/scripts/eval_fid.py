"""
Correct FID evaluation utilities for ECT and EDM.

Why this script exists:

The official ECT `ct_eval.py` routes all requested metrics through the same
`few_step_fn` in evaluation mode, which means that when `mid_t` is set the
reported `fid50k_full` is also measured with the few-step sampler. That makes
`ct_eval.py` unsuitable for clean 1-step vs 2-step comparisons.

This script avoids that problem by:

1. Generating images explicitly with the desired sampler.
2. Evaluating the resulting image directory with EDM's official `fid.py`.

Examples:
    python3 project/scripts/eval_fid.py ect \
        --checkpoint /path/to/network-snapshot.pkl \
        --steps 1 \
        --num 10000

    python3 project/scripts/eval_fid.py ect \
        --checkpoint /path/to/network-snapshot.pkl \
        --steps 2 \
        --mid_t 0.821 \
        --num 50000

    python3 project/scripts/eval_fid.py images \
        --images /path/to/generated_images \
        --ref https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from PIL import Image

import measure_latency


REPO_ROOT = Path(__file__).resolve().parents[2]
ECT_ROOT = REPO_ROOT / "project" / "src" / "ect"
EDM_ROOT = REPO_ROOT / "project" / "src" / "edm"


def run(cmd: list[str], cwd: Path, capture: bool = False) -> str | None:
    pretty = " ".join(str(x) for x in cmd)
    print(f"$ {pretty}")
    if capture:
        result = subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)
        return result.stdout
    subprocess.run(cmd, cwd=cwd, check=True)
    return None


def save_tensor_images(images: torch.Tensor, outdir: Path, start_idx: int) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    images = images.detach().cpu().clamp(-1, 1)
    images = ((images + 1) * 127.5).round().to(torch.uint8)
    images = images.permute(0, 2, 3, 1).numpy()
    for i, img in enumerate(images):
        Image.fromarray(img, "RGB").save(outdir / f"{start_idx + i:06d}.png")


@torch.no_grad()
def generate_ect_images(
    checkpoint: str,
    outdir: Path,
    steps: int,
    mid_t: float,
    num_images: int,
    batch_size: int,
    seed: int,
    device: str,
) -> Path:
    if outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    net = measure_latency._load_network(checkpoint, str(ECT_ROOT), device)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    written = 0
    while written < num_images:
        cur_bs = min(batch_size, num_images - written)
        z = torch.randn(
            cur_bs,
            net.img_channels,
            net.img_resolution,
            net.img_resolution,
            device=device,
            generator=gen,
        )
        class_labels = measure_latency._make_class_labels(net, cur_bs, device)
        images = measure_latency.ect_sampler(net, z, class_labels, steps=steps, mid_t=mid_t)
        save_tensor_images(images, outdir, written)
        written += cur_bs
        if written % max(batch_size * 16, 1024) == 0 or written == num_images:
            print(f"  saved {written}/{num_images} images")
    return outdir


def compute_fid(images: Path, ref: str, num: int, batch: int, seed: int, nproc_per_node: int) -> float:
    cmd = [
        "torchrun",
        "--standalone",
        "--nproc_per_node",
        str(nproc_per_node),
        "fid.py",
        "calc",
        "--images",
        str(images),
        "--ref",
        ref,
        "--num",
        str(num),
        "--seed",
        str(seed),
        "--batch",
        str(batch),
    ]
    stdout = run(cmd, cwd=EDM_ROOT, capture=True)
    assert stdout is not None
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.replace(".", "", 1).isdigit():
            return float(line)
    raise RuntimeError("Could not parse FID value from EDM fid.py output.")


def ect_mode(args: argparse.Namespace) -> None:
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but no CUDA runtime is available.")

    if args.steps not in (1, 2):
        raise ValueError("ECT mode currently supports only 1-step or 2-step evaluation.")

    out_root = Path(args.outdir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    cleanup_images = not args.keep_images
    image_dir = out_root / f"ect_steps{args.steps}_midt_{str(args.mid_t).replace('.', '_')}_n{args.num}"
    if cleanup_images:
        tmpdir = tempfile.TemporaryDirectory(prefix="ect-fid-", dir=out_root)
        image_dir = Path(tmpdir.name)
    else:
        tmpdir = None

    print(
        f"Generating ECT images: steps={args.steps}, mid_t={args.mid_t}, "
        f"num={args.num}, batch={args.gen_batch}, checkpoint={args.checkpoint}"
    )
    generate_ect_images(
        checkpoint=args.checkpoint,
        outdir=image_dir,
        steps=args.steps,
        mid_t=args.mid_t,
        num_images=args.num,
        batch_size=args.gen_batch,
        seed=args.seed,
        device=device,
    )

    fid = compute_fid(
        images=image_dir,
        ref=args.ref,
        num=args.num,
        batch=args.fid_batch,
        seed=args.seed,
        nproc_per_node=args.nproc_per_node,
    )
    print(f"\nCorrect ECT FID: {fid:.6f}")
    print(
        "Summary:",
        {
            "checkpoint": args.checkpoint,
            "steps": args.steps,
            "mid_t": args.mid_t,
            "num_images": args.num,
            "fid": fid,
        },
    )

    if tmpdir is not None:
        tmpdir.cleanup()


def images_mode(args: argparse.Namespace) -> None:
    fid = compute_fid(
        images=Path(args.images),
        ref=args.ref,
        num=args.num,
        batch=args.batch,
        seed=args.seed,
        nproc_per_node=args.nproc_per_node,
    )
    print(f"\nFID: {fid:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)

    ect_parser = subparsers.add_parser(
        "ect",
        help="Generate images from an ECT checkpoint with an explicit 1-step or 2-step sampler and compute FID.",
    )
    ect_parser.add_argument("--checkpoint", required=True, help="ECT checkpoint pickle or URL")
    ect_parser.add_argument("--steps", type=int, choices=[1, 2], required=True)
    ect_parser.add_argument("--mid_t", type=float, default=0.821, help="Midpoint used only for 2-step sampling")
    ect_parser.add_argument("--num", type=int, default=50000, help="Number of generated images")
    ect_parser.add_argument("--gen_batch", type=int, default=64, help="Batch size used during image generation")
    ect_parser.add_argument("--fid_batch", type=int, default=64, help="Batch size used by EDM fid.py")
    ect_parser.add_argument("--seed", type=int, default=0)
    ect_parser.add_argument("--device", default="cuda")
    ect_parser.add_argument("--nproc_per_node", type=int, default=1)
    ect_parser.add_argument("--outdir", default=str(REPO_ROOT / "project" / "results" / "ect-fid"))
    ect_parser.add_argument(
        "--ref",
        default="https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz",
        help="Reference FID stats npz path or URL",
    )
    ect_parser.add_argument("--keep_images", action="store_true", help="Keep generated PNGs instead of using a temp dir")
    ect_parser.set_defaults(func=ect_mode)

    images_parser = subparsers.add_parser(
        "images",
        help="Evaluate an existing generated image directory with EDM's official fid.py path",
    )
    images_parser.add_argument("--images", required=True, help="Directory or zip containing generated images")
    images_parser.add_argument(
        "--ref",
        default="https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz",
        help="Reference FID stats npz path or URL",
    )
    images_parser.add_argument("--num", type=int, default=50000)
    images_parser.add_argument("--seed", type=int, default=0)
    images_parser.add_argument("--batch", type=int, default=64)
    images_parser.add_argument("--nproc_per_node", type=int, default=1)
    images_parser.set_defaults(func=images_mode)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
