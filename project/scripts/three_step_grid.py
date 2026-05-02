"""
Small 3-step ECT schedule grid.

This is intentionally a screening experiment, not final evidence. The question:
does the schedule issue remain when we move from 2-step to 3-step sampling?
If yes, KLUB-CM is not merely picking one midpoint; it is solving a real
multi-knot schedule problem.

For 3-step sampling we use:

    [80, t_high, t_low, 0], with t_high > t_low > 0

and compute FID at a small sample count for each pair.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch


DEFAULT_HIGH = [1.0, 1.5, 2.5, 5.0]
DEFAULT_LOW = [0.3, 0.5, 0.7, 0.821, 1.0]


def add_paths(repo_root: Path) -> None:
    for p in [repo_root / "project" / "scripts", repo_root / "project" / "src" / "ect"]:
        text = str(p)
        if text not in sys.path:
            sys.path.insert(0, text)


@torch.no_grad()
def ect_sampler_any(net, latents, class_labels, mids: list[float]):
    t_steps = torch.tensor([80.0] + list(mids), dtype=torch.float64, device=latents.device)
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])
    x = latents.to(torch.float64) * t_steps[0]
    for t_cur, t_next in zip(t_steps[:-1], t_steps[1:]):
        x = net(x, t_cur, class_labels).to(torch.float64)
        if t_next > 0:
            x = x + t_next * torch.randn_like(x)
    return x


def save_images(images: torch.Tensor, outdir: Path, start_idx: int) -> None:
    from PIL import Image

    outdir.mkdir(parents=True, exist_ok=True)
    arr = images.detach().cpu().clamp(-1, 1)
    arr = ((arr + 1) * 127.5).round().to(torch.uint8)
    arr = arr.permute(0, 2, 3, 1).numpy()
    for i, img in enumerate(arr):
        Image.fromarray(img, "RGB").save(outdir / f"{start_idx + i:06d}.png")


def generate_images(repo_root: Path, checkpoint: Path, mids: list[float], outdir: Path, num: int, batch: int, seed: int, device: str) -> None:
    import measure_latency

    if outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    net = measure_latency._load_network(str(checkpoint), str(repo_root / "project" / "src" / "ect"), device)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    written = 0
    while written < num:
        cur = min(batch, num - written)
        z = torch.randn(cur, net.img_channels, net.img_resolution, net.img_resolution, device=device, generator=gen)
        labels = measure_latency._make_class_labels(net, cur, device)
        images = ect_sampler_any(net, z, labels, mids=mids)
        save_images(images, outdir, written)
        written += cur
        if written % max(batch * 16, 1024) == 0 or written == num:
            print(f"  saved {written}/{num} images")

    del net
    if device.startswith("cuda"):
        torch.cuda.empty_cache()


def compute_fid(repo_root: Path, images: Path, num: int, batch: int, seed: int) -> float:
    edm_root = repo_root / "project" / "src" / "edm"
    cmd = [
        "torchrun",
        "--standalone",
        "--nproc_per_node",
        "1",
        "fid.py",
        "calc",
        "--images",
        str(images),
        "--ref",
        "https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz",
        "--num",
        str(num),
        "--seed",
        str(seed),
        "--batch",
        str(batch),
    ]
    print("$", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(edm_root), capture_output=True, text=True, check=True)
    print(proc.stdout)
    for line in reversed(proc.stdout.splitlines()):
        line = line.strip()
        if line.replace(".", "", 1).isdigit():
            return float(line)
    raise RuntimeError("Could not parse FID from fid.py output.")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["snap_id", "kimg", "t_high", "t_low", "fid", "n_samples", "wall_s"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--snap_id", default="050")
    parser.add_argument("--num", type=int, default=5000)
    parser.add_argument("--gen_batch", type=int, default=64)
    parser.add_argument("--fid_batch", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--t_high", type=float, nargs="*", default=DEFAULT_HIGH)
    parser.add_argument("--t_low", type=float, nargs="*", default=DEFAULT_LOW)
    parser.add_argument("--output_csv", default="project/results/three_step_grid/grid.csv")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    csv_path = (repo_root / args.output_csv).resolve()
    add_paths(repo_root)

    rows: list[dict[str, object]] = []
    if csv_path.exists():
        with csv_path.open() as f:
            rows = list(csv.DictReader(f))
    done = {(float(r["t_high"]), float(r["t_low"])) for r in rows}

    for t_high in args.t_high:
        for t_low in args.t_low:
            if t_high <= t_low:
                continue
            if (float(t_high), float(t_low)) in done:
                print(f"[{t_high}, {t_low}] already done, skipping")
                continue
            print(f"\n=== 3-step schedule [80, {t_high}, {t_low}, 0] ===")
            start = time.perf_counter()
            with tempfile.TemporaryDirectory(prefix="ect-3step-", dir=str(repo_root / "project" / "results")) as tmp:
                image_dir = Path(tmp)
                generate_images(
                    repo_root=repo_root,
                    checkpoint=checkpoint,
                    mids=[float(t_high), float(t_low)],
                    outdir=image_dir,
                    num=args.num,
                    batch=args.gen_batch,
                    seed=args.seed,
                    device=args.device,
                )
                fid = compute_fid(repo_root, image_dir, args.num, args.fid_batch, args.seed)
            wall_s = time.perf_counter() - start
            row = {
                "snap_id": args.snap_id,
                "kimg": int(args.snap_id) * 10,
                "t_high": float(t_high),
                "t_low": float(t_low),
                "fid": round(fid, 6),
                "n_samples": args.num,
                "wall_s": round(wall_s, 2),
            }
            rows.append(row)
            write_csv(csv_path, rows)
            print(row)

    print(f"\nSaved CSV: {csv_path}")
    print("Best rows:")
    for row in sorted(rows, key=lambda r: float(r["fid"]))[:8]:
        print(row)


if __name__ == "__main__":
    main()
