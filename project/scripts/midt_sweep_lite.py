"""
Lite mid_t sensitivity sweep for the COMP447 progress report.

Runs the 1980 kimg ECT checkpoint at 2-step inference for several mid_t
values, computes FID over 5000 generated samples each, and saves a CSV +
matplotlib plot. Designed to fit in roughly 20-40 minutes on a Colab G4.

Usage (from the project root, inside a Colab cell):

    !cd /content/COMP447/project && python3 scripts/midt_sweep_lite.py \
        --checkpoint results_backup/ect_checkpoints/network-snapshot-000198.pkl \
        --num 5000 \
        --output_dir project/results/midt_sweep

Adjust --checkpoint to wherever the 1980 kimg snapshot lives in your
runtime. The script writes:

    project/results/midt_sweep/lite_results.csv
    project/results/midt_sweep/lite_curve.png
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_MID_TS = [0.1, 0.3, 0.5, 0.7, 0.821, 1.0, 1.5, 2.5]


def parse_fid_from_eval(stdout: str) -> float:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("Correct ECT FID:"):
            return float(line.split(":", 1)[1].strip())
        if line.startswith("FID:"):
            return float(line.split(":", 1)[1].strip())
    raise RuntimeError("Could not parse FID from eval_fid.py output.")


def run_one(checkpoint: str, mid_t: float, num: int, gen_batch: int, fid_batch: int,
            seed: int, repo_root: Path) -> tuple[float, float]:
    eval_script = repo_root / "project" / "scripts" / "eval_fid.py"
    cmd = [
        "python3", str(eval_script), "ect",
        "--checkpoint", checkpoint,
        "--steps", "2",
        "--mid_t", str(mid_t),
        "--num", str(num),
        "--gen_batch", str(gen_batch),
        "--fid_batch", str(fid_batch),
        "--seed", str(seed),
    ]
    print(f"[mid_t={mid_t:.3f}] running: {' '.join(cmd)}", flush=True)
    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, cwd=str(repo_root))
    captured = []
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        captured.append(line)
    proc.wait()
    dt = time.perf_counter() - t0
    full_stdout = "".join(captured)
    if proc.returncode != 0:
        raise RuntimeError(f"eval_fid.py failed at mid_t={mid_t} (rc={proc.returncode})")
    fid = parse_fid_from_eval(full_stdout)
    print(f"[mid_t={mid_t:.3f}] FID={fid:.4f}  wall={dt:.1f}s", flush=True)
    return fid, dt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="Path or URL to the 1980 kimg ECT snapshot.")
    parser.add_argument("--num", type=int, default=5000,
                        help="Generated images per mid_t (default 5000).")
    parser.add_argument("--gen_batch", type=int, default=64)
    parser.add_argument("--fid_batch", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mid_ts", type=float, nargs="*", default=DEFAULT_MID_TS,
                        help="mid_t values to sweep (default %(default)s).")
    parser.add_argument("--output_dir",
                        default="project/results/midt_sweep",
                        help="Output directory relative to repo root.")
    parser.add_argument("--repo_root",
                        default=str(Path(__file__).resolve().parents[2]),
                        help="Repo root containing project/ subfolder.")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    out_dir = (repo_root / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "lite_results.csv"
    plot_path = out_dir / "lite_curve.png"

    rows = []
    for mid_t in args.mid_ts:
        fid, wall = run_one(
            checkpoint=args.checkpoint,
            mid_t=float(mid_t),
            num=args.num,
            gen_batch=args.gen_batch,
            fid_batch=args.fid_batch,
            seed=args.seed,
            repo_root=repo_root,
        )
        rows.append({"mid_t": mid_t, "fid": fid, "n_samples": args.num, "wall_s": round(wall, 2)})

        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["mid_t", "fid", "n_samples", "wall_s"])
            writer.writeheader()
            writer.writerows(rows)

    print(f"\nSaved CSV: {csv_path}")
    print("Final results:")
    for r in rows:
        print(f"  mid_t={r['mid_t']:.3f}  FID={r['fid']:.4f}  ({r['wall_s']:.1f}s)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xs = [r["mid_t"] for r in rows]
        ys = [r["fid"] for r in rows]
        best_idx = min(range(len(ys)), key=lambda i: ys[i])
        fig, ax = plt.subplots(figsize=(5.4, 3.4))
        ax.plot(xs, ys, "o-", color="#A83D3D", lw=2, ms=7, label="ECT 2-step (1980 kimg)")
        ax.scatter([xs[best_idx]], [ys[best_idx]], color="#000000", marker="*", s=180, zorder=5,
                   label=f"best mid_t={xs[best_idx]:.3f}, FID={ys[best_idx]:.3f}")
        ax.axvline(0.821, color="#6B6B6B", linestyle="--", lw=1, label="ECT default 0.821")
        ax.set_xlabel(r"$t_{\mathrm{mid}}$")
        ax.set_ylabel(f"FID ({args.num} samples)")
        ax.set_title("mid_t sensitivity sweep · ECT 1980 kimg · CIFAR-10")
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(plot_path, dpi=140)
        print(f"Saved plot: {plot_path}")
    except Exception as e:
        print(f"WARN: plot failed -- {e}")


if __name__ == "__main__":
    main()
