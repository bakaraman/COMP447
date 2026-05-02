"""
Confirm the strongest mid_t drift findings with publication-grade FID.

The lite sweep uses 5k samples and is useful for discovering patterns, but a
reviewer can fairly object that 5k FID is noisy. This script reruns only the
most decision-relevant pairs at a higher sample count:

  - 500 kimg: default 0.821 vs observed best 1.5
  - 1000 kimg: default 0.821 vs observed best 1.0

It streams the underlying eval_fid.py output and saves a compact CSV so partial
progress survives Colab disconnects.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_JOBS = [
    ("050", 0.821),
    ("050", 1.5),
    ("100", 0.821),
    ("100", 1.0),
]


def parse_fid(stdout: str) -> float:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("Correct ECT FID:"):
            return float(line.split(":", 1)[1].strip())
    raise RuntimeError("Could not parse 'Correct ECT FID:' from eval output.")


def find_checkpoint(checkpoint_dir: Path, snap: str) -> Path:
    path = checkpoint_dir / f"network-snapshot-000{snap}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    return path


def run_eval(
    repo_root: Path,
    checkpoint: Path,
    mid_t: float,
    num: int,
    gen_batch: int,
    fid_batch: int,
    seed: int,
) -> tuple[float, float]:
    cmd = [
        "python3",
        str(repo_root / "project" / "scripts" / "eval_fid.py"),
        "ect",
        "--checkpoint",
        str(checkpoint),
        "--steps",
        "2",
        "--mid_t",
        str(mid_t),
        "--num",
        str(num),
        "--gen_batch",
        str(gen_batch),
        "--fid_batch",
        str(fid_batch),
        "--seed",
        str(seed),
    ]
    print("$", " ".join(cmd))
    start = time.perf_counter()
    proc = subprocess.Popen(
        cmd,
        cwd=str(repo_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    collected: list[str] = []
    for line in proc.stdout:
        collected.append(line)
        sys.stdout.write(line)
        sys.stdout.flush()
    rc = proc.wait()
    wall_s = time.perf_counter() - start
    if rc != 0:
        raise RuntimeError(f"eval_fid.py failed with return code {rc}")
    return parse_fid("".join(collected)), wall_s


def write_rows(csv_path: Path, rows: list[dict[str, object]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["snap_id", "kimg", "mid_t", "fid", "n_samples", "wall_s"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument(
        "--checkpoint_dir",
        default="/content/drive/MyDrive/COMP447_checkpoints",
        help="Directory containing network-snapshot-000{050,100,...}.pkl",
    )
    parser.add_argument("--output_csv", default="project/results/midt_sweep/confirm_50k.csv")
    parser.add_argument("--num", type=int, default=50000)
    parser.add_argument("--gen_batch", type=int, default=64)
    parser.add_argument("--fid_batch", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--jobs",
        nargs="*",
        default=[f"{snap}:{mid_t}" for snap, mid_t in DEFAULT_JOBS],
        help="Jobs like 050:0.821 050:1.5",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    csv_path = (repo_root / args.output_csv).resolve()

    rows: list[dict[str, object]] = []
    if csv_path.exists():
        with csv_path.open() as f:
            rows = list(csv.DictReader(f))

    done = {(str(r["snap_id"]), float(r["mid_t"])) for r in rows}

    for job in args.jobs:
        snap, mid_t_text = job.split(":", 1)
        mid_t = float(mid_t_text)
        if (snap, mid_t) in done:
            print(f"[{snap} mid_t={mid_t}] already done, skipping")
            continue

        checkpoint = find_checkpoint(checkpoint_dir, snap)
        print(f"\n=== {snap} checkpoint, mid_t={mid_t}, num={args.num} ===")
        fid, wall_s = run_eval(
            repo_root=repo_root,
            checkpoint=checkpoint,
            mid_t=mid_t,
            num=args.num,
            gen_batch=args.gen_batch,
            fid_batch=args.fid_batch,
            seed=args.seed,
        )
        rows.append(
            {
                "snap_id": snap,
                "kimg": int(snap) * 10,
                "mid_t": mid_t,
                "fid": round(fid, 6),
                "n_samples": args.num,
                "wall_s": round(wall_s, 2),
            }
        )
        write_rows(csv_path, rows)
        print(f"[{snap} mid_t={mid_t}] FID={fid:.6f}, saved {csv_path}")

    print("\nFinal 50k confirmation table:")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
