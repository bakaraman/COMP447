"""
FID evaluation script for ECT project.
Computes FID between generated samples and CIFAR-10 test set.

Usage:
    python eval_fid.py --samples_dir path/to/samples --batch_size 64
"""

import argparse
import torch
import numpy as np
from pathlib import Path


def compute_fid(real_stats_path, samples_dir, batch_size=64, device="cuda"):
    """
    Compute FID between generated samples and precomputed real statistics.
    TODO: integrate with cleanfid or pytorch-fid
    """
    raise NotImplementedError("plug in cleanfid or pytorch-fid here")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples_dir", type=str, required=True)
    parser.add_argument("--real_stats", type=str, default="results/cifar10_stats.npz")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    fid = compute_fid(args.real_stats, args.samples_dir, args.batch_size, args.device)
    print(f"FID: {fid:.4f}")


if __name__ == "__main__":
    main()
