"""
Latency measurement script.
Measures per-image generation time at batch size 1 and 64.

Usage:
    python measure_latency.py --checkpoint path/to/model --sampler ect --steps 2
"""

import argparse
import time
import torch
import numpy as np


def measure_latency(model, sampler_fn, batch_size, num_runs=500, warmup=50, device="cuda"):
    """
    Measure median per-image latency.
    Returns median latency in milliseconds.
    """
    latencies = []

    for i in range(warmup + num_runs):
        z = torch.randn(batch_size, 3, 32, 32, device=device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = sampler_fn(z)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        if i >= warmup:
            latencies.append((t1 - t0) * 1000 / batch_size)  # ms per image

    latencies = np.array(latencies)
    return {
        "median_ms": float(np.median(latencies)),
        "mean_ms": float(np.mean(latencies)),
        "std_ms": float(np.std(latencies)),
        "p95_ms": float(np.percentile(latencies, 95)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--sampler", type=str, choices=["ect", "heun"], required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_runs", type=int, default=500)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # TODO: load model and create sampler_fn based on args
    raise NotImplementedError("load model and wire up sampler here")


if __name__ == "__main__":
    main()
