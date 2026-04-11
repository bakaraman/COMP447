"""
Latency measurement script.
Measures per-image generation time at batch size 1 and 64.

Usage:
    python3 project/scripts/measure_latency.py \
        --checkpoint path/to/model_or_url \
        --sampler ect \
        --steps 2
"""

import argparse
import io
import os
import pickle
import sys
import time
import urllib.request

import numpy as np
import torch


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ECT_ROOT = os.path.join(REPO_ROOT, "project", "src", "ect")
EDM_ROOT = os.path.join(REPO_ROOT, "project", "src", "edm")


def _is_url(path_or_url):
    return path_or_url.startswith("http://") or path_or_url.startswith("https://")


def _open_checkpoint(path_or_url):
    if _is_url(path_or_url):
        with urllib.request.urlopen(path_or_url) as response:
            return io.BytesIO(response.read())
    return open(path_or_url, "rb")


def _load_network(checkpoint, repo_root, device):
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    with _open_checkpoint(checkpoint) as f:
        payload = pickle.load(f)

    net = payload["ema"] if isinstance(payload, dict) and "ema" in payload else payload
    net = net.eval().requires_grad_(False).to(device)
    return net


def _make_class_labels(net, batch_size, device):
    if getattr(net, "label_dim", 0):
        return torch.zeros([batch_size, net.label_dim], device=device)
    return None


@torch.no_grad()
def ect_sampler(net, latents, class_labels=None, steps=2, mid_t=0.821):
    if steps == 1:
        mid_ts = []
    elif steps == 2:
        mid_ts = [mid_t]
    else:
        raise ValueError("ECT latency script currently supports only 1-step or 2-step sampling.")

    t_steps = torch.tensor([80.0] + mid_ts, dtype=torch.float64, device=latents.device)
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

    x = latents.to(torch.float64) * t_steps[0]
    for t_cur, t_next in zip(t_steps[:-1], t_steps[1:]):
        x = net(x, t_cur, class_labels).to(torch.float64)
        if t_next > 0:
            x = x + t_next * torch.randn_like(x)
    return x


@torch.no_grad()
def edm_sampler(net, latents, class_labels=None, steps=18):
    sigma_min = max(0.002, net.sigma_min)
    sigma_max = min(80.0, net.sigma_max)
    rho = 7

    step_indices = torch.arange(steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

    x_next = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next
        denoised = net(x_cur, t_cur, class_labels).to(torch.float64)
        d_cur = (x_cur - denoised) / t_cur
        x_next = x_cur + (t_next - t_cur) * d_cur

        if i < steps - 1:
            denoised = net(x_next, t_next, class_labels).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_cur + (t_next - t_cur) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next


def measure_latency(model, sampler_fn, batch_size, num_runs=500, warmup=50, device="cuda"):
    """
    Measure median per-image latency.
    Returns median latency in milliseconds.
    """
    latencies = []

    for i in range(warmup + num_runs):
        z = torch.randn(batch_size, model.img_channels, model.img_resolution, model.img_resolution, device=device)
        class_labels = _make_class_labels(model, batch_size, device)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = sampler_fn(z, class_labels)
        if device.startswith("cuda"):
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
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--mid_t", type=float, default=0.821, help="ECT intermediate time for 2-step sampling")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but no CUDA runtime is available.")

    if args.sampler == "ect":
        model = _load_network(args.checkpoint, ECT_ROOT, args.device)
        sampler_fn = lambda z, c: ect_sampler(model, z, c, steps=args.steps, mid_t=args.mid_t)
    else:
        if args.steps < 2:
            raise ValueError("Heun / EDM sampling needs at least 2 steps.")
        model = _load_network(args.checkpoint, EDM_ROOT, args.device)
        sampler_fn = lambda z, c: edm_sampler(model, z, c, steps=args.steps)

    stats = measure_latency(
        model=model,
        sampler_fn=sampler_fn,
        batch_size=args.batch_size,
        num_runs=args.num_runs,
        warmup=args.warmup,
        device=args.device,
    )

    print(f"sampler={args.sampler}")
    print(f"steps={args.steps}")
    print(f"batch_size={args.batch_size}")
    print(f"median_ms={stats['median_ms']:.4f}")
    print(f"mean_ms={stats['mean_ms']:.4f}")
    print(f"std_ms={stats['std_ms']:.4f}")
    print(f"p95_ms={stats['p95_ms']:.4f}")


if __name__ == "__main__":
    main()
