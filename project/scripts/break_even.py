"""
Break-even analysis.
Computes N* = T_tune / (t_heun - t_ect) for each latency setting.

Usage:
    python break_even.py --tune_hours 3.5 --ect_ms 12.3 --heun_ms 45.6
"""

import argparse


def compute_break_even(tune_time_ms, ect_latency_ms, heun_latency_ms):
    """
    N* = tune_time / (heun_per_image - ect_per_image)
    Returns break-even image count, or inf if ECT is slower.
    """
    diff = heun_latency_ms - ect_latency_ms
    if diff <= 0:
        return float("inf")
    return tune_time_ms / diff


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune_hours", type=float, required=True, help="ECT tuning time in hours")
    parser.add_argument("--ect_ms", type=float, required=True, help="ECT per-image latency in ms")
    parser.add_argument("--heun_ms", type=float, required=True, help="Heun per-image latency in ms")
    args = parser.parse_args()

    tune_ms = args.tune_hours * 3600 * 1000
    n_star = compute_break_even(tune_ms, args.ect_ms, args.heun_ms)

    if n_star == float("inf"):
        print("ECT is slower per image than Heun at this setting. Tuning never pays off.")
    else:
        print(f"Break-even at N* = {n_star:,.0f} images")
        print(f"That is {n_star/1000:,.1f}k images")


if __name__ == "__main__":
    main()
