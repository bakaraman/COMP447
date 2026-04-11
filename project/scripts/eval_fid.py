"""
Thin wrappers around the official ECT and EDM FID evaluation paths.

Examples:
    python3 project/scripts/eval_fid.py ect \
        --checkpoint /path/to/network-snapshot.pkl \
        --data /path/to/cifar10-32x32.zip

    python3 project/scripts/eval_fid.py images \
        --images /path/to/generated_images \
        --ref https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz
"""

import argparse
import os
import subprocess
import sys


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ECT_ROOT = os.path.join(REPO_ROOT, "project", "src", "ect")
EDM_ROOT = os.path.join(REPO_ROOT, "project", "src", "edm")


def run(cmd, cwd):
    pretty = " ".join(cmd)
    print(f"$ {pretty}")
    subprocess.run(cmd, cwd=cwd, check=True)


def torchrun_cmd(nproc_per_node, script_name):
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        str(nproc_per_node),
        script_name,
    ]


def ect_mode(args):
    cmd = torchrun_cmd(args.nproc_per_node, "ct_eval.py")
    cmd.extend(
        [
            "--outdir",
            args.outdir,
            "--data",
            args.data,
            "--cond=0",
            "--arch=ddpmpp",
            "--metrics=fid50k_full",
            "--resume",
            args.checkpoint,
            "--nosubdir",
        ]
    )
    cmd.extend(["--mid_t", str(args.mid_t)])
    run(cmd, cwd=ECT_ROOT)


def images_mode(args):
    cmd = torchrun_cmd(args.nproc_per_node, "fid.py")
    cmd.extend(
        [
            "calc",
            "--images",
            args.images,
            "--ref",
            args.ref,
            "--num",
            str(args.num),
            "--seed",
            str(args.seed),
            "--batch",
            str(args.batch),
        ]
    )
    run(cmd, cwd=EDM_ROOT)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)

    ect_parser = subparsers.add_parser("ect", help="Evaluate an ECT checkpoint with the official ct_eval.py path")
    ect_parser.add_argument("--checkpoint", required=True, help="ECT checkpoint pickle or URL")
    ect_parser.add_argument("--data", required=True, help="Path to cifar10-32x32.zip")
    ect_parser.add_argument("--outdir", default=os.path.join(REPO_ROOT, "project", "results", "ect-fid"))
    ect_parser.add_argument("--nproc_per_node", type=int, default=1)
    ect_parser.add_argument("--mid_t", type=float, default=0.821, help="Default 2-step ECT midpoint")
    ect_parser.set_defaults(func=ect_mode)

    images_parser = subparsers.add_parser("images", help="Evaluate a generated image directory with the official EDM fid.py path")
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
