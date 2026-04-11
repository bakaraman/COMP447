#!/bin/bash
# Project bootstrap for the COMP447 ECT vs Heun study.
# Run from the repo root or from project/.

set -e

if [ -d ".git" ]; then
    ROOT_DIR="$(pwd)"
elif [ -d "../.git" ]; then
    ROOT_DIR="$(cd .. && pwd)"
else
    echo "Run this script from the repo root or from project/." >&2
    exit 1
fi

mkdir -p "$ROOT_DIR/project/src"

echo "Repo root: $ROOT_DIR"

echo
echo "Cloning or refreshing ECT..."
if [ -d "$ROOT_DIR/project/src/ect/.git" ]; then
    git -C "$ROOT_DIR/project/src/ect" pull --ff-only
else
    git clone https://github.com/locuslab/ect.git "$ROOT_DIR/project/src/ect"
fi

echo
echo "Cloning or refreshing EDM..."
if [ -d "$ROOT_DIR/project/src/edm/.git" ]; then
    git -C "$ROOT_DIR/project/src/edm" pull --ff-only
else
    git clone https://github.com/NVlabs/edm.git "$ROOT_DIR/project/src/edm"
fi

echo
echo "Bootstrap complete."
echo
echo "Next steps:"
echo "  1. Create the runtime environment shown in:"
echo "     - $ROOT_DIR/project/src/ect/env.yml"
echo "     - $ROOT_DIR/project/src/edm/environment.yml"
echo
echo "  2. Prepare CIFAR-10 in EDM format:"
echo "     cd $ROOT_DIR/project/src/ect"
echo "     python3 dataset_tool.py --source=/path/to/cifar-10-python.tar.gz --dest=datasets/cifar10-32x32.zip"
echo
echo "  3. Run an ECT sanity check:"
echo "     cd $ROOT_DIR/project/src/ect"
echo "     bash run_ecm_1hour.sh 1 29500 --desc monday_sanity"
echo
echo "  4. Evaluate an ECT checkpoint:"
echo "     cd $ROOT_DIR/project/src/ect"
echo "     bash eval_ecm.sh 1 29501 --resume /path/to/network-snapshot.pkl"
echo
echo "  5. Measure latency locally once checkpoints are available:"
echo "     cd $ROOT_DIR"
echo "     python3 project/scripts/measure_latency.py --checkpoint /path/to/checkpoint.pkl --sampler ect --steps 2 --batch_size 1"
echo
echo "  6. Heun / EDM baseline reference:"
echo "     cd $ROOT_DIR/project/src/edm"
echo "     python3 generate.py --outdir=out --steps=18 --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/baseline/baseline-cifar10-32x32-uncond-vp.pkl"
