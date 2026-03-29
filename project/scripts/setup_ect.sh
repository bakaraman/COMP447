#!/bin/bash
# Setup script for ECT repo on Colab or local machine
# Run from project/ directory

set -e

echo "Cloning ECT repo..."
if [ -d "src/ect" ]; then
    echo "ECT repo already exists, pulling latest..."
    cd src/ect && git pull && cd ../..
else
    git clone https://github.com/locuslab/ect.git src/ect
fi

echo "Installing dependencies..."
pip install -q torch torchvision
pip install -q -r src/ect/requirements.txt 2>/dev/null || echo "No requirements.txt found, skipping"

echo "Downloading pretrained EDM checkpoint for CIFAR-10..."
# TODO: check ECT repo for exact download command / path
# The repo README should have instructions for downloading the EDM checkpoint
echo "Check src/ect/README.md for checkpoint download instructions"

echo "Setup complete. Next steps:"
echo "  1. Download EDM CIFAR-10 checkpoint"
echo "  2. Run: python src/ect/train.py  (check README for exact command)"
echo "  3. Run: python scripts/measure_latency.py"
