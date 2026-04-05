#!/bin/bash
set -e
# Run this on the pod after SSH-ing in.
# Usage: bash runpod_setup.sh [NUM_TRAIN_SHARDS]
#        bash runpod_setup.sh 1    # quick smoke-test subset
#        bash runpod_setup.sh 80   # full 8B token dataset (default)

NUM_TRAIN_SHARDS=${1:-80}

echo "=== parameter-golf RunPod Setup ==="
echo "Image: runpod/parameter-golf:latest (Python 3.12, PyTorch 2.9.1, CUDA 12.8)"
echo "Train shards: $NUM_TRAIN_SHARDS"
echo ""

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

cd /workspace

# Clone the official repo (matches what the template image expects)
if [ ! -d "parameter-golf" ]; then
    echo "--- Cloning openai/parameter-golf ---"
    git clone https://github.com/openai/parameter-golf.git
fi

cd parameter-golf

# Overlay the watersic submission from the fork
echo "--- Adding watersic submission from Keobkeig/parameter-golf ---"
SUBMISSION_DIR="records/track_10min_16mb/2026-03-29_LeakyReLU_LegalTTT_ParallelMuon_Copy"
if [ ! -d "$SUBMISSION_DIR" ]; then
    git remote add fork https://github.com/Keobkeig/parameter-golf.git 2>/dev/null || true
    git fetch fork copy-pr549-2026-03-29 --depth=1
    git checkout fork/copy-pr549-2026-03-29 -- "$SUBMISSION_DIR"
    git checkout fork/copy-pr549-2026-03-29 -- run_submission.sh run_smoke_test.sh
fi

echo "--- Downloading FineWeb sp1024 ($NUM_TRAIN_SHARDS train shards) ---"
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards "$NUM_TRAIN_SHARDS"

mkdir -p logs

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Smoke test (1 GPU, 200 steps):"
echo "  bash run_smoke_test.sh"
echo ""
echo "Full submission (all GPUs, seed 1337/42/2025):"
echo "  bash run_submission.sh"
echo "  SEED=42 bash run_submission.sh"
echo "  SEED=2025 bash run_submission.sh"
