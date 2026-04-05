#!/bin/bash
# Quick smoke test — 1 GPU, 200 steps, no wallclock cap.
# Use this to verify the submission runs before burning 8xH100 time.
#
# Usage: bash run_smoke_test.sh

set -e

RECORD_DIR="records/track_10min_16mb/2026-03-29_LeakyReLU_LegalTTT_ParallelMuon_Copy"

if [ ! -f "$RECORD_DIR/train_gpt.py" ]; then
    echo "ERROR: $RECORD_DIR/train_gpt.py not found."
    echo "Run from the repository root."
    exit 1
fi

echo "=== Smoke Test (1 GPU, 200 steps) ==="
echo ""

cd "$RECORD_DIR"

NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=200 MAX_WALLCLOCK_SECONDS=0 EVAL_STRIDE=64 \
VAL_LOSS_EVERY=200 TRAIN_LOG_EVERY=50 \
DATA_PATH="$(git rev-parse --show-toplevel)/data/datasets/fineweb10B_sp1024" \
TOKENIZER_PATH="$(git rev-parse --show-toplevel)/data/tokenizers/fineweb_1024_bpe.model" \
SEED=1337 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
