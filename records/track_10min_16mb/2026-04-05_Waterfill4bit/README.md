# Waterfill-4bit: Activation-Weighted 4-bit QAT

Builds on the LeakyReLU² + Legal TTT + Parallel Muon submission, replacing int6+lzma with a
4-bit quantization scheme inspired by the water-filling theorem (arxiv 2603.04956).

## Key Idea

Standard int6 allocates the same resolution to every weight column regardless of how much
input signal that column carries. Water-filling says: give more resolution where the input
energy is high, less where it's low.

**Implementation:**
1. During training: 4-bit fake-quant STE with per-column scales (replaces per-row int6 STE)
2. Post-training: collect input activation statistics via 512 calibration samples
3. Per-column scale = `sqrt(mean(x[:,i]^2)) * max(|w[:,i]|) / 7` — captures activation energy
4. Quantize: `Z = clamp(round(W / scale), -8, 7)` — 16 levels, signed 4-bit
5. Pack: two 4-bit values per byte (50% of int8 storage)
6. Compress: lzma level 6

**Storage per layer:** packed 4-bit weights + fp16 per-column scales (~0.4% overhead)
**No auxiliary matrices** (L, gamma, etc.) — inference is just `W ≈ Z * scale`

## Why 4-bit enables a larger model

| Config | Bits | Raw size | After lzma | Params |
|--------|------|----------|------------|--------|
| L11 int6 (baseline) | 6 | ~20MB | ~16MB | 26.9M |
| L11 wf4 | 4 | ~14MB | ~11MB | 26.9M |
| **L13 wf4 (this)** | **4** | **~16MB** | **~13MB** | **~32M** |

The extra 2 layers give ~20% more parameters in the same 16MB budget, which should
more than offset the quality loss from 4-bit vs 6-bit QAT.

## Run Command (8×H100 SXM)

```bash
NUM_LAYERS=13 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
WATERFILL_ENABLED=1 WATERFILL_CALIBRATION_SAMPLES=512 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=11,12 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Credits

- Base submission: LeakyReLU² + Legal TTT + Parallel Muon (PR #549 stack)
- Waterfilling theory: arxiv 2603.04956 (WaterSIC)
- QAT infrastructure: PR #414 by @signalrush
