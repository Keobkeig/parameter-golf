#!/bin/bash
set -e

echo "🔥 WaterSIC Baseline Test (int6 quantization)"
echo "============================================="

# Configuration for baseline test
export WATERSIC_ENABLED=0  # Use int6 baseline
export SEED=1337
export ITERATIONS=20000
export MAX_WALLCLOCK_SECONDS=600  # 10 minutes
export TRAIN_BATCH_TOKENS=524288
export NUM_LAYERS=11
export MODEL_DIM=512
export NUM_HEADS=8
export NUM_KV_HEADS=4
export MLP_MULT=3.0
export OUT_DIR="baseline_results"
export RUN_ID="baseline_int6_$(date +%s)"

echo "📊 Baseline Test Configuration:"
echo "  - Quantization: int6 baseline"
echo "  - Layers: $NUM_LAYERS"
echo "  - Model dim: $MODEL_DIM"
echo "  - Training time: $MAX_WALLCLOCK_SECONDS seconds"
echo "  - Seed: $SEED"
echo "  - Run ID: $RUN_ID"
echo ""

# Create output directory
mkdir -p $OUT_DIR

# Record start time
START_TIME=$(date +%s)
echo "⏰ Start time: $(date)"
echo ""

# Run training
echo "🚀 Starting baseline training..."
cd watersic_implementation
python train_gpt.py 2>&1 | tee "../$OUT_DIR/training_log.txt"

# Record end time and calculate duration
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo ""
echo "⏰ End time: $(date)"
echo "⌛ Total duration: $DURATION seconds"

# Analyze results
echo ""
echo "📊 Baseline Results Analysis:"
echo "============================="

# Find the final model files
MODEL_FILES=$(find $OUT_DIR -name "*.int6.ptz" | head -1)
if [ -n "$MODEL_FILES" ]; then
    MODEL_SIZE=$(stat -c%s "$MODEL_FILES" 2>/dev/null || stat -f%z "$MODEL_FILES" 2>/dev/null || echo "unknown")
    echo "✅ Final model: $MODEL_FILES"
    echo "📏 Model size: $MODEL_SIZE bytes ($(echo "scale=2; $MODEL_SIZE/1024/1024" | bc -l)MB)"
else
    echo "❌ No final model found"
fi

# Extract final metrics from log
if [ -f "$OUT_DIR/training_log.txt" ]; then
    echo ""
    echo "📈 Training Metrics:"
    echo "-------------------"
    
    # Get final loss
    FINAL_LOSS=$(grep "loss:" "$OUT_DIR/training_log.txt" | tail -1 | grep -o "loss:[0-9.]*" | cut -d: -f2)
    if [ -n "$FINAL_LOSS" ]; then
        echo "🎯 Final loss: $FINAL_LOSS"
    fi
    
    # Get compression info
    COMPRESSION=$(grep "payload_ratio" "$OUT_DIR/training_log.txt" | tail -1)
    if [ -n "$COMPRESSION" ]; then
        echo "🗜️  Compression: $COMPRESSION"
    fi
    
    # Get validation metrics
    VAL_LOSS=$(grep "val_loss" "$OUT_DIR/training_log.txt" | tail -1)
    if [ -n "$VAL_LOSS" ]; then
        echo "✅ Validation: $VAL_LOSS"
    fi
fi

echo ""
echo "✅ Baseline test complete!"
echo "📁 Results saved to: $OUT_DIR/"
echo ""