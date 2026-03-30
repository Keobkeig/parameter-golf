#!/bin/bash
set -e

echo "🌊 WaterSIC Test (WaterSIC quantization)"
echo "========================================"

# Configuration for WaterSIC test
export WATERSIC_ENABLED=1              # Enable WaterSIC!
export WATERSIC_BITS=4.0               # Target 4 bits per parameter
export WATERSIC_CALIBRATION_SAMPLES=512 # Calibration data size
export WATERSIC_LMMSE_CORRECTION=1     # Enable LMMSE correction
export WATERSIC_ACTIVATION_DRIFT=1     # Enable activation drift correction
export WATERSIC_RESIDUAL_CORRECTION=1  # Enable residual stream correction
export WATERSIC_ATTENTION_WEIGHTED=1   # Enable attention weighting
export WATERSIC_ADAPTIVE_MIXING=1      # Enable adaptive mixing
export WATERSIC_DIAGONAL_RESCALERS=1   # Enable diagonal rescaling
export WATERSIC_QR_EPSILON=0.1         # QR mixing factor
export WATERSIC_AW_EPSILON=0.5         # Attention weighting factor

# Training configuration (same as baseline)
export SEED=1337
export ITERATIONS=20000
export MAX_WALLCLOCK_SECONDS=600  # 10 minutes
export TRAIN_BATCH_TOKENS=524288
export NUM_LAYERS=11
export MODEL_DIM=512
export NUM_HEADS=8
export NUM_KV_HEADS=4
export MLP_MULT=3.0
export OUT_DIR="watersic_results"
export RUN_ID="watersic_$(date +%s)"

echo "📊 WaterSIC Test Configuration:"
echo "  - Quantization: WaterSIC @ $WATERSIC_BITS bits"
echo "  - LMMSE correction: $WATERSIC_LMMSE_CORRECTION"
echo "  - Activation drift: $WATERSIC_ACTIVATION_DRIFT"
echo "  - Residual correction: $WATERSIC_RESIDUAL_CORRECTION"
echo "  - Attention weighted: $WATERSIC_ATTENTION_WEIGHTED"
echo "  - Calibration samples: $WATERSIC_CALIBRATION_SAMPLES"
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

# Run pre-training validation
echo "🔍 Pre-training validation..."
cd watersic_implementation

# Test WaterSIC functions
echo "Testing WaterSIC unit tests..."
python test_watersic.py 2>&1 | tee "../$OUT_DIR/pre_test_log.txt"
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "✅ WaterSIC unit tests passed"
else
    echo "❌ WaterSIC unit tests failed - check pre_test_log.txt"
    exit 1
fi

# Test implementation validation
echo "Testing WaterSIC implementation validation..."
python validate_watersic.py 2>&1 | tee -a "../$OUT_DIR/pre_test_log.txt"
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "✅ WaterSIC implementation validation passed"
else
    echo "❌ WaterSIC implementation validation failed - check pre_test_log.txt"
    exit 1
fi

echo ""
echo "🚀 Starting WaterSIC training..."
python train_gpt.py 2>&1 | tee "../$OUT_DIR/training_log.txt"

# Check if training completed successfully
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "✅ WaterSIC training completed successfully"
else
    echo "❌ WaterSIC training failed - check training_log.txt"
fi

# Record end time and calculate duration
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo ""
echo "⏰ End time: $(date)"
echo "⌛ Total duration: $DURATION seconds"

# Analyze results
echo ""
echo "📊 WaterSIC Results Analysis:"
echo "============================="

# Find the final model files
MODEL_FILES=$(find $OUT_DIR -name "*watersic*" -o -name "*.ptz" | head -1)
if [ -n "$MODEL_FILES" ]; then
    MODEL_SIZE=$(stat -c%s "$MODEL_FILES" 2>/dev/null || stat -f%z "$MODEL_FILES" 2>/dev/null || echo "unknown")
    echo "✅ Final model: $MODEL_FILES"
    echo "📏 Model size: $MODEL_SIZE bytes ($(echo "scale=2; $MODEL_SIZE/1024/1024" | bc -l)MB)"
else
    # Look for any .ptz files
    MODEL_FILES=$(find $OUT_DIR -name "*.ptz" | head -1)
    if [ -n "$MODEL_FILES" ]; then
        MODEL_SIZE=$(stat -c%s "$MODEL_FILES" 2>/dev/null || stat -f%z "$MODEL_FILES" 2>/dev/null || echo "unknown")
        echo "✅ Final model: $MODEL_FILES"
        echo "📏 Model size: $MODEL_SIZE bytes ($(echo "scale=2; $MODEL_SIZE/1024/1024" | bc -l)MB)"
    else
        echo "❌ No final model found"
    fi
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
    
    # Get WaterSIC-specific compression info
    WATERSIC_COMPRESSION=$(grep -i "watersic\|compression" "$OUT_DIR/training_log.txt" | tail -3)
    if [ -n "$WATERSIC_COMPRESSION" ]; then
        echo "🌊 WaterSIC Compression:"
        echo "$WATERSIC_COMPRESSION"
    fi
    
    # Get general compression info
    COMPRESSION=$(grep "payload_ratio" "$OUT_DIR/training_log.txt" | tail -1)
    if [ -n "$COMPRESSION" ]; then
        echo "🗜️  Compression: $COMPRESSION"
    fi
    
    # Get validation metrics
    VAL_LOSS=$(grep "val_loss" "$OUT_DIR/training_log.txt" | tail -1)
    if [ -n "$VAL_LOSS" ]; then
        echo "✅ Validation: $VAL_LOSS"
    fi
    
    # Look for any error messages
    ERRORS=$(grep -i "error\|failed\|exception" "$OUT_DIR/training_log.txt" | tail -3)
    if [ -n "$ERRORS" ]; then
        echo ""
        echo "⚠️  Potential Issues:"
        echo "$ERRORS"
    fi
fi

echo ""
echo "✅ WaterSIC test complete!"
echo "📁 Results saved to: $OUT_DIR/"
echo ""