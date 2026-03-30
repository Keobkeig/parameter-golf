#!/bin/bash
set -e

echo "🚀 WaterSIC H100 Parallel Testing Script"
echo "========================================"
echo "Hardware: 8x H100 SXM (640GB VRAM)"
echo "Strategy: Parallel baseline + WaterSIC tests"
echo ""

# System check
echo "🔍 System Check:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Create result directories
mkdir -p h100_baseline_results h100_watersic_results

echo "🎯 Starting parallel tests..."
echo "  - Baseline (int6): GPU 0-3"
echo "  - WaterSIC: GPU 4-7"
echo ""

# Start baseline test in background (GPUs 0-3)
echo "🔥 Starting baseline test..."
(
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    export WATERSIC_ENABLED=0
    export OUT_DIR="h100_baseline_results"
    export SEED=1337
    export ITERATIONS=20000
    export MAX_WALLCLOCK_SECONDS=300  # 5 minutes
    
    cd watersic_implementation
    python train_gpt.py 2>&1 | tee "../$OUT_DIR/training_log.txt"
    echo "✅ Baseline complete: $(date)"
) &
BASELINE_PID=$!

# Start WaterSIC test in background (GPUs 4-7)  
echo "🌊 Starting WaterSIC test..."
(
    export CUDA_VISIBLE_DEVICES=4,5,6,7
    export WATERSIC_ENABLED=1
    export WATERSIC_BITS=4.0
    export WATERSIC_CALIBRATION_SAMPLES=512
    export WATERSIC_LMMSE_CORRECTION=1
    export WATERSIC_ACTIVATION_DRIFT=1
    export OUT_DIR="h100_watersic_results"
    export SEED=1337
    export ITERATIONS=20000
    export MAX_WALLCLOCK_SECONDS=300  # 5 minutes
    
    cd watersic_implementation
    python train_gpt.py 2>&1 | tee "../$OUT_DIR/training_log.txt"
    echo "✅ WaterSIC complete: $(date)"
) &
WATERSIC_PID=$!

# Wait for both tests
echo "⏳ Waiting for both tests to complete..."
wait $BASELINE_PID
BASELINE_EXIT=$?
wait $WATERSIC_PID  
WATERSIC_EXIT=$?

echo ""
echo "📊 H100 Test Results Summary:"
echo "============================="

# Check baseline results
if [ $BASELINE_EXIT -eq 0 ]; then
    echo "✅ Baseline test: SUCCESS"
    if [ -f "h100_baseline_results/training_log.txt" ]; then
        BASELINE_LOSS=$(grep "loss:" "h100_baseline_results/training_log.txt" | tail -1)
        echo "   Final: $BASELINE_LOSS"
    fi
else
    echo "❌ Baseline test: FAILED (exit code: $BASELINE_EXIT)"
fi

# Check WaterSIC results
if [ $WATERSIC_EXIT -eq 0 ]; then
    echo "✅ WaterSIC test: SUCCESS"  
    if [ -f "h100_watersic_results/training_log.txt" ]; then
        WATERSIC_LOSS=$(grep "loss:" "h100_watersic_results/training_log.txt" | tail -1)
        echo "   Final: $WATERSIC_LOSS"
        
        # Look for compression info
        COMPRESSION=$(grep -i "compression\|watersic" "h100_watersic_results/training_log.txt" | tail -3)
        if [ -n "$COMPRESSION" ]; then
            echo "   Compression:"
            echo "$COMPRESSION" | sed 's/^/     /'
        fi
    fi
else
    echo "❌ WaterSIC test: FAILED (exit code: $WATERSIC_EXIT)"
fi

# Model size comparison
echo ""
echo "📏 Model Size Comparison:"
echo "------------------------"

BASELINE_MODEL=$(find h100_baseline_results -name "*.ptz" | head -1)
if [ -n "$BASELINE_MODEL" ]; then
    BASELINE_SIZE=$(stat -c%s "$BASELINE_MODEL" 2>/dev/null || stat -f%z "$BASELINE_MODEL" 2>/dev/null)
    echo "🔥 Baseline: $(echo "scale=2; $BASELINE_SIZE/1024/1024" | bc -l)MB"
fi

WATERSIC_MODEL=$(find h100_watersic_results -name "*.ptz" | head -1)
if [ -n "$WATERSIC_MODEL" ]; then
    WATERSIC_SIZE=$(stat -c%s "$WATERSIC_MODEL" 2>/dev/null || stat -f%z "$WATERSIC_MODEL" 2>/dev/null)
    echo "🌊 WaterSIC: $(echo "scale=2; $WATERSIC_SIZE/1024/1024" | bc -l)MB"
    
    if [ -n "$BASELINE_SIZE" ] && [ -n "$WATERSIC_SIZE" ]; then
        COMPRESSION_RATIO=$(echo "scale=1; $BASELINE_SIZE/$WATERSIC_SIZE" | bc -l)
        echo "🎯 Compression ratio: ${COMPRESSION_RATIO}x"
    fi
fi

echo ""
echo "🎉 H100 parallel testing complete!"
echo "📁 Results: h100_baseline_results/ & h100_watersic_results/"
echo ""