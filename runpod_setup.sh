#!/bin/bash
set -e

echo "🚀 WaterSIC RunPod Deployment Script"
echo "===================================="

# Environment setup
export DEBIAN_FRONTEND=noninteractive
export PYTHONUNBUFFERED=1

# System information
echo "📋 System Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo "Python: $(python3 --version)"
echo "CUDA: $(nvcc --version | grep 'release' || echo 'Not found')"
echo "GPU Memory: $(nvidia-ml-py3 2>/dev/null || nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)MB"
echo ""

# Install dependencies
echo "📦 Installing Dependencies..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy tqdm huggingface-hub datasets tiktoken sentencepiece zstandard
echo "✅ Dependencies installed"
echo ""

# Verify installations
echo "🔍 Verifying Installations..."
python3 -c "
import torch
import numpy as np
import zstandard
print(f'✅ PyTorch: {torch.__version__}')
print(f'✅ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✅ GPU: {torch.cuda.get_device_name(0)}')
    print(f'✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB')
print(f'✅ NumPy: {np.__version__}')
print(f'✅ Zstandard: {zstandard.__version__}')
"
echo ""

# Download parameter-golf if not exists
if [ ! -d "parameter-golf" ]; then
    echo "📥 Cloning parameter-golf repository..."
    git clone https://github.com/KellerJordan/parameter-golf.git
    cd parameter-golf
else
    echo "📁 Using existing parameter-golf directory"
    cd parameter-golf
    git pull origin main
fi

# Create WaterSIC test directory
echo "📁 Setting up WaterSIC test environment..."
mkdir -p watersic_test_results
cd watersic_test_results

# Copy our WaterSIC implementation
echo "📋 Copying WaterSIC implementation..."
cp -r ../records/track_10min_16mb/2026-03-29_LeakyReLU_LegalTTT_ParallelMuon_Copy ./watersic_implementation/

echo "✅ RunPod environment setup complete!"
echo ""
echo "🎯 Ready to run WaterSIC tests!"
echo "   - WaterSIC implementation: ./watersic_implementation/"
echo "   - Test results: ./watersic_test_results/"
echo ""
echo "Next steps:"
echo "1. Run baseline test: ./run_baseline.sh"
echo "2. Run WaterSIC test: ./run_watersic.sh"
echo "3. Compare results: ./compare_results.sh"