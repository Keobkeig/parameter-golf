#!/usr/bin/env python3
"""
Test WaterSIC MLX integration
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import mlx.core as mx
import mlx.nn as nn
import numpy as np

# Import WaterSIC functions from the train_gpt_mlx.py
from train_gpt_mlx import (
    quantize_layer_watersic_mlx,
    quantize_state_dict_watersic_mlx,
    dequantize_state_dict_watersic_mlx,
    Hyperparameters
)

def test_watersic_mlx():
    print("🧪 Testing WaterSIC MLX Integration")
    print("=" * 40)
    
    # Test 1: Basic layer quantization
    print("\n📊 Test 1: Basic Layer Quantization")
    weight = mx.random.normal((256, 512))
    activations = mx.random.normal((64, 512))
    
    try:
        result = quantize_layer_watersic_mlx(
            weight=weight,
            activations=activations,
            target_bits=4.0,
            use_lmmse=True
        )
        
        print(f"✅ Layer quantization successful!")
        print(f"  - Compression ratio: {result['compression_ratio']:.2f}x")
        print(f"  - Bits per param: {result['bits_per_param']:.3f}")
        print(f"  - Waterfilling constant c: {result['c']:.6f}")
        print(f"  - Output shape: {result['Z_sic'].shape}")
        
    except Exception as e:
        print(f"❌ Layer quantization failed: {e}")
        return False
    
    # Test 2: State dict quantization
    print("\n📦 Test 2: State Dict Quantization")
    
    # Create a mock state dict
    flat_state = {
        'layer1.weight': mx.random.normal((128, 256)),
        'layer2.weight': mx.random.normal((256, 128)),
        'layer3.bias': mx.random.normal((128,))  # This should be passthrough
    }
    
    # Create calibration data
    calibration_data = {
        'layer1.weight': mx.random.normal((32, 256)),
        'layer2.weight': mx.random.normal((32, 128)),
    }
    
    # Mock hyperparameters
    class MockHparams:
        watersic_bits = 4.0
        watersic_lmmse_correction = True
        watersic_calibration_samples = 32
    
    hparams = MockHparams()
    
    try:
        quant_obj, quant_stats = quantize_state_dict_watersic_mlx(
            flat_state, calibration_data, hparams
        )
        
        print(f"✅ State dict quantization successful!")
        print(f"  - Total tensors: {quant_stats['num_tensors']}")
        print(f"  - WaterSIC tensors: {quant_stats['num_watersic_tensors']}")
        print(f"  - Baseline bytes: {quant_stats['baseline_tensor_bytes']:,}")
        print(f"  - Compressed bytes: {quant_stats['watersic_payload_bytes']:,}")
        
        # Test 3: Dequantization
        print("\n🔄 Test 3: Dequantization")
        reconstructed = dequantize_state_dict_watersic_mlx(quant_obj)
        
        print(f"✅ Dequantization successful!")
        print(f"  - Reconstructed layers: {len(reconstructed)}")
        
        # Check reconstruction error
        for name in ['layer1.weight', 'layer2.weight']:
            if name in reconstructed:
                orig = flat_state[name]
                recon = reconstructed[name]
                error = mx.mean((orig - recon) ** 2)
                print(f"  - {name} MSE error: {float(error):.8f}")
        
    except Exception as e:
        print(f"❌ State dict operations failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n🎉 All WaterSIC MLX tests passed!")
    return True

if __name__ == "__main__":
    test_watersic_mlx()