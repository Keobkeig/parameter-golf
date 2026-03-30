#!/usr/bin/env python3
"""
Standalone comparison script for WaterSIC vs int6 quantization
Tests compression and reconstruction quality without full training dependencies
"""

import torch
import sys
import time
import numpy as np
from typing import Dict, Tuple, Any

# Import WaterSIC functions from train_gpt.py
sys.path.append('.')

def import_watersic_functions():
    """Import WaterSIC functions from train_gpt.py"""
    try:
        from train_gpt import (
            _get_calibration_data,
            _compute_covariance_matrix, 
            _cholesky_decomposition,
            _zsic_quantize,
            _binary_search_rate,
            quantize_layer_watersic
        )
        return {
            'get_calibration_data': _get_calibration_data,
            'compute_covariance_matrix': _compute_covariance_matrix,
            'cholesky_decomposition': _cholesky_decomposition,
            'zsic_quantize': _zsic_quantize,
            'binary_search_rate': _binary_search_rate,
            'quantize_layer_watersic': quantize_layer_watersic
        }
    except ImportError as e:
        print(f"❌ Failed to import WaterSIC functions: {e}")
        return None

def int6_quantize(weight: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Simple int6 quantization baseline
    """
    # Compute scale and zero point
    w_min, w_max = weight.min(), weight.max()
    scale = (w_max - w_min) / 63  # 2^6 - 1 = 63 levels
    zero_point = torch.round(-w_min / scale)
    
    # Quantize
    w_quant = torch.round(weight / scale + zero_point)
    w_quant = torch.clamp(w_quant, 0, 63).to(torch.uint8)
    
    # Dequantize for error measurement
    w_dequant = (w_quant.float() - zero_point) * scale
    
    # Calculate compression metrics
    original_bits = weight.numel() * 32  # fp32
    quantized_bits = w_quant.numel() * 6 + 32 * 2  # 6 bits per param + scale/zero_point
    compression_ratio = original_bits / quantized_bits
    
    return w_dequant, {
        'quantized_data': w_quant,
        'scale': scale,
        'zero_point': zero_point,
        'bits_per_param': 6.0,
        'compression_ratio': compression_ratio,
        'method': 'int6'
    }

def generate_test_weight(shape: Tuple[int, int], distribution: str = 'normal') -> torch.Tensor:
    """Generate test weight matrices with different distributions"""
    if distribution == 'normal':
        return torch.randn(shape) * 0.02
    elif distribution == 'uniform':
        return torch.rand(shape) * 0.1 - 0.05
    elif distribution == 'outliers':
        w = torch.randn(shape) * 0.02
        # Add some outliers
        outlier_mask = torch.rand(shape) < 0.05
        num_outliers = int(outlier_mask.sum().item())
        if num_outliers > 0:
            w[outlier_mask] += torch.randn(num_outliers) * 0.2
        return w
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

def compare_methods(weight: torch.Tensor, watersic_funcs: Dict) -> Dict[str, Any]:
    """Compare WaterSIC vs int6 on a single weight matrix"""
    
    results = {}
    
    # Test int6 quantization
    start_time = time.time()
    w_int6, int6_info = int6_quantize(weight)
    int6_time = time.time() - start_time
    
    int6_error = torch.mean((weight - w_int6) ** 2).item()
    results['int6'] = {
        **int6_info,
        'reconstruction_error': int6_error,
        'time_seconds': int6_time
    }
    
    # Test WaterSIC quantization
    start_time = time.time()
    try:
        # Generate dummy activations for calibration  
        batch_size, seq_len = 16, 128
        in_dim = weight.shape[1]  # Input dimension (weight is [out_dim, in_dim])
        activations = torch.randn(batch_size * seq_len, in_dim)  # [num_samples, in_dim]
        
        watersic_result = watersic_funcs['quantize_layer_watersic'](
            weight=weight,
            activations=activations,
            target_bits=4.0,
            attention_weights=None,
            use_lmmse=True,
            eps_aw=0.5,
            eps_qr=0.1
        )
        watersic_time = time.time() - start_time
        
        # Reconstruct weight (simplified - just use the quantized values)
        w_watersic = watersic_result['Z_sic']
        watersic_error = torch.mean((weight - w_watersic) ** 2).item()
        
        results['watersic'] = {
            'bits_per_param': watersic_result['bits_per_param'],
            'compression_ratio': watersic_result['compression_ratio'],
            'reconstruction_error': watersic_error,
            'time_seconds': watersic_time,
            'method': 'watersic'
        }
        
    except Exception as e:
        results['watersic'] = {
            'error': str(e),
            'reconstruction_error': float('inf'),
            'method': 'watersic'
        }
    
    return results

def main():
    print("🔬 WaterSIC vs int6 Compression Comparison")
    print("=" * 50)
    
    # Import WaterSIC functions
    watersic_funcs = import_watersic_functions()
    if not watersic_funcs:
        print("❌ Cannot run comparison without WaterSIC functions")
        return
    
    # Test configurations
    test_configs = [
        {'shape': (512, 1024), 'dist': 'normal', 'name': 'Small Linear Layer (normal)'},
        {'shape': (1024, 2048), 'dist': 'normal', 'name': 'Medium Linear Layer (normal)'},
        {'shape': (512, 1024), 'dist': 'outliers', 'name': 'Small Linear Layer (with outliers)'},
    ]
    
    all_results = []
    
    for i, config in enumerate(test_configs, 1):
        print(f"\n📊 Test {i}: {config['name']}")
        print("-" * 40)
        
        # Generate test weight
        weight = generate_test_weight(config['shape'], config['dist'])
        print(f"Weight shape: {weight.shape}")
        print(f"Weight range: [{weight.min().item():.6f}, {weight.max().item():.6f}]")
        print(f"Weight std: {weight.std().item():.6f}")
        
        # Compare methods
        results = compare_methods(weight, watersic_funcs)
        all_results.append({**config, 'results': results})
        
        # Display results
        for method in ['int6', 'watersic']:
            if method in results and 'error' not in results[method]:
                r = results[method]
                print(f"\n{method.upper()}:")
                print(f"  Bits per param: {r['bits_per_param']:.2f}")
                print(f"  Compression ratio: {r['compression_ratio']:.2f}x")
                print(f"  Reconstruction error: {r['reconstruction_error']:.8f}")
                print(f"  Time: {r['time_seconds']:.4f}s")
            elif method in results:
                print(f"\n{method.upper()}: ❌ {results[method]['error']}")
    
    # Summary
    print("\n" + "=" * 50)
    print("📈 SUMMARY COMPARISON")
    print("=" * 50)
    
    int6_errors = []
    watersic_errors = []
    int6_ratios = []
    watersic_ratios = []
    
    for result in all_results:
        res = result['results']
        if 'int6' in res and 'error' not in res['int6']:
            int6_errors.append(res['int6']['reconstruction_error'])
            int6_ratios.append(res['int6']['compression_ratio'])
        if 'watersic' in res and 'error' not in res['watersic']:
            watersic_errors.append(res['watersic']['reconstruction_error'])
            watersic_ratios.append(res['watersic']['compression_ratio'])
    
    if int6_errors and watersic_errors:
        print(f"\nAverage Reconstruction Error:")
        print(f"  int6:     {np.mean(int6_errors):.8f} (±{np.std(int6_errors):.8f})")
        print(f"  WaterSIC: {np.mean(watersic_errors):.8f} (±{np.std(watersic_errors):.8f})")
        
        improvement = (np.mean(int6_errors) - np.mean(watersic_errors)) / np.mean(int6_errors) * 100
        print(f"\nWaterSIC Error Improvement: {improvement:+.1f}%")
        
        print(f"\nAverage Compression Ratio:")
        print(f"  int6:     {np.mean(int6_ratios):.2f}x")
        print(f"  WaterSIC: {np.mean(watersic_ratios):.2f}x")
        
        if improvement > 0:
            print("\n🎉 WaterSIC shows better reconstruction quality!")
        else:
            print("\n⚠️  int6 shows better reconstruction quality")
    
    print("\n✅ Comparison complete!")

if __name__ == "__main__":
    main()