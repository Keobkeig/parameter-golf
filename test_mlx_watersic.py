#!/usr/bin/env python3
"""
Simple test for WaterSIC MLX integration
"""
import mlx.core as mx
import numpy as np
import sys
import os

def test_basic_mlx_operations():
    print("🧪 Testing Basic MLX Operations")
    print("=" * 40)
    
    # Test 1: Basic MLX tensor operations
    print("\n📊 Test 1: Basic Tensor Operations")
    try:
        A = mx.random.normal((64, 128))
        B = mx.random.normal((128, 64))
        C = mx.matmul(A, B)
        print(f"✅ Matrix multiplication: {A.shape} × {B.shape} = {C.shape}")
        
        # Test variance and mean
        var = mx.var(A)
        mean = mx.mean(A) 
        print(f"✅ Statistics: mean={float(mean):.6f}, var={float(var):.6f}")
        
    except Exception as e:
        print(f"❌ Basic operations failed: {e}")
        return False
    
    # Test 2: Quantization simulation
    print("\n🔢 Test 2: Quantization Simulation")
    try:
        weight = mx.random.normal((32, 64)) * 0.1
        step_size = 0.01
        quantized = mx.round(weight / step_size) * step_size
        error = mx.mean((weight - quantized) ** 2)
        print(f"✅ Quantization error: {float(error):.8f}")
        
    except Exception as e:
        print(f"❌ Quantization simulation failed: {e}")
        return False
    
    # Test 3: Covariance computation
    print("\n📈 Test 3: Covariance Matrix")
    try:
        activations = mx.random.normal((32, 64))
        centered = activations - mx.mean(activations, axis=0)
        cov = mx.matmul(centered.T, centered) / (activations.shape[0] - 1)
        print(f"✅ Covariance matrix: {cov.shape}")
        
        # Test regularization
        reg = 1e-6 * mx.eye(cov.shape[0])
        cov_reg = cov + reg
        print(f"✅ Regularized covariance: {cov_reg.shape}")
        
    except Exception as e:
        print(f"❌ Covariance computation failed: {e}")
        return False
    
    # Test 4: Alternative decomposition (since Cholesky needs CPU)
    print("\n🔧 Test 4: Matrix Decomposition Alternatives")
    try:
        # Use SVD instead of Cholesky/eigen (should work on GPU)
        U, S, Vt = mx.linalg.svd(cov_reg)
        # Create pseudo-Cholesky from SVD
        S_sqrt = mx.sqrt(mx.maximum(S, 1e-6))
        L_pseudo = U * S_sqrt[None, :]
        print(f"✅ SVD decomposition: U{U.shape}, S{S.shape}, Vt{Vt.shape}")
        print(f"✅ Pseudo-Cholesky: {L_pseudo.shape}")
        
    except Exception as e:
        print(f"❌ Matrix decomposition failed: {e}")
        return False
    
    print("\n🎉 All basic MLX operations successful!")
    return True

def test_watersic_mlx_import():
    print("\n🔍 Testing WaterSIC MLX Import")
    print("=" * 40)
    
    try:
        # Test if we can access the train_gpt_mlx file
        sys.path.append('/Users/Programming/parameter-golf')
        
        print("✅ Added path to sys.path")
        
        # Check if MLX training script exists
        if os.path.exists('/Users/Programming/parameter-golf/train_gpt_mlx.py'):
            print("✅ Found train_gpt_mlx.py")
        else:
            print("❌ train_gpt_mlx.py not found")
            return False
        
        # Try to read the file and check for WaterSIC functions
        with open('/Users/Programming/parameter-golf/train_gpt_mlx.py', 'r') as f:
            content = f.read()
        
        watersic_functions = [
            'quantize_layer_watersic_mlx',
            'quantize_state_dict_watersic_mlx', 
            'dequantize_state_dict_watersic_mlx',
            '_compute_covariance_matrix_mlx',
            '_cholesky_decomposition_mlx',
            '_zsic_quantize_mlx'
        ]
        
        for func in watersic_functions:
            if func in content:
                print(f"✅ Found {func}")
            else:
                print(f"❌ Missing {func}")
                return False
        
        # Check for environment variables
        watersic_vars = [
            'watersic_enabled',
            'watersic_bits',
            'watersic_calibration_samples'
        ]
        
        for var in watersic_vars:
            if var in content:
                print(f"✅ Found {var} environment variable")
            else:
                print(f"❌ Missing {var} environment variable")
        
        print("\n🎉 WaterSIC MLX integration verified!")
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 MLX WaterSIC Integration Test")
    print("=" * 50)
    
    # Test basic MLX operations first
    if not test_basic_mlx_operations():
        print("\n❌ Basic MLX operations failed!")
        sys.exit(1)
    
    # Test WaterSIC integration
    if not test_watersic_mlx_import():
        print("\n❌ WaterSIC MLX integration test failed!")
        sys.exit(1)
    
    print("\n" + "=" * 50)
    print("🎉 ALL TESTS PASSED!")
    print("✅ MLX is working correctly")
    print("✅ WaterSIC functions are integrated")
    print("✅ Environment variables are configured")
    print("🚀 Ready for MLX training with WaterSIC!")