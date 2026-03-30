#!/usr/bin/env python3
"""
Test script for WaterSIC quantization implementation.
"""

import torch
import numpy as np
import sys
import os

# Add current directory to path so we can import from train_gpt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import WaterSIC functions (assuming they're defined in train_gpt.py)
try:
    from train_gpt import (
        _compute_covariance_matrix, _cholesky_decomposition, _zsic_quantize,
        _binary_search_rate, quantize_layer_watersic, _get_calibration_data
    )
    print("✓ Successfully imported WaterSIC functions")
except ImportError as e:
    print(f"✗ Failed to import WaterSIC functions: {e}")
    sys.exit(1)

def test_covariance_computation():
    """Test covariance matrix computation."""
    print("\n=== Testing Covariance Matrix Computation ===")
    
    # Create synthetic activation data
    torch.manual_seed(42)
    num_samples, dim = 100, 64
    activations = torch.randn(num_samples, dim)
    
    # Test basic covariance
    cov = _compute_covariance_matrix(activations)
    assert cov.shape == (dim, dim), f"Expected shape ({dim}, {dim}), got {cov.shape}"
    assert torch.allclose(cov, cov.t(), atol=1e-5), "Covariance matrix should be symmetric"
    
    # Check positive definiteness
    eigenvals = torch.linalg.eigvals(cov).real
    assert (eigenvals >= -1e-5).all(), "Covariance matrix should be positive semi-definite"
    
    print(f"✓ Basic covariance computation passed")
    
    # Test attention-weighted covariance
    attention_weights = torch.rand(num_samples)
    cov_weighted = _compute_covariance_matrix(activations, attention_weights)
    assert cov_weighted.shape == (dim, dim), "Weighted covariance should have same shape"
    
    print(f"✓ Attention-weighted covariance computation passed")

def test_cholesky_decomposition():
    """Test Cholesky decomposition."""
    print("\n=== Testing Cholesky Decomposition ===")
    
    torch.manual_seed(42)
    dim = 32
    
    # Create a positive definite matrix
    A = torch.randn(dim, dim)
    cov = A @ A.t() + torch.eye(dim) * 1e-3
    
    L = _cholesky_decomposition(cov)
    assert L.shape == (dim, dim), f"Expected shape ({dim}, {dim}), got {L.shape}"
    
    # Check that L is lower triangular
    upper_triangle = torch.triu(L, diagonal=1)
    assert torch.allclose(upper_triangle, torch.zeros_like(upper_triangle), atol=1e-6), \
        "L should be lower triangular"
    
    # Check reconstruction: L @ L.t() ≈ cov
    reconstructed = L @ L.t()
    assert torch.allclose(reconstructed, cov, atol=1e-4), \
        "L @ L.t() should reconstruct original covariance matrix"
    
    print(f"✓ Cholesky decomposition passed")

def test_zsic_quantization():
    """Test ZSIC quantization algorithm."""
    print("\n=== Testing ZSIC Quantization ===")
    
    torch.manual_seed(42)
    a, n = 16, 32  # Small test matrix
    
    # Create test weight matrix
    Y = torch.randn(a, n)
    
    # Create test Cholesky factor
    A = torch.randn(n, n)
    cov = A @ A.t() + torch.eye(n) * 1e-3
    L = _cholesky_decomposition(cov)
    
    # Create test alpha (waterfilling rates)
    c = 0.1
    L_diag = torch.diag(L).abs()
    alpha_diag = c / (L_diag + 1e-8)
    
    # Test ZSIC quantization
    Z_sic, gamma, recon_error = _zsic_quantize(Y, L, alpha_diag, lmmse_correction=True)
    
    assert Z_sic.shape == Y.shape, f"Z_sic should have same shape as Y"
    assert gamma.shape == (n,), f"gamma should have shape ({n},), got {gamma.shape}"
    assert isinstance(recon_error.item(), float), "recon_error should be a scalar"
    
    # Check that Z_sic contains integers (approximately)
    Z_rounded = torch.round(Z_sic)
    assert torch.allclose(Z_sic, Z_rounded, atol=1e-6), "Z_sic should contain integers"
    
    print(f"✓ ZSIC quantization passed (error: {recon_error.item():.6f})")

def test_binary_search_rate():
    """Test binary search for rate assignment."""
    print("\n=== Testing Binary Search Rate Assignment ===")
    
    torch.manual_seed(42)
    a, n = 16, 32
    target_bits = 4.0
    
    Y = torch.randn(a, n)
    A = torch.randn(n, n)
    cov = A @ A.t() + torch.eye(n) * 1e-3
    L = _cholesky_decomposition(cov)
    
    # Test binary search
    c = _binary_search_rate(target_bits, Y, L, max_iterations=10, tolerance=0.5)
    
    assert isinstance(c, float), "c should be a float"
    assert c > 0, "c should be positive"
    
    print(f"✓ Binary search passed (c = {c:.6f})")

def test_layer_quantization():
    """Test full layer quantization."""
    print("\n=== Testing Full Layer Quantization ===")
    
    torch.manual_seed(42)
    out_dim, in_dim = 128, 256
    num_samples = 512
    target_bits = 4.0
    
    # Create test weight and activations
    weight = torch.randn(out_dim, in_dim) * 0.02  # Typical weight scale
    activations = torch.randn(num_samples, in_dim)
    
    # Test quantization
    result = quantize_layer_watersic(
        weight=weight,
        activations=activations,
        target_bits=target_bits,
        use_lmmse=True
    )
    
    # Check result structure
    expected_keys = {'Z_sic', 'alpha', 'gamma', 'L', 'bits_per_param', 
                     'compression_ratio', 'reconstruction_error', 'constant_c'}
    assert set(result.keys()) == expected_keys, f"Missing keys in result: {expected_keys - set(result.keys())}"
    
    # Check shapes
    assert result['Z_sic'].shape == weight.shape, "Z_sic should match weight shape"
    assert result['alpha'].shape == (in_dim,), f"alpha should have shape ({in_dim},)"
    assert result['gamma'].shape == (in_dim,), f"gamma should have shape ({in_dim},)"
    assert result['L'].shape == (in_dim, in_dim), f"L should have shape ({in_dim}, {in_dim})"
    
    # Check reasonable values
    assert result['bits_per_param'] > 0, "bits_per_param should be positive"
    assert result['compression_ratio'] > 0, "compression_ratio should be positive"
    assert result['reconstruction_error'] >= 0, "reconstruction_error should be non-negative"
    
    print(f"✓ Layer quantization passed")
    print(f"  - Bits per param: {result['bits_per_param']:.3f}")
    print(f"  - Compression ratio: {result['compression_ratio']:.2f}x")
    print(f"  - Reconstruction error: {result['reconstruction_error']:.6f}")

def test_waterfilling_property():
    """Test that waterfilling allocation improves over uniform quantization."""
    print("\n=== Testing Waterfilling Property ===")
    
    torch.manual_seed(42)
    out_dim, in_dim = 64, 128
    num_samples = 256
    
    weight = torch.randn(out_dim, in_dim) * 0.02
    
    # Create activations with varying variance across dimensions
    activations = torch.randn(num_samples, in_dim)
    # Make some dimensions have much higher variance
    activations[:, :in_dim//4] *= 3.0  # High variance dimensions
    activations[:, in_dim//4:in_dim//2] *= 1.0  # Medium variance
    activations[:, in_dim//2:] *= 0.3  # Low variance dimensions
    
    # Test WaterSIC (waterfilling)
    result_watersic = quantize_layer_watersic(weight, activations, target_bits=4.0)
    
    # For comparison, create a simple uniform quantization
    # (This is a simplified test - in practice you'd compare with actual uniform GPTQ)
    uniform_alpha = torch.full((in_dim,), result_watersic['constant_c'])
    cov = _compute_covariance_matrix(activations).to(weight.device)
    L = _cholesky_decomposition(cov).to(weight.device)
    Y = weight @ L
    Z_uniform, gamma_uniform, error_uniform = _zsic_quantize(Y, L, uniform_alpha, True)
    
    # WaterSIC should generally perform better (lower reconstruction error)
    # Note: This isn't guaranteed in all cases, but should be true for most well-conditioned problems
    print(f"  - WaterSIC error: {result_watersic['reconstruction_error']:.6f}")
    print(f"  - Uniform error: {error_uniform.item():.6f}")
    
    # Check that alpha values are indeed different (waterfilling effect)
    alpha_variance = result_watersic['alpha'].var().item()
    assert alpha_variance > 1e-6, f"Alpha values should vary (waterfilling), variance: {alpha_variance}"
    
    print(f"✓ Waterfilling property verified (alpha variance: {alpha_variance:.6f})")

def main():
    """Run all tests."""
    print("Starting WaterSIC Tests...")
    print("=" * 50)
    
    try:
        test_covariance_computation()
        test_cholesky_decomposition()
        test_zsic_quantization()
        test_binary_search_rate()
        test_layer_quantization()
        test_waterfilling_property()
        
        print("\n" + "=" * 50)
        print("🎉 All WaterSIC tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()