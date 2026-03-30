# WaterSIC: Information-Theoretically Near-Optimal Quantization

This implementation integrates the **WaterSIC** quantization algorithm from ["WaterSIC: information-theoretically (near) optimal linear layer quantization" (arXiv:2603.04956)](https://arxiv.org/html/2603.04956v1) into the parameter-golf training framework.

## Overview

WaterSIC achieves **within 0.255 bits** of the information-theoretic optimum for linear layer quantization through:

1. **Waterfilling Rate Allocation**: Different quantization rates per column based on activation covariance structure
2. **ZSIC Algorithm**: Zero-power Successive Interference Cancellation for optimal quantization
3. **Advanced Corrections**: LMMSE, activation drift, residual stream, and diagonal rescaling optimizations
4. **Entropy Coding**: Lossless compression with zstd for variable-rate integer quantization

## Key Innovation

Unlike existing methods that use uniform quantization rates, WaterSIC allocates rates using the classical **waterfilling** solution:

```
α_i = c / |L[i,i]|
```

where `L` is the Cholesky factor of the activation covariance matrix and `c` is determined by binary search to meet the target bit rate.

## Usage

### Basic Usage

Enable WaterSIC quantization with environment variables:

```bash
WATERSIC_ENABLED=1 WATERSIC_BITS=4.0 python train_gpt.py
```

### Advanced Configuration

```bash
# Core settings
export WATERSIC_ENABLED=1
export WATERSIC_BITS=4.0                      # Target bits per parameter
export WATERSIC_CALIBRATION_SAMPLES=512       # Calibration dataset size

# Algorithm features
export WATERSIC_LMMSE_CORRECTION=1             # Enable LMMSE bias correction
export WATERSIC_ACTIVATION_DRIFT=1             # Enable drift correction
export WATERSIC_RESIDUAL_CORRECTION=1          # Enable residual stream correction
export WATERSIC_ATTENTION_WEIGHTED=1           # Enable attention weighting
export WATERSIC_ADAPTIVE_MIXING=1              # Enable adaptive mixing
export WATERSIC_DIAGONAL_RESCALERS=1           # Enable rescaler optimization

# Fine-tuning parameters
export WATERSIC_QR_EPSILON=0.1                 # Quantization replacement mixing
export WATERSIC_AW_EPSILON=0.5                 # Attention weighting mixing

# Training integration
export QAT_ENABLED=1                           # Enable quantization-aware training

python train_gpt.py
```

## Architecture Integration

### Quantization-Aware Training (QAT)

WaterSIC integrates with the existing `CastedLinear` infrastructure:

- **Fake Quantization**: During training, weights are fake-quantized using waterfilling rates
- **Straight-Through Estimator**: Gradients flow through the quantization operation
- **Late QAT**: Automatically enables when learning rate drops below threshold

### Layer Selection

WaterSIC is applied to:
- ✅ **Attention layers**: `.attn.c_q`, `.attn.c_k`, `.attn.c_v`, `.attn.proj`
- ✅ **MLP layers**: `.mlp.fc`, `.mlp.proj`
- ❌ **Embeddings**: Use existing int8 quantization (better for small parameters)
- ❌ **Layer norms**: Kept in higher precision

### Compression Pipeline

```
1. Training with fake quantization (optional)
2. Calibration data collection (512 samples)
3. Covariance matrix estimation
4. Cholesky decomposition: Σ_X = L L^T
5. Waterfilling rate calculation: α_i = c / |L[i,i]|
6. ZSIC quantization with LMMSE correction
7. Diagonal rescaler optimization
8. Entropy coding with zstd-22
```

## Expected Performance

Based on the paper's results, WaterSIC should provide:

- **Better compression quality** than existing int6 approaches
- **Theoretical guarantees**: Within 0.255 bits of optimal
- **Adaptive rate allocation**: Automatically adjusts to activation statistics
- **Robust to rotations**: Performance invariant to coordinate transformations

### Comparison with Parameter Golf Baseline

| Method | Compression | Quality | Theoretical |
|--------|-------------|---------|-------------|
| GPTQ-lite int6 | ~6 bits | Good | Ad-hoc |
| WaterSIC | 4-6 bits | Better | Near-optimal |

## Implementation Details

### Core Algorithm: ZSIC

The Zero-power Successive Interference Cancellation algorithm:

```python
def _zsic_quantize(Y, L, alpha_diag, lmmse_correction=True):
    # Process columns from n to 1
    for i in range(n-1, -1, -1):
        # Round to nearest integer scaled by alpha_i
        z_i = torch.round(Y_work[:, i] / alpha_i)
        
        # LMMSE correction
        if lmmse_correction:
            gamma[i] = (Y_work[:, i] * z_i).sum() / (alpha_i * (z_i * z_i).sum())
        
        # Subtract interference from remaining columns
        correction = gamma[i] * alpha_i * z_i
        Y_work[:, :i] -= correction.unsqueeze(1) * L[i, :i].unsqueeze(0)
```

### Waterfilling Property

The key insight is allocating more bits to "important" directions:

```python
# Compute activation covariance
Sigma_X = _compute_covariance_matrix(activations)
L = torch.linalg.cholesky(Sigma_X)

# Waterfilling: allocate rates inversely proportional to L diagonal
L_diag = torch.diag(L).abs()
alpha_diag = c / (L_diag + 1e-8)  # Higher alpha = more quantization noise
```

### Advanced Corrections

1. **LMMSE Correction**: Reduces quantization bias
   ```python
   gamma[i] = numerator / (denominator + 1e-8)
   ```

2. **Activation Drift Correction**: Handles distribution shift between unquantized and quantized inputs
   ```python
   # Use cross-covariance Σ_{X,X̂} instead of Σ_X
   ```

3. **Residual Stream Correction**: Special handling for down-projection layers
   ```python
   # Account for residual connections: Y = WX + R
   ```

## File Structure

- `train_gpt.py`: Main training script with WaterSIC integration
- `validate_watersic.py`: Implementation validator (syntax and structure check)
- `test_watersic.py`: Unit tests (requires PyTorch environment)

## Implementation Status

✅ **Completed Features**:
- Core ZSIC algorithm with waterfilling
- LMMSE correction for bias reduction
- Activation drift correction (Qronos approach)
- Residual stream correction for down-projections
- Diagonal rescaler optimization
- Entropy coding with zstd compression
- Attention-weighted calibration for QKV layers
- Binary search rate assignment
- QAT fake quantization integration
- Environment variable configuration

⚠️ **Limitations**:
- Simplified attention weighting (uses uniform weights as placeholder)
- No TTT (Test-Time Training) integration yet
- Performance not fully optimized for large models

## Theoretical Background

WaterSIC is based on classical information theory:

1. **Rate-Distortion Theory**: Fundamental limits of lossy compression
2. **Waterfilling**: Optimal rate allocation across parallel Gaussian channels
3. **Successive Interference Cancellation**: Near-optimal quantization algorithm
4. **LMMSE Estimation**: Optimal linear reconstruction

The algorithm achieves the **waterfilling lower bound** up to a constant gap of **0.255 bits**, making it theoretically near-optimal.

## Future Improvements

- **Performance optimization**: GPU-accelerated matrix operations
- **Memory efficiency**: Streaming calibration data collection
- **TTT integration**: Test-time training with WaterSIC
- **Adaptive mixing**: Smarter attention weight collection
- **Model-specific tuning**: Per-architecture optimization

## Citation

If you use this implementation, please cite the original paper:

```bibtex
@article{lifar2026watersic,
  title={WaterSIC: information-theoretically (near) optimal linear layer quantization},
  author={Lifar, Egor and Savkin, Semyon and Ordentlich, Or and Polyanskiy, Yury},
  journal={arXiv preprint arXiv:2603.04956},
  year={2026}
}
```

---

**Note**: This implementation is based on the paper available at https://arxiv.org/html/2603.04956v1 and has been integrated into the parameter-golf challenge framework for practical evaluation.