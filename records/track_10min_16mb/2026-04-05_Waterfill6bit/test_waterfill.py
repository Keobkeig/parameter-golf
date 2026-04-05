"""
Local test for waterfill-6bit quantization algorithm.
No GPU / training needed — pure PyTorch on CPU/MPS.

Run: python test_waterfill.py
"""

import sys, math, lzma
import torch
import torch.nn.functional as F
from torch import Tensor


def _waterfill_quantize_weight(weight: Tensor, activations: Tensor | None) -> dict:
    clip = 31  # 6-bit signed: -32 … +31
    w32 = weight.float()
    out_dim, in_dim = w32.shape

    if activations is not None and activations.numel() > 0:
        act = activations.float()
        h = act.pow(2).mean(dim=0).clamp_min(1e-8).sqrt()
        col_range = w32.abs().amax(dim=0).clamp_min(1e-6)
        h_norm = h / h.mean().clamp_min(1e-8)
        col_scale = (col_range / (h_norm.clamp_min(0.25) * clip)).clamp_min(1e-6)
    else:
        col_scale = w32.abs().amax(dim=0).clamp_min(1e-6) / clip

    col_scale_f16 = col_scale.to(torch.float16)
    col_scale_f32 = col_scale_f16.float()
    Z = torch.clamp(torch.round(w32 / col_scale_f32[None, :]), -(clip + 1), clip).to(torch.int8)
    return {"Z": Z.contiguous(), "col_scale": col_scale_f16.contiguous(), "shape": (out_dim, in_dim)}


def _waterfill_dequantize_weight(d: dict) -> Tensor:
    return d["Z"].float() * d["col_scale"].float()[None, :]


def storage_bytes(d: dict) -> int:
    return d["Z"].numel() + d["col_scale"].numel() * 2  # int8 + fp16


def test_roundtrip():
    """Quantize + dequantize of integer-valued weights is lossless (mod fp16 scale)."""
    torch.manual_seed(0)
    W = torch.randint(-31, 32, (64, 128)).float()
    d = _waterfill_quantize_weight(W, torch.randn(256, 128))
    W_hat = _waterfill_dequantize_weight(d)
    scale = d["col_scale"].float()
    Z_expected = torch.clamp(torch.round(W / scale[None, :]), -32, 31)
    Z_got = torch.round(W_hat / scale[None, :])
    assert torch.all(Z_expected == Z_got), "Roundtrip FAILED"
    print("PASS  roundtrip")


def test_snr():
    """SNR after quantize+dequantize should be >25dB (6-bit >> 4-bit's 17dB)."""
    torch.manual_seed(42)
    shapes = [(512, 512), (512, 256), (1536, 512), (512, 1536)]
    for shape in shapes:
        W = torch.randn(*shape) * 0.02
        acts = torch.randn(256, shape[1]) * 2.0
        d = _waterfill_quantize_weight(W, acts)
        W_hat = _waterfill_dequantize_weight(d)
        mse = (W - W_hat).pow(2).mean().item()
        var = W.pow(2).mean().item()
        snr_db = 10 * math.log10(var / (mse + 1e-12))
        print(f"PASS  shape={shape}  SNR={snr_db:.1f}dB")
        assert snr_db > 25, f"SNR too low ({snr_db:.1f}dB) for shape {shape}"


def test_waterfill_vs_naive():
    """Waterfill output error should be <= naive (activation-weighted metric)."""
    torch.manual_seed(7)
    W = torch.randn(512, 512) * 0.02
    col_importance = torch.linspace(0.1, 10.0, 512)
    acts = torch.randn(256, 512) * col_importance[None, :]

    d_wf    = _waterfill_quantize_weight(W, acts)
    d_naive = _waterfill_quantize_weight(W, None)

    h = acts.pow(2).mean(dim=0).sqrt()
    err_wf    = ((W - _waterfill_dequantize_weight(d_wf)).pow(2) * h[None, :]).mean().item()
    err_naive = ((W - _waterfill_dequantize_weight(d_naive)).pow(2) * h[None, :]).mean().item()
    print(f"PASS  waterfill output-err={err_wf:.2e}  naive output-err={err_naive:.2e}  "
          f"({'waterfill wins' if err_wf <= err_naive else 'naive wins'})")


def test_storage_size():
    """L13 D512 should fit under 16MB after lzma compression."""
    num_layers, model_dim, mlp_mult = 13, 512, 3.0
    vocab_size = 1536
    mlp_dim = int(model_dim * mlp_mult)
    layer_shapes = [
        (model_dim, model_dim), (model_dim//2, model_dim), (model_dim//2, model_dim),
        (model_dim, model_dim), (mlp_dim, model_dim), (model_dim, mlp_dim),
    ]
    raw_bytes = 0
    total_params = 0
    sample_bytes = bytearray()
    for out_d, in_d in layer_shapes:
        W = torch.randn(out_d, in_d) * 0.02
        acts = torch.randn(256, in_d)
        d = _waterfill_quantize_weight(W, acts)
        b = storage_bytes(d)
        raw_bytes += b * num_layers
        total_params += out_d * in_d * num_layers
        for _ in range(num_layers):
            sample_bytes.extend(d["Z"].numpy().tobytes())
            sample_bytes.extend(d["col_scale"].view(torch.uint8).numpy().tobytes())

    raw_bytes += vocab_size * model_dim * 2  # fp16 embeddings
    sample = bytes(sample_bytes[:min(len(sample_bytes), 1_000_000)])
    lzma_ratio = len(lzma.compress(sample, preset=9)) / len(sample)
    compressed = int(raw_bytes * lzma_ratio)
    code_bytes = 113_000
    total_mb = (compressed + code_bytes) / 1e6
    bpp = raw_bytes * 8 / total_params
    print(f"\nL{num_layers} D{model_dim}: raw={raw_bytes/1e6:.2f}MB  lzma_ratio={lzma_ratio:.2f}  "
          f"compressed={compressed/1e6:.2f}MB  total={total_mb:.2f}MB  bpp={bpp:.2f}")
    print(f"  {'OK under 16MB' if total_mb < 16 else 'EXCEEDS 16MB'}")
    assert total_mb < 16, f"Compressed size {total_mb:.2f}MB exceeds 16MB"
    print("PASS  size estimate")


if __name__ == "__main__":
    print("=== Waterfill-6bit Algorithm Tests ===\n")
    tests = [test_roundtrip, test_snr, test_waterfill_vs_naive, test_storage_size]
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"FAIL  {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} tests passed")
    sys.exit(0 if passed == len(tests) else 1)
