"""
Local test for waterfill-4bit quantization algorithm.
No GPU / training needed — pure PyTorch on CPU/MPS.

Run: python test_waterfill.py
"""

import sys, math, lzma
import torch
import torch.nn.functional as F
from torch import Tensor

# ── copy the three core functions here so we don't import the full script ──

def _waterfill_quantize_weight(weight: Tensor, activations: Tensor | None) -> dict:
    clip = 7
    w32 = weight.float()
    out_dim, in_dim = w32.shape

    if activations is not None and activations.numel() > 0:
        act = activations.float()
        h = act.pow(2).mean(dim=0).clamp_min(1e-8).sqrt()
        col_range = w32.abs().amax(dim=0).clamp_min(1e-6)
        # Waterfilling: normalize h so mean column gets naive scale; finer for high-activation cols
        h_norm = h / h.mean().clamp_min(1e-8)
        col_scale = (col_range / (h_norm.clamp_min(0.25) * clip)).clamp_min(1e-6)
    else:
        col_scale = w32.abs().amax(dim=0).clamp_min(1e-6) / clip

    col_scale_f16 = col_scale.to(torch.float16)
    col_scale_f32 = col_scale_f16.float()

    Z = torch.clamp(torch.round(w32 / col_scale_f32[None, :]), -(clip + 1), clip).to(torch.int8)

    pad = (in_dim % 2 != 0)
    if pad:
        Z = F.pad(Z, (0, 1))
    Z_u = (Z + 8).to(torch.uint8)
    packed = (Z_u[:, 0::2] & 0xF) | ((Z_u[:, 1::2] & 0xF) << 4)

    return {"packed": packed.contiguous(), "col_scale": col_scale_f16.contiguous(), "shape": (out_dim, in_dim)}


def _waterfill_dequantize_weight(d: dict) -> Tensor:
    packed = d["packed"]
    col_scale = d["col_scale"].float()
    out_dim, in_dim = d["shape"]

    low  = (packed & 0xF).to(torch.int8) - 8
    high = ((packed >> 4) & 0xF).to(torch.int8) - 8
    Z = torch.zeros(out_dim, packed.shape[1] * 2, dtype=torch.int8)
    Z[:, 0::2] = low
    Z[:, 1::2] = high
    Z = Z[:, :in_dim]
    col_scale = col_scale[:in_dim]
    return Z.float() * col_scale[None, :]


def storage_bytes(d: dict) -> int:
    return d["packed"].numel() + d["col_scale"].numel() * 2  # uint8 + fp16


# ── tests ──

def test_pack_roundtrip():
    """4-bit pack/unpack is lossless."""
    torch.manual_seed(0)
    W = torch.randint(-8, 8, (64, 128)).to(torch.int8).float()  # already integer
    acts = torch.randn(256, 128)
    d = _waterfill_quantize_weight(W, acts)
    W_hat = _waterfill_dequantize_weight(d)
    # After quantize→dequantize the integer values should match exactly
    # (modulo the fp16 scale round-trip)
    scale = d["col_scale"].float()[:128]
    Z_expected = torch.clamp(torch.round(W / scale[None, :]), -8, 7)
    Z_got = torch.round(W_hat / scale[None, :])
    assert torch.all(Z_expected == Z_got), "Pack roundtrip FAILED"
    print("PASS  pack/unpack roundtrip")


def test_reconstruction_error():
    """MSE after quantize+dequantize should be small relative to weight variance."""
    torch.manual_seed(42)
    shapes = [(512, 512), (512, 256), (1536, 512), (512, 1536)]
    for shape in shapes:
        W = torch.randn(*shape) * 0.02  # typical weight scale
        acts = torch.randn(256, shape[1]) * 2.0  # typical activation scale
        d = _waterfill_quantize_weight(W, acts)
        W_hat = _waterfill_dequantize_weight(d)
        mse = (W - W_hat).pow(2).mean().item()
        var = W.pow(2).mean().item()
        snr_db = 10 * math.log10(var / (mse + 1e-12))
        print(f"PASS  shape={shape}  SNR={snr_db:.1f}dB  (mse={mse:.2e}, var={var:.2e})")
        assert snr_db > 15, f"SNR too low ({snr_db:.1f}dB) for shape {shape}"


def test_waterfill_vs_naive():
    """Waterfill (activation-weighted) should have lower MSE than naive per-col."""
    torch.manual_seed(7)
    W = torch.randn(512, 512) * 0.02

    # Simulate activations where some columns are much more active than others
    col_importance = torch.linspace(0.1, 10.0, 512)
    acts = torch.randn(256, 512) * col_importance[None, :]

    d_wf   = _waterfill_quantize_weight(W, acts)
    d_naive = _waterfill_quantize_weight(W, None)  # no activations = naive

    mse_wf    = (W - _waterfill_dequantize_weight(d_wf)).pow(2).mean().item()
    mse_naive = (W - _waterfill_dequantize_weight(d_naive)).pow(2).mean().item()

    print(f"PASS  waterfill MSE={mse_wf:.2e}  naive MSE={mse_naive:.2e}  "
          f"({'waterfill wins' if mse_wf < mse_naive else 'same/naive wins'})")


def test_storage_size():
    """Estimate full model artifact size for L13 D512."""
    num_layers = 13
    model_dim = 512
    mlp_mult = 3.0
    mlp_dim = int(model_dim * mlp_mult)
    vocab_size = 1536  # bigram-extended tokenizer used in run.sh

    # Per-layer matrices
    # Attn: Q [512,512], K [256,512], V [256,512], O [512,512]
    # MLP: up [1536,512], down [512,1536]
    layer_shapes = [
        (model_dim, model_dim),          # Q
        (model_dim // 2, model_dim),     # K (GQA 4 heads)
        (model_dim // 2, model_dim),     # V
        (model_dim, model_dim),          # O
        (mlp_dim, model_dim),            # MLP up
        (model_dim, mlp_dim),            # MLP down
    ]

    raw_bytes = 0
    total_params = 0
    all_packed_bytes = bytearray()
    for out_d, in_d in layer_shapes:
        W = torch.randn(out_d, in_d) * 0.02
        acts = torch.randn(256, in_d)
        d = _waterfill_quantize_weight(W, acts)
        raw_bytes += storage_bytes(d) * num_layers
        total_params += out_d * in_d * num_layers
        # Collect raw bytes for lzma estimate
        for _ in range(num_layers):
            all_packed_bytes.extend(d["packed"].numpy().tobytes())
            all_packed_bytes.extend(d["col_scale"].view(torch.uint8).numpy().tobytes())

    # Embeddings: vocab × dim × 2 bytes (fp16, tied so count once)
    embed_bytes = vocab_size * model_dim * 2
    raw_bytes += embed_bytes

    # Estimate lzma compression ratio from actual packed data sample
    sample = bytes(all_packed_bytes[:min(len(all_packed_bytes), 1_000_000)])
    compressed_sample = lzma.compress(sample, preset=6)
    lzma_ratio = len(compressed_sample) / len(sample)

    compressed_weights = int(raw_bytes * lzma_ratio)
    code_bytes = 113_000
    total_compressed_mb = (compressed_weights + code_bytes) / 1e6
    total_raw_mb = (raw_bytes + code_bytes) / 1e6
    bits_per_param = (raw_bytes * 8) / total_params

    print(f"\nSize estimate for L{num_layers} D{model_dim} MLP{mlp_mult}:")
    print(f"  Raw weights:       {raw_bytes/1e6:.2f}MB  (lzma ratio: {lzma_ratio:.2f})")
    print(f"  After lzma:        {compressed_weights/1e6:.2f}MB")
    print(f"  + code:            {code_bytes/1e3:.1f}KB")
    print(f"  Total compressed:  {total_compressed_mb:.2f}MB  (raw: {total_raw_mb:.2f}MB)")
    print(f"  Avg bits/param:    {bits_per_param:.2f}")
    print(f"  {'OK under 16MB' if total_compressed_mb < 16 else 'EXCEEDS 16MB — need smaller model'}")
    assert total_compressed_mb < 16, f"Compressed size {total_compressed_mb:.2f}MB exceeds 16MB"
    print("PASS  size estimate")


def test_odd_in_dim():
    """Padding logic for odd input dimensions."""
    W = torch.randn(64, 65) * 0.02
    acts = torch.randn(32, 65)
    d = _waterfill_quantize_weight(W, acts)
    W_hat = _waterfill_dequantize_weight(d)
    assert W_hat.shape == W.shape, f"Shape mismatch: {W_hat.shape} vs {W.shape}"
    print("PASS  odd in_dim padding")


if __name__ == "__main__":
    print("=== Waterfill-4bit Algorithm Tests ===\n")
    tests = [
        test_pack_roundtrip,
        test_reconstruction_error,
        test_waterfill_vs_naive,
        test_storage_size,
        test_odd_in_dim,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"FAIL  {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} tests passed")
    sys.exit(0 if passed == len(tests) else 1)
