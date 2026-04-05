#!/usr/bin/env python3
"""
Corrected WaterSIC Size Calculator - accounts for actual file structure
"""

def analyze_current_model():
    """Analyze actual model file to understand size breakdown"""
    
    # Current config from train_gpt.py
    config = {
        'num_layers': 11,
        'model_dim': 512, 
        'num_heads': 8,
        'num_kv_heads': 4,
        'mlp_mult': 3.0,
        'vocab_size': 1024
    }
    
    # Calculate actual parameters
    embed_params = config['vocab_size'] * config['model_dim']  # 1024 * 512 = 524K
    
    # Per layer calculations
    d = config['model_dim']
    mlp_d = int(d * config['mlp_mult'])
    
    # Attention weights (Q, K, V, O)
    attn_params = d * d + d * d + d * d + d * d  # Q, K, V, O projections
    
    # MLP weights (up, down) 
    mlp_params = d * mlp_d + mlp_d * d  # up + down
    
    # Layer norms (minimal)
    norm_params = d * 2  # 2 norms per layer
    
    layer_params = attn_params + mlp_params + norm_params
    total_layer_params = layer_params * config['num_layers']
    
    # Output head (often tied to embeddings)
    head_params = config['model_dim'] * config['vocab_size']
    
    total_params = embed_params + total_layer_params + head_params
    
    print(f"🔍 Parameter Breakdown:")
    print(f"   Embeddings: {embed_params:,} params ({embed_params*2/1024/1024:.2f}MB @ fp16)")
    print(f"   Layers ({config['num_layers']}): {total_layer_params:,} params")
    print(f"     - Per layer: {layer_params:,} params")  
    print(f"     - Attention: {attn_params:,} params")
    print(f"     - MLP: {mlp_params:,} params")
    print(f"   Output head: {head_params:,} params ({head_params*2/1024/1024:.2f}MB @ fp16)")
    print(f"   TOTAL: {total_params:,} params")
    print()
    
    # Size with different quantizations
    print(f"📏 Size Analysis:")
    
    # WaterSIC: only quantize layer weights, others stay fp16
    watersic_layer_bits = total_layer_params * 4  # 4 bits per param
    watersic_other_bits = (embed_params + head_params) * 16  # fp16
    watersic_total_mb = (watersic_layer_bits + watersic_other_bits) / 8 / 1024 / 1024
    
    # int6: quantize layer weights to 6 bits
    int6_layer_bits = total_layer_params * 6
    int6_other_bits = (embed_params + head_params) * 16  # fp16
    int6_total_mb = (int6_layer_bits + int6_other_bits) / 8 / 1024 / 1024
    
    print(f"   WaterSIC (4-bit layers): {watersic_total_mb:.2f}MB")
    print(f"   int6 (6-bit layers): {int6_total_mb:.2f}MB")
    print(f"   Compression ratio: {int6_total_mb/watersic_total_mb:.1f}x")
    print()
    
    # Calculate room for expansion
    target_mb = 16.0
    remaining_mb = target_mb - watersic_total_mb
    
    print(f"🎯 Expansion Analysis:")
    print(f"   Target limit: {target_mb}MB")
    print(f"   WaterSIC current: {watersic_total_mb:.2f}MB") 
    print(f"   Remaining room: {remaining_mb:.2f}MB")
    print()
    
    # How many more parameters can we fit?
    remaining_bits = remaining_mb * 1024 * 1024 * 8
    additional_layer_params = remaining_bits / 4  # 4 bits per WaterSIC param
    
    print(f"💡 Expansion Potential:")
    print(f"   Additional layer params: {additional_layer_params:,.0f}")
    print(f"   Current layer params: {total_layer_params:,}")
    print(f"   Possible scaling: {1 + additional_layer_params/total_layer_params:.1f}x")
    
    # Suggest specific expansions
    print(f"\n🚀 Suggested Expansions (staying under 16MB):")
    
    # Try increasing layers
    max_layers = int(config['num_layers'] * (1 + additional_layer_params/total_layer_params))
    if max_layers > config['num_layers']:
        print(f"   Layers: {config['num_layers']} → {max_layers} (+{max_layers-config['num_layers']})")
    
    # Try increasing MLP
    max_mlp_mult = config['mlp_mult'] * (1 + additional_layer_params/(mlp_params * config['num_layers']))
    if max_mlp_mult > config['mlp_mult']:
        print(f"   MLP mult: {config['mlp_mult']} → {max_mlp_mult:.1f}")
    
    # Try a balanced approach
    balanced_scale = (1 + additional_layer_params/total_layer_params) ** 0.5
    balanced_layers = int(config['num_layers'] * balanced_scale)
    balanced_mlp = config['mlp_mult'] * balanced_scale
    
    if balanced_scale > 1.1:
        print(f"   Balanced: L{balanced_layers} MLP{balanced_mlp:.1f} ({balanced_scale:.1f}x scale)")

if __name__ == "__main__":
    print("🌊 WaterSIC Model Expansion Analysis")
    print("="*50)
    analyze_current_model()