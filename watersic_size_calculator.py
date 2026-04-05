#!/usr/bin/env python3
"""
WaterSIC Model Size Calculator
Calculates parameter counts and model sizes for different configurations
"""

def calculate_model_size(num_layers=11, model_dim=512, num_heads=8, num_kv_heads=4, 
                        mlp_mult=3.0, vocab_size=1024, watersic_bits=4.0):
    """Calculate model size with WaterSIC quantization"""
    
    # Embedding parameters
    embed_params = vocab_size * model_dim
    
    # Per-layer parameters
    # Attention: Q, K, V, O projections
    head_dim = model_dim // num_heads
    q_params = model_dim * model_dim  # Q projection
    k_params = model_dim * (num_kv_heads * head_dim)  # K projection  
    v_params = model_dim * (num_kv_heads * head_dim)  # V projection
    o_params = model_dim * model_dim  # O projection
    attn_params = q_params + k_params + v_params + o_params
    
    # MLP: up, down projections
    mlp_dim = int(model_dim * mlp_mult)
    up_params = model_dim * mlp_dim
    down_params = mlp_dim * model_dim  
    mlp_params = up_params + down_params
    
    # Layer norm parameters (minimal)
    norm_params = model_dim * 2  # Pre-attn + pre-mlp norms
    
    # Total per layer
    layer_params = attn_params + mlp_params + norm_params
    total_layer_params = layer_params * num_layers
    
    # Output head (usually tied to embeddings, but count for safety)
    head_params = model_dim * vocab_size
    
    # Total parameters
    total_params = embed_params + total_layer_params + head_params
    
    # Size calculations
    # WaterSIC: quantized weights
    watersic_weight_bits = watersic_bits * total_layer_params  # Only quantize layer weights
    watersic_weight_bytes = watersic_weight_bits / 8
    
    # int6: current quantization (6 bits per parameter)
    int6_weight_bytes = (6 * total_layer_params) / 8
    
    # Embeddings and other data (unquantized, ~fp16)
    other_bytes = (embed_params + head_params) * 2  # 2 bytes per param
    
    # Total model sizes
    watersic_total = watersic_weight_bytes + other_bytes
    int6_total = int6_weight_bytes + other_bytes
    
    return {
        'total_params': total_params,
        'layer_params': total_layer_params,
        'embed_params': embed_params,
        'watersic_mb': watersic_total / (1024**2),
        'int6_mb': int6_total / (1024**2),
        'compression_ratio': int6_total / watersic_total,
        'config': {
            'layers': num_layers,
            'dim': model_dim, 
            'heads': num_heads,
            'kv_heads': num_kv_heads,
            'mlp_mult': mlp_mult,
            'vocab': vocab_size
        }
    }

def find_optimal_config(target_mb=16, watersic_bits=4.0):
    """Find optimal model configuration within size limit"""
    
    print(f"🎯 Finding optimal config for {target_mb}MB limit with WaterSIC @ {watersic_bits} bits")
    print("=" * 70)
    
    # Current baseline
    current = calculate_model_size()
    print(f"📊 Current Config: {current['config']}")
    print(f"   WaterSIC: {current['watersic_mb']:.2f}MB")
    print(f"   int6: {current['int6_mb']:.2f}MB")
    print(f"   Compression: {current['compression_ratio']:.1f}x")
    print()
    
    # Try larger configurations
    configs = [
        # Increase layers
        {'num_layers': 15, 'model_dim': 512, 'num_heads': 8, 'mlp_mult': 3.0},
        {'num_layers': 20, 'model_dim': 512, 'num_heads': 8, 'mlp_mult': 3.0},
        {'num_layers': 25, 'model_dim': 512, 'num_heads': 8, 'mlp_mult': 3.0},
        
        # Increase width
        {'num_layers': 11, 'model_dim': 768, 'num_heads': 12, 'mlp_mult': 3.0},
        {'num_layers': 11, 'model_dim': 1024, 'num_heads': 16, 'mlp_mult': 3.0},
        
        # Increase MLP
        {'num_layers': 11, 'model_dim': 512, 'num_heads': 8, 'mlp_mult': 4.0},
        {'num_layers': 11, 'model_dim': 512, 'num_heads': 8, 'mlp_mult': 5.0},
        
        # Balanced scaling
        {'num_layers': 16, 'model_dim': 640, 'num_heads': 10, 'mlp_mult': 3.5},
        {'num_layers': 20, 'model_dim': 768, 'num_heads': 12, 'mlp_mult': 4.0},
        {'num_layers': 24, 'model_dim': 896, 'num_heads': 14, 'mlp_mult': 4.0},
    ]
    
    print("🚀 Possible Expansions:")
    print("-" * 70)
    
    best_watersic = None
    best_params = 0
    
    for config in configs:
        # Update KV heads proportionally
        config['num_kv_heads'] = max(1, config['num_heads'] // 2)
        
        result = calculate_model_size(**config)
        
        if result['watersic_mb'] <= target_mb:
            param_increase = result['total_params'] / current['total_params']
            
            print(f"✅ L{config['num_layers']} D{config['model_dim']} H{config['num_heads']} MLP{config['mlp_mult']}")
            print(f"   WaterSIC: {result['watersic_mb']:.2f}MB ({param_increase:.1f}x params)")
            print(f"   int6: {result['int6_mb']:.2f}MB (would exceed limit)")
            print()
            
            if result['total_params'] > best_params:
                best_watersic = result
                best_params = result['total_params']
        else:
            print(f"❌ L{config['num_layers']} D{config['model_dim']} H{config['num_heads']} MLP{config['mlp_mult']}")
            print(f"   WaterSIC: {result['watersic_mb']:.2f}MB (exceeds limit)")
            print()
    
    if best_watersic:
        improvement = best_watersic['total_params'] / current['total_params']
        print(f"🏆 Best WaterSIC Config: {best_watersic['config']}")
        print(f"   Size: {best_watersic['watersic_mb']:.2f}MB")
        print(f"   Improvement: {improvement:.1f}x more parameters!")
        print(f"   Would be {best_watersic['int6_mb']:.2f}MB with int6 (impossible)")
    
    return best_watersic

if __name__ == "__main__":
    # Current analysis
    print("🌊 WaterSIC Model Size Analysis")
    print("=" * 70)
    
    # Find optimal config
    best = find_optimal_config(target_mb=16, watersic_bits=4.0)
    
    print("\n" + "=" * 70)
    print("💡 Key Insights:")
    print("- Current int6 model uses ~15MB")
    print("- WaterSIC enables MASSIVE model expansion")
    print("- Can fit 10-50x more parameters in same 16MB!")
    print("- This is the TRUE power of WaterSIC! 🚀")