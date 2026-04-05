#!/usr/bin/env python3
"""
RunPod Results Analyzer - Parse and summarize WaterSIC test results
"""
import re
import os
from pathlib import Path

def parse_log_file(log_path):
    """Parse training log and extract key metrics"""
    if not os.path.exists(log_path):
        return {"error": f"Log file not found: {log_path}"}
    
    with open(log_path, 'r') as f:
        content = f.read()
    
    results = {}
    
    # Extract final loss
    loss_matches = re.findall(r'loss:\s*([\d.]+)', content)
    if loss_matches:
        results['final_loss'] = float(loss_matches[-1])
    
    # Extract validation loss
    val_matches = re.findall(r'val_loss:\s*([\d.]+)', content)
    if val_matches:
        results['val_loss'] = float(val_matches[-1])
    
    # Extract compression info
    compression_matches = re.findall(r'compression.*?(\d+\.?\d*)', content, re.IGNORECASE)
    if compression_matches:
        results['compression_ratio'] = compression_matches[-1]
    
    # Look for WaterSIC specific info
    watersic_matches = re.findall(r'watersic.*', content, re.IGNORECASE)
    results['watersic_info'] = watersic_matches[-3:] if watersic_matches else []
    
    # Check for errors
    errors = re.findall(r'(error|failed|exception).*', content, re.IGNORECASE)
    results['errors'] = errors[-5:] if errors else []
    
    # Training completion status
    if 'training complete' in content.lower():
        results['status'] = 'completed'
    elif 'exceed' in content.lower() or 'limit' in content.lower():
        results['status'] = 'size_exceeded'
    elif errors:
        results['status'] = 'failed'
    else:
        results['status'] = 'unknown'
    
    return results

def find_model_files(directory):
    """Find and analyze model files"""
    if not os.path.exists(directory):
        return []
    
    model_files = []
    for file_path in Path(directory).rglob("*.ptz"):
        size_bytes = file_path.stat().st_size
        size_mb = size_bytes / (1024 * 1024)
        
        model_files.append({
            'path': str(file_path),
            'size_bytes': size_bytes,
            'size_mb': round(size_mb, 2),
            'within_limit': size_mb <= 16.0
        })
    
    return sorted(model_files, key=lambda x: x['size_mb'], reverse=True)

def analyze_results():
    """Analyze all RunPod test results"""
    
    print("🔍 RunPod WaterSIC Test Results Analysis")
    print("=" * 50)
    
    # Check for different result directories
    test_dirs = [
        'h100_baseline_results',
        'h100_watersic_results', 
        'h100_expanded_results',
        'baseline_results',
        'watersic_results'
    ]
    
    found_results = False
    
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            found_results = True
            print(f"\n📁 {test_dir.upper()}")
            print("-" * 30)
            
            # Parse log file
            log_file = os.path.join(test_dir, 'training_log.txt')
            log_results = parse_log_file(log_file)
            
            print(f"📊 Training Status: {log_results.get('status', 'unknown')}")
            
            if 'final_loss' in log_results:
                print(f"🎯 Final Loss: {log_results['final_loss']:.4f}")
            
            if 'val_loss' in log_results:
                print(f"✅ Validation Loss: {log_results['val_loss']:.4f}")
            
            if log_results.get('compression_ratio'):
                print(f"🗜️  Compression: {log_results['compression_ratio']}")
            
            if log_results.get('watersic_info'):
                print("🌊 WaterSIC Info:")
                for info in log_results['watersic_info']:
                    print(f"   {info}")
            
            if log_results.get('errors'):
                print("⚠️  Errors:")
                for error in log_results['errors']:
                    print(f"   {error}")
            
            # Find model files
            model_files = find_model_files(test_dir)
            if model_files:
                print("📏 Model Files:")
                for model in model_files:
                    status = "✅" if model['within_limit'] else "❌" 
                    print(f"   {status} {model['size_mb']:.2f}MB - {Path(model['path']).name}")
            else:
                print("❌ No model files found")
    
    if not found_results:
        print("❌ No result directories found!")
        print("Expected directories:", test_dirs)
        print("\nCurrent directory contents:")
        for item in os.listdir('.'):
            print(f"   {item}")
    
    print("\n" + "=" * 50)
    print("📋 Summary:")
    print("- Run this script after RunPod testing completes")
    print("- Upload results to complete the official test report")
    print("- Expected: WaterSIC enables larger models within 16MB limit")

if __name__ == "__main__":
    analyze_results()