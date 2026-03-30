#!/usr/bin/env python3
"""
Example usage of WaterSIC quantization in the parameter-golf framework.

This script demonstrates how to configure and run training with WaterSIC quantization.
"""

import os
import subprocess
import sys

def set_environment_variables():
    """Set up environment variables for WaterSIC training."""
    
    # Core WaterSIC settings
    env_vars = {
        # Enable WaterSIC quantization
        'WATERSIC_ENABLED': '1',
        'WATERSIC_BITS': '4.0',                      # Target 4 bits per parameter
        'WATERSIC_CALIBRATION_SAMPLES': '512',       # Use 512 samples for calibration
        
        # Algorithm features (all enabled for best quality)
        'WATERSIC_LMMSE_CORRECTION': '1',             # Enable LMMSE bias correction
        'WATERSIC_ACTIVATION_DRIFT': '1',             # Enable activation drift correction
        'WATERSIC_RESIDUAL_CORRECTION': '1',          # Enable residual stream correction
        'WATERSIC_ATTENTION_WEIGHTED': '1',           # Enable attention-weighted calibration
        'WATERSIC_ADAPTIVE_MIXING': '1',              # Enable adaptive mixing
        'WATERSIC_DIAGONAL_RESCALERS': '1',           # Enable diagonal rescaler optimization
        
        # Fine-tuning parameters
        'WATERSIC_QR_EPSILON': '0.1',                 # Quantization replacement mixing factor
        'WATERSIC_AW_EPSILON': '0.5',                 # Attention weighting mixing factor
        
        # Enable quantization-aware training
        'QAT_ENABLED': '1',
        'LATE_QAT_THRESHOLD': '0.15',                 # Enable QAT when LR drops below 15%
        
        # Standard parameter-golf settings (adapt as needed)
        'NUM_LAYERS': '11',
        'MODEL_DIM': '512',
        'NUM_HEADS': '8',
        'NUM_KV_HEADS': '4',
        'MLP_MULT': '3.0',
        'ITERATIONS': '7500',                         # Shorter run for testing
        'SEED': '1337',
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
        print(f"Set {key}={value}")
    
    return env_vars

def run_training():
    """Run the training with WaterSIC enabled."""
    print("\n" + "="*60)
    print("🚀 Starting Parameter Golf Training with WaterSIC")
    print("="*60)
    
    try:
        # Run the training script
        result = subprocess.run([
            sys.executable, 'train_gpt.py'
        ], capture_output=True, text=True, timeout=3600)  # 1 hour timeout
        
        if result.returncode == 0:
            print("✅ Training completed successfully!")
            print("\nTraining output:")
            print(result.stdout[-2000:])  # Show last 2000 characters
        else:
            print("❌ Training failed!")
            print("Error output:")
            print(result.stderr[-1000:])  # Show last 1000 characters of error
            
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("⏰ Training timed out after 1 hour")
        return False
    except FileNotFoundError:
        print("❌ train_gpt.py not found! Make sure you're in the correct directory.")
        return False

def validate_results():
    """Check that WaterSIC quantization worked correctly."""
    print("\n" + "="*60)
    print("🔍 Validating WaterSIC Results")
    print("="*60)
    
    # Check for WaterSIC output files
    expected_files = [
        'final_model.watersic.ptz',  # WaterSIC quantized model
        'logs/',                     # Training logs
    ]
    
    found_files = []
    missing_files = []
    
    for file_path in expected_files:
        if os.path.exists(file_path):
            found_files.append(file_path)
            if file_path.endswith('.ptz'):
                size = os.path.getsize(file_path)
                print(f"✅ Found {file_path} ({size:,} bytes)")
        else:
            missing_files.append(file_path)
    
    # Check log files for WaterSIC messages
    log_dir = 'logs'
    if os.path.exists(log_dir):
        log_files = [f for f in os.listdir(log_dir) if f.endswith('.txt')]
        if log_files:
            latest_log = os.path.join(log_dir, log_files[-1])
            with open(latest_log, 'r') as f:
                log_content = f.read()
                
            watersic_indicators = [
                'WaterSIC',
                'watersic',
                'waterfilling',
                'ZSIC',
                'LMMSE',
            ]
            
            found_indicators = [ind for ind in watersic_indicators if ind in log_content]
            
            if found_indicators:
                print(f"✅ Found WaterSIC indicators in log: {found_indicators}")
                
                # Extract key metrics
                lines = log_content.split('\n')
                watersic_lines = [line for line in lines if 'watersic' in line.lower()]
                
                print("\n📊 WaterSIC Metrics:")
                for line in watersic_lines[-5:]:  # Show last 5 WaterSIC-related lines
                    print(f"   {line.strip()}")
            else:
                print("⚠️ No WaterSIC indicators found in log files")
    
    # Summary
    success = len(found_files) > 0 and len(missing_files) == 0
    
    if success:
        print("\n🎉 WaterSIC validation passed!")
    else:
        print(f"\n⚠️ Validation issues: missing files {missing_files}")
    
    return success

def compare_with_baseline():
    """Compare WaterSIC results with baseline int6 if available."""
    print("\n" + "="*60)
    print("📈 Comparing WaterSIC with Baseline")
    print("="*60)
    
    watersic_file = 'final_model.watersic.ptz'
    int6_file = 'final_model.int6.ptz'
    
    if os.path.exists(watersic_file) and os.path.exists(int6_file):
        watersic_size = os.path.getsize(watersic_file)
        int6_size = os.path.getsize(int6_file)
        
        compression_improvement = (int6_size - watersic_size) / int6_size * 100
        
        print(f"📁 File sizes:")
        print(f"   WaterSIC: {watersic_size:,} bytes")
        print(f"   Int6 baseline: {int6_size:,} bytes")
        print(f"   Improvement: {compression_improvement:+.1f}%")
        
        if compression_improvement > 0:
            print("✅ WaterSIC achieved better compression!")
        else:
            print("ℹ️ WaterSIC used more space (possibly due to metadata overhead)")
    else:
        print("ℹ️ Cannot compare - missing baseline or WaterSIC file")

def main():
    """Main execution flow."""
    print("WaterSIC Training Example")
    print("="*40)
    
    # Check that we're in the right directory
    if not os.path.exists('train_gpt.py'):
        print("❌ Error: train_gpt.py not found!")
        print("Please run this script from the parameter-golf training directory.")
        return 1
    
    # Set up environment
    print("\n1. Setting up environment variables...")
    env_vars = set_environment_variables()
    
    # Run training
    print("\n2. Running training...")
    training_success = run_training()
    
    if not training_success:
        print("Training failed. Check the error messages above.")
        return 1
    
    # Validate results
    print("\n3. Validating results...")
    validation_success = validate_results()
    
    # Compare with baseline
    print("\n4. Comparing with baseline...")
    compare_with_baseline()
    
    # Final summary
    print("\n" + "="*60)
    if training_success and validation_success:
        print("🎉 WaterSIC training completed successfully!")
        print("\nNext steps:")
        print("• Analyze the compression ratios and model quality")
        print("• Compare bits-per-byte (BPB) with the baseline")
        print("• Experiment with different WATERSIC_BITS settings")
        print("• Try different calibration sample sizes")
    else:
        print("⚠️ Some issues were encountered. Check the logs for details.")
    
    return 0 if training_success and validation_success else 1

if __name__ == '__main__':
    sys.exit(main())